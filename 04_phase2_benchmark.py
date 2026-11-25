"""
04_phase2_benchmark.py
=============================================================================
Phase II — Multi-Seed, Parameter-Parity Comparative Benchmark
=============================================================================
Evaluates KAN, MLP, and XGBoost across ten imbalanced KEEL datasets under
three training conditions: baseline, SMOTE-Tomek resampling, and Focal Loss.

This script is self-contained: it does not read the Phase I JSON files.
The KAN architecture (grid=7, k=3, n_layers=1, width=10) was inferred from
the Phase I grid sensitivity sweep and is fixed here as a global constant.

"""

import os
import gc
import time
import warnings
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
from imblearn.combine import SMOTETomek
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from scipy.stats import friedmanchisquare, wilcoxon
from kan import KAN

# ===================== COMPATIBILITY PATCH =====================

def patch_imbalance_xgboost():
    """
    Patches imxgboost for XGBoost >= 2.0 compatibility.
   
    """
    _orig = xgb.train

    def _patched(*args, **kwargs):
        if 'feval' in kwargs:
            kwargs['custom_metric'] = kwargs.pop('feval')
        return _orig(*args, **kwargs)

    xgb.train = _patched
    print("[Patch] imbalance-xgboost compatibility applied.")


# ===================== A100 MIG HARDWARE SETTINGS =====================
# Applied symmetrically to KAN and MLP — does not bias the comparison.

torch.backends.cudnn.benchmark    = True    
torch.backends.cudnn.deterministic = False 

USE_AMP   = True
AMP_DTYPE = torch.bfloat16 

# ===================== GLOBAL CONFIGURATION =====================

SEEDS      = [42, 123, 999, 7, 21]
EPOCHS     = 100
BATCH_SIZE = 64    # applied equally to KAN and MLP
NUM_WORKERS = 0    

KEEL_DATASET_DIR = Path("keel_datasets")   
RESULTS_DIR      = Path("final_benchmark_results")
PLOT_DIR         = RESULTS_DIR / "plots"

for d in [RESULTS_DIR, PLOT_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# KAN architecture: inferred from Phase I grid sensitivity sweep.
GLOBAL_KAN_CONFIG = {
    'grid'          : 7,
    'k'             : 3,
    'n_layers'      : 1,
    'layer_0_width' : 10
}

HYPERPARAMS = {
    'KAN': {'lr': 0.0005, 'weight_decay': 1e-4},
    'MLP': {'lr': 0.001,  'weight_decay': 1e-5},
    'XGB': {'lr': 0.1}
}

STRATEGY_PALETTE = {
    'baseline' : '#4C72B0',
    'resampled': '#55A868',
    'focal'    : '#DD8452'
}

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hardware] Device : {device}")
if device.type == 'cuda':
    print(f"[Hardware] GPU    : {torch.cuda.get_device_name(0)}")
    print(f"[Hardware] VRAM   : "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ===================== LOSS FUNCTIONS =====================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    """
    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(inputs, targets, reduction='none')
        pt   = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ===================== PARAMETER-PARITY UTILITIES =====================

def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters, including KAN B-spline coefficients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_mlp_width(input_dim, target_params: int,
                        n_layers: int = 1) -> int:
    """
    Derives MLP hidden width W such that total params ≈ target_params.

    n_layers=1 (closed-form, Eq. 1 in paper):
        W = (target_params - D_out) / (d_in + D_out + 1)

    n_layers>1 (quadratic, Eq. 2 in paper):
        (n_layers-1)*W^2 + W*(d_in + D_out + n_layers) + D_out = target_params

    Minimum width = 8 prevents degenerate architectures.
    """
    in_d  = int(input_dim[0] if isinstance(input_dim,
                (torch.Size, tuple, list)) else input_dim)
    out_d = 2

    if n_layers == 1:
        width = int((target_params - out_d) / (in_d + out_d + 1))
    else:
        a    = n_layers - 1
        b    = in_d + out_d + n_layers
        c    = out_d - target_params
        disc = b ** 2 - 4 * a * c
        width = 8 if disc < 0 else int((-b + np.sqrt(disc)) / (2 * a))

    return max(8, width)


class MLP(nn.Module):
    """Standard ReLU-MLP. Width derived analytically for parameter parity."""
    def __init__(self, input_dim, width: int, n_layers: int = 1):
        super().__init__()
        layers  = []
        curr_in = int(input_dim[0] if isinstance(input_dim,
                      (torch.Size, tuple, list)) else input_dim)
        for _ in range(n_layers):
            layers += [nn.Linear(curr_in, width), nn.ReLU()]
            curr_in  = width
        layers.append(nn.Linear(curr_in, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===================== DATA LOADING =====================

def load_keel_dataset(dataset_path: Path):
    """
    Parses KEEL .dat files 
    """
    try:
        with open(dataset_path, 'r') as f:
            lines = f.readlines()

        attr_names, data_idx = [], 0
        for i, line in enumerate(lines):
            if line.lower().startswith("@attribute"):
                attr_names.append(line.split()[1].strip())
            if line.lower().startswith("@data"):
                data_idx = i + 1
                break

        data = [l.strip().split(',') for l in lines[data_idx:] if l.strip()]
        df   = pd.DataFrame(data, columns=attr_names)

        for col in df.columns[:-1]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        le             = LabelEncoder()
        y              = le.fit_transform(df.iloc[:, -1])
        counts         = np.bincount(y)
        minority_class = np.argmin(counts)
        y              = np.where(y == minority_class, 1, 0)

        X  = df.iloc[:, :-1].values.astype(np.float32)
        ir = max(counts) / min(counts)
        return X, y, ir

    except Exception as e:
        print(f"  [Error] {dataset_path.name}: {e}")
        return None, None, None


# ===================== MEMORY TRACKING =====================

def reset_gpu_stats():
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_mb() -> float:
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


# ===================== PARITY DIAGNOSTICS =====================

def log_parity_check(dataset_name: str, X: np.ndarray):
    """Prints KAN vs MLP parameter counts. Run once per dataset at startup."""
    kan = KAN(width=[X.shape[1], GLOBAL_KAN_CONFIG['layer_0_width'], 2],
              grid=GLOBAL_KAN_CONFIG['grid'], k=GLOBAL_KAN_CONFIG['k'])
    target = count_parameters(kan)
    width  = calculate_mlp_width(X.shape[1], target, n_layers=1)
    mlp    = MLP(X.shape[1], width, n_layers=1)
    mlp_p  = count_parameters(mlp)
    ratio  = mlp_p / target if target > 0 else 0
    print(f"  [Parity] {dataset_name}: KAN={target:,}  MLP={mlp_p:,}  "
          f"ratio={ratio:.3f}  MLP_width={width}")


# ===================== CORE EXPERIMENT =====================

def run_experiment(X: np.ndarray, y: np.ndarray,
                   model_type: str, strategy: str,
                   dataset_name: str, seed: int) -> dict:
    """
    5-fold stratified CV for one (model, strategy, seed) configuration.
    """
    skf          = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_metrics = []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr_raw, y_te     = y[tr_idx], y[te_idx]

        
        sc          = StandardScaler()
        X_tr_sc     = sc.fit_transform(X_tr_raw)
        X_te_sc     = sc.transform(X_te_raw)

       
        if strategy == "resampled":
            X_tr, y_tr = SMOTETomek(random_state=seed).fit_resample(
                X_tr_sc, y_tr_raw)
        else:
            X_tr, y_tr = X_tr_sc, y_tr_raw

        
        focal_alpha = int(np.sum(y_tr == 1)) / len(y_tr)

        criterion = (FocalLoss(alpha=focal_alpha, gamma=2.0)
                     if strategy == "focal"
                     else nn.CrossEntropyLoss())

        reset_gpu_stats()
        tracemalloc.start()
        t0 = time.perf_counter()

        # ── KAN / MLP ────────────────────────────────────────────────────────
        if model_type in ["KAN", "MLP"]:

            scaler_amp = GradScaler() if (USE_AMP and device.type == 'cuda') \
                         else None

            loader = DataLoader(
                TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                              torch.tensor(y_tr, dtype=torch.long)),
                batch_size         = BATCH_SIZE,
                shuffle            = True,
                num_workers        = NUM_WORKERS,
                pin_memory         = (device.type == 'cuda'),
                persistent_workers = False
            )

            if model_type == "KAN":
                model = KAN(
                    width = [X.shape[1], GLOBAL_KAN_CONFIG['layer_0_width'], 2],
                    grid  = GLOBAL_KAN_CONFIG['grid'],
                    k     = GLOBAL_KAN_CONFIG['k']
                ).to(device)
            else:
                kan_ref       = KAN(
                    width = [X.shape[1], GLOBAL_KAN_CONFIG['layer_0_width'], 2],
                    grid  = GLOBAL_KAN_CONFIG['grid'],
                    k     = GLOBAL_KAN_CONFIG['k']
                )
                target_params = count_parameters(kan_ref)
                width         = calculate_mlp_width(X.shape[1], target_params)
                del kan_ref
                model = MLP(X.shape[1], width).to(device)

            opt = optim.AdamW(model.parameters(),
                             lr=HYPERPARAMS[model_type]['lr'],
                             weight_decay=HYPERPARAMS[model_type]['weight_decay'])

            best_loss, patience_ctr, PATIENCE = float('inf'), 0, 5

            for _ in range(EPOCHS):
                model.train()
                epoch_loss = 0.0
                for bX, by in loader:
                    bX = bX.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)

                    if USE_AMP and device.type == 'cuda':
                        with autocast(dtype=AMP_DTYPE):
                            loss = criterion(model(bX), by)
                        scaler_amp.scale(loss).backward()
                        scaler_amp.step(opt)
                        scaler_amp.update()
                    else:
                        loss = criterion(model(bX), by)
                        loss.backward()
                        opt.step()
                    epoch_loss += loss.item()

                avg = epoch_loss / len(loader)
                if avg < best_loss:
                    best_loss, patience_ctr = avg, 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= PATIENCE:
                        break

            model.eval()
            with torch.no_grad():
                t_in = torch.tensor(X_te_sc, dtype=torch.float32, device=device)
                if USE_AMP and device.type == 'cuda':
                    with autocast(dtype=AMP_DTYPE):
                        logits = model(t_in)
                else:
                    logits = model(t_in)
                
                y_pred = logits.argmax(dim=1).cpu().numpy()
                y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        # ── XGBoost ──────────────────────────────────────────────────────────
        elif model_type == "XGB":
            np.random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)

            if strategy == "focal":
                xgb_model = imb_xgb(
                    special_objective = 'focal',
                    imbalance_alpha   = focal_alpha,
                    focal_gamma       = 2.0,
                    num_round         = 100,
                    eta               = HYPERPARAMS['XGB']['lr'],
                    eval_metric       = 'logloss'
                )
            else:
                xgb_model = imb_xgb(
                    num_round   = 100,
                    eta         = HYPERPARAMS['XGB']['lr'],
                    eval_metric = 'logloss'
                )

            xgb_model.fit(X_tr, y_tr)

            y_pred = (xgb_model.predict(X_te_sc) > 0.5).astype(int)
            dtest   = xgb.DMatrix(X_te_sc)
            margins = xgb_model.boosting_model.predict(
                dtest, output_margin=True)
            y_prob  = 1.0 / (1.0 + np.exp(-margins))

        # ── Metrics ───────────────────────────────────────────────────────────
        elapsed      = time.perf_counter() - t0
        _, ram_peak  = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        fold_metrics.append({
            'gmean'        : geometric_mean_score(y_te, y_pred),
            'f1'           : f1_score(y_te, y_pred, zero_division=0),
            'auc'          : roc_auc_score(y_te, y_prob),
            'balanced_acc' : balanced_accuracy_score(y_te, y_pred),
            'time'         : elapsed,
            'ram_mb'       : ram_peak / (1024 ** 2),
            'gpu_mb'       : peak_gpu_mb()
        })

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    df_f    = pd.DataFrame(fold_metrics)
    m_gmean = df_f['gmean'].mean()
    s_gmean = df_f['gmean'].std(ddof=1)

    return {
        'model'             : model_type,
        'strategy'          : strategy,
        'dataset'           : dataset_name,
        'gmean_mean'        : m_gmean,
        'gmean_std'         : s_gmean,
        'gmean_cv'          : s_gmean / m_gmean if m_gmean > 0 else 0,
        'f1_mean'           : df_f['f1'].mean(),
        'auc_mean'          : df_f['auc'].mean(),
        'balanced_acc_mean' : df_f['balanced_acc'].mean(),
        'time_avg'          : df_f['time'].mean(),
        'ram_memory_avg_mb' : df_f['ram_mb'].mean(),
        'gpu_memory_avg_mb' : df_f['gpu_mb'].mean()
    }


# ===================== SEED AGGREGATION =====================

def aggregate_seed_results(seed_results: list) -> dict:
    """Collapses 5-seed trials into one averaged record."""
    df   = pd.DataFrame(seed_results)
    base = seed_results[0]
    mu   = df['gmean_mean'].mean()
    return {
        'model'             : base['model'],
        'strategy'          : base['strategy'],
        'dataset'           : base['dataset'],
        'IR'                : base.get('IR', 0),
        'gmean_mean'        : mu,
        'gmean_std'         : df['gmean_mean'].std(ddof=1),
        'gmean_cv'          : df['gmean_mean'].std(ddof=1) / mu if mu > 0 else 0,
        'f1_mean'           : df['f1_mean'].mean(),
        'auc_mean'          : df['auc_mean'].mean(),
        'balanced_acc_mean' : df['balanced_acc_mean'].mean(),
        'time_avg'          : df['time_avg'].mean(),
        'ram_memory_avg_mb' : df['ram_memory_avg_mb'].mean(),
        'gpu_memory_avg_mb' : df['gpu_memory_avg_mb'].mean()
    }


# ===================== STATISTICAL TESTS =====================

def _pooled_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pooled Cohen's d (independent samples).
    
    """
    nx, ny = len(x), len(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pool   = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx+ny-2))
    return float((np.mean(x) - np.mean(y)) / pool) if pool > 0 else 0.0


def _dz(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d_z for paired observations.
    
    """
    diff = np.asarray(x) - np.asarray(y)
    sd   = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def _effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:   return "Negligible"
    if a < 0.5:   return "Small"
    if a < 0.8:   return "Medium"
    return "Large"


def perform_statistical_tests(df: pd.DataFrame):
    """
    Two-stage non-parametric statistical suite.


    """
    print("\n  [Stats] Running statistical suite...")

    df = df.copy()
    df['ms'] = df['model'] + "_" + df['strategy']

    pivot_g = df.pivot(index='dataset', columns='ms',
                       values='gmean_mean').dropna()
    pivot_t = df.pivot(index='dataset', columns='ms',
                       values='time_avg').dropna()

    if pivot_g.empty:
        print("  [Stats] ERROR: insufficient data.")
        return

    stat, p_f = friedmanchisquare(*[pivot_g[c] for c in pivot_g.columns])
    print(f"  [Stats] Friedman χ²={stat:.2f}, p={p_f:.4f}")

    comparisons = [
        ('KAN_baseline', 'KAN_focal',      'G-mean',  pivot_g),
        ('KAN_baseline', 'KAN_resampled',  'G-mean',  pivot_g),
        ('KAN_baseline', 'MLP_baseline',   'G-mean',  pivot_g),
        ('KAN_resampled','MLP_resampled',  'G-mean',  pivot_g),
        ('KAN_resampled','XGB_resampled',  'G-mean',  pivot_g),
        ('KAN_baseline', 'MLP_baseline',   'Latency', pivot_t),
        ('KAN_resampled','MLP_resampled',  'Latency', pivot_t),
    ]

    rows = []
    for a, b, metric, pivot in comparisons:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        x, y = pivot[a].values, pivot[b].values
        _, p  = wilcoxon(x, y)
        dp    = _pooled_d(x, y)
        dz    = _dz(x, y)
        rows.append({
            'Metric'           : metric,
            'Comparison'       : f"{a} vs {b}",
            'p_value'          : round(p, 4),
            'Cohens_d_pooled'  : round(dp, 2),   
            'Effect_Size'      : _effect_label(dz),
            'Significant'      : p < 0.05
        })
        print(f"  [Stats] {a:22s} vs {b:22s} | {metric:7s} | "
              f"p={p:.4f} | d_pool={dp:+.2f} | d_z={dz:+.2f} | "
              f"{'Sig' if p<0.05 else 'Non-sig'}")

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "statistical_outcomes_table.csv", index=False)

    ranks    = pivot_g.rank(axis=1, ascending=False).mean().sort_values()
    ranks_df = ranks.reset_index()
    ranks_df.columns = ['Model_Strategy', 'Mean_Rank']
    ranks_df.to_csv(RESULTS_DIR / "cd_diagram_ranks.csv", index=False)
    print(f"  [Stats] Top ranked: {ranks.index[0]} (R={ranks.iloc[0]:.2f})")


# ===================== TABLE 1 (PAPER FORMAT) =====================

def generate_table1_csv(results_list: list, output_dir: Path):
    """
   csv
    """
    df = pd.DataFrame(results_list)
    rows = []

    for model in ['KAN', 'MLP', 'XGB']:
        for strategy in ['baseline', 'focal', 'resampled']:
            sub = df[(df['model'] == model) & (df['strategy'] == strategy)]
            if sub.empty:
                continue
            gm  = sub['gmean_mean'].values * 100
            f1  = sub['f1_mean'].values    * 100
            auc = sub['auc_mean'].values   * 100
            rows.append({
                'Model'                  : model,
                'Strategy'               : strategy,
                'GMean_MeanPlusMinusStd' : f"{gm.mean():.2f} ± {gm.std(ddof=1):.2f}",
                'GMean_Mean_pct'         : round(gm.mean(), 2),
                'GMean_Std_pct'          : round(gm.std(ddof=1), 2),
                'MacroF1_pct'            : round(f1.mean(), 2),
                'AUCROC_pct'             : round(auc.mean(), 2),
                'Min_GMean_pct'          : round(gm.min(), 2)
            })

    out = output_dir / "table1_global_performance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  [Table 1] {out}")


# ===================== PLOTS =====================

def generate_publication_plots(results_list: list, output_dir: Path):
    df       = pd.DataFrame(results_list)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.2)

    
    df['Configuration'] = df['model'] + " (" + df['strategy'] + ")"
    config_order = [
        'KAN (baseline)', 'KAN (resampled)', 'KAN (focal)',
        'MLP (baseline)', 'MLP (resampled)', 'MLP (focal)',
        'XGB (baseline)', 'XGB (resampled)', 'XGB (focal)'
    ]
    color_mapping = [
        STRATEGY_PALETTE[c.split('(')[1].replace(')', '')]
        for c in config_order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, metric, title in zip(axes,
                                  ['gmean_mean', 'f1_mean'],
                                  ['G-Mean', 'F1-Score']):
        sns.boxplot(data=df, y='Configuration', x=metric,
                    order=config_order, palette=color_mapping,
                    ax=ax, width=0.7)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel("Performance Score", fontweight='bold')
        ax.set_ylabel("")
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_dir / "performance_box_plots.pdf", bbox_inches='tight')
    plt.close()

    metrics_r = ['gmean_mean', 'f1_mean', 'auc_mean']
    kan_r     = (df[df['model'] == 'KAN']
                 .groupby('strategy')[metrics_r].mean().reset_index())
    labels    = ['G-Mean', 'F1', 'AUC']
    angles    = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles   += angles[:1]
    fig, ax   = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for _, row in kan_r.iterrows():
        vals = row[metrics_r].values.tolist() + [row[metrics_r[0]]]
        ax.plot(angles, vals, label=f"KAN ({row['strategy']})",
                linewidth=2, color=STRATEGY_PALETTE[row['strategy']])
        ax.fill(angles, vals, alpha=0.1, color=STRATEGY_PALETTE[row['strategy']])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylim(0, 1)
    plt.title("KAN Strategy Interplay", y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(plot_dir / "kan_strategy_radar.pdf", bbox_inches='tight')
    plt.close()

   
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='time_avg', y='gmean_mean',
                    hue='model', style='strategy',
                    size='gpu_memory_avg_mb', sizes=(100, 1000),
                    alpha=0.6, palette='Set1')
    plt.xscale('log')
    plt.title("Efficiency Frontier: G-Mean vs Training Time",
              fontsize=15, fontweight='bold')
    plt.xlabel("Training Time (s, log scale)", fontweight='bold')
    plt.ylabel("G-Mean", fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig(plot_dir / "resource_bubble_frontier.pdf", bbox_inches='tight')
    plt.close()

    (df.groupby(['model', 'strategy'])
       .agg(gmean_mean    =('gmean_mean',       'mean'),
            gmean_std     =('gmean_std',         'mean'),
            f1_mean       =('f1_mean',           'mean'),
            auc_mean      =('auc_mean',          'mean'),
            time_avg      =('time_avg',          'mean'),
            gpu_mem_avg_mb=('gpu_memory_avg_mb', 'mean'),
            ram_mem_avg_mb=('ram_memory_avg_mb', 'mean'))
       .sort_values('gmean_mean', ascending=False)
       .to_csv(plot_dir / "detailed_summary_table.csv"))

    print(f"  [Plots] Saved to {plot_dir}")


# ===================== MAIN PIPELINE =====================

def main_pipeline():
    datasets = sorted(KEEL_DATASET_DIR.glob("*.dat"))
    if not datasets:
        print(f"[Error] No .dat files in {KEEL_DATASET_DIR}")
        return

    all_results = []
    print(f"\n[Start] {len(datasets)} datasets | device={device} | "
          f"AMP={USE_AMP} | batch={BATCH_SIZE} | seeds={SEEDS}\n")

    for di, ds_path in enumerate(datasets):
        print(f"\n--- [{di+1}/{len(datasets)}] {ds_path.stem} ---")
        X, y, ir = load_keel_dataset(ds_path)
        if X is None:
            continue
        log_parity_check(ds_path.stem, X)

        for m_type in ["KAN", "MLP", "XGB"]:
            for strat in ["baseline", "resampled", "focal"]:
                trials = []
                print(f"  {m_type:<4} {strat:<10} seeds: ",
                      end="", flush=True)

                for seed in SEEDS:
                    print(f"{seed} ", end="", flush=True)

                    
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                   
                    res       = run_experiment(X, y, m_type, strat,
                                               ds_path.stem, seed)
                    res['IR'] = ir
                    trials.append(res)

                avg = aggregate_seed_results(trials)
                all_results.append(avg)
                print("| done")

        pd.DataFrame(all_results).to_csv(
            RESULTS_DIR / "benchmark_checkpoint.csv", index=False)

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(RESULTS_DIR / "final_benchmark_results.csv", index=False)

    generate_publication_plots(all_results, RESULTS_DIR)
    generate_table1_csv(all_results, RESULTS_DIR)
    perform_statistical_tests(final_df)
    print("\n[Done] Benchmark complete.")

if __name__ == "__main__":
    patch_imbalance_xgboost()
    main_pipeline()
