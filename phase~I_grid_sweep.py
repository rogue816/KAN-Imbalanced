"""

=============================================================================
Phase I — Step 3: Isolated Grid Resolution Sweep
=============================================================================
With all hyperparameters frozen from HPO (other scripts ), this script
varies G_grid as the sole free parameter across {3, 5, 7, 9, 12} and
evaluates each configuration over five independent random seeds.

Inputs  (produced by other scripts )
-------
 .json    baseline HPO configurations
  .json        resampled HPO configurations
  kan_sensitivity_results/
      {dataset}_phase3_raw_assets.pkl
      {dataset}_phase3_resampled_assets.pkl

Outputs
-------
  kan_sensitivity_results 
    


"""

import os
import json
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.metrics import geometric_mean_score
from imblearn.combine import SMOTETomek
from torch.utils.data import DataLoader, TensorDataset
from kan import KAN

warnings.filterwarnings("ignore")
os.environ['KAN_SAVE_PLOT'] = '0'
os.environ['KAN_VERBOSE']   = '0'

# ===================== CONFIGURATION =====================

RANDOM_STATE       = 42
SENSITIVITY_SEEDS  = [42, 123, 999, 7, 21]
SENSITIVITY_GRIDS  = [3, 5, 7, 9, 12]
SENSITIVITY_EPOCHS = 120      # full budget for final sweep evaluation
TUNING_GRID_VAL    = 5        # HPO anchor grid 
PHASE2_GRID_VAL    = 7      

RESULTS_DIR      = Path("kan_sensitivity_results")
TUNED_PARAMS_RAW = Path("tuned_params_cache.json")
TUNED_PARAMS_RES = Path("tuned_params_resampled.json")

RESULTS_DIR.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hardware] Device: {device}")

# ===================== UTILITIES =====================

def set_seed(seed: int):
    """
    Full deterministic seeding for reproducible sweep results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def clear_memory():
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_train_loader(X: np.ndarray, y: np.ndarray,
                     batch_size: int) -> DataLoader:
    """
    Returns a DataLoader for training data.
    """
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ),
        batch_size = batch_size,
        shuffle    = True,
        drop_last  = True
    )


# ===================== SWEEP TRAINING =====================

def train_and_evaluate(model: nn.Module, train_loader: DataLoader,
                       X_test: np.ndarray, y_test: np.ndarray,
                       lr: float, epochs: int) -> float:
    """
    Trains KAN for a fixed number of epochs using the hyperparameters
    (lr, batch_size) from the HPO JSON, then evaluates on the held-out
    test set.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

  
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(X_b), y_b).backward()
            optimizer.step()

    
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        preds = model(X_t).argmax(dim=1).cpu().numpy()

    return geometric_mean_score(y_test, preds)


# ===================== MAIN SWEEP =====================

def main_sensitivity_sweep() -> pd.DataFrame:
    """
    Iterates over [raw, resampled] x [ecoli3, yeast5] x GRIDS x SEEDS.
    """
    with open(TUNED_PARAMS_RAW, 'r') as f:
        raw_cache = json.load(f)
    with open(TUNED_PARAMS_RES, 'r') as f:
        res_cache = json.load(f)

    rows           = []
    target_datasets = ["ecoli3", "yeast5"]

    for mode in ['raw', 'resampled']:
        cache = res_cache if mode == 'resampled' else raw_cache

        for ds_name in target_datasets:
            asset_path = RESULTS_DIR / f"{ds_name}_phase3_{mode}_assets.pkl"
            if not asset_path.exists():
                print(f"  [Skip] Missing asset: {asset_path.name}")
                continue

            assets     = joblib.load(asset_path)
            X_tr_raw   = assets['X_train_raw']
            y_tr_raw   = assets['y_train']
            X_test_sc  = assets['X_test_scaled']   
            y_test     = assets['y_test']
            scaler     = assets['scaler']
            input_dim  = assets['input_dim']

            kan_cfg = cache[ds_name]['kan']
            mlp_cfg = cache[ds_name]['mlp']

            print(f"\n  [{mode.upper():10s}] {ds_name} — "
                  f"k={kan_cfg['k']}, widths={kan_cfg['layer_widths']}, "
                  f"lr={kan_cfg['lr']:.5f}")

            for grid in SENSITIVITY_GRIDS:
                for seed in SENSITIVITY_SEEDS:
                    set_seed(seed)

                    
                    if mode == 'resampled':
                        smt = SMOTETomek(random_state=seed)
                        X_tr, y_tr = smt.fit_resample(X_tr_raw, y_tr_raw)
                    else:
                        X_tr, y_tr = X_tr_raw, y_tr_raw

                    
                    X_tr_scaled = scaler.transform(X_tr).astype(np.float32)

                    model = KAN(
                        width = [input_dim] + kan_cfg['layer_widths'] + [2],
                        grid  = grid,
                        k     = kan_cfg['k']
                    ).to(device)

                    n_params = count_parameters(model)

                    loader = get_train_loader(
                        X_tr_scaled, y_tr, kan_cfg['batch_size'])

                    gmean = train_and_evaluate(
                        model, loader, X_test_sc, y_test,
                        lr=kan_cfg['lr'], epochs=SENSITIVITY_EPOCHS)

                    rows.append({
                        'Dataset'     : ds_name,
                        'Mode'        : mode,
                        'Grid'        : grid,
                        'Seed'        : seed,
                        'G_Mean'      : gmean,
                        'Params'      : n_params,
                        
                        'MLP_HPO_GMean': mlp_cfg['best_gmean'],
                        'MLP_Params'  : mlp_cfg['best_model_params']
                    })

                    del model
                    clear_memory()

            print(f"    Grid sweep complete for {ds_name} [{mode}]")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "sensitivity_final_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\n  [Saved] {out}")
    return df


# ===================== FIGURE 2 =====================

def plot_ieee_master_sensitivity(master_df: pd.DataFrame):
    """
    Generates plots.
    """
    with open(TUNED_PARAMS_RAW, 'r') as f:
        raw_cache = json.load(f)
    with open(TUNED_PARAMS_RES, 'r') as f:
        res_cache = json.load(f)

    plt.rcParams['font.family'] = 'serif'
    sns.set_context("paper", font_scale=1.4)

    datasets = ["ecoli3", "yeast5"]
    fig, axes = plt.subplots(2, len(datasets),
                             figsize=(6 * len(datasets), 10),
                             constrained_layout=True)

    colors = {'raw': '#d95f02', 'resampled': '#7570b3'}

    for col, ds in enumerate(datasets):
        df_ds = master_df[master_df['Dataset'] == ds]

        
        for row in [0, 1]:
            axes[row, col].axvspan(
                TUNING_GRID_VAL - 0.25, TUNING_GRID_VAL + 0.25,
                color='grey', alpha=0.06, zorder=0)

        
        axes[0, col].set_ylim(0.0, 1.05)
        axes[0, col].grid(True, linestyle='--', alpha=0.4)

        for mode in ['raw', 'resampled']:
            stats = (df_ds[df_ds['Mode'] == mode]
                     .groupby('Grid')['G_Mean']
                     .agg(['mean', 'std'])
                     .reset_index())

            axes[0, col].plot(
                stats['Grid'], stats['mean'],
                marker='o', markersize=5, linewidth=2.5,
                color=colors[mode],
                label=f"KAN ({mode.capitalize()})")

            
            axes[0, col].fill_between(
                stats['Grid'],
                stats['mean'] - stats['std'],
                stats['mean'] + stats['std'],
                color=colors[mode], alpha=0.18)

           
            g5 = stats.loc[stats['Grid'] == TUNING_GRID_VAL, 'mean']
            if not g5.empty:
                axes[0, col].scatter(
                    [TUNING_GRID_VAL], [g5.values[0]],
                    color=colors[mode], edgecolor='black',
                    s=130, zorder=10, marker='X')

           
            g7 = stats.loc[stats['Grid'] == PHASE2_GRID_VAL, 'mean']
            if not g7.empty:
                axes[0, col].scatter(
                    [PHASE2_GRID_VAL], [g7.values[0]],
                    color=colors[mode], edgecolor='black',
                    s=110, zorder=11, marker='o')

        axes[0, col].axhline(
            raw_cache[ds]['mlp']['best_gmean'],
            ls='--', color='gray', alpha=0.8,
            label='MLP (Raw Baseline)')
        axes[0, col].axhline(
            res_cache[ds]['mlp']['best_gmean'],
            ls=':', color='green', alpha=0.8,
            label='MLP (Resampled Baseline)')

        axes[0, col].set_title(f"Dataset: {ds.upper()}",
                               fontweight='bold', pad=15)
        axes[0, col].set_ylabel(r"Geometric Mean ($G_{\mathrm{mean}}$)")
        axes[0, col].set_xticks(SENSITIVITY_GRIDS)

       
        axes[1, col].grid(True, linestyle='--', alpha=0.4)

        for mode in ['raw', 'resampled']:
            param_stats = (df_ds[df_ds['Mode'] == mode]
                           .groupby('Grid')['Params']
                           .mean()
                           .reset_index())
            axes[1, col].plot(
                param_stats['Grid'], param_stats['Params'],
                marker='o', markersize=5, linewidth=2,
                color=colors[mode], label=f"KAN ({mode.capitalize()})")

   
        mlp_p_raw = raw_cache[ds]['mlp']['best_model_params']
        axes[1, col].axhline(
            mlp_p_raw, ls='--', color='gray', alpha=0.6, linewidth=1.5)

       
        mlp_p_res = res_cache[ds]['mlp']['best_model_params']
        if mlp_p_res != mlp_p_raw:
            axes[1, col].axhline(
                mlp_p_res, ls=':', color='green', alpha=0.6, linewidth=1.5)

     
        y_mid = df_ds['Params'].median()
        axes[1, col].annotate(
            r'Parity Point ($N_p$)',
            xy     = (TUNING_GRID_VAL, mlp_p_raw),
            xytext = (TUNING_GRID_VAL + 1.0,
                      mlp_p_raw + 0.12 * (df_ds['Params'].max() - df_ds['Params'].min())),
            arrowprops = dict(arrowstyle='->', color='black', lw=1.5),
            fontsize   = 10,
            fontweight = 'bold',
            bbox       = dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black", alpha=0.8)
        )

        axes[1, col].set_ylabel(r"Trainable Parameters ($N_p$)")
        axes[1, col].set_xlabel(r"Spline Grid Resolution ($G_{\mathrm{grid}}$)")
        axes[1, col].set_xticks(SENSITIVITY_GRIDS)

    # Legend on top-left panel only
    axes[0, 0].legend(loc='lower right', frameon=True, fontsize='small')

    out = RESULTS_DIR / "IEEE_SENSITIVITY_MASTER.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  [Figure] Saved to {out}")
    plt.show()


# ===================== ENTRY POINT =====================

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[Phase I — Step 3] Grid sensitivity sweep")
    print(f"  Grids : {SENSITIVITY_GRIDS}")
    print(f"  Seeds : {SENSITIVITY_SEEDS}")
    print(f"  Epochs: {SENSITIVITY_EPOCHS}")
    print()

    results_df = main_sensitivity_sweep()
    plot_ieee_master_sensitivity(results_df)

    print("\n[Done] Sensitivity sweep complete.")