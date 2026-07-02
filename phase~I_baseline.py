"""
=============================================================================
Phase I — Step 1 & 2: Hyperparameter Optimisation (Baseline Condition)
=============================================================================
Tunes KAN and parameter-matched MLP architectures on the native (unmitigated)
class distribution for two calibration datasets: ecoli3 and yeast5.

Outputs
-------
  tuned_params_cache.json          KAN and MLP best configurations
  kan_sensitivity_results/
      {dataset}_phase3_raw_assets.pkl   scaler + data assets for grid sweep
  optuna_studies/
      kan_{dataset}.db             Optuna study databases (reproducibility)

Hardware
--------
  NVIDIA A100 MIG 3g.20gb (20 GB HBM2e, CUDA 12.x)
  All other hardware will work but runtimes will differ.

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
import optuna
import joblib
import re

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.metrics import geometric_mean_score
from imblearn.combine import SMOTETomek
from torch.utils.data import DataLoader, TensorDataset
from kan import KAN
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ['KAN_SAVE_PLOT'] = '0'
os.environ['KAN_VERBOSE']   = '0'

# ===================== CONFIGURATION =====================

RANDOM_STATE = 42
N_TRIALS     = 25       
TUNING_GRID  = 5        
HPO_EPOCHS   = 60       
PATIENCE_INT = 10       

KEEL_DIR    = Path("keel_datasets")
RESULTS_DIR = Path("kan_sensitivity_results")
OPTUNA_DIR  = Path("optuna_studies")
PARAMS_OUT  = Path("tuned_params_cache.json")

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
OPTUNA_DIR.mkdir(exist_ok=True)

# Load existing cache if resuming
TUNED_PARAMS = {}
if PARAMS_OUT.exists():
    with open(PARAMS_OUT, 'r') as f:
        TUNED_PARAMS = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hardware] Device: {device}")
if device.type == 'cuda':
    print(f"[Hardware] GPU   : {torch.cuda.get_device_name(0)}")


# ===================== UTILITIES =====================

def clear_memory():
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters — used to verify parameter parity."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_mlp_width(input_dim, target_params: int, n_layers: int,
                        output_dim: int = 2) -> int:
    """
    Derives MLP hidden width W such that total params ≈ target_params.

    n_layers = 1 (closed-form, exact):
        W = (target_params - out_d) / (in_d + out_d + 1)

    n_layers = 2 (quadratic, positive root):
        W = [-(in_d + out_d + 2) + sqrt((in_d+out_d+2)^2 - 4*(out_d-N))] / 2

    Minimum width = 8~prevent degenerate architectures.
    """
    in_d  = int(input_dim[0] if isinstance(input_dim, (torch.Size, tuple, list))
                else input_dim)
    out_d = output_dim

    if n_layers == 1:
        width = int((target_params - out_d) / (in_d + out_d + 1))
    else:
        a    = 1
        b    = in_d + out_d + 2
        c    = out_d - target_params
        disc = b ** 2 - 4 * a * c
        width = 8 if disc < 0 else int((-b + np.sqrt(disc)) / (2 * a))

    return max(8, width)


# ===================== DATA LOADING =====================

def load_keel_dataset(dataset_path: Path):
    """
    Parses KEEL .dat files into (X: float32, y: int, IR: float).
    
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        attributes, data_start = [], None
        for i, line in enumerate(lines):
            s = line.strip()
            if s.lower().startswith("@data"):
                data_start = i + 1
                break
            elif s.lower().startswith("@attribute"):
                parts = re.split(r'\s+', s)
                if len(parts) >= 3:
                    attributes.append((parts[1].strip("'\""), parts[2]))

        if not attributes or data_start is None:
            return None, None, None

        data_lines = [
            l.split('%')[0].strip()
            for l in lines[data_start:]
            if l.strip() and not l.strip().startswith('@')
        ]
        data = [re.split(r',\s*', l) for l in data_lines if l]
        cols = [a[0] for a in attributes]
        df   = pd.DataFrame(data, columns=cols)

        for col, (_, atype) in zip(cols[:-1], attributes[:-1]):
            if any(t in atype.lower() for t in ['real', 'integer', 'numeric']):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        le  = LabelEncoder()
        y   = le.fit_transform(df[cols[-1]])
        df  = df.dropna()

        if len(np.unique(y)) > 2:
            minority = np.argmin(np.bincount(y))
            y = np.where(y == minority, 1, 0)

        counts = np.bincount(y)
        X  = df.drop(columns=[cols[-1]]).values.astype(np.float32)
        ir = max(counts) / min(counts)
        return X, y, ir

    except Exception as e:
        print(f"  [Error] Loading {dataset_path}: {e}")
        return None, None, None


# ===================== ASSET SERIALISATION =====================

def save_phase3_assets(dataset_name: str, X_train: np.ndarray,
                       y_train: np.ndarray, X_test: np.ndarray,
                       y_test: np.ndarray, scaler, input_dim: int,
                       mode: str = "raw"):
    """
    Serialises training data and the training-fitted scaler for the grid
    sweep. Phase II does not use these assets.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{dataset_name}_phase3_{mode}_assets.pkl"
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    joblib.dump({
        'X_train_raw'  : X_train,
        'y_train'      : y_train,
        'X_test_scaled': X_test_scaled,
        'y_test'       : y_test,
        'scaler'       : scaler,
        'input_dim'    : input_dim
    }, path)
    print(f"  [Assets] Saved {mode.upper()} assets for {dataset_name}")


# ===================== MLP MODEL =====================

class MLP(nn.Module):
    """Standard ReLU-MLP. Width is derived analytically for parameter parity."""
    def __init__(self, input_dim, hidden_dims: list):
        super().__init__()
        layers  = []
        curr_in = int(input_dim[0] if isinstance(input_dim, (torch.Size, tuple, list))
                      else input_dim)
        for dim in hidden_dims:
            layers += [nn.Linear(curr_in, dim), nn.ReLU()]
            curr_in = dim
        layers.append(nn.Linear(curr_in, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===================== HPO: KAN (BASELINE) =====================

def tune_kan_baseline(dataset_name: str, X: np.ndarray, y: np.ndarray,
                      n_trials: int = N_TRIALS) -> dict:
    """
    Optuna HPO for KAN on the native (unmitigated) class distribution.
    Grid is fixed at TUNING_GRID=5.
    """
    db_path    = (OPTUNA_DIR / f"kan_{dataset_name}.db").absolute()
    study_name = f"kan_{dataset_name}_baseline_grid{TUNING_GRID}_cv5"
    skf        = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=RANDOM_STATE)

    def objective(trial):
        clear_memory()

        lr      = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        wd      = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        bs      = trial.suggest_categorical("batch_size", [16, 32])
        k       = trial.suggest_int("k", 2, 3)
        n_lay   = trial.suggest_int("n_layers", 1, 2)
        widths  = [trial.suggest_int(f"layer_{i}_width", 2, 12)
                   for i in range(n_lay)]

        fold_gmeans, n_params = [], 0

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            sc    = StandardScaler()
            X_trs = torch.from_numpy(sc.fit_transform(X_tr)).float().to(device)
            X_vas = torch.from_numpy(sc.transform(X_va)).float().to(device)
            y_trt = torch.from_numpy(y_tr).long().to(device)

            model = KAN(width=[X.shape[1]] + widths + [2],
                        grid=TUNING_GRID, k=k).to(device)
            if n_params == 0:
                n_params = count_parameters(model)

            opt      = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            crit     = nn.CrossEntropyLoss()
            loader   = DataLoader(TensorDataset(X_trs, y_trt),
                                  batch_size=bs, shuffle=True)

            best_g, no_improve = 0.0, 0
            for epoch in range(HPO_EPOCHS):
                model.train()
                for xb, yb in loader:
                    opt.zero_grad(set_to_none=True)
                    crit(model(xb), yb).backward()
                    opt.step()

                if epoch % 2 == 0:
                    model.eval()
                    with torch.no_grad():
                        preds = model(X_vas).argmax(1).cpu().numpy()
                    g = geometric_mean_score(y_va, preds)
                    if g > best_g:
                        best_g, no_improve = g, 0
                    else:
                        no_improve += 1
                    if no_improve >= PATIENCE_INT:
                        break

            fold_gmeans.append(best_g)
            # Early pruning: abort trial if first two folds are near-zero
            if len(fold_gmeans) == 2 and np.mean(fold_gmeans) < 0.01:
                raise optuna.TrialPruned()
            del model
            clear_memory()

        trial.set_user_attr("total_params", n_params)
        return float(np.mean(fold_gmeans))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner  = optuna.pruners.MedianPruner()
    study   = optuna.create_study(
        direction  = "maximize",
        storage    = f"sqlite:///{db_path}",
        study_name = study_name,
        sampler    = sampler,
        pruner     = pruner,
        load_if_exists = True
    )
    study.optimize(objective, n_trials=n_trials)

    bp = study.best_params
    return {
        'lr'              : bp["lr"],
        'batch_size'      : bp["batch_size"],
        'k'               : bp["k"],
        'n_layers'        : bp["n_layers"],
        'layer_widths'    : [bp[f"layer_{i}_width"] for i in range(bp["n_layers"])],
        'best_gmean'      : study.best_value,
        'best_params_count': int(study.best_trial.user_attrs.get("total_params", 0))
    }


# ===================== HPO: MLP (BASELINE) =====================

def tune_mlp_baseline(dataset_name: str, X: np.ndarray, y: np.ndarray,
                      target_params: int, n_trials: int = N_TRIALS) -> dict:
    """
    Optuna HPO for MLP on the native class distribution.
    """
    db_path = (OPTUNA_DIR / f"mlp_{dataset_name}_baseline.db").absolute()
    skf     = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=RANDOM_STATE)

    def objective(trial):
        clear_memory()

        n_lay  = trial.suggest_int("n_layers", 1, 2)
        width  = calculate_mlp_width(X.shape[1], target_params, n_lay)
        dims   = [width] * n_lay
        lr     = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd     = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        bs     = trial.suggest_categorical("batch_size", [16, 32])

        fold_gmeans, n_params = [], 0

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            sc    = StandardScaler()
            X_trs = torch.from_numpy(sc.fit_transform(X_tr)).float().to(device)
            X_vas = torch.from_numpy(sc.transform(X_va)).float().to(device)
            y_trt = torch.from_numpy(y_tr).long().to(device)

            model = MLP(X.shape[1], dims).to(device)
            if n_params == 0:
                n_params = count_parameters(model)

            opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            crit   = nn.CrossEntropyLoss()
            loader = DataLoader(TensorDataset(X_trs, y_trt),
                                batch_size=bs, shuffle=True)

            best_g, no_improve = 0.0, 0
            for epoch in range(HPO_EPOCHS):
                model.train()
                for xb, yb in loader:
                    opt.zero_grad(set_to_none=True)
                    crit(model(xb), yb).backward()
                    opt.step()

                if epoch % 2 == 0:
                    model.eval()
                    with torch.no_grad():
                        preds = model(X_vas).argmax(1).cpu().numpy()
                    g = geometric_mean_score(y_va, preds)
                    if g > best_g:
                        best_g, no_improve = g, 0
                    else:
                        no_improve += 1
                    if no_improve >= PATIENCE_INT:
                        break

            fold_gmeans.append(best_g)
            del model
            clear_memory()

        trial.set_user_attr("total_params", n_params)
        return float(np.mean(fold_gmeans))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner  = optuna.pruners.MedianPruner()
    study   = optuna.create_study(
        direction      = "maximize",
        storage        = f"sqlite:///{db_path}",
        sampler        = sampler,
        pruner         = pruner,
        load_if_exists = True
    )
    study.optimize(objective, n_trials=n_trials)

    bp     = study.best_params
    best_nl = bp["n_layers"]
    width  = calculate_mlp_width(X.shape[1], target_params, best_nl)

    return {
        'lr'              : bp["lr"],
        'batch_size'      : bp["batch_size"],
        'n_layers'        : best_nl,
        'hidden_dims'     : [width] * best_nl,
        'best_gmean'      : study.best_value,
        'best_model_params': int(study.best_trial.user_attrs.get("total_params", 0))
    }


# ===================== MAIN =====================

def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    datasets     = ["ecoli3", "yeast5"]
    dataset_pbar = tqdm(datasets, desc="Phase I Baseline HPO")

    for ds_name in dataset_pbar:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}  (Baseline / no resampling)")
        print(f"{'='*60}")

        ds_path = KEEL_DIR / f"{ds_name}.dat"
        X, y, ir = load_keel_dataset(ds_path)
        if X is None:
            print(f"  [Skip] Could not load {ds_name}")
            continue


        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

        print(f"  Tuning KAN  ({N_TRIALS} trials, G_grid={TUNING_GRID} fixed)...")
        kan_res = tune_kan_baseline(ds_name, X_temp, y_temp)

        kan_budget = kan_res['best_params_count']
        print(f"  Tuning MLP  ({N_TRIALS} trials, target_params={kan_budget})...")
        mlp_res = tune_mlp_baseline(ds_name, X_temp, y_temp, kan_budget)

        mlp_params   = mlp_res['best_model_params']
        parity_ratio = mlp_params / kan_budget if kan_budget > 0 else 0.0

        print(f"  KAN params : {kan_budget:,}   G-mean: {kan_res['best_gmean']:.4f}")
        print(f"  MLP params : {mlp_params:,}   G-mean: {mlp_res['best_gmean']:.4f}")
        print(f"  Parity ratio: {parity_ratio:.4f}")

        
        final_scaler = StandardScaler().fit(X_temp)
        save_phase3_assets(ds_name, X_temp, y_temp, X_test, y_test,
                           final_scaler, X.shape[1], mode="raw")

        TUNED_PARAMS[ds_name] = {
            'metadata': {
                'imbalance_ratio': float(ir),
                'input_dim'      : int(X.shape[1]),
                'parity_ratio'   : parity_ratio
            },
            'kan': kan_res,
            'mlp': mlp_res
        }

        with open(PARAMS_OUT, 'w') as f:
            json.dump(TUNED_PARAMS, f, indent=2)
        print(f"  [Saved] {PARAMS_OUT}")


if __name__ == "__main__":
    main()