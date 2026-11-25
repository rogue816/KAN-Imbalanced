# KAN vs MLP vs XGBoost: Imbalanced Tabular Classification

Reproducibility repository for:

> **Are Kolmogorov-Arnold Networks (KANs) Suitable for Imbalanced Data Classification?**  
> Submitted to IEEE Transactions on Emerging Topics in Computational Intelligence (TETCI)

---

## Repository Structure

```
.
├── 01_phase1_hpo_baseline.py     Phase I — KAN/MLP HPO on native distribution
├── 02_phase1_hpo_resampled.py    Phase I — KAN/MLP HPO under SMOTE-Tomek
├── 03_phase1_grid_sweep.py       Phase I — Grid sensitivity sweep (produces Fig. 2)
├── 04_phase2_benchmark.py        Phase II — Full 10-dataset comparative benchmark
├── requirements.txt              Python environment specification
├── README.md                     This file
└── results/
    ├── tuned_params_cache.json        Phase I outputs (baseline HPO)
    ├── tuned_params_resampled.json    Phase I outputs (resampled HPO)
    └── final_benchmark_results.csv    Phase II outputs (all results)
```

---

## Datasets

All ten datasets are drawn from the
[KEEL repository](https://sci2s.ugr.es/keel/datasets.php) (public access).

Download the following `.dat` files and place them in a `keel_datasets/` folder
in the same directory as the scripts:

| Dataset | IR |
|---|---|
| ecoli3 | 8.60 |
| yeast4 | 28.10 |
| yeast5 | 32.73 |
| yeast6 | 41.40 |
| glass2 | 11.59 |
| glass4 | 15.47 |
| glass6 | 6.02 |
| new-thyroid1 | 5.14 |
| winequality-red-8_vs_6 | 35.44 |
| winequality-red-8_vs_6-7 | 46.50 |

---

## Environment

```bash
pip install -r requirements.txt
```

Tested on Python 3.10, CUDA 12.x, NVIDIA A100 MIG 3g.20gb (20 GB HBM2e).  
All scripts run on CPU as well; runtimes will be substantially longer.

---

## Execution Order

### Option A — Full reproduction (all phases)

```bash
# Step 1: Phase I HPO — baseline condition
python 01_phase1_hpo_baseline.py

# Step 2: Phase I HPO — resampled condition  
python 02_phase1_hpo_resampled.py

# Step 3: Phase I grid sensitivity sweep (requires outputs of steps 1 and 2)
python 03_phase1_grid_sweep.py

# Step 4: Phase II benchmark (standalone — reads datasets directly)
python 04_phase2_benchmark.py
```

### Option B — Verify Phase II results only (skip HPO)

Phase I outputs are already provided in `results/`. Script 04 does not depend
on the `.pkl` assets produced by scripts 01–03; it reads datasets directly and
uses the fixed architecture from Phase I (`grid=7, k=3, n_layers=1, width=10`).

```bash
python 04_phase2_benchmark.py
```

Results are saved to `final_benchmark_results/final_benchmark_results.csv`.  
Compare against `results/final_benchmark_results.csv` to verify.

### Option C — Regenerate plots and statistics from saved CSV

```python
from 04_phase2_benchmark import plot_from_csv
plot_from_csv("results/final_benchmark_results.csv")
```

---

## Key Design Decisions

**Parameter parity.**  
MLP hidden width is derived analytically from the KAN parameter count using
closed-form equations (Eq. 1–2 in the paper), enforcing Parity Ratio ≈ 1.0
per dataset. The MLP is not independently tuned via Optuna; width is computed
at each trial to match the KAN's parameter budget exactly.

**Focal loss alpha.**  
Alpha is set to `n_minority / n_total` computed from each training fold after
any resampling, making it adaptive to the actual imbalance degree. Gamma = 2.0
throughout (Lin et al., 2017). This convention is applied identically to KAN,
MLP, and XGBoost (via `imxgboost`).

**Timing and memory.**  
Wall-clock time (`time.perf_counter`) captures end-to-end fold training time
including Python interpreter overhead inherent to `pykan`'s sequential spline
evaluation. GPU memory is tracked via `torch.cuda.max_memory_allocated` to
capture structural device-side cost separately from Python-heap overhead
(tracemalloc).

**Phase I → Phase II transfer.**  
Phase II uses `grid=7, k=3, n_layers=1, layer_width=10` for KAN across all
ten datasets. These are not the same as the Phase I HPO-selected architectures
(which used `n_layers=2`); the single-layer configuration was selected for
Phase II to prevent overfitting on the small-to-medium cohort datasets and to
hold architectural depth constant across the KAN–MLP comparison.

---

## Compatibility Note

`imxgboost` requires a patch for XGBoost >= 2.0 (the deprecated `feval`
argument is renamed to `custom_metric`). This patch is applied automatically
at startup via `patch_imbalance_xgboost()` in script 04.
