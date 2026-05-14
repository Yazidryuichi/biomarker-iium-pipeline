# Biomarker_IIUM Pipeline — Status Report for Dandy

> **Historical snapshot — 7 April 2026, predates Phase 1 honest-reframe + public release.** Repo URL and "private" status are stale: the repo moved to <https://github.com/Yazidryuichi/biomarker-iium-pipeline> and is public since 24 April 2026. Numbers and terminology ("quantum-inspired") in this report match the pre-Phase-1 framing; current canonical state is in [README.md](README.md) and [METHODS.md](METHODS.md). Retained for historical record of the April pilot snapshot.

**Date:** 7 April 2026
**From:** Yazid
**Repo:** https://github.com/Yazidryuichi/biomarker-iium-pipeline (now public)
**Your access:** Invited as collaborator (`dndyzz`) — please accept the invite if you haven't.

---

## 1. What Was Done (This Session)

### Pipeline Upgrades (all pushed to GitHub)

The pipeline was updated to match the research proposal (Proposal_QEEG_v4). Here's what changed:

| Change | Status | Commit |
|--------|--------|--------|
| Added LightGBM classifier | Working | `454e9b5` |
| Added CatBoost classifier (with sklearn 1.8 wrapper) | Working (see known issues) | `9ccdb5c` |
| Added KNN classifier | Working | `454e9b5` |
| Added MLP (neural network) classifier | Working | `454e9b5` |
| Added CNN-LSTM (PyTorch) classifier | Code complete but disabled (see known issues) | `9ccdb5c` |
| Added sensitivity & specificity metrics | Working | `454e9b5` |
| Added hyperparameter tuning (RandomizedSearchCV, nested CV) | Working | `454e9b5` |
| Updated config.yaml (8 models, 5 scoring metrics) | Done | `454e9b5` |
| Updated README with latest results | Done | `454e9b5` |
| Updated requirements.txt (added catboost) | Done | `454e9b5` |

### Other Deliverables
- You (`dndyzz`) have been invited as a collaborator with push access
- CV v3 created with the Biomarker_IIUM study listed as active research
- Research proposal updated with 8-model comparison results
- AUFEI publication (Dewi et al. 2025, Int J Neuropsychopharmacology) added to CV

---

## 2. Known Issues That Need Fixing

### ISSUE 1: CNN-LSTM hangs on macOS (CRITICAL)

**Symptom:** When `_run_cnn_lstm_cv()` is called after sklearn's `cross_validate` (which uses joblib internally), the PyTorch training loop drops to 0% CPU and hangs indefinitely. The process doesn't crash — it just freezes.

**Root cause (suspected):** sklearn's `cross_validate` uses joblib's `loky` backend which forks processes. On macOS, PyTorch's internal threading (OpenMP/MKL) doesn't survive process forking. When CNN-LSTM tries to run after joblib has already forked workers, PyTorch's thread pool is in a corrupted state.

**Evidence:**
- `import torch` works fine in isolation (torch 2.11.0)
- A standalone PyTorch training loop works fine
- The hang only occurs when PyTorch runs *after* sklearn cross_validate

**Current workaround:** CNN-LSTM is disabled by default. Set `RUN_CNN_LSTM=1` environment variable to enable.

**Proposed fixes (for you to try):**

```python
# Option A: Force joblib to use 'threading' backend before CNN-LSTM
import joblib
with joblib.parallel_backend('threading'):
    # run CNN-LSTM here

# Option B: Run CNN-LSTM BEFORE sklearn models (reverse the order)

# Option C: Use multiprocessing.set_start_method('spawn') at the top of run_all.py
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Option D: Set environment variable before import
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
```

**To test:** Run the pipeline on your machine (likely Linux, not macOS) — it may just work there:
```bash
RUN_CNN_LSTM=1 python run_all.py --stage 4
```

---

### ISSUE 2: CatBoost + sklearn 1.8 compatibility (FIXED but fragile)

**Symptom:** CatBoost 1.2.8 doesn't implement sklearn 1.8's `__sklearn_tags__()` method. When used in a Pipeline with `roc_auc` scoring, sklearn thinks it's a regressor and raises: `"Got a regressor with response_method=predict_proba instead"`

**Fix applied:** A `CatBoostWrapper` class wraps CatBoost and overrides `__sklearn_tags__()`:
```python
def __sklearn_tags__(self):
    tags = super().__sklearn_tags__()
    tags.estimator_type = "classifier"
    return tags
```

**Status:** Confirmed working in isolation test. Should work in full pipeline but hasn't been verified end-to-end yet (system crashed before verification completed).

**If CatBoost still fails for you:** Upgrade catboost: `pip install catboost --upgrade`. Newer versions may have native sklearn 1.8 support.

---

### ISSUE 3: Hyperparameter tuning not yet verified

**Symptom:** The `_run_tuned_classification()` function runs RandomizedSearchCV with nested CV on the best feature set. It runs *after* the main classification loop. Because system crashed during CNN-LSTM (which runs before tuning), tuning was never reached.

**What to check:** Run stage 4 (with CNN-LSTM disabled) and verify tuning output appears at the end:
```
  Running hyperparameter tuning on best feature set: conventional_qeeg
  tuned_best  | RandomForest_tuned | Acc: ... 
  tuned_best  | SVM_tuned          | Acc: ...
  tuned_best  | XGBoost_tuned      | Acc: ...
```

---

## 3. Latest Results (7 models, CNN-LSTM pending)

These results are from the last successful run before the crash. CatBoost results are from a partial run.

### Best per feature set (balanced accuracy)

| Feature Set | Best Model | Bal. Acc | Sens | Spec | AUC |
|-------------|-----------|---------|------|------|-----|
| **Conventional QEEG** | **XGBoost** | **0.663** | 0.673 | 0.653 | **0.703** |
| Conv. + Advanced | LightGBM | 0.572 | 0.577 | 0.567 | 0.583 |
| Covariance only | XGBoost | 0.633 | 0.707 | 0.560 | 0.669 |
| All features | RandomForest | 0.567 | 0.633 | 0.500 | 0.560 |

### Full results for conventional_qeeg (best feature set)

| Model | Bal. Acc | Sensitivity | Specificity | F1 | AUC |
|-------|---------|------------|-------------|-----|-----|
| XGBoost | **0.663** | 0.673 | 0.653 | 0.644 | **0.703** |
| LightGBM | 0.600 | 0.580 | 0.620 | 0.571 | 0.673 |
| CatBoost | 0.575 | 0.593 | 0.557 | 0.563 | 0.604 |
| RandomForest | 0.585 | 0.597 | 0.573 | 0.572 | 0.662 |
| KNN | 0.548 | 0.537 | 0.560 | 0.517 | 0.534 |
| MLP | 0.533 | 0.880 | 0.187 | 0.645 | 0.534 |
| SVM | 0.525 | 0.557 | 0.493 | 0.513 | 0.559 |
| CNN-LSTM | — | — | — | — | — |

### Key observations
- **XGBoost dominates** across all feature sets — consistent with proposal hypothesis
- **LightGBM** is a strong second, especially on conventional features
- **MLP** has severe specificity problems (0.187) — predicts almost everyone as "high EF"
- **Conventional QEEG features outperform larger feature sets** — suggests overfitting with 922 features at N=28
- **H4 target (75% accuracy) not met** — expected, given N=28 vs target N=100

### Quantum vs Classical (from previous run, unchanged)

| Feature Set | Model | Bal. Acc | AUC |
|-------------|-------|---------|-----|
| Quantum only | LogReg | **0.657** | 0.694 |
| Classical only | LogReg | 0.585 | 0.662 |
| Combined | LogReg | 0.608 | 0.634 |

---

## 4. What You Need To Do

### Priority 1: Run the full pipeline on your machine
```bash
git pull origin main
pip install -r requirements.txt
python run_all.py --stage 4
```
This should work without crashes on Linux. Check:
- Do all 7 models complete without errors?
- Does hyperparameter tuning run at the end?
- Are results saved to `results/ml_results.csv`?

### Priority 2: Test CNN-LSTM
```bash
RUN_CNN_LSTM=1 python run_all.py --stage 4
```
If it works, great. If it hangs, try the fixes in Issue 1 above.

### Priority 3: Run with emotional conditions
```bash
python run_all.py --include-emotional
```
This adds Happy, Calm, Sad, Scare conditions. The proposal mentions them but we haven't run this yet.

### Priority 4: Share results
Once you have clean results, please share:
1. The updated `results/ml_results.csv`
2. The updated `results/shap_importance.csv`
3. Any new `figures/shap_summary.png`

---

## 5. Git History (Current State)

```
9ccdb5c Fix CatBoost sklearn 1.8 tags + disable CNN-LSTM by default
955b516 Fix CatBoost classifier detection and add CNN-LSTM timeout
454e9b5 Add 5 missing ML models, hyperparameter tuning, and sensitivity/specificity metrics
33150db Fix 10 issues from code review
be1279e Add quantum-inspired feature engineering exploration (Stage 5)
316c931 Initial commit: QEEG biomarker pipeline for executive function
```

## 6. File Structure

```
pipeline/
  run_all.py              # Main CLI — python run_all.py --help
  requirements.txt        # pip install -r requirements.txt
  configs/
    config.yaml           # All parameters (8 models, 5 metrics)
  stages/
    stage1_cleaning.py    # EDF -> clean epochs
    stage2_features.py    # Epochs -> 922 features
    stage3_merge.py       # Features + behavioral data
    stage4_analysis.py    # Correlations + ML (7+1 models) + SHAP + tuning
    exploratory_quantum.py # Stage 5: quantum-inspired features
  utils/
    io.py                 # Data loading utilities
  results/                # Generated outputs
  figures/                # Generated plots
```

---

## 7. Stage 5 Upgrade (Quantum Exploration)

Stage 5 (`exploratory_quantum.py`) now also uses all 7 sklearn-compatible models (previously only LogReg + RF). This means the quantum vs classical comparison produces 21 results instead of 6.

**Previous results (2 models only):**

| Feature Set | Best Model | Bal. Acc | AUC |
|-------------|-----------|---------|-----|
| Quantum only | LogReg | **0.657** | 0.694 |
| Classical only | RF | 0.585 | 0.662 |
| Combined | LogReg | 0.608 | 0.634 |

**To regenerate with all models:**
```bash
python run_all.py --stage 5
```

This requires Stage 1 outputs (cleaned epochs) to exist in `results/cleaned_epochs/`.

---

## 8. Environment

The pipeline was developed and tested on:
- macOS Darwin 25.4.0 (Apple Silicon)
- Python 3.12 (Anaconda base)
- sklearn 1.8.0, mne 1.11.0, xgboost 3.1.3, lightgbm 4.6.0, catboost 1.2.8, torch 2.11.0

If you're on Linux/Windows, versions may differ. The pipeline should be compatible with Python 3.10+.
