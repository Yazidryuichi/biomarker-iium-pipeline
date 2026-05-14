"""Stage 5 — Fair comparison: density-matrix features vs classical QEEG features.

Replaces `per_fold_dm_svm_vs_classical_rf.py` which compared DM-SVM-linear vs
Classical-RF (different feature sets AND different model classes). That
apples-to-oranges design conflated feature-set effect with model-class effect
and produced the portfolio's headline "AUC 0.780 vs 0.615" — a comparison a
senior reviewer flagged as not making sense.

This module runs a 2×2 design (feature-set × model-class) under identical
preprocessing pipelines, and adds proper subject-level uncertainty:

    Feature set       Model class
    DM ─┬─ Linear SVM      ──>  cell A
        └─ Random Forest   ──>  cell B
    CL ─┬─ Linear SVM      ──>  cell C   (new — was missing)
        └─ Random Forest   ──>  cell D

Reports per fold (matched CV across all four cells) AND per subject
(leave-one-subject-out predictions, used for unbiased AUC and DeLong test).

Statistical rules:
  * Subject-level AUC computed from leave-one-subject-out probabilities.
    This is the correct AUC for an N-subject study, NOT the average of
    per-fold AUCs (which is biased by fold size and ignores cross-fold
    rank inconsistency).
  * Paired subject-bootstrap CIs (10000 resamples of SUBJECTS, not folds).
    Per-fold bootstrap on 100 splits with only 28 actual subjects produces
    CIs too tight by a factor of sqrt(100/28) ≈ 1.9.
  * DeLong test on paired subject-level AUC (compares AUCs at SAME subject
    set). This is the standard test for paired ROC curves; Wilcoxon on
    per-fold AUCs treats correlated folds as independent and inflates type-I.
  * Permutation test on cell-by-cell BAcc: 1000 label permutations, two-sided
    p-value reports whether each cell beats chance.

Run:
    # Canonical 2x2:
    python -m stages.stage5_fair_comparison \
        --results-dir results \
        --out-json results/stage5_fair_comparison.json

    # With L1-regularisation sensitivity (P0.1):
    python -m stages.stage5_fair_comparison \
        --results-dir results \
        --out-json results/stage5_fair_comparison.json \
        --include-l1-sensitivity

Writes:
    results/stage5_per_fold.csv      (per-fold BAcc + AUC, all cells)
    results/stage5_per_subject.csv   (LOSO predicted probability per subject)
    results/stage5_fair_comparison.json  (consolidated stats + CIs + DeLong)

Sensitivity cells (when --include-l1-sensitivity):
    classical+svm_l1         LinearSVC with L1 penalty (calibrated sigmoid)
    classical+lr_l1          LogisticRegression L1
    classical+lr_elasticnet  LogisticRegression elasticnet (l1_ratio=0.5)

The motivating question: the canonical classical+svm_linear cell produced
LOSO AUC = 0.32 (below chance) at N=28, which the README reads as overfitting
of L2-SVC on top-10 SelectKBest. If L1 regularisation lifts classical above
chance, the matched-SVM DM-vs-Classical DeLong p=0.001 gap may narrow. If L1
stays below chance, the gap is robust to regularisation choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

CLASSICAL_PREFIXES = ("psd_", "coh_", "cov_", "tbr_", "faa_", "alpha_reactivity")
TARGET = "ef_group_global"
N_SELECT = 10
N_SPLITS, N_REPEATS, SEED = 10, 10, 42
N_BOOT = 10000
N_PERM = 1000


@dataclass
class CellResult:
    """One cell of the 2x2 design: per-fold and per-subject results."""
    name: str
    feature_set: str
    model_class: str
    per_fold_bacc: np.ndarray
    per_fold_auc: np.ndarray
    per_subject_proba: np.ndarray
    per_subject_label: np.ndarray
    per_subject_pred: np.ndarray


def _make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=N_SELECT)),
        ("clf", model),
    ])


def _evaluate_per_fold(X: np.ndarray, y: np.ndarray, model_factory,
                       splits: list) -> Tuple[np.ndarray, np.ndarray]:
    """Per-fold BAcc + AUC for the matched CV splits."""
    accs, aucs = [], []
    for tr, te in splits:
        pipe = _make_pipeline(model_factory())
        pipe.fit(X[tr], y[tr])
        yhat = pipe.predict(X[te])
        try:
            yprob = pipe.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], yprob)
        except Exception:
            auc = np.nan
        accs.append(balanced_accuracy_score(y[te], yhat))
        aucs.append(auc)
    return np.asarray(accs), np.asarray(aucs)


def _evaluate_loso(X: np.ndarray, y: np.ndarray,
                   model_factory) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-subject-out predicted probabilities.

    Returns (proba, label, pred), each shape (n_subjects,).
    """
    n = len(y)
    proba = np.full(n, np.nan, dtype=np.float64)
    pred = np.zeros(n, dtype=int)
    cv = LeaveOneOut()
    for tr, te in cv.split(X):
        pipe = _make_pipeline(model_factory())
        pipe.fit(X[tr], y[tr])
        try:
            proba[te] = pipe.predict_proba(X[te])[0, 1]
        except Exception:
            proba[te] = np.nan
        pred[te] = pipe.predict(X[te])[0]
    return proba, y.copy(), pred


def _delong_var(proba1: np.ndarray, proba2: np.ndarray,
                labels: np.ndarray) -> Tuple[float, float, float, float]:
    """DeLong covariance for paired AUCs.

    Returns (auc1, auc2, var_delta, z).
    Following DeLong et al. (1988) for paired ROC curves.
    """
    # Standard one-sample DeLong implementation
    mask = ~np.isnan(proba1) & ~np.isnan(proba2)
    p1, p2, lab = proba1[mask], proba2[mask], labels[mask]
    pos = lab == 1
    neg = lab == 0
    n1, n2 = pos.sum(), neg.sum()
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, np.nan

    def _midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        ranks = np.empty(len(x))
        i = 0
        while i < len(x):
            j = i
            while j < len(x) - 1 and x_sorted[j + 1] == x_sorted[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return ranks

    def _delong_components(probas: np.ndarray) -> Tuple[float, np.ndarray,
                                                         np.ndarray]:
        x = probas[pos]
        y = probas[neg]
        tx = _midrank(x)
        ty = _midrank(y)
        tz = _midrank(np.concatenate([x, y]))
        auc = (tz[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
        v01 = (tz[:n1] - tx) / n2
        v10 = 1 - (tz[n1:] - ty) / n1
        return auc, v01, v10

    auc1, v01_1, v10_1 = _delong_components(p1)
    auc2, v01_2, v10_2 = _delong_components(p2)
    sx = np.cov(np.column_stack([v01_1, v01_2]).T) / n1
    sy = np.cov(np.column_stack([v10_1, v10_2]).T) / n2
    cov = sx + sy
    var_delta = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var_delta <= 0:
        return float(auc1), float(auc2), var_delta, np.nan
    z = (auc1 - auc2) / np.sqrt(var_delta)
    return float(auc1), float(auc2), var_delta, float(z)


def _delong_paired_p(proba1: np.ndarray, proba2: np.ndarray,
                     labels: np.ndarray) -> Tuple[float, float, float, float]:
    """Two-sided p-value from DeLong paired test.

    Returns (auc1, auc2, z, p_two_sided).
    """
    auc1, auc2, var_delta, z = _delong_var(proba1, proba2, labels)
    if np.isnan(z):
        return auc1, auc2, z, np.nan
    p = 2 * (1 - norm.cdf(abs(z)))
    return auc1, auc2, z, float(p)


def _permutation_bacc(X: np.ndarray, y: np.ndarray, model_factory,
                      splits: list, n_perm: int = N_PERM,
                      seed: int = SEED) -> Tuple[float, float]:
    """Permutation p-value for per-fold mean BAcc. Returns (observed, p)."""
    accs_obs, _ = _evaluate_per_fold(X, y, model_factory, splits)
    obs = float(accs_obs.mean())
    rng = np.random.default_rng(seed)
    perm_means = np.zeros(n_perm)
    for k in range(n_perm):
        y_perm = rng.permutation(y)
        accs_perm, _ = _evaluate_per_fold(X, y_perm, model_factory, splits)
        perm_means[k] = float(accs_perm.mean())
    p = float((np.sum(perm_means >= obs) + 1) / (n_perm + 1))
    return obs, p


def _subject_bootstrap_ci(
    proba: np.ndarray, labels: np.ndarray, n_boot: int = N_BOOT,
    seed: int = SEED, alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Bootstrap CI for AUC at the SUBJECT level.

    Resamples subjects (not folds) — this is the correct level of resampling
    for an N-subject CV study and produces honest CIs at small N.
    """
    mask = ~np.isnan(proba)
    p, lab = proba[mask], labels[mask]
    n = len(p)
    rng = np.random.default_rng(seed)
    aucs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(lab[idx])) < 2:
            continue
        try:
            aucs[b] = roc_auc_score(lab[idx], p[idx])
        except Exception:
            pass
    aucs = aucs[~np.isnan(aucs)]
    if len(aucs) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(aucs))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


def _subject_bootstrap_bacc_ci(
    pred: np.ndarray, labels: np.ndarray, n_boot: int = N_BOOT,
    seed: int = SEED, alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Bootstrap CI for BAcc at the SUBJECT level."""
    n = len(labels)
    rng = np.random.default_rng(seed + 1)
    baccs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            baccs[b] = balanced_accuracy_score(labels[idx], pred[idx])
        except Exception:
            pass
    baccs = baccs[~np.isnan(baccs)]
    if len(baccs) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(baccs))
    lo, hi = np.percentile(baccs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


def _load_data(results_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                            list]:
    """Load DM + classical features aligned by subject_id, with target."""
    full = pd.read_csv(results_dir / "full_dataset.csv")
    dm = pd.read_csv(results_dir / "density_matrix_features.csv")
    full = full.dropna(subset=[TARGET])
    labels = full[["subject_id", TARGET]].drop_duplicates("subject_id")
    merged_dm = dm.merge(labels, on="subject_id", how="inner")
    merged_dm = merged_dm.drop_duplicates("subject_id", keep="first")
    dm_cols = [c for c in merged_dm.columns if c.startswith("dm_")]
    X_dm = merged_dm[dm_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = merged_dm[TARGET].astype(int).to_numpy()
    subject_order = merged_dm["subject_id"].tolist()
    cl_cols = [c for c in full.columns if c.startswith(CLASSICAL_PREFIXES)]
    cl_df = full.drop_duplicates("subject_id", keep="first").set_index(
        "subject_id")
    cl_df = cl_df.loc[subject_order, cl_cols]
    X_cl = cl_df.fillna(0.0).to_numpy(dtype=np.float64)
    return X_dm, X_cl, y, subject_order


def _model_factory(kind: str):
    if kind == "svm_linear":
        return lambda: SVC(kernel="linear", C=1.0, probability=True,
                           random_state=SEED)
    if kind == "rf_shallow":
        return lambda: RandomForestClassifier(n_estimators=200, max_depth=3,
                                              random_state=SEED)
    # L1-regularised variants (--include-l1-sensitivity opt-in).
    # Each runs through the same SelectKBest(k=10) + StandardScaler pipeline
    # as the baseline cells — apples-to-apples sensitivity: only the model
    # class differs. The motivating question is whether the Classical+SVM-linear
    # cell's LOSO AUC = 0.32 (below chance) reflects an artifact of L2-SVC
    # overfitting on top-10 SelectKBest at N=28, or a deeper limitation of
    # classical features. If L1 lifts classical above chance, the
    # matched-SVM DM-vs-Classical DeLong gap may narrow; if L1 stays below
    # chance, the gap is robust to regularisation choice.
    if kind == "svm_l1":
        # LinearSVC with L1 penalty needs dual=False. No native predict_proba,
        # so wrap with sigmoid calibration via 3-fold inner CV.
        return lambda: CalibratedClassifierCV(
            LinearSVC(penalty="l1", dual=False, C=1.0,
                      random_state=SEED, max_iter=5000),
            method="sigmoid", cv=3,
        )
    if kind == "lr_l1":
        return lambda: LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0,
            random_state=SEED, max_iter=2000,
        )
    if kind == "lr_elasticnet":
        return lambda: LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            random_state=SEED, max_iter=5000,
        )
    raise ValueError(f"unknown model kind: {kind}")


def _run_cell(name: str, feature_set: str, model_class: str, X: np.ndarray,
              y: np.ndarray, splits: list) -> CellResult:
    factory = _model_factory(model_class)
    per_fold_bacc, per_fold_auc = _evaluate_per_fold(X, y, factory, splits)
    per_subject_proba, per_subject_label, per_subject_pred = _evaluate_loso(
        X, y, factory)
    return CellResult(
        name=name, feature_set=feature_set, model_class=model_class,
        per_fold_bacc=per_fold_bacc, per_fold_auc=per_fold_auc,
        per_subject_proba=per_subject_proba,
        per_subject_label=per_subject_label, per_subject_pred=per_subject_pred,
    )


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-json", type=Path,
                        default=Path("results/stage5_fair_comparison.json"))
    parser.add_argument("--portfolio-json", type=Path, default=None,
                        help="Optional: also write the JSON to a portfolio "
                             "asset path so the portfolio + repo stay in sync.")
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--include-l1-sensitivity", action="store_true",
                        help=("Append 3 L1-regularised sensitivity cells on the "
                              "classical feature set (svm_l1, lr_l1, lr_elasticnet). "
                              "Default off to preserve the canonical 2x2 grid; "
                              "set this to investigate whether L1 regularisation "
                              "lifts Classical+SVM out of the below-chance LOSO AUC "
                              "0.32 at N=28."))
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    X_dm, X_cl, y, subject_order = _load_data(results_dir)
    print(f"N subjects: {len(y)}  DM features: {X_dm.shape[1]}  "
          f"classical features: {X_cl.shape[1]}", file=sys.stderr)
    print(f"class balance: {np.bincount(y).tolist()}", file=sys.stderr)

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    splits = list(cv.split(np.zeros(len(y)), y))
    print(f"matched CV: {N_SPLITS}-fold x {N_REPEATS} repeats = "
          f"{len(splits)} folds", file=sys.stderr)

    cells = []
    for fset_name, X in [("density_matrix", X_dm), ("classical", X_cl)]:
        for model in ["svm_linear", "rf_shallow"]:
            name = f"{fset_name}+{model}"
            print(f"  running {name} ...", file=sys.stderr)
            cells.append(_run_cell(name, fset_name, model, X, y, splits))

    # Optional L1-regularised sensitivity cells (classical features only).
    # Triggered by --include-l1-sensitivity. The classical+svm_linear baseline
    # produced LOSO AUC = 0.32 (below chance) at N=28 — this rerun tests
    # whether L1 regularisation lifts the classical baseline. If it does,
    # the matched-SVM DM-vs-Classical DeLong gap may narrow.
    if args.include_l1_sensitivity:
        print("  --include-l1-sensitivity ON: running 3 additional cells",
              file=sys.stderr)
        for model in ["svm_l1", "lr_l1", "lr_elasticnet"]:
            name = f"classical+{model}"
            print(f"  running {name} ...", file=sys.stderr)
            cells.append(_run_cell(name, "classical", model, X_cl, y, splits))

    # Per-fold CSV
    per_fold_df = pd.DataFrame({"fold": np.arange(len(splits))})
    for c in cells:
        per_fold_df[f"{c.name}_bacc"] = c.per_fold_bacc
        per_fold_df[f"{c.name}_auc"] = c.per_fold_auc
    per_fold_out = results_dir / "stage5_per_fold.csv"
    per_fold_df.to_csv(per_fold_out, index=False)
    print(f"wrote {per_fold_out}", file=sys.stderr)

    # Per-subject CSV
    per_subj_df = pd.DataFrame({"subject_id": subject_order,
                                "label": cells[0].per_subject_label})
    for c in cells:
        per_subj_df[f"{c.name}_proba"] = c.per_subject_proba
        per_subj_df[f"{c.name}_pred"] = c.per_subject_pred
    per_subj_out = results_dir / "stage5_per_subject.csv"
    per_subj_df.to_csv(per_subj_out, index=False)
    print(f"wrote {per_subj_out}", file=sys.stderr)

    # Summary stats — subject level (the honest ones)
    cell_summary = []
    for c in cells:
        loso_auc, auc_lo, auc_hi = _subject_bootstrap_ci(
            c.per_subject_proba, c.per_subject_label)
        loso_bacc, bacc_lo, bacc_hi = _subject_bootstrap_bacc_ci(
            c.per_subject_pred, c.per_subject_label)
        cell_summary.append({
            "cell": c.name, "feature_set": c.feature_set,
            "model_class": c.model_class,
            "loso_auc_mean": loso_auc, "loso_auc_ci_lo": auc_lo,
            "loso_auc_ci_hi": auc_hi,
            "loso_bacc_mean": loso_bacc, "loso_bacc_ci_lo": bacc_lo,
            "loso_bacc_ci_hi": bacc_hi,
            "per_fold_bacc_mean": float(c.per_fold_bacc.mean()),
            "per_fold_bacc_std": float(c.per_fold_bacc.std()),
            "per_fold_auc_mean": float(np.nanmean(c.per_fold_auc)),
            "per_fold_auc_std": float(np.nanstd(c.per_fold_auc)),
        })

    # Pairwise DeLong tests on subject-level AUC
    pairwise = []
    labels_ref = cells[0].per_subject_label
    for i, c1 in enumerate(cells):
        for c2 in cells[i + 1:]:
            auc1, auc2, z, p = _delong_paired_p(
                c1.per_subject_proba, c2.per_subject_proba, labels_ref)
            pairwise.append({
                "cell_1": c1.name, "cell_2": c2.name,
                "auc_1": auc1, "auc_2": auc2, "delta_auc": auc1 - auc2,
                "delong_z": z, "delong_p_two_sided": p,
            })

    # Permutation tests for each cell (against shuffled labels)
    print("  running permutation tests (this is the slow part) ...",
          file=sys.stderr)
    perm = []
    for c in cells:
        factory = _model_factory(c.model_class)
        X = X_dm if c.feature_set == "density_matrix" else X_cl
        obs, p_perm = _permutation_bacc(X, y, factory, splits,
                                        n_perm=args.n_perm)
        perm.append({"cell": c.name, "per_fold_bacc_obs": obs,
                     "permutation_p": p_perm})

    # The headline comparisons the supervisor will ask for
    def _by_cell(name: str) -> dict:
        for r in cell_summary:
            if r["cell"] == name:
                return r
        return {}

    headline = {
        "n_subjects": int(len(y)),
        "class_balance": np.bincount(y).tolist(),
        "design": "2x2 (feature_set x model_class), matched CV across all cells",
        "feature_set_within_model_class": {
            "svm_linear": {
                "delta_auc_dm_minus_cl_subject_level": (
                    _by_cell("density_matrix+svm_linear").get("loso_auc_mean",
                                                              float("nan"))
                    - _by_cell("classical+svm_linear").get("loso_auc_mean",
                                                            float("nan"))),
                "delta_bacc_dm_minus_cl_subject_level": (
                    _by_cell("density_matrix+svm_linear").get("loso_bacc_mean",
                                                              float("nan"))
                    - _by_cell("classical+svm_linear").get("loso_bacc_mean",
                                                            float("nan"))),
            },
            "rf_shallow": {
                "delta_auc_dm_minus_cl_subject_level": (
                    _by_cell("density_matrix+rf_shallow").get("loso_auc_mean",
                                                              float("nan"))
                    - _by_cell("classical+rf_shallow").get("loso_auc_mean",
                                                            float("nan"))),
                "delta_bacc_dm_minus_cl_subject_level": (
                    _by_cell("density_matrix+rf_shallow").get("loso_bacc_mean",
                                                              float("nan"))
                    - _by_cell("classical+rf_shallow").get("loso_bacc_mean",
                                                            float("nan"))),
            },
        },
    }

    out = {
        "n_subjects": int(len(y)),
        "n_folds_total": int(len(splits)),
        "cv_design": (f"matched RepeatedStratifiedKFold "
                      f"({N_SPLITS}-fold x {N_REPEATS} repeats); "
                      "LOSO per-subject probabilities used for unbiased AUC + "
                      "DeLong; subject-bootstrap CIs (N_BOOT=10000)."),
        "cells": cell_summary,
        "pairwise_delong": pairwise,
        "permutation_tests": perm,
        "headline": headline,
        "honesty_notes": [
            "AUC and BAcc here are LOSO (subject-level), NOT averages of "
            "per-fold metrics. Per-fold means are also reported for "
            "comparison with the prior version of this analysis.",
            "Bootstrap CIs resample SUBJECTS (n=N), not folds (n=100). "
            "Per-fold bootstrap was too tight at small N.",
            "DeLong tests use paired subject-level AUCs. Wilcoxon on per-fold "
            "AUCs treated correlated folds as independent and is not reported "
            "as the primary inference.",
            "Permutation p-values report whether each cell's per-fold mean "
            "BAcc exceeds chance under label permutation.",
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {args.out_json}", file=sys.stderr)

    if args.portfolio_json is not None:
        args.portfolio_json.parent.mkdir(parents=True, exist_ok=True)
        args.portfolio_json.write_text(json.dumps(out, indent=2, default=float))
        print(f"wrote {args.portfolio_json}", file=sys.stderr)

    # Console summary
    print("\n--- Cell summary (subject-level LOSO) ---", file=sys.stderr)
    for r in cell_summary:
        print(f"  {r['cell']:30s}  AUC = {r['loso_auc_mean']:.3f} "
              f"[{r['loso_auc_ci_lo']:.3f}-{r['loso_auc_ci_hi']:.3f}]   "
              f"BAcc = {r['loso_bacc_mean']:.3f} "
              f"[{r['loso_bacc_ci_lo']:.3f}-{r['loso_bacc_ci_hi']:.3f}]",
              file=sys.stderr)
    print("\n--- Pairwise DeLong (subject-level AUC) ---", file=sys.stderr)
    for r in pairwise:
        print(f"  {r['cell_1']:30s} vs {r['cell_2']:30s}  "
              f"ΔAUC = {r['delta_auc']:+.3f}   "
              f"z = {r['delong_z']:.2f}   p = {r['delong_p_two_sided']:.3f}",
              file=sys.stderr)
    print("\n--- Permutation p (per-fold BAcc vs chance) ---", file=sys.stderr)
    for r in perm:
        print(f"  {r['cell']:30s}  obs BAcc = {r['per_fold_bacc_obs']:.3f}   "
              f"p = {r['permutation_p']:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
