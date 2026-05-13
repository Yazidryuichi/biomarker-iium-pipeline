"""Per-fold audit of DM-SVM-linear vs classical-RF for the density-matrix
portfolio's Figure 5 numbers. Reproduces the stage6 matched CV exactly, dumps
per-fold balanced accuracy + AUC, and adds:
  - Paired Wilcoxon test (DM-SVM vs classical-RF) on AUC and BAcc.
  - 95% bootstrap CIs (10000 resamples) for the mean of each classifier's
    per-fold AUC and BAcc.

The result is a JSON sidecar the Vega-Lite Figure 5 spec can consume directly
for error bars + a paragraph injection in section 04-density-matrix.qmd.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULTS = Path("/Users/ryuichiyazid/Desktop/Second Brain/Academic/On going research/Biomarker_IIUM/pipeline/results")
OUT_CSV = RESULTS / "per_fold_dm_svm_vs_classical_rf.csv"
OUT_JSON = Path("/Users/ryuichiyazid/Desktop/Second Brain/Academic/Projects/Density-Matrix-Brainwaves/assets/figure5_stats.json")

CLASSICAL_PREFIXES = ("psd_", "coh_", "cov_", "tbr_", "faa_", "alpha_reactivity")
TARGET = "ef_group_global"
N_SELECT = 10
N_SPLITS, N_REPEATS, SEED = 10, 10, 42
N_BOOT = 10000


def main() -> int:
    full = pd.read_csv(RESULTS / "full_dataset.csv")
    dm = pd.read_csv(RESULTS / "density_matrix_features.csv")

    full = full.dropna(subset=[TARGET])
    labels = full[["subject_id", TARGET]].drop_duplicates("subject_id")

    merged_dm = dm.merge(labels, on="subject_id", how="inner")
    merged_dm = merged_dm.drop_duplicates("subject_id", keep="first")
    dm_cols = [c for c in merged_dm.columns if c.startswith("dm_")]
    X_dm = merged_dm[dm_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = merged_dm[TARGET].astype(int).to_numpy()
    subject_order = merged_dm["subject_id"].tolist()
    print(f"N subjects (DM): {len(y)}  features: {X_dm.shape[1]}", file=sys.stderr)
    print(f"class balance: {np.bincount(y).tolist()}", file=sys.stderr)

    cl_cols = [c for c in full.columns if c.startswith(CLASSICAL_PREFIXES)]
    cl_df = full.drop_duplicates("subject_id", keep="first").set_index("subject_id")
    cl_df = cl_df.loc[subject_order, cl_cols]
    X_cl = cl_df.fillna(0.0).to_numpy(dtype=np.float64)
    print(f"classical features: {X_cl.shape[1]}", file=sys.stderr)

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    splits = list(cv.split(np.zeros(len(y)), y))
    print(f"matched CV: {N_SPLITS}-fold x {N_REPEATS} repeats = {len(splits)} folds",
          file=sys.stderr)

    def evaluate(X, model):
        accs, aucs = [], []
        for tr, te in splits:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=min(N_SELECT, X.shape[1]))),
                ("clf", model),
            ])
            pipe.fit(X[tr], y[tr])
            yhat = pipe.predict(X[te])
            try:
                yprob = pipe.predict_proba(X[te])[:, 1]
                auc = roc_auc_score(y[te], yprob)
            except Exception:
                auc = np.nan
            accs.append(balanced_accuracy_score(y[te], yhat))
            aucs.append(auc)
        return np.array(accs), np.array(aucs)

    dm_acc, dm_auc = evaluate(
        X_dm, SVC(kernel="linear", C=1.0, probability=True, random_state=SEED)
    )
    cl_acc, cl_auc = evaluate(
        X_cl,
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=SEED),
    )

    df = pd.DataFrame({
        "fold": np.arange(len(splits)),
        "dm_svm_lin_bacc": dm_acc, "dm_svm_lin_auc": dm_auc,
        "classical_rf_bacc": cl_acc, "classical_rf_auc": cl_auc,
        "delta_bacc_dm_minus_cl": dm_acc - cl_acc,
        "delta_auc_dm_minus_cl": dm_auc - cl_auc,
    })
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote per-fold CSV: {OUT_CSV}", file=sys.stderr)

    print(f"\n--- Summary ---", file=sys.stderr)
    print(f"DM SVM-lin   BAcc: {dm_acc.mean():.4f} ± {dm_acc.std():.4f}"
          f"   AUC: {np.nanmean(dm_auc):.4f}", file=sys.stderr)
    print(f"Classical RF BAcc: {cl_acc.mean():.4f} ± {cl_acc.std():.4f}"
          f"   AUC: {np.nanmean(cl_auc):.4f}", file=sys.stderr)

    w_bacc, p_bacc = wilcoxon(dm_acc, cl_acc, zero_method="zsplit")
    w_auc, p_auc = wilcoxon(dm_auc[~np.isnan(dm_auc) & ~np.isnan(cl_auc)],
                            cl_auc[~np.isnan(dm_auc) & ~np.isnan(cl_auc)],
                            zero_method="zsplit")
    print(f"\nPaired Wilcoxon (DM-SVM vs classical-RF):", file=sys.stderr)
    print(f"  BAcc: W = {w_bacc:.2f},  p = {p_bacc:.4g}", file=sys.stderr)
    print(f"  AUC : W = {w_auc:.2f},  p = {p_auc:.4g}", file=sys.stderr)

    rng = np.random.default_rng(SEED)
    def boot_ci(x, n_boot=N_BOOT, alpha=0.05):
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return float("nan"), float("nan"), float("nan")
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        means = x[idx].mean(axis=1)
        lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(np.mean(x)), float(lo), float(hi)

    classifiers = {
        "DM linear SVM":            ("density_matrix", dm_auc, dm_acc),
        "DM Random Forest":         ("density_matrix",
            evaluate(X_dm, RandomForestClassifier(n_estimators=200, max_depth=3,
                                                  random_state=SEED))[1],
            evaluate(X_dm, RandomForestClassifier(n_estimators=200, max_depth=3,
                                                  random_state=SEED))[0]),
        "Classical RF":             ("classical", cl_auc, cl_acc),
        "QSVM 6q ZZ":               ("kernel_proxy", None, None),
        "Hilbert-Schmidt SVM":      ("kernel_proxy", None, None),
    }

    summary_rows = []
    for label, (group, auc_arr, acc_arr) in classifiers.items():
        if auc_arr is None:
            continue
        m_auc, lo_auc, hi_auc = boot_ci(np.array(auc_arr))
        m_acc, lo_acc, hi_acc = boot_ci(np.array(acc_arr))
        summary_rows.append({
            "model": label, "set": group,
            "auc_mean": m_auc, "auc_ci_lo": lo_auc, "auc_ci_hi": hi_auc,
            "bacc_mean": m_acc, "bacc_ci_lo": lo_acc, "bacc_ci_hi": hi_acc,
        })

    out = {
        "n_subjects": int(len(y)),
        "n_folds_total": int(len(splits)),
        "wilcoxon_dm_svm_vs_classical_rf": {
            "bacc": {"w": float(w_bacc), "p": float(p_bacc)},
            "auc":  {"w": float(w_auc),  "p": float(p_auc)},
        },
        "boot_ci": summary_rows,
        "headline_delta_auc_mean": float(np.nanmean(dm_auc) - np.nanmean(cl_auc)),
        "headline_delta_bacc_mean": float(dm_acc.mean() - cl_acc.mean()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote JSON: {OUT_JSON}", file=sys.stderr)
    print("\n" + json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
