"""
Apples-to-apples comparison: classification (binary at-risk label) vs
regression (continuous, post-hoc threshold) for the six pre-specified
targets.

Both arms:
  - Same X (curated feature pool, target-blind)
  - Same age-residualized target (residualization runs once per target)
  - Common evaluation label: y_binary = (y_residual <= tertile_bottom)
  - Same CV partition: KFold(n_splits=5, shuffle=True, random_state=42)
  - Paired models: L1 vs L1, ElasticNet vs ElasticNet, RF vs RF
  - Headline metric: ROC-AUC against the common y_binary label
    + bootstrap 95% CI (1000 resamples)

Output (under scripts/runs/<ts>/):
  - cls_vs_reg_long.csv         every (target, fset, model_pair) row
  - cls_vs_reg_summary.csv      best classifier vs best regressor per target
  - cls_vs_reg_heatmap.png      AUC delta heatmap (cls − reg) per target

Run: python scripts/compare_classification_vs_regression.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNetCV, LassoCV, LogisticRegressionCV,
)
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

from utils.io import latest_stage_dir, load_config, load_stage_config  # noqa: E402
from stages.analysis.feature_curation import (  # noqa: E402
    drop_collinear_hierarchical, drop_low_variance,
)
from stages.analysis.target_preparation import (  # noqa: E402
    derive_clinical_threshold, residualize_target,
)
from stages.analysis.analysis import get_feature_sets  # noqa: E402

TARGETS = [
    "Global_EF",
    "IC_score",
    "WM_score",
    "BW_Span",
    "ddm_v_incongruent",
    "ddm_delta_v",
]

RANDOM_STATE = 42
N_BOOT = 1000


# Paired model factories. Hyperparameters mirror the regression-first path
# in stages/analysis/analysis.py:_build_regressors so the only difference
# between arms is the loss function.
def _regressors():
    return {
        "Lasso":        LassoCV(cv=5, n_alphas=50, random_state=RANDOM_STATE,
                                max_iter=5000, n_jobs=1),
        "ElasticNet":   ElasticNetCV(cv=5, n_alphas=50,
                                     l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                                     random_state=RANDOM_STATE, max_iter=5000,
                                     n_jobs=1),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=3,
                                              min_samples_leaf=3,
                                              random_state=RANDOM_STATE, n_jobs=1),
    }


def _classifiers():
    return {
        "Lasso":        LogisticRegressionCV(
            cv=5, Cs=10, penalty="l1", solver="saga",
            scoring="roc_auc", max_iter=5000,
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        "ElasticNet":   LogisticRegressionCV(
            cv=5, Cs=10, penalty="elasticnet", solver="saga",
            l1_ratios=[0.5], scoring="roc_auc", max_iter=5000,
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
    }


def _build_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def _bootstrap_auc_ci(y_bin, score, n_boot=N_BOOT, rs=RANDOM_STATE):
    rng = np.random.RandomState(rs)
    n = len(y_bin)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y_bin[idx]
        sc = score[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yb, sc))
        except Exception:
            continue
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _screen_metrics(y_bin, score):
    """Sensitivity / specificity at the 33rd-percentile threshold of `score`,
    so prevalence matches the at-risk fraction of the true label."""
    if len(np.unique(y_bin)) < 2:
        return {k: float("nan") for k in
                ("auc", "sens", "spec", "f1", "balacc")}
    thr = float(np.quantile(score, 1.0 / 3.0))
    pred = (score >= thr).astype(int)
    try:
        auc = float(roc_auc_score(y_bin, score))
    except Exception:
        auc = float("nan")
    return {
        "auc":    auc,
        "sens":   float(recall_score(y_bin, pred, pos_label=1, zero_division=0)),
        "spec":   float(recall_score(y_bin, pred, pos_label=0, zero_division=0)),
        "f1":     float(f1_score(y_bin, pred, zero_division=0)),
        "balacc": float(balanced_accuracy_score(y_bin, pred)),
    }


def _curate(X_pool, fc_cfg):
    var_thresh = float(fc_cfg.get("variance_threshold", 1e-6))
    corr_thresh = float(fc_cfg.get("collinearity_threshold", 0.95))
    var_out = drop_low_variance(X_pool, threshold=var_thresh)
    X_after = var_out["X_filtered"]
    if X_after.shape[1] > 1:
        col_out = drop_collinear_hierarchical(
            X_after, list(X_after.columns), corr_threshold=corr_thresh,
        )
        X_curated = col_out["X_filtered"]
        kept = col_out["kept_names"]
    else:
        X_curated = X_after
        kept = list(X_after.columns)
    return X_curated, kept


def _run_target(name, full_df, X_curated_full, feature_sets_curated):
    print("\n" + "-" * 60)
    print(f"TARGET: {name}")
    print("-" * 60)

    if name not in full_df.columns or full_df[name].isna().all():
        print("  SKIP: missing or all-NaN")
        return []

    cov_df = full_df[["age_months"]]
    res = residualize_target(full_df[name], cov_df)
    y_residual = res["y_residual"]

    valid_mask = y_residual.notna()
    y_res_valid = y_residual.loc[valid_mask]
    thr_obj = derive_clinical_threshold(y_res_valid, method="tertile_bottom")
    y_bin_valid = thr_obj["y_binary"].astype(int).values
    print(f"  age R²={res['age_model_r2']:.3f}  threshold={thr_obj['threshold']:.4f}  "
          f"prevalence={thr_obj['prevalence']:.0%}  N={int(valid_mask.sum())}")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    regressors = _regressors()
    classifiers = _classifiers()

    for fs_name, fs_cols in feature_sets_curated.items():
        cols = [c for c in fs_cols if c in X_curated_full.columns]
        if not cols:
            continue
        X_fs = X_curated_full.loc[valid_mask, cols]

        for model_name in regressors:
            row = {"target": name, "feature_set": fs_name, "model_pair": model_name}
            try:
                # Regression arm.
                pipe_r = _build_pipeline(regressors[model_name])
                # cross_val_predict on continuous target.
                y_pred_cont = cross_val_predict(
                    pipe_r, X_fs.values, y_res_valid.values, cv=cv, n_jobs=1,
                )
                # Score for "lower residual = at-risk": flip sign.
                score_reg = -np.asarray(y_pred_cont, dtype=float)
                metrics_r = _screen_metrics(y_bin_valid, score_reg)
                ci_lo_r, ci_hi_r = _bootstrap_auc_ci(y_bin_valid, score_reg)

                # Classification arm.
                pipe_c = _build_pipeline(classifiers[model_name])
                y_proba = cross_val_predict(
                    pipe_c, X_fs.values, y_bin_valid, cv=cv, n_jobs=1,
                    method="predict_proba",
                )[:, 1]
                metrics_c = _screen_metrics(y_bin_valid, y_proba)
                ci_lo_c, ci_hi_c = _bootstrap_auc_ci(y_bin_valid, y_proba)

                for arm, m, ci_lo, ci_hi in [
                    ("regression",     metrics_r, ci_lo_r, ci_hi_r),
                    ("classification", metrics_c, ci_lo_c, ci_hi_c),
                ]:
                    rows.append({
                        **row,
                        "arm":      arm,
                        "auc":      round(m["auc"], 4),
                        "auc_ci_lo": round(ci_lo, 4),
                        "auc_ci_hi": round(ci_hi, 4),
                        "sens":     round(m["sens"], 4),
                        "spec":     round(m["spec"], 4),
                        "f1":       round(m["f1"], 4),
                        "balacc":   round(m["balacc"], 4),
                    })

                d_auc = metrics_c["auc"] - metrics_r["auc"]
                print(f"  {fs_name:30s} {model_name:12s}  "
                      f"reg AUC={metrics_r['auc']:+.3f}  "
                      f"cls AUC={metrics_c['auc']:+.3f}  "
                      f"delta={d_auc:+.3f}")
            except Exception as e:
                print(f"  {fs_name} | {model_name} | ERROR: {e}")

    return rows


def _summarize(long_df, out_dir):
    """Best regression vs best classification per target by AUC."""
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    for tgt, grp in long_df.groupby("target"):
        for arm in ("regression", "classification"):
            sub = grp[grp["arm"] == arm].dropna(subset=["auc"])
            if sub.empty:
                continue
            best = sub.loc[sub["auc"].idxmax()]
            rows.append({
                "target":      tgt,
                "arm":         arm,
                "best_model":  best["model_pair"],
                "best_fset":   best["feature_set"],
                "auc":         best["auc"],
                "auc_ci_lo":   best["auc_ci_lo"],
                "auc_ci_hi":   best["auc_ci_hi"],
                "sens":        best["sens"],
                "spec":        best["spec"],
            })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    # Pivot to side-by-side delta.
    wide = summary.pivot(index="target", columns="arm",
                         values=["auc", "auc_ci_lo", "auc_ci_hi", "sens", "spec"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide["delta_auc_cls_minus_reg"] = wide["auc_classification"] - wide["auc_regression"]
    wide = wide.reset_index()
    wide.to_csv(os.path.join(out_dir, "cls_vs_reg_summary.csv"), index=False)
    return wide


def _heatmap(summary, out_path):
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(summary) + 2))
    targets = summary["target"].tolist()
    reg = summary["auc_regression"].values
    cls = summary["auc_classification"].values
    delta = summary["delta_auc_cls_minus_reg"].values

    y = np.arange(len(targets))
    width = 0.38
    ax.barh(y - width / 2, reg, width, label="regression",
            color="#3a7", edgecolor="black")
    ax.barh(y + width / 2, cls, width, label="classification",
            color="#a37", edgecolor="black")

    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, label="chance")
    ax.set_yticks(y)
    ax.set_yticklabels(targets)
    ax.set_xlabel("Best ROC-AUC at tertile-bottom label")
    ax.set_title("Classification vs regression-then-threshold (best per target)")
    for i, d in enumerate(delta):
        ax.text(max(reg[i], cls[i]) + 0.01, i, f"Δ={d:+.3f}",
                va="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = load_config(os.path.join(PIPELINE_ROOT, "configs", "config.yaml"))
    eng_dir = latest_stage_dir(cfg, "engineering")
    if eng_dir is None:
        sys.exit("No engineering output found.")
    full_path = os.path.join(eng_dir, "full_dataset.csv")
    full_df = pd.read_csv(full_path)
    print(f"Loaded {full_path}  shape={full_df.shape}")

    # Borrow analysis stage's feature_sets and curation params from config.
    ana_cfg = load_stage_config("analysis", create_output_dir=False)
    fc_params = ana_cfg["analysis"].get("feature_curation", {}) or {}
    feature_sets_all = get_feature_sets(full_df, ana_cfg)
    pool_cols = sorted({c for cols in feature_sets_all.values() for c in cols
                        if c in full_df.columns})
    X_pool = full_df[pool_cols].apply(pd.to_numeric, errors="coerce")
    X_curated, kept_names = _curate(X_pool, fc_params)
    print(f"Feature curation: {len(pool_cols)} -> {X_curated.shape[1]} kept")

    # Restrict each feature_set to surviving columns.
    kept_set = set(kept_names)
    feature_sets_curated = {fs: [c for c in cols if c in kept_set]
                            for fs, cols in feature_sets_all.items()}
    feature_sets_curated = {k: v for k, v in feature_sets_curated.items() if v}

    X_curated_full = full_df[kept_names].apply(pd.to_numeric, errors="coerce")

    out_dir = os.path.join(
        PIPELINE_ROOT, "scripts", "runs",
        "cls_vs_reg_" + datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing outputs to {out_dir}\n")

    all_rows = []
    for tgt in TARGETS:
        all_rows.extend(_run_target(tgt, full_df, X_curated_full, feature_sets_curated))

    long_df = pd.DataFrame(all_rows)
    long_df.to_csv(os.path.join(out_dir, "cls_vs_reg_long.csv"), index=False)

    summary = _summarize(long_df, out_dir)
    if not summary.empty:
        print("\n" + "=" * 60)
        print("SUMMARY: best classifier vs best regressor (by AUC)")
        print("=" * 60)
        cols_show = ["target", "auc_regression", "auc_classification",
                     "delta_auc_cls_minus_reg"]
        print(summary[cols_show].to_string(index=False))
        _heatmap(summary, os.path.join(out_dir, "cls_vs_reg_heatmap.png"))

    print(f"\nDone. See {out_dir}")


if __name__ == "__main__":
    main()
