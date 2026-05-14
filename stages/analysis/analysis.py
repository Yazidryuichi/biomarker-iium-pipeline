"""
Stage 4: Statistical Analysis + ML (Regression-First)
======================================================
4A. Descriptive statistics and correlations (H1-H3)
4B. Pre-modeling preprocessing (unsupervised feature curation +
    age-residualization of the continuous target)
4C. Regression CV (LassoCV / ElasticNetCV / RandomForestRegressor)
    and post-hoc clinical screening from cross_val_predict.
4D. Permutation test + bootstrap CI on the best model.
4E. Legacy binary-classification path (kept behind config flag).
4F. SHAP analysis on the best regressor.
"""

import gc
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

logger = logging.getLogger("biomarker_iium.stage4")


# ──────────────────────────────────────────────────────────────────
# 4A. Descriptive & Correlational Analysis
# ──────────────────────────────────────────────────────────────────

def run_descriptives(df, config):
    """Descriptive statistics for all variables."""
    print("\n" + "-" * 40)
    print("4A. Descriptive Statistics")
    print("-" * 40)

    print(f"\n  N = {len(df)}")
    if "age_years" in df.columns:
        print(f"  Age: {df['age_years'].mean():.1f} +/- {df['age_years'].std():.1f} years")
    if "Sex" in df.columns:
        sex_counts = df["Sex"].value_counts()
        print(f"  Sex: {dict(sex_counts)}")

    beh_cols = ["Global_EF", "WM_score", "IC_score", "CF_score", "P_score", "SF_score",
                "flanker_effect", "ddm_v", "ddm_a", "ddm_t", "ddm_delta_v",
                "ddm_v_incongruent", "FW_Span", "BW_Span", "Total_Span"]
    beh_cols = [c for c in beh_cols if c in df.columns]

    desc = df[beh_cols].describe().round(3)
    print(f"\n  Behavioral measures:")
    print(desc.to_string())

    tbr_cols = [c for c in df.columns if "tbr_" in c and c.startswith("eo_")]
    if tbr_cols:
        print(f"\n  TBR features:")
        print(df[tbr_cols].describe().round(4).to_string())

    return desc


def run_correlations(df, config):
    """
    Test hypotheses H1-H3 (pre-specified, not data-driven).

    NOTE on FDR scope: correction is applied across these pre-specified
    tests only, not across all 200+ feature-outcome pairs.
    """
    print("\n" + "-" * 40)
    print("4A. Hypothesis Testing (Correlations)")
    print("-" * 40)

    results = []

    test_pairs = [
        ("eo_tbr_frontal_mean", "Global_EF", "H1: TBR(EO) vs Global EF (expected: negative)"),
        ("eo_psd_abs_theta_Fz", "Global_EF", "H2: Theta_Fz(EO) vs Global EF (expected: negative)"),
        ("eo_tbr_frontal_mean", "flanker_effect", "H3: TBR(EO) vs Flanker Effect (expected: positive)"),
        ("eo_tbr_Fz", "Global_EF", "TBR_Fz(EO) vs Global EF"),
        ("eo_tbr_Cz", "Global_EF", "TBR_Cz(EO) vs Global EF"),
        ("eo_faa_F4_F3", "Global_EF", "FAA(EO) vs Global EF"),
        ("alpha_reactivity_global", "Global_EF", "Alpha Reactivity vs Global EF"),
        ("eo_tbr_frontal_mean", "BW_Span", "TBR(EO) vs Digit Span Backward"),
        ("eo_tbr_frontal_mean", "ddm_v", "TBR(EO) vs DDM Drift Rate (expected: negative)"),
        ("eo_tbr_frontal_mean", "ddm_delta_v", "TBR(EO) vs DDM Delta-v (expected: negative)"),
        ("eo_psd_abs_theta_Fz", "ddm_v", "Theta_Fz(EO) vs DDM Drift Rate (expected: negative)"),
    ]

    for x_col, y_col, label in test_pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue

        x = df[x_col].dropna()
        y = df[y_col].reindex(x.index).dropna()
        x = x.reindex(y.index)

        if len(x) < 5:
            continue

        r_spearman, p_spearman = stats.spearmanr(x, y)
        r_pearson, p_pearson = stats.pearsonr(x, y)
        cohens_d = 2 * r_spearman / np.sqrt(1 - r_spearman**2 + 1e-10)

        results.append({
            "hypothesis": label,
            "x": x_col,
            "y": y_col,
            "method": "Spearman",
            "r": round(r_spearman, 3),
            "p": round(p_spearman, 4),
            "r_pearson": round(r_pearson, 3),
            "p_pearson": round(p_pearson, 4),
            "effect_size_d": round(cohens_d, 3),
            "n": len(x),
            "sig": "***" if p_spearman < 0.001 else "**" if p_spearman < 0.01 else "*" if p_spearman < 0.05 else "ns",
        })

        print(f"  {label}")
        print(f"    Spearman r = {r_spearman:.3f}, p = {p_spearman:.4f}, n = {len(x)} {'*' if p_spearman < 0.05 else 'ns'}")
        print(f"    (Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f})")

    if results:
        from statsmodels.stats.multitest import multipletests

        pvals = [r["p"] for r in results]
        reject, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh")
        for i, r in enumerate(results):
            r["p_fdr"] = round(pvals_corrected[i], 4)
            r["sig_fdr"] = "***" if pvals_corrected[i] < 0.001 else \
                           "**" if pvals_corrected[i] < 0.01 else \
                           "*" if pvals_corrected[i] < 0.05 else "ns"

        print("\n  After FDR correction:")
        for r in results:
            print(f"    {r['hypothesis']}: p_fdr = {r['p_fdr']} {r['sig_fdr']}")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────
# Feature sets (shared by regression + legacy classification paths)
# ──────────────────────────────────────────────────────────────────

def get_feature_sets(df, config):
    """
    Define feature sets for comparison. Returns dict {set_name: [columns]}.

    Each combined set (EO+EC) is accompanied by condition-specific subsets
    so the results table shows whether Eyes-Open or Eyes-Closed features
    drive predictive performance independently.
    """
    all_cols = df.columns.tolist()

    def _match(col, patterns):
        return any(p in col for p in patterns)

    CONV_PATTERNS    = ["psd_", "tbr_", "faa_", "alpha_reactivity", "coh_"]
    ADV_PATTERNS     = ["cwt_", "hjorth_", "spectral_entropy_", "pac_"]
    COV_PATTERN      = "cov_"
    QUANTUM_PATTERNS = ["qepp_", "qi_", "tn_"]

    conventional = [c for c in all_cols if _match(c, CONV_PATTERNS)]
    advanced     = list(dict.fromkeys(
        conventional + [c for c in all_cols if _match(c, ADV_PATTERNS)]
    ))
    covariance   = [c for c in all_cols if COV_PATTERN in c]
    all_features = list(dict.fromkeys(
        conventional + [c for c in all_cols if _match(c, ADV_PATTERNS + [COV_PATTERN])]
    ))
    quantum      = [c for c in all_cols if _match(c, QUANTUM_PATTERNS)]

    conventional_eo = [c for c in conventional if not c.startswith("ec_")]
    advanced_eo     = [c for c in advanced     if not c.startswith("ec_")]
    covariance_eo   = [c for c in covariance   if not c.startswith("ec_")]
    all_features_eo = [c for c in all_features if not c.startswith("ec_")]

    conventional_ec = [c for c in conventional if not c.startswith("eo_")]
    advanced_ec     = [c for c in advanced     if not c.startswith("eo_")]
    covariance_ec   = [c for c in covariance   if not c.startswith("eo_")]
    all_features_ec = [c for c in all_features if not c.startswith("eo_")]

    feature_sets = {
        "conventional_qeeg":             conventional,
        "conventional_plus_advanced":    advanced,
        "covariance_only":               covariance,
        "all_features":                  all_features,
        "conventional_qeeg_eo":          conventional_eo,
        "conventional_plus_advanced_eo": advanced_eo,
        "covariance_only_eo":            covariance_eo,
        "all_features_eo":               all_features_eo,
        "conventional_qeeg_ec":          conventional_ec,
        "conventional_plus_advanced_ec": advanced_ec,
        "covariance_only_ec":            covariance_ec,
        "all_features_ec":               all_features_ec,
    }

    if quantum:
        feature_sets["quantum_only"] = quantum
        feature_sets["classical_plus_quantum"] = list(dict.fromkeys(all_features + quantum))

    requested = config.get("analysis", {}).get("feature_sets")
    if requested:
        requested = [s.split("#")[0].strip() for s in requested if s]
        unknown = [s for s in requested if s not in feature_sets]
        if unknown:
            print(f"  [WARN] feature_sets not recognised, skipping: {unknown}")
        feature_sets = {k: v for k, v in feature_sets.items() if k in requested}

    return feature_sets


# ──────────────────────────────────────────────────────────────────
# 4B. Pre-modelling preprocessing
#     (unsupervised feature curation + target residualization)
# ──────────────────────────────────────────────────────────────────

def run_feature_curation(X_df, params, output_dir=None):
    """
    Run unsupervised feature curation (variance + collinearity) and write
    a JSON QC report. Returns (X_filtered, kept_names, report).
    """
    from stages.analysis.feature_curation import (
        drop_low_variance, drop_collinear_hierarchical,
    )

    var_thresh = float(params.get("variance_threshold", 1e-6))
    corr_thresh = float(params.get("collinearity_threshold", 0.95))

    feature_names = list(X_df.columns)
    n_input = len(feature_names)

    # Pass 1: low variance
    var_out = drop_low_variance(X_df, threshold=var_thresh)
    X_after_var = var_out["X_filtered"]
    n_after_var = X_after_var.shape[1]
    dropped_var = [feature_names[i] for i in var_out["dropped_idx"].tolist()]

    # Pass 2: collinearity
    if X_after_var.shape[1] > 1:
        col_out = drop_collinear_hierarchical(
            X_after_var, list(X_after_var.columns), corr_threshold=corr_thresh,
        )
    else:
        col_out = {
            "X_filtered": X_after_var,
            "kept_names": list(X_after_var.columns),
            "dropped_names": [],
            "cluster_map": {c: [] for c in X_after_var.columns},
        }
    X_final = col_out["X_filtered"]
    n_final = X_final.shape[1]

    report = {
        "n_input": int(n_input),
        "n_after_variance": int(n_after_var),
        "n_after_collinearity": int(n_final),
        "variance_threshold": var_thresh,
        "collinearity_threshold": corr_thresh,
        "dropped_low_variance": dropped_var,
        "dropped_collinear": col_out["dropped_names"],
        "kept_features": col_out["kept_names"],
        "cluster_map": col_out["cluster_map"],
    }
    if output_dir is not None and params.get("save_diagnostics", True):
        with open(os.path.join(output_dir, "feature_curation_report.json"),
                  "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    print(f"  Feature curation: {n_input} -> {n_after_var} (variance) -> "
          f"{n_final} (collinearity)")
    return X_final, list(col_out["kept_names"]), report


def run_target_preparation(df, target_cfg, output_dir=None):
    """
    Residualize the primary target and derive the clinical threshold.
    Returns dict with y_continuous, y_residual, y_binary, threshold,
    valid_idx (intersection of feature/target/covariate non-NaN), and a
    JSON-serializable report.
    """
    from stages.analysis.target_preparation import (
        residualize_target, derive_clinical_threshold,
    )

    primary = target_cfg.get("primary", {}) or {}
    target_name = primary.get("name", "ddm_v_incongruent")
    covariates = primary.get("residualize_covariates") or ["age_months"]
    threshold_cfg = target_cfg.get("clinical_threshold", {}) or {}
    method = threshold_cfg.get("method", "tertile_bottom")
    sens_methods = threshold_cfg.get("sensitivity_methods") or []

    if target_name not in df.columns:
        raise KeyError(
            f"Target '{target_name}' not found in dataset. "
            f"Available continuous targets: "
            f"{[c for c in df.columns if c.startswith(('ddm_', 'flanker_'))][:20]}"
        )

    missing_covs = [c for c in covariates if c not in df.columns]
    if missing_covs:
        raise KeyError(
            f"Residualization covariates missing: {missing_covs}. "
            f"Available age columns: {[c for c in df.columns if 'age' in c]}"
        )

    cov_df = df[covariates]
    y_raw = df[target_name]

    res = residualize_target(y_raw, cov_df)
    y_residual = res["y_residual"]

    # Primary clinical threshold
    primary_thr = derive_clinical_threshold(y_residual, method=method)

    # Sensitivity analysis: also compute thresholds for alternate methods
    sensitivity = {}
    for m in sens_methods:
        sensitivity[m] = derive_clinical_threshold(y_residual, method=m)

    valid_mask = y_residual.notna() & cov_df.notna().all(axis=1)
    valid_idx = df.index[valid_mask]

    # Stats on residual
    res_valid = y_residual.loc[valid_idx]
    raw_valid = pd.to_numeric(y_raw, errors="coerce").loc[valid_idx]

    report = {
        "target_name": target_name,
        "covariates": covariates,
        "n_used_for_residualization": int(res["n_used"]),
        "age_model_r2": round(float(res["age_model_r2"]), 4),
        "model_coef": {k: round(v, 6) for k, v in res["model_coef"].items()},
        "intercept": round(float(res["intercept"]), 6),
        "raw_target_stats": {
            "mean": round(float(raw_valid.mean()), 4),
            "std":  round(float(raw_valid.std()), 4),
            "skew": round(float(stats.skew(raw_valid.dropna())), 4),
            "n":    int(raw_valid.notna().sum()),
        },
        "residual_stats": {
            "mean": round(float(res_valid.mean()), 6),
            "std":  round(float(res_valid.std()), 6),
            "skew": round(float(stats.skew(res_valid.dropna())), 4),
            "n":    int(res_valid.notna().sum()),
        },
        "primary_threshold": {
            "method":     primary_thr["method"],
            "quantile":   primary_thr["quantile"],
            "threshold":  round(float(primary_thr["threshold"]), 6),
            "prevalence": round(float(primary_thr["prevalence"]), 4),
            "n_at_risk":  int((primary_thr["y_binary"] == 1).sum()),
        },
        "sensitivity_thresholds": {
            m: {
                "method":     v["method"],
                "quantile":   v["quantile"],
                "threshold":  round(float(v["threshold"]), 6),
                "prevalence": round(float(v["prevalence"]), 4),
                "n_at_risk":  int((v["y_binary"] == 1).sum()),
            }
            for m, v in sensitivity.items()
        },
    }
    if output_dir is not None:
        with open(os.path.join(output_dir, "target_preparation_report.json"),
                  "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    print(f"  Target: {target_name}")
    print(f"  Covariate model R^2 (raw target ~ {covariates}): {res['age_model_r2']:.3f}")
    print(f"  Threshold ({method}, q={primary_thr['quantile']:.3f}): "
          f"{primary_thr['threshold']:.4f} | prevalence={primary_thr['prevalence']:.2%}")

    return {
        "target_name": target_name,
        "y_continuous_raw": pd.to_numeric(y_raw, errors="coerce"),
        "y_residual": y_residual,
        "y_binary":   primary_thr["y_binary"],
        "threshold":  primary_thr["threshold"],
        "primary_thr_obj": primary_thr,
        "sensitivity_thr_objs": sensitivity,
        "valid_idx": valid_idx,
        "report": report,
    }


# ──────────────────────────────────────────────────────────────────
# 4C. Regression
# ──────────────────────────────────────────────────────────────────

def _build_regressors(random_state, enabled_models):
    """
    Return dict of {name: regressor} for those names listed in enabled_models.

    Hyperparameter policy (CV-only protocol — no train/test holdout, no
    nested RandomizedSearchCV):
      - LassoCV / ElasticNetCV: built-in k-fold tuning over the
        regularisation path. No outer wrapper needed.
      - RandomForestRegressor: FIXED hyperparameters
        (n_estimators=100, max_depth=3, min_samples_leaf=3) chosen
        for N=26. No tuning — small N + tree-search makes data-driven
        tuning produce optimistic estimates (Vabalas et al. 2019).
      - SVR / GradientBoostingRegressor: fixed conservative defaults.
    """
    from sklearn.linear_model import LassoCV, ElasticNetCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR

    factory = {
        "LassoCV": lambda: LassoCV(cv=5, n_alphas=50, random_state=random_state,
                                   max_iter=5000, n_jobs=1),
        "ElasticNetCV": lambda: ElasticNetCV(
            cv=5, n_alphas=50, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            random_state=random_state, max_iter=5000, n_jobs=1,
        ),
        "RandomForestRegressor": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=3,
            random_state=random_state, n_jobs=1,
        ),
        "SVR": lambda: SVR(kernel="rbf", C=1.0, gamma="scale"),
        "GradientBoostingRegressor": lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=random_state,
        ),
    }
    out = {}
    for name in enabled_models:
        name_clean = name.split("#")[0].strip()
        if name_clean not in factory:
            print(f"  [WARN] Unknown regressor '{name_clean}', skipping.")
            continue
        out[name_clean] = factory[name_clean]()
    return out


def _make_cv(cv_cfg, random_state_default):
    from sklearn.model_selection import RepeatedKFold
    return RepeatedKFold(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        n_repeats=int(cv_cfg.get("n_repeats", 5)),
        random_state=int(cv_cfg.get("random_state", random_state_default)),
    )


def _build_regression_pipeline(model, univariate_cfg, n_features_in):
    """
    Build sklearn Pipeline: SimpleImputer -> StandardScaler ->
    [optional univariate filter] -> regressor.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]
    if univariate_cfg.get("enable", False):
        from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
        method = univariate_cfg.get("method", "mutual_info_regression")
        score_func = mutual_info_regression if method == "mutual_info_regression" else f_regression
        if univariate_cfg.get("k_adaptive", True):
            k = max(5, min(40, n_features_in // 5))
        else:
            k = int(univariate_cfg.get("k", 20))
        k = min(k, max(n_features_in, 1))
        steps.append(("select", SelectKBest(score_func, k=k)))
    steps.append(("reg", model))
    return Pipeline(steps)


def _pearson_safe(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def _spearman_safe(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    r, _ = stats.spearmanr(y_true, y_pred)
    return float(r)


def run_regression(X_df, y_residual, feature_sets, config, output_dir):
    """
    Run regression CV across (feature_set, model) combinations.

    Returns:
        regression_results_df: per-fold and per-repeat metrics.
        best_info: dict {feature_set, model, mean_pearson_r}.
        oof_predictions: dict (fs_name, model_name) -> dict with
            y_true, y_pred (OOF concatenation across all folds/repeats),
            and per-repeat mean Pearson r used for downstream screening.
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_absolute_error, r2_score

    ana = config["analysis"]
    cv_cfg = ana.get("cv", {}) or {}
    random_state = config["random_state"]
    cv = _make_cv(cv_cfg, random_state)

    enabled_models = [m.split("#")[0].strip() for m in (ana.get("models") or [])
                      if isinstance(m, str) and m.strip()]
    models = _build_regressors(random_state, enabled_models)
    if not models:
        print("  ERROR: No regressors enabled. Check analysis.models in config.yaml")
        return pd.DataFrame(), {}, {}
    print(f"  Regressors enabled: {list(models.keys())}")

    univariate_cfg = ana.get("univariate_filter", {}) or {}

    # Restrict to subjects with non-NaN target (residualization may yield NaN
    # for rows missing covariates).
    valid_mask = y_residual.notna()
    y_use = y_residual.loc[valid_mask]
    X_use = X_df.loc[valid_mask]
    print(f"  Regression sample: N={len(y_use)} after dropping NaN target")

    rows = []
    oof_predictions = {}

    for fs_name, fs_cols in feature_sets.items():
        cols_present = [c for c in fs_cols if c in X_use.columns]
        if not cols_present:
            continue
        X_fs = X_use[cols_present]

        for model_name, factory_model in models.items():
            try:
                # Per-fold metrics: walk CV manually so we get fold-level r/spearman.
                fold_pearson, fold_spearman, fold_r2, fold_mae = [], [], [], []
                # Manual CV loop over the repeated splitter for per-fold scores.
                n_splits = int(cv_cfg.get("n_splits", 5))
                for k, (train_idx, test_idx) in enumerate(cv.split(X_fs, y_use)):
                    pipe_k = _build_regression_pipeline(
                        factory_model, univariate_cfg, len(cols_present)
                    )
                    pipe_k.fit(X_fs.iloc[train_idx], y_use.iloc[train_idx])
                    y_hat = pipe_k.predict(X_fs.iloc[test_idx])
                    y_te  = y_use.iloc[test_idx].values

                    fold_pearson.append(_pearson_safe(y_te, y_hat))
                    fold_spearman.append(_spearman_safe(y_te, y_hat))
                    try:
                        fold_r2.append(float(r2_score(y_te, y_hat)))
                    except Exception:
                        fold_r2.append(float("nan"))
                    fold_mae.append(float(mean_absolute_error(y_te, y_hat)))

                    rows.append({
                        "feature_set": fs_name,
                        "n_features_input": len(cols_present),
                        "model": model_name,
                        "repeat": k // n_splits,
                        "fold":   k % n_splits,
                        "pearson_r":  fold_pearson[-1],
                        "spearman_r": fold_spearman[-1],
                        "r2":         fold_r2[-1],
                        "mae":        fold_mae[-1],
                    })

                # Out-of-fold predictions across one full RepeatedKFold pass
                # (used for screening metrics). cross_val_predict here uses
                # only the first repeat's partition; that is fine for binary
                # screening — we just need a single OOF prediction per subject.
                from sklearn.model_selection import KFold
                single_cv = KFold(
                    n_splits=int(cv_cfg.get("n_splits", 5)),
                    shuffle=True, random_state=random_state,
                )
                pipe_oof = _build_regression_pipeline(
                    factory_model, univariate_cfg, len(cols_present),
                )
                y_oof = cross_val_predict(pipe_oof, X_fs, y_use, cv=single_cv, n_jobs=1)
                oof_predictions[(fs_name, model_name)] = {
                    "y_true": y_use.values.copy(),
                    "y_pred": np.asarray(y_oof, dtype=float),
                    "valid_index": y_use.index.copy(),
                    "mean_pearson": float(np.nanmean(fold_pearson)),
                }

                print(f"  {fs_name:30s} | {model_name:22s} | "
                      f"r={np.nanmean(fold_pearson):+.3f} (sd {np.nanstd(fold_pearson):.3f}) | "
                      f"rho={np.nanmean(fold_spearman):+.3f} | "
                      f"R²={np.nanmean(fold_r2):+.3f} | "
                      f"MAE={np.nanmean(fold_mae):.3f}")

            except Exception as e:
                logger.error(f"{fs_name} | {model_name} | {e}")
                print(f"  {fs_name} | {model_name} | ERROR: {e}")
            finally:
                gc.collect()

    df_results = pd.DataFrame(rows)
    if df_results.empty:
        return df_results, {}, oof_predictions

    # Best by mean Pearson r per (feature_set, model)
    summary = (df_results.groupby(["feature_set", "model"])["pearson_r"]
                          .mean().reset_index()
                          .rename(columns={"pearson_r": "mean_pearson_r"}))
    summary = summary.sort_values("mean_pearson_r", ascending=False)
    best_row = summary.iloc[0]
    best_info = {
        "feature_set": best_row["feature_set"],
        "model": best_row["model"],
        "mean_pearson_r": float(best_row["mean_pearson_r"]),
    }
    print(f"\n  BEST regressor: {best_info['model']} on '{best_info['feature_set']}' "
          f"(mean Pearson r = {best_info['mean_pearson_r']:+.3f})")

    df_results.to_csv(os.path.join(output_dir, "regression_results.csv"),
                      index=False)
    return df_results, best_info, oof_predictions


# ──────────────────────────────────────────────────────────────────
# 4C-2. Clinical screening from regression predictions
# ──────────────────────────────────────────────────────────────────

def _binary_metrics_from_continuous(y_true_cont, y_pred_cont, threshold,
                                    direction_at_risk_low=True):
    """
    Apply a continuous threshold to true and predicted values and compute
    standard screening metrics. AUC uses the continuous predictions
    directly so it is threshold-free.
    """
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, recall_score, roc_auc_score,
        precision_score,
    )

    y_true_bin = (y_true_cont <= threshold).astype(int) if direction_at_risk_low \
                 else (y_true_cont >= threshold).astype(int)
    y_pred_bin = (y_pred_cont <= threshold).astype(int) if direction_at_risk_low \
                 else (y_pred_cont >= threshold).astype(int)

    # Score that is HIGHER for at-risk subjects (sklearn AUC convention).
    score_for_risk = -y_pred_cont if direction_at_risk_low else y_pred_cont

    out = {
        "sensitivity":      recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0),
        "specificity":      recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0),
        "ppv":              precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0),
        "npv":              precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0),
        "f1":               f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true_bin, y_pred_bin),
        "n_pos_true":       int((y_true_bin == 1).sum()),
        "n_pos_pred":       int((y_pred_bin == 1).sum()),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true_bin, score_for_risk))
    except Exception:
        out["auc"] = float("nan")
    return out


def run_clinical_screening(oof_predictions, target_prep, output_dir):
    """
    Convert OOF regression predictions into binary screening metrics by
    applying the pre-computed clinical threshold (and sensitivity-analysis
    thresholds).
    Returns the metrics DataFrame.
    """
    rows = []
    primary_thr = target_prep["primary_thr_obj"]
    sens_thrs   = target_prep["sensitivity_thr_objs"]

    threshold_objs = {primary_thr["method"]: primary_thr, **sens_thrs}

    for (fs_name, model_name), oof in oof_predictions.items():
        for thr_method, thr_obj in threshold_objs.items():
            metrics = _binary_metrics_from_continuous(
                oof["y_true"], oof["y_pred"],
                threshold=thr_obj["threshold"],
                direction_at_risk_low=True,
            )
            row = {
                "feature_set":      fs_name,
                "model":            model_name,
                "threshold_method": thr_method,
                "threshold_value":  round(float(thr_obj["threshold"]), 6),
                "is_primary":       thr_method == primary_thr["method"],
            }
            row.update({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in metrics.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["is_primary", "auc"], ascending=[False, False])
        df.to_csv(os.path.join(output_dir, "clinical_screening_metrics.csv"),
                  index=False)

        primary = df[df["is_primary"]]
        if not primary.empty:
            top = primary.iloc[0]
            print(f"\n  Best clinical screening (primary threshold "
                  f"'{top['threshold_method']}'): "
                  f"AUC={top['auc']:.3f}, sens={top['sensitivity']:.3f}, "
                  f"spec={top['specificity']:.3f} "
                  f"({top['model']} on '{top['feature_set']}')")

    return df


# ──────────────────────────────────────────────────────────────────
# 4D. Statistical validation: permutation + bootstrap + FDR
# ──────────────────────────────────────────────────────────────────

def run_permutation_test(X_df, y_residual, feature_sets, regression_results_df,
                         config, output_dir):
    """
    Permutation test on the BEST (feature_set, model) by mean Pearson r.
    Permutes y; uses cross_val_predict to get OOF predictions per
    permutation, then Pearson r on those.
    """
    from sklearn.model_selection import KFold, cross_val_predict

    ana = config["analysis"]
    n_perm = int((ana.get("validation", {}) or {}).get("permutation_n", 200))
    cv_cfg = ana.get("cv", {}) or {}
    random_state = config["random_state"]

    if regression_results_df.empty:
        return {}

    summary = (regression_results_df.groupby(["feature_set", "model"])["pearson_r"]
                                    .mean().reset_index()
                                    .sort_values("pearson_r", ascending=False))
    best = summary.iloc[0]
    best_fs, best_model = best["feature_set"], best["model"]

    enabled_models = ana.get("models", []) or []
    enabled_models = [m.split("#")[0].strip() for m in enabled_models if m]
    models = _build_regressors(random_state, enabled_models)
    model = models.get(best_model)
    if model is None:
        return {}

    cols_present = [c for c in feature_sets[best_fs] if c in X_df.columns]
    valid_mask = y_residual.notna()
    X_fs = X_df.loc[valid_mask, cols_present]
    y_use = y_residual.loc[valid_mask].values

    univariate_cfg = ana.get("univariate_filter", {}) or {}
    cv = KFold(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        shuffle=True, random_state=random_state,
    )

    print(f"\n  Permutation test ({n_perm} perms): {best_model} on '{best_fs}'...")

    pipe = _build_regression_pipeline(model, univariate_cfg, len(cols_present))
    y_pred = cross_val_predict(pipe, X_fs, y_use, cv=cv, n_jobs=1)
    observed_r = _pearson_safe(y_use, y_pred)

    rng = np.random.RandomState(random_state)
    null_r = []
    for i in range(n_perm):
        y_perm = rng.permutation(y_use)
        try:
            pipe_perm = _build_regression_pipeline(
                _build_regressors(random_state, [best_model])[best_model],
                univariate_cfg, len(cols_present),
            )
            y_pred_perm = cross_val_predict(pipe_perm, X_fs, y_perm, cv=cv, n_jobs=1)
            null_r.append(_pearson_safe(y_perm, y_pred_perm))
        except Exception:
            null_r.append(float("nan"))

    null_r = np.array(null_r, dtype=float)
    null_r = null_r[~np.isnan(null_r)]
    p_value = float((np.sum(null_r >= observed_r) + 1) / (len(null_r) + 1))
    print(f"    observed r = {observed_r:+.3f} | null mean = "
          f"{np.mean(null_r):+.3f} | p = {p_value:.4f}")

    payload = {
        "best_feature_set": best_fs,
        "best_model":       best_model,
        "metric":           "pearson_r",
        "observed":         round(float(observed_r), 4),
        "null_mean":        round(float(np.mean(null_r)), 4),
        "null_std":         round(float(np.std(null_r)), 4),
        "p_value":          round(float(p_value), 4),
        "n_permutations":   int(len(null_r)),
        "null_distribution": [round(float(v), 4) for v in null_r],
    }
    with open(os.path.join(output_dir, "permutation_results.json"),
              "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload


def run_bootstrap_ci(oof_predictions, screening_df, target_prep, config,
                     output_dir):
    """
    Bootstrap 95% CI on (a) Pearson r per (feature_set, model) and
    (b) AUC per (feature_set, model, threshold_method).
    """
    from sklearn.metrics import roc_auc_score

    ana = config["analysis"]
    n_boot = int((ana.get("validation", {}) or {}).get("bootstrap_n", 1000))
    rng = np.random.RandomState(config["random_state"])

    primary_thr = target_prep["primary_thr_obj"]
    sens_thrs   = target_prep["sensitivity_thr_objs"]
    threshold_objs = {primary_thr["method"]: primary_thr, **sens_thrs}

    pearson_cis = {}
    auc_cis = {}

    for (fs_name, model_name), oof in oof_predictions.items():
        y_true = oof["y_true"]
        y_pred = oof["y_pred"]
        n = len(y_true)
        if n < 5:
            continue

        boot_r, boot_auc = {m: [] for m in threshold_objs}, {m: [] for m in threshold_objs}
        boot_pearson = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            y_t = y_true[idx]
            y_p = y_pred[idx]
            boot_pearson.append(_pearson_safe(y_t, y_p))
            for thr_method, thr_obj in threshold_objs.items():
                y_bin = (y_t <= thr_obj["threshold"]).astype(int)
                if len(np.unique(y_bin)) < 2:
                    boot_auc[thr_method].append(float("nan"))
                    continue
                try:
                    auc = roc_auc_score(y_bin, -y_p)
                    boot_auc[thr_method].append(float(auc))
                except Exception:
                    boot_auc[thr_method].append(float("nan"))

        boot_pearson = np.array(boot_pearson, dtype=float)
        boot_pearson = boot_pearson[~np.isnan(boot_pearson)]
        if boot_pearson.size:
            pearson_cis[f"{fs_name}|{model_name}"] = {
                "metric": "pearson_r",
                "ci_lower": round(float(np.percentile(boot_pearson, 2.5)), 4),
                "ci_upper": round(float(np.percentile(boot_pearson, 97.5)), 4),
                "median":   round(float(np.percentile(boot_pearson, 50)), 4),
            }
        for thr_method, vals in boot_auc.items():
            arr = np.array(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size:
                auc_cis[f"{fs_name}|{model_name}|{thr_method}"] = {
                    "metric": "auc",
                    "threshold_method": thr_method,
                    "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
                    "ci_upper": round(float(np.percentile(arr, 97.5)), 4),
                    "median":   round(float(np.percentile(arr, 50)), 4),
                }

    payload = {
        "n_bootstrap": n_boot,
        "pearson_r":   pearson_cis,
        "screening_auc": auc_cis,
    }
    with open(os.path.join(output_dir, "bootstrap_ci.json"),
              "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload


def apply_fdr_to_pearson(regression_results_df, output_dir):
    """
    Per-(feature_set, model) one-sample t-test of fold-level Pearson r > 0,
    then BH-FDR across the resulting p-values. Result merged into a
    ``regression_summary.csv`` for the audit trail.
    """
    if regression_results_df.empty:
        return pd.DataFrame()

    from statsmodels.stats.multitest import multipletests

    rows = []
    grouped = regression_results_df.groupby(["feature_set", "model"])
    for (fs, mdl), g in grouped:
        rs = g["pearson_r"].dropna().values
        if rs.size < 2:
            continue
        # One-sample t-test against 0 (one-sided: r > 0 desired).
        t_stat, p_two = stats.ttest_1samp(rs, 0.0)
        p_one = float(p_two / 2 if t_stat > 0 else 1 - p_two / 2)
        rows.append({
            "feature_set":   fs,
            "model":         mdl,
            "mean_pearson":  round(float(np.mean(rs)), 4),
            "std_pearson":   round(float(np.std(rs)), 4),
            "t_stat":        round(float(t_stat), 4),
            "p_one_sided":   round(p_one, 4),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pvals = df["p_one_sided"].values
    _, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    df["p_fdr"] = np.round(p_fdr, 4)
    df["sig_fdr"] = ["**" if p < 0.01 else "*" if p < 0.05 else "ns"
                     for p in p_fdr]
    df = df.sort_values("mean_pearson", ascending=False)
    df.to_csv(os.path.join(output_dir, "regression_summary.csv"), index=False)
    return df


# ──────────────────────────────────────────────────────────────────
# 4E. QC plots
# ──────────────────────────────────────────────────────────────────

def _save_target_distribution_plot(y_raw, y_residual, threshold, output_dir,
                                   target_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [WARN] matplotlib unavailable: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    raw_vals = pd.to_numeric(y_raw, errors="coerce").dropna().values
    res_vals = y_residual.dropna().values

    axes[0].hist(raw_vals, bins=15, color="#888888", edgecolor="black")
    axes[0].set_title(f"Raw {target_name}")
    axes[0].set_xlabel(target_name)
    axes[0].set_ylabel("count")

    axes[1].hist(res_vals, bins=15, color="#3a7", edgecolor="black")
    axes[1].axvline(threshold, color="crimson", linestyle="--",
                    label=f"threshold = {threshold:.3f}")
    axes[1].set_title(f"Age-residualized {target_name}")
    axes[1].set_xlabel(f"residual({target_name} | covariates)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=130)
    plt.close(fig)


def _save_correlation_heatmaps(X_before, X_after, output_dir, max_features=80):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def _heatmap(X, path, title):
        cols = list(X.columns)
        if len(cols) > max_features:
            # Sort by variance and show the top max_features so the heatmap
            # remains legible on wide matrices.
            order = np.argsort(-np.nanvar(X.values, axis=0))[:max_features]
            X = X.iloc[:, order]
            cols = list(X.columns)
        if len(cols) < 2:
            return
        corr = X.corr().values
        fig, ax = plt.subplots(figsize=(min(10, 0.13 * len(cols) + 4),
                                        min(10, 0.13 * len(cols) + 4)))
        im = ax.imshow(np.abs(corr), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{title} (|corr|, top {len(cols)} by variance)")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    _heatmap(X_before,
             os.path.join(output_dir, "feature_correlation_heatmap_before.png"),
             "Before curation")
    _heatmap(X_after,
             os.path.join(output_dir, "feature_correlation_heatmap_after.png"),
             "After curation")


# ──────────────────────────────────────────────────────────────────
# 4F. SHAP (regressor edition)
# ──────────────────────────────────────────────────────────────────

def run_shap_regression(X_df, y_residual, best_info, config, output_dir):
    """
    SHAP for the best regressor on the curated feature matrix. Uses
    TreeExplainer when the model is tree-based, otherwise falls back to
    LinearExplainer / KernelExplainer.
    """
    if not best_info:
        return None
    try:
        import shap
    except ImportError:
        print("  SHAP not installed. pip install shap")
        return None

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LassoCV, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    valid_mask = y_residual.notna()
    y = y_residual.loc[valid_mask].values
    X = X_df.loc[valid_mask].copy()

    print(f"\n  SHAP on best regressor: {best_info['model']} "
          f"(curated features, n={X.shape[1]})")

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_proc = scaler.fit_transform(imputer.fit_transform(X.values))

    name = best_info["model"]
    rs = config["random_state"]
    # Single source of truth for hyperparameters: _build_regressors.
    # Fall back to RandomForestRegressor for SVR / unknown so SHAP can
    # use a tree explainer.
    built = _build_regressors(rs, [name, "RandomForestRegressor"])
    model = built.get(name)
    if model is None:
        model = built["RandomForestRegressor"]

    model.fit(X_proc, y)

    try:
        if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_proc)
        elif isinstance(model, (LassoCV, ElasticNetCV)):
            explainer = shap.LinearExplainer(model, X_proc)
            shap_values = explainer.shap_values(X_proc)
        else:
            explainer = shap.Explainer(model.predict, X_proc[: min(50, len(X_proc))])
            shap_values = explainer(X_proc).values
    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}")
        return None

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance = (pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
                    .sort_values("mean_abs_shap", ascending=False))
    importance.to_csv(os.path.join(output_dir, "shap_importance.csv"), index=False)

    print("\n  Top biomarker candidates (regression SHAP):")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:40s}  SHAP: {row['mean_abs_shap']:.4f}")

    try:
        from utils.bio_interpretation import interpret_biomarkers
        annotated = interpret_biomarkers(importance)
        annotated.to_csv(os.path.join(output_dir, "shap_annotated.csv"), index=False)
    except Exception:
        pass

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        shap.summary_plot(shap_values, X_proc, feature_names=list(X.columns),
                          show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "shap_summary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not save SHAP plot: {e}")

    return importance


# ──────────────────────────────────────────────────────────────────
# 4-LEGACY. Binary classification (kept behind config flag)
# ──────────────────────────────────────────────────────────────────

def _build_legacy_classifiers(random_state, enabled_models, include_qsvm=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=3,
            class_weight="balanced", random_state=random_state),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True,
                   class_weight="balanced", random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             early_stopping=True, random_state=random_state),
    }
    if include_qsvm:
        try:
            from stages.analysis.qsvm_classifier import QuantumKernelSVM
            models["QSVM_4q_ZZ"]   = QuantumKernelSVM(n_qubits=4, n_layers=2, C=1.0, use_entangling=True)
            models["QSVM_6q_ZZ"]   = QuantumKernelSVM(n_qubits=6, n_layers=2, C=1.0, use_entangling=True)
            models["QSVM_6q_prod"] = QuantumKernelSVM(n_qubits=6, n_layers=2, C=1.0, use_entangling=False)
        except ImportError:
            print("  [INFO] QSVM requested but pennylane not installed — skipping")
    return {k: v for k, v in models.items() if k in enabled_models}


def run_legacy_classification(df, config, target, output_dir):
    """
    Pre-refactor classification flow on the same continuous target,
    binarized via fold-internal median split. Kept for sensitivity-analysis
    comparison against the regression pipeline.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, roc_auc_score, recall_score,
    )

    print("\n" + "-" * 40)
    print("4-LEGACY. Binary classification (median split inside CV)")
    print("-" * 40)

    legacy_cfg = (config["analysis"].get("legacy_classification") or {})
    cv_folds = int(legacy_cfg.get("cv_folds", 5))
    cv_repeats = int(legacy_cfg.get("cv_repeats", 5))
    enabled = legacy_cfg.get("models", ["RandomForest", "SVM"])
    enabled = [m.split("#")[0].strip() for m in enabled if m]
    include_qsvm = bool(legacy_cfg.get("include_qsvm", False))
    random_state = config["random_state"]

    if target not in df.columns or df[target].isna().all():
        print(f"  ERROR: target '{target}' missing.")
        return pd.DataFrame()

    y_continuous = df[target].dropna()
    valid_idx = y_continuous.index
    global_median = y_continuous.median()
    y_global = (y_continuous > global_median).astype(int)

    feature_sets = get_feature_sets(df, config)
    requested_fs = legacy_cfg.get("feature_sets")
    if requested_fs:
        feature_sets = {k: v for k, v in feature_sets.items() if k in requested_fs}

    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats,
                                 random_state=random_state)
    models = _build_legacy_classifiers(random_state, enabled, include_qsvm=include_qsvm)
    if not models:
        return pd.DataFrame()

    rows = []
    for fs_name, fs_cols in feature_sets.items():
        cols = [c for c in fs_cols if c in df.columns]
        if not cols:
            continue
        X = df.loc[valid_idx, cols]
        n_select = max(5, min(15, len(cols) // 10))
        n_select = min(n_select, len(cols))

        for model_name, model in models.items():
            fold_metrics = {"bal_acc": [], "f1": [], "auc": [], "sens": [], "spec": []}
            try:
                for train_idx, test_idx in cv.split(X, y_global):
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    train_med = y_continuous.iloc[train_idx].median()
                    y_tr = (y_continuous.iloc[train_idx] > train_med).astype(int)
                    y_te = (y_continuous.iloc[test_idx]  > train_med).astype(int)

                    pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler",  StandardScaler()),
                        ("select",  SelectKBest(f_classif, k=n_select)),
                        ("clf",     model),
                    ])
                    pipe.fit(X_tr, y_tr)
                    preds = pipe.predict(X_te)

                    fold_metrics["bal_acc"].append(balanced_accuracy_score(y_te, preds))
                    fold_metrics["f1"].append(f1_score(y_te, preds, zero_division=0))
                    fold_metrics["sens"].append(recall_score(y_te, preds, pos_label=1, zero_division=0))
                    fold_metrics["spec"].append(recall_score(y_te, preds, pos_label=0, zero_division=0))
                    try:
                        if hasattr(pipe, "predict_proba"):
                            probs = pipe.predict_proba(X_te)[:, 1]
                        else:
                            probs = pipe.decision_function(X_te)
                        fold_metrics["auc"].append(roc_auc_score(y_te, probs))
                    except Exception:
                        fold_metrics["auc"].append(np.nan)

                rows.append({
                    "feature_set":       fs_name,
                    "model":             model_name,
                    "balanced_accuracy": round(float(np.mean(fold_metrics["bal_acc"])), 3),
                    "f1":                round(float(np.mean(fold_metrics["f1"])), 3),
                    "auc":               round(float(np.nanmean(fold_metrics["auc"])), 3),
                    "sensitivity":       round(float(np.mean(fold_metrics["sens"])), 3),
                    "specificity":       round(float(np.mean(fold_metrics["spec"])), 3),
                })
                print(f"  [legacy] {fs_name:30s} | {model_name:12s} | "
                      f"BA={rows[-1]['balanced_accuracy']:.3f}")
            except Exception as e:
                print(f"  [legacy] {fs_name} | {model_name} | ERROR: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.to_csv(os.path.join(output_dir, "legacy_ml_results.csv"),
                      index=False)
    return df_out


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def _resolve_target_list(target_cfg):
    """
    Return a list of target dicts {name, task, residualize_covariates,
    clinical_threshold} from the new or legacy config layouts.

    Accepted layouts (in priority order):
      1. target.names: [a, b, c]      (recommended)
      2. target.primary.name + target.additional_targets: [...]
      3. target.primary.name only      (single-target legacy)
    """
    task = target_cfg.get("task")
    cov  = target_cfg.get("residualize_covariates")
    thr  = target_cfg.get("clinical_threshold", {}) or {}

    primary = target_cfg.get("primary", {}) or {}
    if task is None:
        task = primary.get("task", "regression")
    if cov is None:
        cov = primary.get("residualize_covariates") or ["age_months"]

    names = target_cfg.get("names")
    if names is None:
        names = []
        if primary.get("name"):
            names.append(primary["name"])
        names.extend(target_cfg.get("additional_targets") or [])
    # Dedupe in declared order; strip inline comments.
    seen, ordered = set(), []
    for n in names:
        n = str(n).split("#")[0].strip()
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)

    return [
        {
            "name": n,
            "task": task,
            "residualize_covariates": list(cov),
            "clinical_threshold": thr,
        }
        for n in ordered
    ]


def _build_target_cfg(target_spec):
    """Shape a single-target dict back into the legacy layout that
    run_target_preparation expects."""
    return {
        "primary": {
            "name": target_spec["name"],
            "task": target_spec["task"],
            "residualize_covariates": target_spec["residualize_covariates"],
        },
        "clinical_threshold": target_spec["clinical_threshold"] or {},
    }


def _run_one_target(target_spec, full_df, X_curated_full, feature_sets_curated,
                    config, target_dir):
    """
    Per-target pipeline: target preparation -> regression -> screening ->
    permutation -> bootstrap -> FDR -> SHAP. Writes everything under
    ``target_dir`` (which may equal the run base dir for single-target).
    Returns a dict summarising results for the run_notes audit.
    """
    name = target_spec["name"]
    task = target_spec["task"]

    print("\n" + "#" * 60)
    print(f"# TARGET: {name}")
    print("#" * 60)

    if name not in full_df.columns or full_df[name].isna().all():
        print(f"  SKIP: '{name}' missing or all-NaN in full_dataset.csv")
        return {"target": name, "status": "skipped"}

    os.makedirs(target_dir, exist_ok=True)

    # Target preparation
    target_prep = run_target_preparation(
        full_df, _build_target_cfg(target_spec), output_dir=target_dir,
    )
    y_residual = target_prep["y_residual"]
    _save_target_distribution_plot(
        target_prep["y_continuous_raw"], y_residual,
        target_prep["threshold"], target_dir, target_prep["target_name"],
    )

    # 4C. Regression
    if task != "regression":
        print(f"  [INFO] target.task={task!r}; skipping regression path.")
        return {"target": name, "status": "skipped_task", "task": task}

    print("\n" + "-" * 40)
    print(f"4C. Regression CV — {name}")
    print("-" * 40)
    regression_results, best_info, oof_predictions = run_regression(
        X_curated_full, y_residual, feature_sets_curated, config, target_dir,
    )

    # 4C-2. Clinical screening
    screening_df = pd.DataFrame()
    if oof_predictions:
        print("\n" + "-" * 40)
        print(f"4C-2. Clinical screening — {name}")
        print("-" * 40)
        screening_df = run_clinical_screening(oof_predictions, target_prep, target_dir)

    # 4D. Validation
    perm, bootstrap = {}, {}
    if oof_predictions:
        perm = run_permutation_test(
            X_curated_full, y_residual, feature_sets_curated,
            regression_results, config, target_dir,
        )
        bootstrap = run_bootstrap_ci(oof_predictions, screening_df,
                                     target_prep, config, target_dir)
        apply_fdr_to_pearson(regression_results, target_dir)

    # 4F. SHAP
    importance = None
    if best_info:
        importance = run_shap_regression(
            X_curated_full, y_residual, best_info, config, target_dir,
        )

    return {
        "target":           name,
        "status":           "ok",
        "task":             task,
        "n_used":           int(target_prep["report"]["residual_stats"]["n"]),
        "covariate_r2":     target_prep["report"]["age_model_r2"],
        "threshold":        target_prep["report"]["primary_threshold"]["threshold"],
        "prevalence":       target_prep["report"]["primary_threshold"]["prevalence"],
        "best_regressor":   best_info,
        "permutation_p":    (perm or {}).get("p_value"),
        "shap_top_feature": (importance.iloc[0]["feature"]
                              if importance is not None and len(importance) else None),
    }


def run(config, full_df=None):
    """
    Run regression-first Stage 4 over one or more targets. Writes:

    Run-level (target-agnostic):
      - correlations.csv
      - feature_curation_report.json
      - feature_correlation_heatmap_before.png / _after.png

    Per-target (under <target>/ when N>1, flat when N=1):
      - target_preparation_report.json, target_distribution.png
      - regression_results.csv, regression_summary.csv
      - clinical_screening_metrics.csv
      - permutation_results.json, bootstrap_ci.json
      - shap_importance.csv, shap_annotated.csv (when SHAP available)
      - figures/shap_summary.png

    Optional:
      - legacy_ml_results.csv (when legacy_classification.enable=true)
    """
    from utils.io import write_stage_notes
    from stages.engineering import load_full_dataset

    print("\n" + "=" * 60)
    print("STAGE 4: Analysis (regression-first)")
    print("=" * 60)

    engineering_input = config.get("input_dir")
    if full_df is None:
        if engineering_input is None:
            raise FileNotFoundError(
                "No engineering output found. "
                "Run `python pipeline.py --engineering` first."
            )
        full_df = load_full_dataset(config, stage_dir=engineering_input)

    # 4A. Run-level descriptives + hypothesis correlations.
    desc = run_descriptives(full_df, config)
    corr_results = run_correlations(full_df, config)

    base_dir = config["output_dir"]
    print(f"  Analysis output dir: {base_dir}")
    if not corr_results.empty:
        corr_results.to_csv(os.path.join(base_dir, "correlations.csv"), index=False)

    ana = config["analysis"]
    target_cfg = ana.get("target", {}) or {}
    fc_cfg = ana.get("feature_curation", {}) or {}

    targets = _resolve_target_list(target_cfg)
    if not targets:
        raise ValueError(
            "No targets configured. Set analysis.target.names: [...] "
            "or analysis.target.primary.name in stages/analysis/config.yaml"
        )
    print(f"  Targets to analyse ({len(targets)}): {[t['name'] for t in targets]}")

    # 4B. Build feature pool -> curate (run once, target-agnostic).
    feature_sets_all = get_feature_sets(full_df, config)
    feature_pool_cols = sorted({c for cols in feature_sets_all.values() for c in cols
                                if c in full_df.columns})
    if not feature_pool_cols:
        raise RuntimeError("No features found across requested feature_sets")

    X_pool_raw = full_df[feature_pool_cols].apply(pd.to_numeric, errors="coerce")
    if fc_cfg.get("enable", True):
        X_curated, kept_names, _ = run_feature_curation(
            X_pool_raw, fc_cfg, output_dir=base_dir,
        )
    else:
        X_curated = X_pool_raw
        kept_names = feature_pool_cols
        print("  [INFO] feature_curation.enable=false — skipping curation")

    kept_set = set(kept_names)
    feature_sets_curated = {}
    for fs_name, fs_cols in feature_sets_all.items():
        retained = [c for c in fs_cols if c in kept_set]
        if retained:
            feature_sets_curated[fs_name] = retained
    if not feature_sets_curated:
        raise RuntimeError("No feature sets have any surviving columns after curation")

    try:
        _save_correlation_heatmaps(X_pool_raw, X_curated, base_dir)
    except Exception as e:
        print(f"  [WARN] Correlation heatmaps skipped: {e}")

    X_curated_full = full_df[kept_names].apply(pd.to_numeric, errors="coerce")

    # Per-target loop. Single-target writes to base_dir for backwards-compat;
    # multi-target writes to base_dir/<target>/.
    multi = len(targets) > 1
    per_target = []
    for tgt in targets:
        tgt_dir = os.path.join(base_dir, tgt["name"]) if multi else base_dir
        per_target.append(_run_one_target(
            tgt, full_df, X_curated_full, feature_sets_curated, config, tgt_dir,
        ))

    # 4-LEGACY (optional, primary target only).
    legacy_results = pd.DataFrame()
    if (ana.get("legacy_classification") or {}).get("enable", False):
        legacy_target = targets[0]["name"]
        legacy_results = run_legacy_classification(
            full_df, config, legacy_target, base_dir,
        )

    write_stage_notes(base_dir, {
        "stage": "analysis",
        "input_engineering_dir": engineering_input,
        "n_subjects": int(full_df.shape[0]),
        "n_features_input": int(len(feature_pool_cols)),
        "n_features_curated": int(len(kept_names)),
        "targets":     [t["name"] for t in targets],
        "per_target":  per_target,
        "outputs": [
            "correlations.csv", "feature_curation_report.json",
            "feature_correlation_heatmap_before.png",
            "feature_correlation_heatmap_after.png",
        ] + (["legacy_ml_results.csv"] if not legacy_results.empty else []),
    })

    return {
        "descriptives":   desc,
        "correlations":   corr_results,
        "targets":        [t["name"] for t in targets],
        "per_target":     per_target,
        "legacy_results": legacy_results,
    }
