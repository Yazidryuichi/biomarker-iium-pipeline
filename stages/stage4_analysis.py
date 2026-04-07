"""
Stage 4: Statistical Analysis + ML Classification
====================================================
4A. Descriptive statistics and correlations (H1-H3)
4B. ML classification with feature set comparison (H4)
4C. SHAP analysis for biomarker identification (H5)
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
# 4A. Descriptive & Correlational Analysis
# ──────────────────────────────────────────────────────────────────

def run_descriptives(df, config):
    """Descriptive statistics for all variables."""
    print("\n" + "-" * 40)
    print("4A. Descriptive Statistics")
    print("-" * 40)

    # Demographics
    print(f"\n  N = {len(df)}")
    if "age_years" in df.columns:
        print(f"  Age: {df['age_years'].mean():.1f} +/- {df['age_years'].std():.1f} years")
    if "Sex" in df.columns:
        sex_counts = df["Sex"].value_counts()
        print(f"  Sex: {dict(sex_counts)}")

    # Behavioral measures
    beh_cols = ["Global_EF", "WM_score", "IC_score", "flanker_effect",
                "FW_Span", "BW_Span", "Total_Span"]
    beh_cols = [c for c in beh_cols if c in df.columns]

    desc = df[beh_cols].describe().round(3)
    print(f"\n  Behavioral measures:")
    print(desc.to_string())

    # Key QEEG features
    tbr_cols = [c for c in df.columns if c.startswith("tbr_")]
    if tbr_cols:
        print(f"\n  TBR features:")
        print(df[tbr_cols].describe().round(4).to_string())

    return desc


def run_correlations(df, config):
    """
    Test hypotheses H1-H3 (pre-specified, not data-driven):
      H1: negative correlation TBR_frontal ↔ Global_EF
      H2: negative correlation theta_frontal ↔ Global_EF
      H3: positive correlation TBR_frontal ↔ Flanker_Effect

    NOTE on FDR scope: correction is applied across these 8 pre-specified
    tests only, not across all 200+ feature-outcome pairs. These hypotheses
    were derived from the literature review (Arns et al. 2013, Zhang et al.
    2017, Tan et al. 2024) prior to data analysis. Exploratory correlations
    across all features would require a separate, broader FDR correction
    and should be reported as exploratory in any publication.
    """
    print("\n" + "-" * 40)
    print("4A. Hypothesis Testing (Correlations)")
    print("-" * 40)

    results = []

    # Pairs to test
    test_pairs = [
        ("tbr_frontal_mean", "Global_EF", "H1: TBR ↔ Global EF (expected: negative)"),
        ("psd_abs_theta_Fz", "Global_EF", "H2: Theta_Fz ↔ Global EF (expected: negative)"),
        ("tbr_frontal_mean", "flanker_effect", "H3: TBR ↔ Flanker Effect (expected: positive)"),
        ("tbr_Fz", "Global_EF", "TBR_Fz ↔ Global EF"),
        ("tbr_Cz", "Global_EF", "TBR_Cz ↔ Global EF"),
        ("faa_F4_F3", "Global_EF", "FAA ↔ Global EF"),
        ("alpha_reactivity_global", "Global_EF", "Alpha Reactivity ↔ Global EF"),
        ("tbr_frontal_mean", "BW_Span", "TBR ↔ Digit Span Backward"),
    ]

    for x_col, y_col, label in test_pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue

        x = df[x_col].dropna()
        y = df[y_col].reindex(x.index).dropna()
        x = x.reindex(y.index)

        if len(x) < 5:
            continue

        # Normality test
        _, p_norm_x = stats.shapiro(x)
        _, p_norm_y = stats.shapiro(y)

        # Use Pearson if both normal, else Spearman
        if p_norm_x > 0.05 and p_norm_y > 0.05:
            r, p = stats.pearsonr(x, y)
            method = "Pearson"
        else:
            r, p = stats.spearmanr(x, y)
            method = "Spearman"

        results.append({
            "hypothesis": label,
            "x": x_col,
            "y": y_col,
            "method": method,
            "r": round(r, 3),
            "p": round(p, 4),
            "n": len(x),
            "sig": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns",
        })

        print(f"  {label}")
        print(f"    {method} r = {r:.3f}, p = {p:.4f}, n = {len(x)} {'*' if p < 0.05 else 'ns'}")

    # FDR correction
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
# 4B. ML Classification
# ──────────────────────────────────────────────────────────────────

def get_feature_sets(df, config):
    """
    Define feature sets for comparison.
    Returns dict: {set_name: list of column names}
    """
    all_cols = df.columns.tolist()

    # Set 1: Conventional QEEG only
    conventional = [c for c in all_cols if any(c.startswith(p) for p in
                    ["psd_", "tbr_", "faa_", "alpha_reactivity", "coh_"])]

    # Set 2: Conventional + Advanced (wavelet, Hjorth, entropy, PAC)
    advanced = conventional + [c for c in all_cols if any(c.startswith(p) for p in
                ["cwt_", "hjorth_", "spectral_entropy_", "pac_"])]

    # Set 3: Covariance features only
    covariance = [c for c in all_cols if c.startswith("cov_")]

    # Set 4: All features combined
    all_features = conventional + [c for c in all_cols if any(c.startswith(p) for p in
                   ["cwt_", "hjorth_", "spectral_entropy_", "pac_", "cov_"])]

    return {
        "conventional_qeeg": conventional,
        "conventional_plus_advanced": advanced,
        "covariance_only": covariance,
        "all_features": all_features,
    }


def run_classification(df, config):
    """
    ML classification comparing feature sets and models.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif

    print("\n" + "-" * 40)
    print("4B. ML Classification")
    print("-" * 40)

    target_col = "ef_group_global"
    if target_col not in df.columns or df[target_col].isna().all():
        print("  ERROR: No valid classification target")
        return pd.DataFrame()

    y = df[target_col].dropna().astype(int)
    valid_idx = y.index

    feature_sets = get_feature_sets(df, config)
    cv_folds = config["ml"]["cv_folds"]
    cv_repeats = config["ml"]["cv_repeats"]
    random_state = config["ml"]["random_state"]

    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state
    )

    # Models
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=0.1, random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=3, random_state=random_state
        ),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=random_state),
    }

    # Try XGBoost if available
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=random_state, use_label_encoder=False,
            eval_metric="logloss", verbosity=0,
        )
    except ImportError:
        pass

    results = []

    for fs_name, fs_cols in feature_sets.items():
        fs_cols_valid = [c for c in fs_cols if c in df.columns]
        if not fs_cols_valid:
            continue

        X = df.loc[valid_idx, fs_cols_valid].fillna(0)

        # Feature selection: keep at most 10 features (given N=28)
        n_select = min(10, len(fs_cols_valid))

        for model_name, model in models.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_select)),
                ("clf", model),
            ])

            try:
                scores = cross_validate(
                    pipe, X, y, cv=cv,
                    scoring=["balanced_accuracy", "f1", "roc_auc"],
                    return_train_score=False,
                    error_score="raise",
                )

                result = {
                    "feature_set": fs_name,
                    "n_features_input": len(fs_cols_valid),
                    "n_features_selected": n_select,
                    "model": model_name,
                    "balanced_accuracy": round(np.mean(scores["test_balanced_accuracy"]), 3),
                    "bal_acc_std": round(np.std(scores["test_balanced_accuracy"]), 3),
                    "f1": round(np.mean(scores["test_f1"]), 3),
                    "auc": round(np.mean(scores["test_roc_auc"]), 3),
                }
                results.append(result)

                print(
                    f"  {fs_name:30s} | {model_name:20s} | "
                    f"Acc: {result['balanced_accuracy']:.3f} "
                    f"(+/-{result['bal_acc_std']:.3f}) | "
                    f"F1: {result['f1']:.3f} | AUC: {result['auc']:.3f}"
                )

            except Exception as e:
                print(f"  {fs_name} | {model_name} | ERROR: {e}")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────
# 4C. SHAP Analysis
# ──────────────────────────────────────────────────────────────────

def run_shap_analysis(df, config, best_feature_set="conventional_qeeg"):
    """
    SHAP analysis on the best-performing model to identify
    candidate biomarker features.
    """
    print("\n" + "-" * 40)
    print("4C. SHAP Feature Importance")
    print("-" * 40)

    try:
        import shap
    except ImportError:
        print("  SHAP not installed. pip install shap")
        return None

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif

    target_col = "ef_group_global"
    y = df[target_col].dropna().astype(int)
    valid_idx = y.index

    feature_sets = get_feature_sets(df, config)
    fs_cols = feature_sets.get(best_feature_set, [])
    fs_cols = [c for c in fs_cols if c in df.columns]

    if not fs_cols:
        print("  No features found for SHAP analysis")
        return None

    X = df.loc[valid_idx, fs_cols].fillna(0)

    # CV-stable feature selection: run SelectKBest inside CV folds,
    # keep features selected in >= 3/5 folds. Avoids data leakage
    # from fitting selector on full dataset.
    n_select = min(10, len(fs_cols))
    from sklearn.model_selection import StratifiedKFold

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_votes = np.zeros(len(fs_cols))
    for train_idx, _ in cv_inner.split(X, y):
        sel = SelectKBest(f_classif, k=n_select)
        sel.fit(X.iloc[train_idx], y.iloc[train_idx])
        feature_votes[sel.get_support()] += 1

    stable_mask = feature_votes >= 3
    if stable_mask.sum() == 0:
        stable_mask = feature_votes >= 2  # fallback if too strict
    selected_names = [fs_cols[i] for i, m in enumerate(stable_mask) if m]
    X_selected = X[selected_names].values

    # Fit model for SHAP interpretation (not for prediction claims)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=3,
        random_state=config["ml"]["random_state"]
    )
    model.fit(X_scaled, y)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Feature importance ranking
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (high EF)

    # Handle multi-dimensional SHAP values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # class 1

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0).ravel()
    importance = pd.DataFrame({
        "feature": selected_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n  Top biomarker candidates (by SHAP importance):")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:40s}  SHAP: {row['mean_abs_shap']:.4f}")

    # Save SHAP summary plot
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(
            shap_values, X_scaled,
            feature_names=selected_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(figures_dir, "shap_summary.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"\n  SHAP summary plot saved: {figures_dir}/shap_summary.png")
    except Exception as e:
        print(f"  Could not save SHAP plot: {e}")

    return importance


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def run_stage4(config, full_df):
    """Run full Stage 4 analysis."""
    print("\n" + "=" * 60)
    print("STAGE 4: Analysis")
    print("=" * 60)

    # 4A: Descriptives + Correlations
    desc = run_descriptives(full_df, config)
    corr_results = run_correlations(full_df, config)

    # 4B: ML Classification
    ml_results = run_classification(full_df, config)

    # 4C: SHAP
    shap_importance = run_shap_analysis(full_df, config)

    # Save all results
    output_dir = config["paths"]["output_dir"]

    if not corr_results.empty:
        corr_results.to_csv(
            os.path.join(output_dir, "correlations.csv"), index=False
        )

    if not ml_results.empty:
        ml_results.to_csv(
            os.path.join(output_dir, "ml_results.csv"), index=False
        )

        # Find best model
        best = ml_results.loc[ml_results["balanced_accuracy"].idxmax()]
        print(f"\n  BEST MODEL:")
        print(f"    Feature set: {best['feature_set']}")
        print(f"    Model: {best['model']}")
        print(f"    Balanced accuracy: {best['balanced_accuracy']:.3f}")
        print(f"    H4 target (>=0.75): {'MET' if best['balanced_accuracy'] >= 0.75 else 'NOT MET'}")

    if shap_importance is not None:
        shap_importance.to_csv(
            os.path.join(output_dir, "shap_importance.csv"), index=False
        )

    return {
        "descriptives": desc,
        "correlations": corr_results,
        "ml_results": ml_results,
        "shap_importance": shap_importance,
    }
