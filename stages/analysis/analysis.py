"""
Stage 4: Statistical Analysis + ML Classification
====================================================
4A. Descriptive statistics and correlations (H1-H3)
4B. ML classification with feature set comparison (H4)
4C. SHAP analysis for biomarker identification (H5)
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

    # Demographics
    print(f"\n  N = {len(df)}")
    if "age_years" in df.columns:
        print(f"  Age: {df['age_years'].mean():.1f} +/- {df['age_years'].std():.1f} years")
    if "Sex" in df.columns:
        sex_counts = df["Sex"].value_counts()
        print(f"  Sex: {dict(sex_counts)}")

    # Behavioral measures
    beh_cols = ["Global_EF", "WM_score", "IC_score", "CF_score", "P_score", "SF_score",
                "flanker_effect", "ddm_v", "ddm_a", "ddm_t", "ddm_delta_v",
                "FW_Span", "BW_Span", "Total_Span"]
    beh_cols = [c for c in beh_cols if c in df.columns]

    desc = df[beh_cols].describe().round(3)
    print(f"\n  Behavioral measures:")
    print(desc.to_string())

    # Key QEEG features (EO condition)
    tbr_cols = [c for c in df.columns if "tbr_" in c and c.startswith("eo_")]
    if tbr_cols:
        print(f"\n  TBR features:")
        print(df[tbr_cols].describe().round(4).to_string())

    return desc


def run_correlations(df, config):
    """
    Test hypotheses H1-H3 (pre-specified, not data-driven):
      H1: negative correlation TBR_frontal vs Global_EF
      H2: negative correlation theta_frontal vs Global_EF
      H3: positive correlation TBR_frontal vs Flanker_Effect

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
    # EO condition used for TBR/theta hypotheses (resting-state, eyes open)
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

        # Use Spearman for all tests (robust to non-normality with N=28).
        # Pearson reported as supplementary. Do not condition method on
        # normality test — that is a form of double-dipping.
        r_spearman, p_spearman = stats.spearmanr(x, y)
        r_pearson, p_pearson = stats.pearsonr(x, y)

        # Effect size: r-to-d conversion (Cohen 1988; Fritz et al. 2012)
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

    Each combined set (EO+EC) is accompanied by condition-specific subsets
    so the ML table shows whether Eyes-Open or Eyes-Closed features drive
    predictive performance independently.
    """
    all_cols = df.columns.tolist()

    def _match(col, patterns):
        return any(p in col for p in patterns)

    CONV_PATTERNS    = ["psd_", "tbr_", "faa_", "alpha_reactivity", "coh_"]
    ADV_PATTERNS     = ["cwt_", "hjorth_", "spectral_entropy_", "pac_"]
    COV_PATTERN      = "cov_"
    QUANTUM_PATTERNS = ["qepp_", "qi_", "tn_"]

    # --- Combined (EO + EC) sets ---
    conventional = [c for c in all_cols if _match(c, CONV_PATTERNS)]
    advanced     = list(dict.fromkeys(
        conventional + [c for c in all_cols if _match(c, ADV_PATTERNS)]
    ))
    covariance   = [c for c in all_cols if COV_PATTERN in c]
    all_features = list(dict.fromkeys(
        conventional + [c for c in all_cols if _match(c, ADV_PATTERNS + [COV_PATTERN])]
    ))
    quantum      = [c for c in all_cols if _match(c, QUANTUM_PATTERNS)]

    # --- Eyes-Open only ---
    conventional_eo = [c for c in conventional if c.startswith("eo_") or not (c.startswith("ec_"))]
    # alpha_reactivity is cross-condition (no prefix); keep in both EO and EC sets
    conventional_eo = [c for c in conventional if not c.startswith("ec_")]
    advanced_eo     = [c for c in advanced     if not c.startswith("ec_")]
    covariance_eo   = [c for c in covariance   if not c.startswith("ec_")]
    all_features_eo = [c for c in all_features if not c.startswith("ec_")]

    # --- Eyes-Closed only ---
    conventional_ec = [c for c in conventional if not c.startswith("eo_")]
    advanced_ec     = [c for c in advanced     if not c.startswith("eo_")]
    covariance_ec   = [c for c in covariance   if not c.startswith("eo_")]
    all_features_ec = [c for c in all_features if not c.startswith("eo_")]

    feature_sets = {
        # Combined EO+EC
        "conventional_qeeg":          conventional,
        "conventional_plus_advanced":  advanced,
        "covariance_only":             covariance,
        "all_features":                all_features,
        # Eyes-Open only
        "conventional_qeeg_eo":        conventional_eo,
        "conventional_plus_advanced_eo": advanced_eo,
        "covariance_only_eo":          covariance_eo,
        "all_features_eo":             all_features_eo,
        # Eyes-Closed only
        "conventional_qeeg_ec":        conventional_ec,
        "conventional_plus_advanced_ec": advanced_ec,
        "covariance_only_ec":          covariance_ec,
        "all_features_ec":             all_features_ec,
    }

    # Quantum-inspired sets (only when those columns are present, i.e.
    # features.include_quantum was true at extraction time)
    if quantum:
        feature_sets["quantum_only"] = quantum
        feature_sets["classical_plus_quantum"] = list(dict.fromkeys(all_features + quantum))

    return feature_sets


def _build_models(random_state, include_qsvm=False):
    """
    Build models for classification.
    Always tries: RF, SVM, KNN, MLP + (XGBoost, LightGBM, CatBoost) if available.
    CNN-LSTM is built in a separate path (PyTorch).
    QSVM (quantum-kernel) models are added when ``include_qsvm`` is true and
    pennylane is installed.
    Gracefully skips models whose dependencies are missing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=3,
            class_weight="balanced", random_state=random_state
        ),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True,
                   class_weight="balanced", random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            early_stopping=True, random_state=random_state
        ),
    }

    # XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=random_state, use_label_encoder=False,
            eval_metric="logloss", verbosity=0,
        )
    except ImportError:
        print("  [INFO] XGBoost not installed — skipping")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            num_leaves=8, min_child_samples=5,
            random_state=random_state, verbose=-1,
        )
    except ImportError:
        print("  [INFO] LightGBM not installed — skipping")

    # CatBoost — wrapped for sklearn 1.8+ compatibility
    try:
        from catboost import CatBoostClassifier as _CatBoost
        from sklearn.base import BaseEstimator, ClassifierMixin

        class CatBoostWrapper(BaseEstimator, ClassifierMixin):
            """Thin wrapper to fix CatBoost/sklearn tags compatibility."""
            _estimator_type = "classifier"

            def __init__(self, iterations=100, depth=3, learning_rate=0.1,
                         random_state=42, verbose=0):
                self.iterations = iterations
                self.depth = depth
                self.learning_rate = learning_rate
                self.random_state = random_state
                self.verbose = verbose

            def fit(self, X, y):
                self._model = _CatBoost(
                    iterations=self.iterations, depth=self.depth,
                    learning_rate=self.learning_rate,
                    random_seed=self.random_state, verbose=self.verbose,
                )
                self._model.fit(X, y)
                self.classes_ = np.array(sorted(set(y)))
                return self

            def predict(self, X):
                return self._model.predict(X).flatten().astype(int)

            def predict_proba(self, X):
                return self._model.predict_proba(X)

            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.estimator_type = "classifier"
                return tags

        models["CatBoost"] = CatBoostWrapper(
            iterations=100, depth=3, learning_rate=0.1,
            random_state=random_state, verbose=0,
        )
    except ImportError:
        print("  [INFO] CatBoost not installed — skipping")

    # Quantum-kernel SVM models (opt-in via ml.include_qsvm)
    if include_qsvm:
        try:
            from stages.analysis.qsvm_classifier import QuantumKernelSVM
            models["QSVM_4q_ZZ"]   = QuantumKernelSVM(n_qubits=4, n_layers=2, C=1.0, use_entangling=True)
            models["QSVM_6q_ZZ"]   = QuantumKernelSVM(n_qubits=6, n_layers=2, C=1.0, use_entangling=True)
            models["QSVM_6q_prod"] = QuantumKernelSVM(n_qubits=6, n_layers=2, C=1.0, use_entangling=False)
        except ImportError:
            print("  [INFO] QSVM requested but pennylane not installed — skipping")

    return models


def _make_scoring():
    """Return scoring dict with all metrics from the proposal."""
    from sklearn.metrics import make_scorer, balanced_accuracy_score, \
        f1_score, roc_auc_score, recall_score

    return {
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "sensitivity": make_scorer(recall_score, pos_label=1),
        "specificity": make_scorer(recall_score, pos_label=0),
    }


def run_classification(df, config, target=None):
    """
    ML classification comparing feature sets and models.
    Implements all 8 algorithms from the research proposal (Section 2.4.2):
    RF, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP, CNN-LSTM.

    Args:
        target: continuous behavioral column to predict. Falls back to
                ml.target/ml.targets[0] in config when None.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, \
        permutation_test_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import balanced_accuracy_score, f1_score, \
        roc_auc_score, recall_score

    logger.info("4B. ML Classification")
    print("\n" + "-" * 40)
    print("4B. ML Classification")
    print("-" * 40)

    # Use continuous EF scores — median split done INSIDE each CV fold
    # to prevent target leakage (threshold from training fold only)
    if target is None:
        from utils.io import get_targets
        target = get_targets(config)[0]
    continuous_col = target
    if continuous_col not in df.columns or df[continuous_col].isna().all():
        print(f"  ERROR: Target '{continuous_col}' not found or all-NaN. "
              f"Check ml.targets in config.yaml")
        return pd.DataFrame(), {}
    print(f"  Target variable: {continuous_col}")

    y_continuous = df[continuous_col].dropna()
    valid_idx = y_continuous.index

    # Global median split for permutation test (sklearn requires fixed y)
    global_median = y_continuous.median()
    y_global = (y_continuous > global_median).astype(int)

    feature_sets = get_feature_sets(df, config)
    ana = config["analysis"]
    cv_folds = ana["cv_folds"]
    cv_repeats = ana["cv_repeats"]
    random_state = config["random_state"]

    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state
    )

    enabled_models = ana.get("models", [])
    # Strip inline YAML comments that survive as trailing text (e.g. "XGBoost # requires...")
    enabled_models = [m.split("#")[0].strip() for m in enabled_models if m]
    include_qsvm = ana.get("include_qsvm", False)
    if include_qsvm:
        # Auto-include QSVM model names so the user doesn't need to list each one
        for q in ("QSVM_4q_ZZ", "QSVM_6q_ZZ", "QSVM_6q_prod"):
            if q not in enabled_models:
                enabled_models.append(q)
    models = {k: v for k, v in _build_models(random_state, include_qsvm=include_qsvm).items()
              if k in enabled_models}
    if not models:
        print("  ERROR: No models enabled. Check ml.models in config.yaml")
        return pd.DataFrame(), {}
    print(f"  Models enabled: {list(models.keys())}")

    results = []

    for fs_name, fs_cols in feature_sets.items():
        fs_cols_valid = [c for c in fs_cols if c in df.columns]
        if not fs_cols_valid:
            continue

        X = df.loc[valid_idx, fs_cols_valid]

        # Feature selection: adaptive k based on feature set size
        n_select = max(5, min(15, len(fs_cols_valid) // 10))
        n_select = min(n_select, len(fs_cols_valid))

        for model_name, model in models.items():
            # Manual CV loop with fold-internal median split
            # This prevents target leakage: the binarization threshold
            # is computed on TRAINING data only in each fold
            fold_metrics = {
                "bal_acc": [], "f1": [], "auc": [],
                "sens": [], "spec": [],
            }

            is_qsvm = model_name.startswith("QSVM")

            try:
                for train_idx, test_idx in cv.split(X, y_global):
                    X_train = X.iloc[train_idx]
                    X_test = X.iloc[test_idx]

                    # Compute median on training fold only
                    train_median = y_continuous.iloc[train_idx].median()
                    y_train = (y_continuous.iloc[train_idx] > train_median).astype(int)
                    y_test = (y_continuous.iloc[test_idx] > train_median).astype(int)

                    if is_qsvm:
                        # QSVM does its own PCA + scaling internally; only impute NaNs
                        imputer = SimpleImputer(strategy="median")
                        X_train_in = imputer.fit_transform(X_train)
                        X_test_in  = imputer.transform(X_test)
                        model.fit(X_train_in, y_train)
                        pipe = model  # used for predict_proba/decision_function below
                    else:
                        pipe = Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("select", SelectKBest(f_classif, k=n_select)),
                            ("clf", model),
                        ])
                        pipe.fit(X_train, y_train)
                        X_test_in = X_test

                    preds = pipe.predict(X_test_in)

                    fold_metrics["bal_acc"].append(
                        balanced_accuracy_score(y_test, preds))
                    fold_metrics["f1"].append(
                        f1_score(y_test, preds, zero_division=0))
                    fold_metrics["sens"].append(
                        recall_score(y_test, preds, pos_label=1, zero_division=0))
                    fold_metrics["spec"].append(
                        recall_score(y_test, preds, pos_label=0, zero_division=0))
                    try:
                        if len(np.unique(y_test)) < 2:
                            raise ValueError("single class in fold")
                        if hasattr(pipe, "predict_proba"):
                            probs = pipe.predict_proba(X_test_in)[:, 1]
                        else:
                            probs = pipe.decision_function(X_test_in)
                        probs = np.nan_to_num(np.array(probs, dtype=float), nan=0.5)
                        fold_metrics["auc"].append(roc_auc_score(y_test, probs))
                    except Exception:
                        fold_metrics["auc"].append(np.nan)

                # Bootstrap 95% CI for balanced accuracy
                ba_scores = np.array(fold_metrics["bal_acc"])
                n_boot = 1000
                rng = np.random.RandomState(random_state)
                boot_means = [
                    np.mean(rng.choice(ba_scores, size=len(ba_scores), replace=True))
                    for _ in range(n_boot)
                ]
                ci_lower = np.percentile(boot_means, 2.5)
                ci_upper = np.percentile(boot_means, 97.5)

                result = {
                    "feature_set": fs_name,
                    "n_features_input": len(fs_cols_valid),
                    "n_features_selected": n_select,
                    "model": model_name,
                    "balanced_accuracy": round(np.mean(ba_scores), 3),
                    "bal_acc_std": round(np.std(ba_scores), 3),
                    "ci_lower": round(ci_lower, 3),
                    "ci_upper": round(ci_upper, 3),
                    "f1": round(np.mean(fold_metrics["f1"]), 3),
                    "auc": round(float(np.nanmean(fold_metrics["auc"])) if fold_metrics["auc"] else float("nan"), 3),
                    "sensitivity": round(np.mean(fold_metrics["sens"]), 3),
                    "specificity": round(np.mean(fold_metrics["spec"]), 3),
                }
                results.append(result)

                logger.info(f"{fs_name} | {model_name} | BA={result['balanced_accuracy']:.3f}")
                print(
                    f"  {fs_name:30s} | {model_name:12s} | "
                    f"Acc: {result['balanced_accuracy']:.3f} "
                    f"[{ci_lower:.3f}-{ci_upper:.3f}] | "
                    f"Sens: {result['sensitivity']:.3f} | "
                    f"Spec: {result['specificity']:.3f} | "
                    f"F1: {result['f1']:.3f} | AUC: {result['auc']:.3f}"
                )

            except Exception as e:
                logger.error(f"{fs_name} | {model_name} | {e}")
                print(f"  {fs_name} | {model_name} | ERROR: {e}")
            finally:
                gc.collect()  # Free memory between model runs

    # CNN-LSTM (PyTorch-based, separate CV loop)
    # Enable by adding "CNN-LSTM" to ml.models in config.yaml.
    # Env var RUN_CNN_LSTM=1 also works as an override.
    import os as _os
    cnn_lstm_in_config = "CNN-LSTM" in enabled_models
    cnn_lstm_env = _os.environ.get("RUN_CNN_LSTM", "0") == "1"
    if cnn_lstm_in_config or cnn_lstm_env:
        try:
            cnn_lstm_results = _run_cnn_lstm_cv(df, valid_idx, feature_sets, y, config)
            results.extend(cnn_lstm_results)
        except Exception as e:
            print(f"  [INFO] CNN-LSTM failed: {e}")
    else:
        print("  [INFO] CNN-LSTM skipped (add 'CNN-LSTM' to ml.models in config.yaml to enable)")

    # Permutation test on best model to verify above-chance performance
    # Uses global median split (required by sklearn: fixed y for permutation)
    best_info = {}
    if results:
        results_df = pd.DataFrame(results)
        best_row = results_df.loc[results_df["balanced_accuracy"].idxmax()]
        best_fs = best_row["feature_set"]
        best_model_name = best_row["model"]
        best_info = {"feature_set": best_fs, "model": best_model_name}

        print(f"\n  Permutation test on best model ({best_model_name}, {best_fs})...")
        fs_cols_best = [c for c in feature_sets[best_fs] if c in df.columns]
        X_perm = df.loc[valid_idx, fs_cols_best]
        n_sel_perm = max(5, min(15, len(fs_cols_best) // 10))
        n_sel_perm = min(n_sel_perm, len(fs_cols_best))

        best_model = _build_models(random_state)[best_model_name]
        pipe_perm = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=n_sel_perm)),
            ("clf", best_model),
        ])

        cv_perm = RepeatedStratifiedKFold(
            n_splits=cv_folds, n_repeats=3, random_state=random_state
        )
        try:
            # n_jobs=1 to prevent OOM on macOS (18GB RAM).
            # n_jobs=-1 spawns parallel workers that each clone the full
            # dataset — causes OOM kill with 941 features x 8 models.
            score_real, perm_scores, perm_pvalue = permutation_test_score(
                pipe_perm, X_perm, y_global,
                scoring="balanced_accuracy", cv=cv_perm,
                n_permutations=200, random_state=random_state,
                n_jobs=1,
            )
            print(f"    Real score: {score_real:.3f}")
            print(f"    Permutation mean: {np.mean(perm_scores):.3f}")
            print(f"    p-value: {perm_pvalue:.4f}")
            print(f"    Above chance: {'YES' if perm_pvalue < 0.05 else 'NO'}")
            logger.info(f"Permutation test: p={perm_pvalue:.4f}")

            for r in results:
                if r["feature_set"] == best_fs and r["model"] == best_model_name:
                    r["perm_p_value"] = round(perm_pvalue, 4)
                    r["perm_mean"] = round(np.mean(perm_scores), 3)
        except Exception as e:
            print(f"    Permutation test failed: {e}")
            logger.error(f"Permutation test failed: {e}")

        # Hyperparameter tuning on best feature set (proposal Section 3.5.2)
        print(f"\n  Running hyperparameter tuning on best feature set: {best_fs}")
        tuned = _run_tuned_classification(df, valid_idx, feature_sets[best_fs], y_global, config)
        results.extend(tuned)

    return pd.DataFrame(results), best_info


def _run_tuned_classification(df, valid_idx, fs_cols, y, config):
    """
    Hyperparameter tuning using RandomizedSearchCV (proposal Section 3.5.2).
    Uses nested CV: inner loop for tuning, outer loop for evaluation.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, \
        RandomizedSearchCV, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score, f1_score, \
        roc_auc_score, recall_score
    from scipy.stats import randint, uniform

    random_state = config["random_state"]
    fs_cols_valid = [c for c in fs_cols if c in df.columns]
    X = df.loc[valid_idx, fs_cols_valid]  # no fillna — imputer handles NaN
    n_select = max(5, min(15, len(fs_cols_valid) // 10))
    n_select = min(n_select, len(fs_cols_valid))

    param_grids = {
        "RandomForest": {
            "clf__n_estimators": randint(50, 300),
            "clf__max_depth": randint(2, 6),
            "clf__min_samples_leaf": randint(2, 8),
        },
        "SVM": {
            "clf__C": uniform(0.01, 10),
            "clf__gamma": ["scale", "auto"],
        },
    }

    # XGBoost
    try:
        from xgboost import XGBClassifier
        param_grids["XGBoost"] = {
            "clf__n_estimators": randint(30, 200),
            "clf__max_depth": randint(2, 5),
            "clf__learning_rate": uniform(0.01, 0.3),
        }
    except ImportError:
        pass

    results = []
    inner_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=random_state)
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_state)

    base_models = _build_models(random_state)

    for model_name, param_dist in param_grids.items():
        if model_name not in base_models:
            continue

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=n_select)),
            ("clf", base_models[model_name]),
        ])

        search = RandomizedSearchCV(
            pipe, param_dist, n_iter=20, cv=inner_cv,
            scoring="balanced_accuracy", random_state=random_state,
            n_jobs=1, refit=True,  # n_jobs=1 to prevent OOM on macOS
        )

        # Nested CV evaluation
        fold_metrics = {"bal_acc": [], "f1": [], "auc": [], "sens": [], "spec": []}
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            search.fit(X_train, y_train)
            preds = search.predict(X_test)

            fold_metrics["bal_acc"].append(balanced_accuracy_score(y_test, preds))
            fold_metrics["f1"].append(f1_score(y_test, preds, zero_division=0))
            fold_metrics["sens"].append(recall_score(y_test, preds, pos_label=1, zero_division=0))
            fold_metrics["spec"].append(recall_score(y_test, preds, pos_label=0, zero_division=0))
            try:
                if hasattr(search, "predict_proba"):
                    probs = search.predict_proba(X_test)[:, 1]
                else:
                    probs = search.decision_function(X_test)
                fold_metrics["auc"].append(roc_auc_score(y_test, probs))
            except Exception:
                fold_metrics["auc"].append(0.5)

        result = {
            "feature_set": "tuned_best",
            "n_features_input": len(fs_cols_valid),
            "n_features_selected": n_select,
            "model": f"{model_name}_tuned",
            "balanced_accuracy": round(np.mean(fold_metrics["bal_acc"]), 3),
            "bal_acc_std": round(np.std(fold_metrics["bal_acc"]), 3),
            "f1": round(np.mean(fold_metrics["f1"]), 3),
            "auc": round(float(np.nanmean(fold_metrics["auc"])) if fold_metrics["auc"] else float("nan"), 3),
            "sensitivity": round(np.mean(fold_metrics["sens"]), 3),
            "specificity": round(np.mean(fold_metrics["spec"]), 3),
        }
        results.append(result)

        print(
            f"  {'tuned_best':30s} | {model_name+'_tuned':12s} | "
            f"Acc: {result['balanced_accuracy']:.3f} "
            f"(+/-{result['bal_acc_std']:.3f}) | "
            f"Sens: {result['sensitivity']:.3f} | "
            f"Spec: {result['specificity']:.3f} | "
            f"F1: {result['f1']:.3f} | AUC: {result['auc']:.3f}"
        )

    return results


def _run_cnn_lstm_cv(df, valid_idx, feature_sets, y, config):
    """
    CNN-LSTM classifier using PyTorch (proposal Section 2.4.2).
    Runs manual stratified k-fold CV since PyTorch models
    don't integrate with sklearn cross_validate.
    """
    try:
        import torch
        import torch.nn as nn
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.metrics import balanced_accuracy_score, f1_score, \
            roc_auc_score, recall_score
    except ImportError:
        print("  [INFO] PyTorch not installed — skipping CNN-LSTM")
        return []

    class CNNLSTM(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(16, 32, batch_first=True)
            self.fc = nn.Linear(32, 1)

        def forward(self, x):
            # x: (batch, n_features) -> (batch, 1, n_features)
            x = x.unsqueeze(1)
            x = self.relu(self.conv1(x))       # (batch, 16, n_features)
            x = x.permute(0, 2, 1)             # (batch, n_features, 16)
            _, (h_n, _) = self.lstm(x)          # h_n: (1, batch, 32)
            out = self.fc(h_n.squeeze(0))       # (batch, 1)
            return out.squeeze(-1)

    ana = config["analysis"]
    random_state = config["random_state"]
    cv_folds = ana["cv_folds"]
    cv_repeats = ana["cv_repeats"]
    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state
    )

    results = []

    for fs_name, fs_cols in feature_sets.items():
        fs_cols_valid = [c for c in fs_cols if c in df.columns]
        if not fs_cols_valid:
            continue

        X_full = df.loc[valid_idx, fs_cols_valid].fillna(0).values
        y_np = y.values

        n_select = min(10, len(fs_cols_valid))
        fold_metrics = {"bal_acc": [], "f1": [], "auc": [], "sens": [], "spec": []}

        for train_idx, test_idx in cv.split(X_full, y_np):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]

            # Feature selection + scaling
            sel = SelectKBest(f_classif, k=n_select).fit(X_train, y_train)
            X_train = sel.transform(X_train)
            X_test = sel.transform(X_test)
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Train
            torch.manual_seed(random_state)
            model = CNNLSTM(n_select)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCEWithLogitsLoss()

            X_t = torch.FloatTensor(X_train)
            y_t = torch.FloatTensor(y_train)

            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                loss = criterion(model(X_t), y_t)
                loss.backward()
                optimizer.step()

            # Predict
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_test))
                probs = torch.sigmoid(logits).numpy()
                preds = (probs >= 0.5).astype(int)

            fold_metrics["bal_acc"].append(balanced_accuracy_score(y_test, preds))
            fold_metrics["f1"].append(f1_score(y_test, preds, zero_division=0))
            fold_metrics["sens"].append(recall_score(y_test, preds, pos_label=1, zero_division=0))
            fold_metrics["spec"].append(recall_score(y_test, preds, pos_label=0, zero_division=0))
            try:
                fold_metrics["auc"].append(roc_auc_score(y_test, probs))
            except ValueError:
                fold_metrics["auc"].append(0.5)

        result = {
            "feature_set": fs_name,
            "n_features_input": len(fs_cols_valid),
            "n_features_selected": n_select,
            "model": "CNN-LSTM",
            "balanced_accuracy": round(np.mean(fold_metrics["bal_acc"]), 3),
            "bal_acc_std": round(np.std(fold_metrics["bal_acc"]), 3),
            "f1": round(np.mean(fold_metrics["f1"]), 3),
            "auc": round(float(np.nanmean(fold_metrics["auc"])) if fold_metrics["auc"] else float("nan"), 3),
            "sensitivity": round(np.mean(fold_metrics["sens"]), 3),
            "specificity": round(np.mean(fold_metrics["spec"]), 3),
        }
        results.append(result)

        print(
            f"  {fs_name:30s} | {'CNN-LSTM':12s} | "
            f"Acc: {result['balanced_accuracy']:.3f} "
            f"(+/-{result['bal_acc_std']:.3f}) | "
            f"Sens: {result['sensitivity']:.3f} | "
            f"Spec: {result['specificity']:.3f} | "
            f"F1: {result['f1']:.3f} | AUC: {result['auc']:.3f}"
        )

    return results


# ──────────────────────────────────────────────────────────────────
# 4C. SHAP Analysis
# ──────────────────────────────────────────────────────────────────

def run_shap_analysis(df, config, best_info=None, target=None, output_dir=None):
    """
    SHAP analysis on the best-performing model/feature-set combination
    to identify candidate biomarker features.

    Args:
        best_info: dict with 'feature_set' and 'model' from classification
        target: continuous column for binary labels. Falls back to first
                configured target.
        output_dir: directory for SHAP outputs (figures + annotated csv).
                Falls back to config.paths.analysis_dir.
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
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_classif

    # Use best feature set from classification, or fall back to conventional
    best_feature_set = (best_info or {}).get("feature_set", "conventional_qeeg")

    if target is None:
        from utils.io import get_targets
        target = get_targets(config)[0]
    continuous_col = target
    if continuous_col not in df.columns or df[continuous_col].isna().all():
        print(f"  SKIP: Target '{continuous_col}' not found or all-NaN.")
        return None
    y_cont = df[continuous_col].dropna()
    y = (y_cont > y_cont.median()).astype(int)
    valid_idx = y.index
    print(f"  Target variable: {continuous_col}")

    feature_sets = get_feature_sets(df, config)
    fs_cols = feature_sets.get(best_feature_set, [])
    fs_cols = [c for c in fs_cols if c in df.columns]

    print(f"  Using feature set: {best_feature_set} ({len(fs_cols)} features)")

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

    # Compute SHAP values per CV fold and average — avoids overfitting
    # SHAP importance on a model trained on all 28 subjects is misleading.
    from sklearn.model_selection import StratifiedKFold as _SKF

    random_state_shap = config["random_state"]
    cv_shap = _SKF(n_splits=5, shuffle=True, random_state=random_state_shap)

    all_shap_values = []
    feature_selection_counts = np.zeros(len(selected_names))

    for fold_idx, (train_idx, test_idx) in enumerate(cv_shap.split(X_selected, y)):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_selected[train_idx])
        X_test_s = scaler.transform(X_selected[test_idx])

        model = RandomForestClassifier(
            n_estimators=100, max_depth=3,
            class_weight="balanced",
            random_state=random_state_shap,
        )
        model.fit(X_train_s, y.iloc[train_idx])

        explainer = shap.TreeExplainer(model)
        shap_fold = explainer.shap_values(X_test_s)

        if isinstance(shap_fold, list):
            shap_fold = shap_fold[1]
        if shap_fold.ndim == 3:
            shap_fold = shap_fold[:, :, 1]

        all_shap_values.append(np.mean(np.abs(shap_fold), axis=0).ravel())

    # Average SHAP importance across folds
    mean_abs_shap = np.mean(all_shap_values, axis=0)
    shap_std = np.std(all_shap_values, axis=0)

    # Also compute SHAP on full data for the summary plot only
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X_selected)
    model_full = RandomForestClassifier(
        n_estimators=100, max_depth=3,
        class_weight="balanced",
        random_state=random_state_shap,
    )
    model_full.fit(X_scaled, y)
    explainer_full = shap.TreeExplainer(model_full)
    shap_values = explainer_full.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    importance = pd.DataFrame({
        "feature": selected_names,
        "mean_abs_shap": mean_abs_shap,
        "shap_std_across_folds": shap_std,
        "shap_cv": shap_std / (mean_abs_shap + 1e-10),  # coefficient of variation
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n  Top biomarker candidates (by SHAP importance, CV-averaged):")
    for _, row in importance.head(10).iterrows():
        stability = "stable" if row["shap_cv"] < 0.5 else "unstable"
        print(f"    {row['feature']:40s}  SHAP: {row['mean_abs_shap']:.4f} "
              f"(+/-{row['shap_std_across_folds']:.4f}, {stability})")

    # Biological interpretation of top biomarkers
    try:
        from utils.bio_interpretation import print_biomarker_report
        print_biomarker_report(importance, top_n=10)
    except Exception as e:
        logger.warning(f"Biological interpretation skipped: {e}")

    # Save SHAP summary plot
    out_dir = output_dir or config["output_dir"]
    figures_dir = os.path.join(out_dir, "figures")
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

    # Save annotated importance with biological interpretation
    try:
        from utils.bio_interpretation import interpret_biomarkers
        annotated = interpret_biomarkers(importance)
        annotated.to_csv(
            os.path.join(out_dir, "shap_annotated.csv"),
            index=False,
        )
        logger.info("Annotated SHAP importance saved")
    except Exception:
        pass

    return importance


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def run(config, full_df=None):
    """
    Run full Stage 4 analysis. Loops 4B (classification) and 4C (SHAP)
    over every target listed in ``config['analysis']['targets']``.
    Descriptives and correlations are target-agnostic and run once.

    Writes correlations.csv at ``config['output_dir']``, plus per-target
    ml_results.csv, shap_importance.csv, shap_annotated.csv, and figures/
    subdir. Auto-loads full_dataset.csv from ``config['input_dir']``
    (latest engineering run) when ``full_df`` is None.
    """
    from utils.io import get_targets, write_stage_notes
    from stages.engineering import load_full_dataset

    print("\n" + "=" * 60)
    print("STAGE 4: Analysis")
    print("=" * 60)

    engineering_input = config.get("input_dir")
    if full_df is None:
        if engineering_input is None:
            raise FileNotFoundError(
                "No engineering output found. "
                "Run `python pipeline.py --engineering` first."
            )
        full_df = load_full_dataset(config, stage_dir=engineering_input)

    # 4A: Descriptives + Correlations (target-agnostic)
    desc = run_descriptives(full_df, config)
    corr_results = run_correlations(full_df, config)

    base_dir = config["output_dir"]
    print(f"  Analysis output dir: {base_dir}")

    if not corr_results.empty:
        corr_results.to_csv(
            os.path.join(base_dir, "correlations.csv"), index=False
        )

    targets = get_targets(config)
    multi = len(targets) > 1
    print(f"\n  Targets to analyse: {targets}")

    per_target = {}
    for target in targets:
        print("\n" + "#" * 60)
        print(f"# TARGET: {target}")
        print("#" * 60)

        # Per-target output directory (subdir only when >1 target — preserves
        # the legacy single-target layout)
        target_dir = os.path.join(base_dir, target) if multi else base_dir
        os.makedirs(target_dir, exist_ok=True)

        if target not in full_df.columns or full_df[target].isna().all():
            print(f"  SKIP: '{target}' missing or all-NaN in full_dataset.")
            per_target[target] = {"ml_results": pd.DataFrame(), "shap_importance": None}
            continue

        # 4B: ML classification
        ml_results, best_info = run_classification(full_df, config, target=target)

        # 4C: SHAP on best-performing model/feature-set
        shap_importance = run_shap_analysis(
            full_df, config, best_info=best_info,
            target=target, output_dir=target_dir,
        )

        if not ml_results.empty:
            ml_results.to_csv(
                os.path.join(target_dir, "ml_results.csv"), index=False
            )
            best = ml_results.loc[ml_results["balanced_accuracy"].idxmax()]
            print(f"\n  BEST MODEL ({target}):")
            print(f"    Feature set: {best['feature_set']}")
            print(f"    Model: {best['model']}")
            print(f"    Balanced accuracy: {best['balanced_accuracy']:.3f}")
            print(f"    H4 target (>=0.75): {'MET' if best['balanced_accuracy'] >= 0.75 else 'NOT MET'}")

        if shap_importance is not None:
            shap_importance.to_csv(
                os.path.join(target_dir, "shap_importance.csv"), index=False
            )

        per_target[target] = {
            "ml_results": ml_results,
            "shap_importance": shap_importance,
            "best_info": best_info,
            "output_dir": target_dir,
        }

    write_stage_notes(base_dir, {
        "stage": "analysis",
        "input_engineering_dir": engineering_input,
        "targets": targets,
        "n_subjects": int(full_df.shape[0]),
        "outputs": (
            ["correlations.csv"]
            + [f"{t}/ml_results.csv" if multi else "ml_results.csv" for t in targets]
            + [f"{t}/shap_importance.csv" if multi else "shap_importance.csv" for t in targets]
        ),
    })

    return {
        "descriptives": desc,
        "correlations": corr_results,
        "targets": targets,
        "per_target": per_target,
    }
