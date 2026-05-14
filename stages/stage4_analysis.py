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

    Optional config-driven filter: if `ml.feature_sets` is a non-empty list,
    keep only the named sets. Unknown names are skipped with a warning so
    a typo doesn't silently produce empty results.
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

    feature_sets = {
        "conventional_qeeg": conventional,
        "conventional_plus_advanced": advanced,
        "covariance_only": covariance,
        "all_features": all_features,
    }

    # Optional config-driven selector. Strip inline YAML comments so users
    # can leave explanatory comments next to each entry in config.yaml.
    requested = (config.get("ml", {}) or {}).get("feature_sets")
    if requested:
        requested = [s.split("#")[0].strip() for s in requested if s]
        unknown = [s for s in requested if s not in feature_sets]
        if unknown:
            print(f"  [WARN] ml.feature_sets entries not recognised, "
                  f"skipping: {unknown}")
        feature_sets = {k: v for k, v in feature_sets.items() if k in requested}
        if not feature_sets:
            print("  [WARN] ml.feature_sets filter eliminated ALL sets; "
                  "falling back to the full default lineup")
            feature_sets = {
                "conventional_qeeg": conventional,
                "conventional_plus_advanced": advanced,
                "covariance_only": covariance,
                "all_features": all_features,
            }

    return feature_sets


def _build_models(random_state):
    """
    Build all 8 models specified in the research proposal:
    RF, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP, CNN-LSTM.
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


def run_classification(df, config):
    """
    ML classification comparing feature sets and models.
    Implements all 8 algorithms from the research proposal (Section 2.4.2):
    RF, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP, CNN-LSTM.
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
    continuous_col = "Global_EF"
    if continuous_col not in df.columns or df[continuous_col].isna().all():
        print("  ERROR: No valid classification target")
        return pd.DataFrame()

    y_continuous = df[continuous_col].dropna()
    valid_idx = y_continuous.index

    # Global median split for permutation test (sklearn requires fixed y)
    global_median = y_continuous.median()
    y_global = (y_continuous > global_median).astype(int)

    feature_sets = get_feature_sets(df, config)
    cv_folds = config["ml"]["cv_folds"]
    cv_repeats = config["ml"]["cv_repeats"]
    random_state = config["ml"]["random_state"]

    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state
    )

    models = _build_models(random_state)

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

            try:
                for train_idx, test_idx in cv.split(X, y_global):
                    X_train = X.iloc[train_idx]
                    X_test = X.iloc[test_idx]

                    # Compute median on training fold only
                    train_median = y_continuous.iloc[train_idx].median()
                    y_train = (y_continuous.iloc[train_idx] > train_median).astype(int)
                    y_test = (y_continuous.iloc[test_idx] > train_median).astype(int)

                    pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("select", SelectKBest(f_classif, k=n_select)),
                        ("clf", model),
                    ])

                    pipe.fit(X_train, y_train)
                    preds = pipe.predict(X_test)

                    fold_metrics["bal_acc"].append(
                        balanced_accuracy_score(y_test, preds))
                    fold_metrics["f1"].append(
                        f1_score(y_test, preds, zero_division=0))
                    fold_metrics["sens"].append(
                        recall_score(y_test, preds, pos_label=1, zero_division=0))
                    fold_metrics["spec"].append(
                        recall_score(y_test, preds, pos_label=0, zero_division=0))
                    try:
                        if hasattr(pipe, "predict_proba"):
                            probs = pipe.predict_proba(X_test)[:, 1]
                        else:
                            probs = pipe.decision_function(X_test)
                        fold_metrics["auc"].append(roc_auc_score(y_test, probs))
                    except Exception:
                        fold_metrics["auc"].append(0.5)

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
                    "auc": round(np.mean(fold_metrics["auc"]), 3),
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
    # Disabled by default: PyTorch hangs after sklearn joblib on macOS.
    # Set RUN_CNN_LSTM=1 to enable.
    import os as _os
    if _os.environ.get("RUN_CNN_LSTM", "0") == "1":
        try:
            cnn_lstm_results = _run_cnn_lstm_cv(df, valid_idx, feature_sets, y, config)
            results.extend(cnn_lstm_results)
        except Exception as e:
            print(f"  [INFO] CNN-LSTM failed: {e}")
    else:
        print("  [INFO] CNN-LSTM skipped (set RUN_CNN_LSTM=1 to enable)")

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

    random_state = config["ml"]["random_state"]
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
            "auc": round(np.mean(fold_metrics["auc"]), 3),
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

    random_state = config["ml"]["random_state"]
    cv_folds = config["ml"]["cv_folds"]
    cv_repeats = config["ml"]["cv_repeats"]
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
            "auc": round(np.mean(fold_metrics["auc"]), 3),
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

def run_shap_analysis(df, config, best_info=None):
    """
    SHAP analysis on the best-performing model/feature-set combination
    to identify candidate biomarker features.

    Args:
        best_info: dict with 'feature_set' and 'model' from classification
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

    # Use continuous EF with global median for SHAP labels
    continuous_col = "Global_EF"
    y_cont = df[continuous_col].dropna()
    y = (y_cont > y_cont.median()).astype(int)
    valid_idx = y.index

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

    random_state_shap = config["ml"]["random_state"]
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

    # Save annotated importance with biological interpretation
    try:
        from utils.bio_interpretation import interpret_biomarkers
        annotated = interpret_biomarkers(importance)
        annotated.to_csv(
            os.path.join(config["paths"]["output_dir"], "shap_annotated.csv"),
            index=False,
        )
        logger.info("Annotated SHAP importance saved")
    except Exception:
        pass

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
    ml_results, best_info = run_classification(full_df, config)

    # 4C: SHAP on best-performing model/feature-set
    # CI can set SKIP_SHAP=1 to bypass — SHAP's tree explainer can stall on
    # degenerate-input synthetic fixtures (e.g. CI fair-comparison run was
    # hanging > 40 min at this step on N=28 synthetic data).
    if os.environ.get("SKIP_SHAP") == "1":
        print("  SKIP_SHAP=1 set — skipping SHAP analysis.")
        shap_importance = pd.DataFrame()
    else:
        shap_importance = run_shap_analysis(full_df, config, best_info=best_info)

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
