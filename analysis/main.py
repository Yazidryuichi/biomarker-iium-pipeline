"""
Stage 5: Analysis (full regression)
===================================
Hierarchical OLS hypothesis-test per (target, feature_set):
  Restricted:  y ~ covariates                       (default: age_months)
  Full:        y ~ covariates + composite features
  Block F:     incremental R^2 of the composite block over covariates-only

Per-composite:
  beta, SE, t, p_two_sided, p_one_sided (directional per `directions`),
  p_corrected (Bonferroni or FDR-BH across composites in the feature set).
  Cronbach's alpha gate per composite (where components are listed).
  Subject-resample bootstrap 95% CI on every coefficient.

Why hierarchical OLS instead of CV regression:
  - At N=26 a CV regression with R^2 selection is dominated by noise.
  - OLS with covariates is the Frisch-Waugh-Lovell equivalent of
    residualizing both y and X against age — handles the age-confound
    critique without leakage. Diagnostics (covariate R^2, full R^2,
    block F) remain interpretable.
  - No model search, no SHAP, no feature curation — pre-registered
    inference only.

Output per target: output/<ts>/<target>/{composite_alpha.csv, coef_inference.csv,
  block_f_test.json, bootstrap_ci.csv, summary.json}.
Plus output/<ts>/{headline.json, run_notes.json}.
"""
from __future__ import annotations

import json
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore")

STAGE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = STAGE_DIR.parent
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def load_config():
    with open(STAGE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def make_output_dir():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = STAGE_DIR / "output" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"


def latest_output(root):
    root = resolve(root)
    if not root.is_dir():
        return None
    runs = sorted([d for d in root.iterdir()
                   if d.is_dir() and TS_RE.match(d.name)], reverse=True)
    return runs[0] if runs else None


# ──────────────────────────────────────────────────────────────────
# Cronbach's alpha
# ──────────────────────────────────────────────────────────────────

def cronbach_alpha(items_df):
    X = items_df.apply(pd.to_numeric, errors="coerce").dropna()
    k = X.shape[1]; n = X.shape[0]
    if k < 2 or n < 3:
        return float("nan"), n
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan"), n
    return float((k / (k - 1.0)) * (1.0 - item_var / total_var)), n


# ──────────────────────────────────────────────────────────────────
# Hierarchical OLS
# ──────────────────────────────────────────────────────────────────

def fit_ols(df, target, predictors):
    import statsmodels.api as sm
    X = sm.add_constant(df[list(predictors)], has_constant="add")
    return sm.OLS(df[target], X).fit()


def bootstrap_ols_ci(df, target, predictors, n_boot, seed):
    """Subject-resample bootstrap on the FULL OLS coefficients."""
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    n = len(df)
    coefs = {p: [] for p in predictors}
    coefs["const"] = []
    idx = np.arange(n)
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        sub = df.iloc[b]
        try:
            X = sm.add_constant(sub[list(predictors)], has_constant="add")
            res = sm.OLS(sub[target], X).fit()
            for p in res.params.index:
                if p in coefs:
                    coefs[p].append(float(res.params[p]))
        except Exception:
            continue
    rows = []
    for name, arr in coefs.items():
        if not arr:
            continue
        a = np.asarray(arr, dtype=float)
        rows.append({
            "predictor": name,
            "median":   round(float(np.percentile(a, 50)), 4),
            "ci_lo":    round(float(np.percentile(a, 2.5)), 4),
            "ci_hi":    round(float(np.percentile(a, 97.5)), 4),
            "n_iter":   int(a.size),
        })
    return pd.DataFrame(rows)


def apply_correction(pvals_one, method, k, alpha):
    arr = np.asarray(pvals_one, dtype=float)
    if method == "bonferroni":
        return np.minimum(arr * k, 1.0)
    if method in ("fdr_bh", "bh"):
        from statsmodels.stats.multitest import multipletests
        _, p_corr, _, _ = multipletests(arr, alpha=alpha, method="fdr_bh")
        return p_corr
    return arr


def run_hypothesis_test(full_df, target, feature_set_name, feature_cols,
                        covariates, directions, components_map,
                        alpha_min, correction, alpha_level, bootstrap_n,
                        seed, out_dir):
    """One hierarchical OLS pass for one (target, feature_set) combo."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cronbach gate per composite
    alpha_rows = []
    for comp in feature_cols:
        comps = components_map.get(comp)
        if not comps:
            alpha_rows.append({"composite": comp, "k_items": None, "n_obs": None,
                               "alpha": None, "pass_gate": None,
                               "note": "no components listed (skip gate)"})
            continue
        present = [c for c in comps if c in full_df.columns]
        if len(present) < 2:
            alpha_rows.append({"composite": comp, "k_items": len(present),
                               "n_obs": 0, "alpha": None, "pass_gate": False,
                               "note": "<2 components present"})
            continue
        a, n = cronbach_alpha(full_df[present])
        alpha_rows.append({
            "composite": comp, "k_items": len(present), "n_obs": int(n),
            "alpha": round(a, 4) if np.isfinite(a) else None,
            "pass_gate": bool(np.isfinite(a) and a >= alpha_min),
            "note": "",
        })
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(out_dir / "composite_alpha.csv", index=False)
    gate_states = [r["pass_gate"] for r in alpha_rows if r["pass_gate"] is not None]
    all_pass = bool(gate_states) and all(gate_states)

    # 2. Build OLS design matrix
    needed = list(dict.fromkeys(covariates + list(feature_cols) + [target]))
    missing = [c for c in needed if c not in full_df.columns]
    if missing:
        raise RuntimeError(f"hypothesis_test: missing columns {missing}")
    df = full_df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    n_used = len(df); n_drop = len(full_df) - n_used
    if n_used < (len(covariates) + len(feature_cols) + 3):
        raise RuntimeError(f"insufficient N for OLS: n_used={n_used}, "
                           f"need >= {len(covariates) + len(feature_cols) + 3}")

    restricted = fit_ols(df, target, covariates) if covariates else None
    full = fit_ols(df, target, list(covariates) + list(feature_cols))

    # 3. Block F-test for incremental R^2
    block = {}
    if restricted is not None:
        r2_r, r2_f = float(restricted.rsquared), float(full.rsquared)
        q = int(full.df_model) - int(restricted.df_model)
        df_resid = n_used - int(full.df_model) - 1
        if q > 0 and df_resid > 0 and (1.0 - r2_f) > 0:
            f_stat = ((r2_f - r2_r) / q) / ((1.0 - r2_f) / df_resid)
            block = {
                "delta_r2": round(r2_f - r2_r, 4),
                "F":        round(float(f_stat), 4),
                "df_num":   int(q),
                "df_den":   int(df_resid),
                "p_value":  round(float(1.0 - stats.f.cdf(f_stat, q, df_resid)), 4),
            }
    with open(out_dir / "block_f_test.json", "w") as f:
        json.dump(block, f, indent=2, default=str)

    # 4. Per-composite inference
    rows = []; pvals_one = []
    for comp in feature_cols:
        beta = float(full.params.get(comp, np.nan))
        se = float(full.bse.get(comp, np.nan))
        t = float(full.tvalues.get(comp, np.nan))
        p2 = float(full.pvalues.get(comp, np.nan))
        d = directions.get(comp)
        if d in ("positive", "negative") and np.isfinite(beta) and np.isfinite(p2):
            matches = (d == "positive" and beta > 0) or (d == "negative" and beta < 0)
            p1 = p2 / 2.0 if matches else 1.0 - p2 / 2.0
            sign_ok = bool(matches)
        else:
            p1 = p2; d = "two_sided"; sign_ok = None
        rows.append({
            "composite": comp, "direction": d,
            "beta": round(beta, 4), "se": round(se, 4), "t": round(t, 3),
            "p_two_sided": round(p2, 4), "p_one_sided": round(p1, 4),
            "sign_matches_hypothesis": sign_ok,
        })
        pvals_one.append(p1)

    k = len(feature_cols)
    p_corr = apply_correction(pvals_one, correction, k, alpha_level)
    for r, pc in zip(rows, p_corr):
        r["p_corrected"] = round(float(pc), 4)
        r["correction"] = correction
        r["significant"] = bool(np.isfinite(pc) and pc < alpha_level)
    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(out_dir / "coef_inference.csv", index=False)

    # 5. Bootstrap CIs
    boot_df = bootstrap_ols_ci(df, target, list(covariates) + list(feature_cols),
                               n_boot=bootstrap_n, seed=seed)
    boot_df.to_csv(out_dir / "bootstrap_ci.csv", index=False)

    # 6. Summary + console
    summary = {
        "target": target,
        "feature_set": feature_set_name,
        "covariates": list(covariates),
        "composites": list(feature_cols),
        "n_used": int(n_used),
        "n_dropped_nan": int(n_drop),
        "covariate_r2": round(float(restricted.rsquared), 4) if restricted is not None else None,
        "full_r2": round(float(full.rsquared), 4),
        "block_f_test": block,
        "correction": correction,
        "alpha_level": alpha_level,
        "alpha_min_acceptable": alpha_min,
        "cronbach_gate_all_pass": all_pass,
        "bootstrap_n": bootstrap_n,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  [{feature_set_name}] N={n_used} (dropped {n_drop} for NaN)")
    print(f"    Cronbach gate (>={alpha_min}): "
          f"{'PASS' if all_pass else 'see composite_alpha.csv'}")
    for r in alpha_rows:
        print(f"      {r['composite']:30s} k={r['k_items']} "
              f"n={r['n_obs']} alpha={r['alpha']} pass={r['pass_gate']}")
    print(f"    Restricted R^2 = {summary['covariate_r2']}  |  "
          f"Full R^2 = {summary['full_r2']}")
    if block:
        print(f"    Block F({block['df_num']},{block['df_den']})={block['F']}  "
              f"deltaR^2={block['delta_r2']:+.4f}  p={block['p_value']}")
    print(f"    Per-composite ({correction}, k={k}, alpha={alpha_level}):")
    for r in rows:
        flag = "  *" if r["significant"] else ""
        print(f"      {r['composite']:30s} dir={r['direction']:10s} "
              f"beta={r['beta']:+.4f}  t={r['t']:+.3f}  "
              f"p_1={r['p_one_sided']}  p_corr={r['p_corrected']}{flag}")
    return summary


# ──────────────────────────────────────────────────────────────────
# Exploratory ML grid (sensitivity track)
# ──────────────────────────────────────────────────────────────────

def _build_regressors(random_state):
    from sklearn.linear_model import ElasticNetCV
    from sklearn.ensemble import RandomForestRegressor
    out = {}
    out["ElasticNetCV"] = lambda: ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        cv=5, random_state=random_state, max_iter=10000, n_jobs=1,
    )
    out["RandomForest"] = lambda: RandomForestRegressor(
        n_estimators=300, max_depth=3, min_samples_leaf=3,
        random_state=random_state, n_jobs=1,
    )
    try:
        from xgboost import XGBRegressor
        out["XGBoost"] = lambda: XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=random_state, n_jobs=1, verbosity=0,
        )
    except ImportError:
        pass
    return out


def _make_ml_pipeline(model_factory, n_features, k_cap, use_kbest):
    """Imputer -> Scaler -> [SelectKBest] -> model. SelectKBest only when
    the pool is wider than k_cap AND the model isn't doing its own L1.
    """
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if use_kbest and n_features > k_cap:
        steps.append(("kbest", SelectKBest(score_func=f_regression, k=k_cap)))
    steps.append(("model", model_factory()))
    return Pipeline(steps)


def _pooled_oof_r2(pipeline_factory, X, y, cv):
    """Mean OOF R^2 across CV repeats, MAE against mean predictor baseline."""
    from sklearn.metrics import r2_score, mean_absolute_error
    fold_r2, fold_mae = [], []
    for tr, te in cv.split(X):
        pipe = pipeline_factory()
        pipe.fit(X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr], y[tr])
        yp = pipe.predict(X.iloc[te] if isinstance(X, pd.DataFrame) else X[te])
        if np.std(y[te]) > 0:
            fold_r2.append(r2_score(y[te], yp))
        fold_mae.append(mean_absolute_error(y[te], yp))
    baseline_mae = float(np.mean(np.abs(y - np.mean(y))))
    return (float(np.mean(fold_r2)) if fold_r2 else float("nan"),
            float(np.mean(fold_mae)),
            baseline_mae)


def _resolve_scheme_columns(scheme_name, schemes_resolved, full_cols):
    """Given a scheme name, return the column list available in `full_cols`."""
    info = schemes_resolved.get(scheme_name, {})
    if scheme_name == "theory":
        cols = info.get("members", [])
    elif scheme_name == "data_driven":
        cols = info.get("members", [])
    elif scheme_name == "hybrid":
        cols = list(info.get("apriori_members", [])) + list(
            info.get("rest_pool", []))
    else:
        cols = []
    return [c for c in cols if c in full_cols]


def run_exploratory_ml(full_df, targets, schemes_resolved, models_cfg,
                       cv_cfg, perm_cfg, covariates, k_cap, out_dir,
                       random_state):
    """Per-(target × scheme × model) RepeatedKFold OOF R^2 + MAE, plus
    permutation null with selection-corrected max-stat per target."""
    from sklearn.model_selection import RepeatedKFold, KFold

    n_splits = int(cv_cfg.get("n_splits", 5))
    n_repeats = int(cv_cfg.get("n_repeats", 3))
    n_perm = int(perm_cfg.get("n_perm", 100))
    perm_n_splits = int(perm_cfg.get("perm_n_splits", 5))

    models = _build_regressors(random_state)
    if not models:
        return pd.DataFrame()

    cv_main = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                             random_state=random_state)
    cv_perm = KFold(n_splits=perm_n_splits, shuffle=True,
                     random_state=random_state)

    scheme_names = list(schemes_resolved.keys())
    rng = np.random.default_rng(random_state)

    print(f"\n[Exploratory ML grid]")
    print(f"  Models: {list(models.keys())}")
    print(f"  Schemes: {scheme_names}")
    print(f"  Targets: {targets}")
    print(f"  CV: {n_splits}x{n_repeats}  |  perm: {n_perm}x"
          f"{perm_n_splits}")

    rows = []
    # Cache observed R² per (target, scheme, model) for selection-corrected p
    observed_r2 = {}
    null_max_per_target = {t: [] for t in targets}
    null_combo = {}

    for target in targets:
        if target not in full_df.columns or full_df[target].isna().all():
            print(f"  SKIP target '{target}': missing or all NaN")
            continue
        print(f"\n  Target: {target}")
        cov_present = [c for c in covariates if c in full_df.columns]
        for scheme in scheme_names:
            scheme_cols = _resolve_scheme_columns(scheme, schemes_resolved,
                                                   full_df.columns)
            if not scheme_cols:
                print(f"    SKIP scheme {scheme}: no columns")
                continue
            # Include covariates in X — they enter the model as predictors so
            # the ML grid does the same Frisch-Waugh handling of age as OLS.
            X_cols = list(dict.fromkeys(cov_present + scheme_cols))
            needed = [target] + X_cols
            sub = full_df[needed].apply(pd.to_numeric,
                                          errors="coerce").dropna()
            if len(sub) < n_splits * 2:
                print(f"    SKIP {scheme}: N={len(sub)} too small")
                continue
            y = sub[target].values
            X = sub[X_cols]
            n_feat = X.shape[1]

            for model_name, factory in models.items():
                # data_driven + non-linear -> use_kbest
                # data_driven + ElasticNet -> rely on L1
                # theory + any -> no selection (already small)
                # hybrid + non-linear -> use_kbest on union
                use_kbest = (
                    scheme in ("data_driven", "hybrid")
                    and model_name != "ElasticNetCV"
                )

                def _pf(_factory=factory, _nfeat=n_feat,
                        _use_kbest=use_kbest):
                    return _make_ml_pipeline(_factory, _nfeat, k_cap,
                                              _use_kbest)

                r2, mae, base_mae = _pooled_oof_r2(_pf, X, y, cv_main)
                observed_r2[(target, scheme, model_name)] = r2
                rows.append({
                    "target": target, "scheme": scheme,
                    "model": model_name,
                    "n_used": int(len(sub)),
                    "n_features_input": int(n_feat),
                    "use_kbest": bool(use_kbest),
                    "oof_r2": round(r2, 4) if np.isfinite(r2) else None,
                    "oof_mae": round(mae, 4),
                    "baseline_mae": round(base_mae, 4),
                })
                print(f"    {scheme:13s} | {model_name:14s} | "
                      f"R²={r2:+.3f}  MAE={mae:.3f} "
                      f"(base {base_mae:.3f})")

        # Permutation null for this target across the full scheme×model grid
        print(f"  Permutation null ({n_perm} perms; selection-corrected "
              f"max-stat per target)...")
        for perm_i in range(n_perm):
            y_perm = rng.permutation(y)
            perm_combo_r2 = []
            for scheme in scheme_names:
                scheme_cols = _resolve_scheme_columns(scheme,
                                                       schemes_resolved,
                                                       full_df.columns)
                if not scheme_cols:
                    continue
                X_cols = list(dict.fromkeys(cov_present + scheme_cols))
                # Re-build sub with the same y_perm aligned to last sub
                # (use the sub computed for this target — assume all schemes
                # see same N for simplicity; if NaN patterns differ they
                # only differ in feature space)
                # For permutation, hold y_perm fixed and rebuild X each scheme
                sub_p = full_df[[target] + X_cols].apply(
                    pd.to_numeric, errors="coerce").dropna()
                if len(sub_p) < perm_n_splits * 2:
                    continue
                yp = rng.permutation(sub_p[target].values)
                Xp = sub_p[X_cols]
                for model_name, factory in models.items():
                    use_kbest = (
                        scheme in ("data_driven", "hybrid")
                        and model_name != "ElasticNetCV"
                    )

                    def _pf(_factory=factory, _nfeat=Xp.shape[1],
                            _use_kbest=use_kbest):
                        return _make_ml_pipeline(_factory, _nfeat, k_cap,
                                                  _use_kbest)

                    try:
                        r2_p, _, _ = _pooled_oof_r2(_pf, Xp, yp, cv_perm)
                    except Exception:
                        r2_p = float("nan")
                    null_combo.setdefault(
                        (target, scheme, model_name), []).append(r2_p)
                    perm_combo_r2.append(r2_p)
            if perm_combo_r2:
                null_max_per_target[target].append(
                    float(np.nanmax(perm_combo_r2)))
            if (perm_i + 1) % 20 == 0:
                print(f"    perm {perm_i+1}/{n_perm}")

    # Compute permutation p-values
    for r in rows:
        key = (r["target"], r["scheme"], r["model"])
        obs = observed_r2.get(key, float("nan"))
        nulls = np.asarray(null_combo.get(key, []), dtype=float)
        nulls = nulls[np.isfinite(nulls)]
        if nulls.size and np.isfinite(obs):
            p_combo = float((nulls >= obs).mean())
        else:
            p_combo = float("nan")
        # Selection-corrected per target (max across schemes×models)
        target_nulls = np.asarray(
            null_max_per_target.get(r["target"], []), dtype=float)
        target_nulls = target_nulls[np.isfinite(target_nulls)]
        if target_nulls.size and np.isfinite(obs):
            p_corrected = float((target_nulls >= obs).mean())
        else:
            p_corrected = float("nan")
        r["perm_p_uncorrected"] = round(p_combo, 4)
        r["perm_p_selection_corrected_within_target"] = round(p_corrected, 4)
        r["n_perm"] = int(nulls.size)

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(out_dir / "exploratory_ml_grid.csv", index=False)

    summary = {
        "n_combos": len(rows),
        "n_perm": n_perm,
        "perm_n_splits": perm_n_splits,
        "cv": {"n_splits": n_splits, "n_repeats": n_repeats},
        "models": list(models.keys()),
        "schemes": scheme_names,
        "targets_run": [t for t in targets if t in
                        {r["target"] for r in rows}],
        "k_cap_for_kbest": k_cap,
    }
    with open(out_dir / "exploratory_ml_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Grid CSV: {out_dir / 'exploratory_ml_grid.csv'}")
    return grid_df


# ──────────────────────────────────────────────────────────────────
# Screening index (ONE pre-specified combo, from confirmatory OLS)
# ──────────────────────────────────────────────────────────────────

def compute_screening_index(full_df, target, covariates, composite_cols,
                             out_dir, random_state, label_prefix):
    """Per-subject LOO-OLS predicted score with Bayesian shrinkage,
    prediction interval, and confidence_flag.

    This is run for ONE pre-specified combination only (per Rule 7 in
    feedback-no-manufactured-validation memory): the primary confirmatory
    analysis path (OLS on the a_priori_union for `target`). Not run across
    the exploratory grid.

    Computation:
      - LOO-CV with OLS y ~ covariates + composites on N-1 -> predict i held out.
      - Bayesian shrinkage on LOO predictions toward the sample mean of y,
        weighted by tau^2 / (tau^2 + sigma_loo^2) where tau^2 = var of LOO
        preds, sigma_loo^2 = LOO residual variance.
      - Prediction interval per subject = LOO residual SD * t(N-1, 0.025).
      - confidence_flag = "incremental_eeg_signal" if the full-model LOO R^2
        beats age-only LOO R^2 at incremental F-test p < 0.05, else
        "score_approximates_age_only_proxy".

    Naming: this is a CANDIDATE EF SCREENING INDEX. Not a diagnostic. Not
    anchored to clinical norms. Percentile within sample only — no tier
    label suggesting clinical categories.
    """
    import statsmodels.api as sm

    cov_cols = [c for c in covariates if c in full_df.columns]
    comp_cols = [c for c in composite_cols if c in full_df.columns]
    needed = [target] + cov_cols + comp_cols
    sub = full_df[["subject_id"] + needed].copy()
    sub_num = sub[needed].apply(pd.to_numeric, errors="coerce")
    mask = sub_num.notna().all(axis=1)
    sub = sub.loc[mask].reset_index(drop=True)
    if len(sub) < 5:
        print(f"    SKIP screening index: N={len(sub)} too small")
        return None

    n = len(sub)
    y = sub[target].astype(float).values
    X_full = sub[cov_cols + comp_cols].astype(float).values
    X_cov = sub[cov_cols].astype(float).values

    def _loo_predict(X_arr, y_arr):
        preds = np.zeros(n)
        for i in range(n):
            mask_i = np.ones(n, dtype=bool); mask_i[i] = False
            X_tr = sm.add_constant(X_arr[mask_i], has_constant="add")
            X_te = sm.add_constant(X_arr[i:i+1], has_constant="add")
            if X_te.shape[1] < X_tr.shape[1]:
                X_te = np.concatenate(
                    [X_te, np.zeros((1, X_tr.shape[1] - X_te.shape[1]))],
                    axis=1)
            res = sm.OLS(y_arr[mask_i], X_tr).fit()
            preds[i] = float(res.predict(X_te)[0])
        return preds

    loo_full = _loo_predict(X_full, y)
    loo_age = _loo_predict(X_cov, y)

    res_full = y - loo_full
    res_age = y - loo_age
    sigma_full = float(np.std(res_full, ddof=1))
    ssr_full = float(np.sum(res_full ** 2))
    ssr_age = float(np.sum(res_age ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2_full = 1.0 - ssr_full / sst if sst > 0 else float("nan")
    r2_age = 1.0 - ssr_age / sst if sst > 0 else float("nan")

    # Incremental F on LOO residuals (treat as if they were OOF predictions)
    k_block = len(comp_cols)
    df_resid = n - len(cov_cols) - k_block - 1
    if k_block > 0 and df_resid > 0 and r2_age < 1.0:
        f_inc = ((r2_full - r2_age) / k_block) / ((1 - r2_full) / df_resid) \
            if r2_full < 1.0 else float("inf")
        p_inc = (1.0 - stats.f.cdf(f_inc, k_block, df_resid)
                 if np.isfinite(f_inc) else 0.0)
    else:
        f_inc, p_inc = float("nan"), float("nan")
    incremental_significant = bool(np.isfinite(p_inc) and p_inc < 0.05)
    confidence_flag = (
        "incremental_eeg_signal" if incremental_significant
        else "score_approximates_age_only_proxy"
    )

    # Bayesian shrinkage toward sample mean of y
    y_mean = float(np.mean(y))
    tau2 = float(np.var(loo_full, ddof=1))
    sigma2 = float(sigma_full ** 2)
    if (tau2 + sigma2) > 0:
        w = tau2 / (tau2 + sigma2)
    else:
        w = 1.0
    shrunk = w * loo_full + (1 - w) * y_mean

    # Per-subject prediction interval: LOO residual SD scaled by t(0.025, df)
    t_crit = float(stats.t.ppf(0.975, df=max(1, n - 1)))
    pi_lo = shrunk - t_crit * sigma_full
    pi_hi = shrunk + t_crit * sigma_full

    # Percentile within sample
    percentiles = pd.Series(shrunk).rank(pct=True).values * 100.0

    out_df = pd.DataFrame({
        "ID": sub["subject_id"].astype(str).values,
        f"raw_{target}": y,
        "loo_predicted_score": np.round(loo_full, 4),
        "shrunk_score": np.round(shrunk, 4),
        "percentile_in_sample": np.round(percentiles, 1),
        "pred_interval_lower": np.round(pi_lo, 4),
        "pred_interval_upper": np.round(pi_hi, 4),
        "confidence_flag": confidence_flag,
        "n_features_used": k_block,
    })
    fname = f"screening_index_{label_prefix}.csv"
    out_df.to_csv(out_dir / fname, index=False)

    meta = {
        "candidate_label": "Candidate EF screening index (illustrative).",
        "non_clinical_warning": (
            "Not a diagnostic. Not anchored to external clinical norms "
            "(BRIEF-2 / Conners-3 pending). Percentile is within-sample only. "
            "This index inherits the construct validity of its target."),
        "target": target,
        "covariates": cov_cols,
        "composites": comp_cols,
        "n_used": int(n),
        "shrinkage_weight": round(w, 4),
        "loo_r2_full": round(float(r2_full), 4),
        "loo_r2_age_only": round(float(r2_age), 4),
        "incremental_F": round(float(f_inc), 4) if np.isfinite(f_inc) else None,
        "incremental_p_value": round(float(p_inc), 4) if np.isfinite(p_inc) else None,
        "confidence_flag": confidence_flag,
        "incremental_significant_alpha_05": incremental_significant,
        "label_prefix": label_prefix,
    }
    meta_fname = f"screening_index_{label_prefix}_meta.json"
    with open(out_dir / meta_fname, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"    Screening index ({label_prefix}): "
          f"LOO R²={r2_full:.3f} (age-only {r2_age:.3f}) "
          f"incrF_p={p_inc} flag={confidence_flag}")
    return out_df


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def _load_per_target_directions(fe_dir, base_directions):
    """Per-target directional priors for union composites come from the
    apriori_theory_mapping.csv emitted by feature_engineering. Returns a
    dict {target: {composite: 'positive'|'negative'|'two_sided'}}.
    """
    out = {}
    mapping_path = fe_dir / "apriori_theory_mapping.csv" if fe_dir else None
    if mapping_path and mapping_path.exists():
        m = pd.read_csv(mapping_path)
        for _, r in m.iterrows():
            for col in m.columns:
                if col.startswith("dir_"):
                    target = col[len("dir_"):]
                    out.setdefault(target, {})[r["composite"]] = r[col]
    return out


def main():
    cfg = load_config()
    p = cfg["params"]

    fe_dir = latest_output(cfg["paths"]["feature_engineering_root"])
    if fe_dir is None:
        raise FileNotFoundError(
            f"No feature_engineering output under {cfg['paths']['feature_engineering_root']}. "
            f"Run `python feature_engineering/main.py` first.")
    print(f"Reading from: {fe_dir}")

    out_dir = make_output_dir()
    print(f"Output: {out_dir}")

    full = pd.read_csv(fe_dir / "full_dataset.csv")
    print(f"  full_dataset.csv: {full.shape}")

    # Per-target directional priors (union composites). Falls back to global
    # `directions:` block for composites not in the theory mapping.
    per_target_dirs = _load_per_target_directions(fe_dir, p.get("directions", {}))

    targets = list(p["targets"])
    covariates = list(p["covariates"])
    feature_sets = p["feature_sets"]
    base_directions = p.get("directions", {}) or {}
    components_map = p.get("composite_components", {}) or {}
    alpha_min = float(p["alpha_min_acceptable"])
    correction = str(p.get("correction", "bonferroni")).lower()
    alpha_level = float(p["alpha_level"])
    bootstrap_n = int(p["bootstrap_n"])

    # ─── Confirmatory hypothesis_test (existing OLS path) ─────────
    headline = []
    for target in targets:
        if target not in full.columns:
            print(f"\nSKIP target '{target}': not in full_dataset.csv")
            continue
        if full[target].isna().all():
            print(f"\nSKIP target '{target}': all NaN")
            continue
        target_dir = out_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 60)
        print(f"# TARGET: {target}")
        print("#" * 60)

        # Merge global directions with per-target (theory-mapping) directions
        dirs_for_target = dict(base_directions)
        dirs_for_target.update(per_target_dirs.get(target, {}))

        for fs_name, fs_cols in feature_sets.items():
            cols = [c for c in fs_cols if c in full.columns]
            if not cols:
                print(f"\n  SKIP feature_set '{fs_name}': no composite columns in data")
                continue
            try:
                summary = run_hypothesis_test(
                    full, target, fs_name, cols,
                    covariates, dirs_for_target, components_map,
                    alpha_min, correction, alpha_level, bootstrap_n,
                    cfg["random_state"], target_dir / fs_name)
                headline.append(summary)
            except Exception as e:
                print(f"  ERROR [{target}/{fs_name}]: {type(e).__name__}: {e}")
                headline.append({
                    "target": target, "feature_set": fs_name,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                })

    with open(out_dir / "headline.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)

    # ─── Exploratory ML grid (sensitivity track) ────────────────
    ml_cfg = p.get("exploratory_ml") or {}
    grid_df = None
    if ml_cfg.get("enable", False):
        schemes_path = fe_dir / "feature_selection_schemes_resolved.json"
        if not schemes_path.exists():
            print(f"\nSKIP exploratory_ml: missing {schemes_path}. "
                  f"Re-run feature_engineering to emit schemes.")
        else:
            with open(schemes_path, "r") as f:
                schemes_resolved = json.load(f)
            ml_out = out_dir / "exploratory_ml"
            ml_out.mkdir(parents=True, exist_ok=True)
            grid_df = run_exploratory_ml(
                full, targets, schemes_resolved,
                models_cfg=ml_cfg.get("models", {}),
                cv_cfg=ml_cfg.get("cv", {}),
                perm_cfg=ml_cfg.get("perm", {}),
                covariates=covariates,
                k_cap=int(ml_cfg.get("kbest_cap", 10)),
                out_dir=ml_out,
                random_state=cfg["random_state"],
            )

    # ─── Screening index (ONE pre-spec combo from confirmatory OLS) ─
    si_cfg = p.get("screening_index") or {}
    si_path = None
    if si_cfg.get("enable", False):
        si_target = si_cfg.get("target")
        si_fs = si_cfg.get("feature_set")
        si_cols = [c for c in feature_sets.get(si_fs, [])
                   if c in full.columns]
        if si_target in full.columns and si_cols:
            si_out = out_dir / "screening_index"
            si_out.mkdir(parents=True, exist_ok=True)
            si_df = compute_screening_index(
                full, si_target, covariates, si_cols, si_out,
                cfg["random_state"],
                label_prefix=si_cfg.get(
                    "label_prefix",
                    f"{si_target}__{si_fs}__OLS"),
            )
            if si_df is not None:
                si_path = str(si_out)
        else:
            print(f"\nSKIP screening_index: target={si_target}, "
                  f"fs={si_fs} missing cols")

    notes = {
        "stage": "analysis",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "feature_engineering_consumed": str(fe_dir),
        "targets": targets,
        "retracted_targets": p.get("retracted_targets", []),
        "covariates": covariates,
        "feature_sets_run": list(feature_sets.keys()),
        "n_hypothesis_test_combos": len(headline),
        "exploratory_ml_enabled": bool(ml_cfg.get("enable", False)),
        "exploratory_ml_n_combos": (int(grid_df.shape[0])
                                     if grid_df is not None else 0),
        "screening_index_enabled": bool(si_cfg.get("enable", False)),
        "screening_index_path": si_path,
        "outputs": ["<target>/<feature_set>/{composite_alpha.csv, coef_inference.csv,"
                    " block_f_test.json, bootstrap_ci.csv, summary.json}",
                    "headline.json",
                    "exploratory_ml/exploratory_ml_grid.csv",
                    "exploratory_ml/exploratory_ml_summary.json",
                    "screening_index/screening_index_<label>.csv",
                    "screening_index/screening_index_<label>_meta.json"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
