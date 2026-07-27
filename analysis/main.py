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
import os
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

# Pin native math libraries to a single thread BEFORE numpy/scipy/sklearn load.
# The affective decoder does many tiny repeated MLP fits + predict_proba + per-
# subject precursor refits; nested BLAS/OpenMP threadpools collide and segfault
# natively on Windows (no Python traceback). threadpoolctl scoping was not
# sufficient — the limit must precede the numpy import. Impact on the rest of
# the stage is negligible (OLS bootstrap is BLAS-light; the ML grid is
# permutation-bound, not BLAS-bound).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
# Univariate scan (one predictor + covariates at a time)
# ──────────────────────────────────────────────────────────────────

def run_univariate_scan(full_df, target, feature_set_name, feature_cols,
                        covariates, directions, correction, alpha_level,
                        out_dir):
    """For ONE (target, feature_set): fit y ~ covariates + <one feature> for
    each feature separately, collect into a single table.

    This is the per-feature companion to the block OLS in run_hypothesis_test.
    Each feature is tested in its own model (and on its own complete-case N,
    so a feature with fewer NaNs is not penalised by listwise deletion across
    the whole block). Reported per feature:
      beta, se, t, p_two_sided, p_one_sided (if a direction prior exists),
      partial_r of the feature net of covariates, delta_r2 / block-F over the
      covariate-only model.
    Multiplicity correction is applied ACROSS the features in this scan and is
    EXPLORATORY in scope (flagged as such) — distinct from the confirmatory
    within-block correction in run_hypothesis_test.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cov = [c for c in covariates if c in full_df.columns]

    rows, pvals_two = [], []
    for feat in feature_cols:
        if feat not in full_df.columns:
            rows.append({"feature": feat, "note": "column absent"})
            pvals_two.append(np.nan)
            continue
        needed = list(dict.fromkeys(cov + [feat, target]))
        df = full_df[needed].apply(pd.to_numeric, errors="coerce").dropna()
        n_used = len(df)
        if n_used < len(cov) + 1 + 3:
            rows.append({"feature": feat, "n_used": int(n_used),
                         "note": "insufficient N"})
            pvals_two.append(np.nan)
            continue

        restricted = fit_ols(df, target, cov) if cov else None
        full_m = fit_ols(df, target, cov + [feat])

        beta = float(full_m.params.get(feat, np.nan))
        se = float(full_m.bse.get(feat, np.nan))
        t = float(full_m.tvalues.get(feat, np.nan))
        p2 = float(full_m.pvalues.get(feat, np.nan))
        df_resid = int(full_m.df_resid)
        partial_r = (float(np.sign(t) * np.sqrt(t ** 2 / (t ** 2 + df_resid)))
                     if np.isfinite(t) and df_resid > 0 else float("nan"))

        r2_f = float(full_m.rsquared)
        r2_r = float(restricted.rsquared) if restricted is not None else 0.0
        delta_r2 = r2_f - r2_r
        if df_resid > 0 and (1.0 - r2_f) > 0:
            f_stat = (delta_r2 / 1.0) / ((1.0 - r2_f) / df_resid)
            p_block = float(1.0 - stats.f.cdf(f_stat, 1, df_resid))
        else:
            f_stat, p_block = float("nan"), float("nan")

        d = directions.get(feat)
        if d in ("positive", "negative") and np.isfinite(beta) and np.isfinite(p2):
            matches = (d == "positive" and beta > 0) or (d == "negative" and beta < 0)
            p1 = p2 / 2.0 if matches else 1.0 - p2 / 2.0
        else:
            p1, d = p2, "two_sided"

        rows.append({
            "feature": feat, "direction": d, "n_used": int(n_used),
            "beta": round(beta, 4), "se": round(se, 4), "t": round(t, 3),
            "partial_r": round(partial_r, 4) if np.isfinite(partial_r) else None,
            "delta_r2_over_cov": round(delta_r2, 4),
            "block_F_1df": round(float(f_stat), 4) if np.isfinite(f_stat) else None,
            "p_two_sided": round(p2, 4),
            "p_one_sided": round(p1, 4),
            "p_block_F": round(p_block, 4) if np.isfinite(p_block) else None,
            "note": "",
        })
        pvals_two.append(p2)

    # Exploratory multiplicity correction across the features in this scan
    valid = [i for i, p in enumerate(pvals_two) if np.isfinite(p)]
    p_corr = np.full(len(pvals_two), np.nan)
    if valid:
        corr_vals = apply_correction([pvals_two[i] for i in valid],
                                     correction, len(valid), alpha_level)
        for i, v in zip(valid, corr_vals):
            p_corr[i] = v
    for r, pc in zip(rows, p_corr):
        r["p_corrected"] = round(float(pc), 4) if np.isfinite(pc) else None
        r["correction"] = f"{correction} (exploratory, across scan)"
        r["significant"] = bool(np.isfinite(pc) and pc < alpha_level)

    scan_df = pd.DataFrame(rows)
    fname = f"univariate_{target}__{feature_set_name}.csv"
    scan_df.to_csv(out_dir / fname, index=False)

    print(f"\n[Univariate scan: {target} ~ age + <one feature> | "
          f"set={feature_set_name}]")
    print(f"  correction={correction} (exploratory) | alpha={alpha_level}")
    for r in rows:
        if r.get("note"):
            print(f"    {r['feature']:14s} SKIP ({r['note']})")
            continue
        flag = "  *" if r.get("significant") else ""
        print(f"    {r['feature']:14s} N={r['n_used']} "
              f"beta={r['beta']:+.4f} t={r['t']:+.3f} "
              f"partial_r={r['partial_r']} dR2={r['delta_r2_over_cov']:+.4f} "
              f"p={r['p_two_sided']} p_corr={r['p_corrected']}{flag}")
    return scan_df


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
# EXPLORATORY: ASM affective decoder (Wahab lineage; decode_IAPS.pdf)
# ──────────────────────────────────────────────────────────────────
# Within-subject 4-emotion window classifier. Step 2 of the affective-decoder
# branch: it produces the per-subject decode accuracy + confusion matrix and
# (later, step 3) the affective-space geometry features. NOT cross-subject.

def _load_decode_epochs(preproc_dir, emotion_conditions):
    """Load all <sid>_<emotion>_decode-epo.fif under <preproc_dir>/cleaned_epochs/.

    Returns {(sid, emotion): Epochs}. These are the full-block 2 s windows
    preprocessing emits when params.iaps.decode.enable is true.
    """
    import mne
    epoch_dir = preproc_dir / "cleaned_epochs"
    if not epoch_dir.is_dir():
        return {}
    conds = sorted(set(emotion_conditions), key=len, reverse=True)
    out = {}
    for f in sorted(epoch_dir.glob("*_decode-epo.fif")):
        base = f.stem.removesuffix("-epo")          # {sid}_{cond}_decode
        if not base.endswith("_decode"):
            continue
        core = base[:-len("_decode")]               # {sid}_{cond}
        sid = cond = None
        for c in conds:
            if core.endswith(f"_{c}"):
                cond = c; sid = core[:-(len(c) + 1)]; break
        if cond is None:
            print(f"  WARN: skip unmatched decode file {f.name}")
            continue
        out[(sid, cond)] = mne.read_epochs(str(f), verbose=False)
    return out


def _decode_window_features(epochs, bands, kde_points, families):
    """One feature row per 2 s window. Families:
      band_power : abs + rel band power per channel (Welch + np.trapezoid),
                   matching the feature_building primitive (computed per WINDOW
                   here, not averaged over epochs).
      kde        : mean + SD of a Gaussian KDE evaluated on `kde_points` over
                   each channel's amplitude range (the paper's feature).
    """
    from scipy import signal as sig
    from scipy.stats import gaussian_kde
    data = epochs.get_data()              # (n_win, n_ch, n_times)
    sfreq = float(epochs.info["sfreq"])
    ch_names = list(epochs.ch_names)
    use_bp = "band_power" in families
    use_kde = "kde" in families
    nperseg = min(data.shape[2], int(sfreq))
    rows = []
    for w in range(data.shape[0]):
        feat = {}
        for ci, ch in enumerate(ch_names):
            x = data[w, ci]
            if use_bp:
                f, pxx = sig.welch(x, fs=sfreq, nperseg=nperseg)
                total = float(np.trapezoid(pxx, f))
                for band, (lo, hi) in bands.items():
                    m = (f >= lo) & (f < hi)
                    bp = float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0
                    feat[f"bp_abs_{band}_{ch}"] = bp
                    feat[f"bp_rel_{band}_{ch}"] = bp / total if total > 0 else 0.0
            if use_kde:
                try:
                    kde = gaussian_kde(x)
                    grid = np.linspace(float(x.min()), float(x.max()), kde_points)
                    dens = kde(grid)
                    feat[f"kde_mean_{ch}"] = float(np.mean(dens))
                    feat[f"kde_sd_{ch}"] = float(np.std(dens))
                except Exception:
                    feat[f"kde_mean_{ch}"] = float("nan")
                    feat[f"kde_sd_{ch}"] = float("nan")
        rows.append(feat)
    return pd.DataFrame(rows)


# Russell circumplex canonical coordinates (V = valence, A = arousal) on the
# unit circle. Each basic emotion sits in one quadrant — this is the fixed
# "affective model basis" of the paper. A window's V-A point is the posterior-
# weighted average of these anchors, so the affective space is cross-subject
# comparable (it does NOT depend on a per-subject decode threshold).
ASM_EMO_COORD = {
    "1_Happy": (0.7071, 0.7071),    # +valence, +arousal
    "2_Calm":  (0.7071, -0.7071),   # +valence, -arousal
    "3_Sad":   (-0.7071, -0.7071),  # -valence, -arousal
    "4_Scare": (-0.7071, 0.7071),   # -valence, +arousal (Fear)
}


def _va_geometry(va, y, classes):
    """Per-subject affective-space geometry from window V-A coordinates.

    va      : (n_win, 2) array of (valence, arousal) per window
    y       : (n_win,) true emotion label per window
    classes : ordered emotion labels present

    Returns a dict of cross-subject-comparable scalars:
      asm_centroid_V/A      mean window position
      asm_centroid_offset   |centroid - origin(0,0)|  (the paper's centroid shift)
      asm_sep_centroid      mean pairwise distance between per-emotion centroids
      asm_silhouette        silhouette of the emotion clusters in V-A space
      asm_dispersion        mean within-emotion distance to the emotion centroid
    """
    import itertools
    from sklearn.metrics import silhouette_score
    centroid = va.mean(axis=0)
    offset = float(np.hypot(centroid[0], centroid[1]))
    cents = {c: va[y == c].mean(axis=0) for c in classes}
    if len(classes) >= 2:
        sep = float(np.mean([np.hypot(*(cents[a] - cents[b]))
                             for a, b in itertools.combinations(classes, 2)]))
    else:
        sep = float("nan")
    try:
        sil = float(silhouette_score(va, y)) if (len(classes) >= 2 and
                                                 len(va) > len(classes)) else float("nan")
    except Exception:
        sil = float("nan")
    disp = float(np.mean([np.hypot(*(va[i] - cents[y[i]]))
                          for i in range(len(y))]))
    return {
        "asm_centroid_V": round(float(centroid[0]), 5),
        "asm_centroid_A": round(float(centroid[1]), 5),
        "asm_centroid_offset": round(offset, 5),
        "asm_sep_centroid": round(sep, 5) if np.isfinite(sep) else None,
        "asm_silhouette": round(sil, 5) if np.isfinite(sil) else None,
        "asm_dispersion": round(disp, 5),
    }


def run_affective_decoder(preproc_dir, cfg_block, out_dir, random_state):
    """Step 2-3: per-subject within-subject 4-emotion window decoder + the
    affective-space geometry features derived from its out-of-fold posteriors.

    For each subject: build a window-level feature matrix across the emotion
    blocks, train MLP (primary, per paper) + LDA (baseline) with StratifiedKFold
    WITHIN that subject, report mean fold accuracy + an aggregated confusion
    matrix. Prints per subject; writes decoder_accuracy.csv,
    decoder_confusion_matrices.csv, decoder_meta.json.

    GUARDRAILS: within-subject only (windows of one subject share train/test by
    design — that is the point); accuracy is NOT a headline; everything labelled
    exploratory; SAM validation skipped when no SAM columns exist.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, confusion_matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    emotions = list(cfg_block.get("emotion_conditions", []))
    bands = {k: tuple(v) for k, v in (cfg_block.get("bands", {}) or {}).items()}
    kde_points = int(cfg_block.get("kde_points", 100))
    families = list(cfg_block.get("features", ["band_power", "kde"]))
    n_folds = int(cfg_block.get("cv_folds", 5))
    mlp_hidden = tuple(cfg_block.get("mlp_hidden", [32]))
    mlp_max_iter = int(cfg_block.get("mlp_max_iter", 500))
    min_per_class = int(cfg_block.get("min_windows_per_class", 5))
    geo_cfg = cfg_block.get("geometry", {}) or {}
    geo_enable = bool(geo_cfg.get("enable", False))
    precursor_ec = bool(geo_cfg.get("precursor_from_ec", False))

    epdict = _load_decode_epochs(preproc_dir, emotions)
    if not epdict:
        print("\n[Affective decoder] no *_decode-epo.fif found — re-run "
              "preprocessing with params.iaps.decode.enable=true. SKIP.")
        return None, None

    by_subject = {}
    for (sid, cond), ep in epdict.items():
        by_subject.setdefault(sid, {})[cond] = ep

    print("\n" + "#" * 60)
    print("# EXPLORATORY: ASM affective decoder (within-subject)")
    print("#" * 60)
    print("  WITHIN-SUBJECT window classification — NOT cross-subject "
          "generalization.")
    print(f"  Classes: {emotions} (chance = {100.0 / max(1, len(emotions)):.0f}%)")
    print(f"  Features: {families} | CV: {n_folds}-fold StratifiedKFold")
    print("  Decode accuracy is NOT a headline result (exploratory substrate "
          "for step 3 geometry features).")

    def _build_models():
        return {
            "MLP": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=mlp_hidden,
                                      max_iter=mlp_max_iter,
                                      random_state=random_state)),
            ]),
            "LDA": Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis()),
            ]),
        }

    # Preload resting Eyes_Closed epochs UP FRONT (not inside the fitting loop).
    # Interleaving mne.read_epochs file I/O with sklearn fits across subjects was
    # the source of a nondeterministic native crash on Windows; reading all fiff
    # files before any model is fit keeps the loop pure numpy/sklearn.
    ec_dict = {}
    if geo_enable and precursor_ec:
        import mne
        for f in (preproc_dir / "cleaned_epochs").glob("*_Eyes_Closed-epo.fif"):
            sid_ec = f.stem.removesuffix("-epo")[:-len("_Eyes_Closed")]
            try:
                ec_dict[sid_ec] = mne.read_epochs(str(f), verbose=False)
            except Exception:
                pass

    # Pin BLAS/OpenMP to a single thread for the duration of the per-subject
    # fitting loop. Repeated small MLP fits + predict_proba + per-subject
    # precursor refits segfault natively on Windows when nested threadpools
    # collide; single-threaded is reliable and these fits are tiny. Restored
    # after the loop so the exploratory ML grid keeps full parallelism.
    _tpl = None
    try:
        from threadpoolctl import threadpool_limits
        _tpl = threadpool_limits(limits=1)
    except Exception:
        pass

    summary_rows, confusions, geom_rows = [], {}, []
    for sid in sorted(by_subject):
        emo_present = [c for c in emotions if c in by_subject[sid]]
        frames, labels = [], []
        for c in emo_present:
            fdf = _decode_window_features(by_subject[sid][c], bands,
                                          kde_points, families)
            frames.append(fdf); labels += [c] * len(fdf)
        if not frames:
            continue
        X = pd.concat(frames, ignore_index=True)
        y = np.array(labels)
        vc = pd.Series(y).value_counts()
        keep = vc[vc >= min_per_class].index.tolist()
        mask = np.isin(y, keep)
        X = X.loc[mask].reset_index(drop=True); y = y[mask]
        n_classes = len(set(y))
        if n_classes < 2:
            print(f"\n  [{sid}] SKIP: <2 emotion classes with >= "
                  f"{min_per_class} windows")
            summary_rows.append({"subject_id": sid,
                                 "status": "skip_insufficient_classes",
                                 "n_classes": int(n_classes)})
            continue
        min_count = int(pd.Series(y).value_counts().min())
        folds = max(2, min(n_folds, min_count))
        skf = StratifiedKFold(n_splits=folds, shuffle=True,
                              random_state=random_state)
        labels_sorted = sorted(set(y))
        row = {"subject_id": sid, "status": "ok", "n_windows": int(len(y)),
               "n_classes": int(n_classes), "classes": ",".join(labels_sorted),
               "cv_folds": int(folds), "chance": round(1.0 / n_classes, 4)}
        # OOF MLP posteriors (collected from the same fits used for accuracy) ->
        # affective-space coordinates. Out-of-fold so a window's own coordinate
        # is not produced by a model that trained on it.
        oof_proba = np.full((len(y), len(labels_sorted)), np.nan)
        for mname, pipe in _build_models().items():
            accs, yt_all, yp_all = [], [], []
            for tr, te in skf.split(X, y):
                pipe.fit(X.iloc[tr], y[tr])
                yp = pipe.predict(X.iloc[te])
                accs.append(accuracy_score(y[te], yp))
                yt_all.extend(y[te]); yp_all.extend(yp)
                if mname == "MLP":
                    pr = pipe.predict_proba(X.iloc[te])
                    cls = list(pipe.named_steps["clf"].classes_)
                    idx = [cls.index(c) for c in labels_sorted]
                    oof_proba[te] = pr[:, idx]
            row[f"{mname}_acc_mean"] = round(float(np.mean(accs)), 4)
            row[f"{mname}_acc_sd"] = round(float(np.std(accs)), 4)
            cm = confusion_matrix(yt_all, yp_all, labels=labels_sorted)
            confusions[f"{sid}|{mname}"] = (labels_sorted, cm)
        summary_rows.append(row)

        print(f"\n  [{sid}] N_win={row['n_windows']} classes=[{row['classes']}] "
              f"folds={folds} chance={row['chance']:.2f}")
        for mname in ("MLP", "LDA"):
            print(f"    {mname}: acc={row[f'{mname}_acc_mean']:.3f} "
                  f"(SD {row[f'{mname}_acc_sd']:.3f})")
        labs, cm = confusions[f"{sid}|MLP"]
        print(f"    MLP confusion (rows=true, cols=pred) "
              f"[{', '.join(labs)}]:")
        for i, lab in enumerate(labs):
            print(f"      {lab:10s} {cm[i].tolist()}")

        # ── Step 3: affective-space geometry from OOF posteriors ──
        if geo_enable:
            Vc = np.array([ASM_EMO_COORD[c][0] for c in labels_sorted])
            Ac = np.array([ASM_EMO_COORD[c][1] for c in labels_sorted])
            va = np.column_stack([oof_proba @ Vc, oof_proba @ Ac])
            grow = {"subject_id": sid, "n_windows": int(len(y)),
                    "n_classes": int(n_classes)}
            grow.update(_va_geometry(va, y, labels_sorted))

            # Per-emotion decoded V-A centroid (decode_IAPS.pdf Fig 8: each
            # emotion's position in Russell's affective space). 4 emotions x
            # (valence, arousal) = 8 features asm_va_{H,C,S,F}{v,a}. This is the
            # IAPS-decode feature set fed to the step-4 OLS, DISTINCT from the 9
            # aggregate geometry scalars above. A subject missing an emotion
            # leaves those two columns NaN (dropped row-wise in the OLS).
            emo_short = {"1_Happy": "H", "2_Calm": "C",
                         "3_Sad": "S", "4_Scare": "F"}
            for c in labels_sorted:
                cen_c = va[y == c].mean(axis=0)
                sh = emo_short.get(c, c)
                grow[f"asm_va_{sh}v"] = round(float(cen_c[0]), 5)
                grow[f"asm_va_{sh}a"] = round(float(cen_c[1]), 5)

            # Optional resting-EC "precursor" centroid: project EC windows
            # through a classifier trained on ALL of this subject's emotion
            # windows, then take the EC centroid offset in the same V-A space.
            if precursor_ec and sid in ec_dict:
                try:
                    Xec = _decode_window_features(ec_dict[sid], bands,
                                                  kde_points, families)
                    clf = _build_models()["MLP"]
                    clf.fit(X, y)
                    cls = list(clf.named_steps["clf"].classes_)
                    idx = [cls.index(c) for c in labels_sorted]
                    pr_ec = clf.predict_proba(Xec)[:, idx]
                    va_ec = np.column_stack([pr_ec @ Vc, pr_ec @ Ac])
                    cen = va_ec.mean(axis=0)
                    grow["asm_precursor_V"] = round(float(cen[0]), 5)
                    grow["asm_precursor_A"] = round(float(cen[1]), 5)
                    grow["asm_precursor_offset"] = round(
                        float(np.hypot(cen[0], cen[1])), 5)
                except Exception as e:
                    grow["asm_precursor_offset"] = None
                    print(f"    (precursor EC skipped: {type(e).__name__})")
            geom_rows.append(grow)
            print(f"    geometry: centroid_offset="
                  f"{grow['asm_centroid_offset']} sep={grow['asm_sep_centroid']} "
                  f"silhouette={grow['asm_silhouette']} "
                  f"dispersion={grow['asm_dispersion']}"
                  + (f" precursor_offset={grow.get('asm_precursor_offset')}"
                     if precursor_ec else ""))

    if _tpl is not None:
        _tpl.unregister()          # restore original thread limits

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out_dir / "decoder_accuracy.csv", index=False)
    geom_df = pd.DataFrame(geom_rows) if geom_rows else None
    if geom_df is not None:
        geom_df.to_csv(out_dir / "affective_geometry.csv", index=False)
    crows = []
    for key, (labs, cm) in confusions.items():
        sid, mname = key.split("|", 1)
        for i, tl in enumerate(labs):
            for j, pl in enumerate(labs):
                crows.append({"subject_id": sid, "model": mname,
                              "true": tl, "pred": pl, "count": int(cm[i, j])})
    pd.DataFrame(crows).to_csv(out_dir / "decoder_confusion_matrices.csv",
                              index=False)

    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    meta = {
        "exploratory": True,
        "method": "ASM affective decoder (Wahab lineage; decode_IAPS.pdf)",
        "interpretation_guardrail": (
            "WITHIN-SUBJECT window classification, NOT cross-subject "
            "generalization. Windows from the same subject appear in both train "
            "and test folds by design. Decode accuracy is the substrate for the "
            "affective-space geometry features (step 3), NOT a headline result."),
        "sam_validation": ("skipped — no SAM / valence / arousal columns in the "
                           "behavioral export"),
        "emotion_conditions": emotions,
        "feature_families": families,
        "bands": {k: list(v) for k, v in bands.items()},
        "kde_points": kde_points,
        "cv_folds_requested": n_folds,
        "n_subjects_decoded": len(ok_rows),
        "models": ["MLP", "LDA"],
        "chance_level": round(1.0 / max(1, len(emotions)), 4),
        "geometry_enabled": geo_enable,
        "geometry_oof_posteriors": True,
        "geometry_precursor_from_ec": precursor_ec,
        "asm_canonical_coords": {k: list(v) for k, v in ASM_EMO_COORD.items()},
        "n_subjects_with_geometry": (int(geom_df.shape[0])
                                     if geom_df is not None else 0),
    }
    with open(out_dir / "decoder_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if ok_rows:
        for mname in ("MLP", "LDA"):
            accs = [r[f"{mname}_acc_mean"] for r in ok_rows
                    if f"{mname}_acc_mean" in r]
            if accs:
                print(f"\n  [exploratory] mean within-subject {mname} acc across "
                      f"{len(accs)} subjects = {np.mean(accs):.3f} "
                      f"(chance {meta['chance_level']:.2f})")
    print(f"  Decoder outputs: {out_dir}")
    return sdf, geom_df


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


def plot_ols_scatter(full, targets, covariates, feature_cols, out_dir,
                     set_label="a_priori_union"):
    """One observed-vs-fitted scatter for the confirmatory OLS, all targets
    overlaid. Because targets live on different scales, observed and fitted
    are z-scored within each target (using the observed mean/SD), so a point
    on the y = x line is a perfectly fitted subject regardless of target.
    Each target's legend entry carries its in-sample R^2 and its leave-one-out
    R^2, so the in-sample fit and the out-of-sample collapse are both visible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SKIP ols_scatter: matplotlib not installed")
        return None

    def _fit_predict(X_tr, y_tr, X_te):
        A = np.column_stack([np.ones(len(X_tr)), X_tr])
        beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
        return np.column_stack([np.ones(len(X_te)), X_te]) @ beta

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    plotted = 0
    for i, target in enumerate(targets):
        cols = [c for c in feature_cols if c in full.columns]
        if target not in full.columns or not cols:
            continue
        d = full[covariates + cols + [target]].apply(
            pd.to_numeric, errors="coerce").dropna()
        if len(d) < len(covariates) + len(cols) + 3:
            continue
        y = d[target].to_numpy(float)
        X = d[covariates + cols].to_numpy(float)
        n = len(d)
        fitted = _fit_predict(X, y, X)
        sst = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - np.sum((y - fitted) ** 2) / sst if sst > 0 else float("nan")
        loo = np.array([
            _fit_predict(X[np.arange(n) != j], y[np.arange(n) != j],
                         X[j:j + 1])[0]
            for j in range(n)])
        loo_r2 = 1.0 - np.sum((y - loo) ** 2) / sst if sst > 0 else float("nan")
        mu, sd = y.mean(), y.std(ddof=1)
        if sd <= 0:
            continue
        ax.scatter((y - mu) / sd, (fitted - mu) / sd, s=45, alpha=0.8,
                   color=colors[i % len(colors)], edgecolor="white",
                   linewidth=0.6,
                   label=f"{target}  (in-sample R²={r2:.2f}, "
                         f"LOO R²={loo_r2:.2f})")
        plotted += 1
    if not plotted:
        plt.close(fig)
        print("  SKIP ols_scatter: no plottable targets")
        return None

    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6,
            label="y = x (perfect fit)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Observed value (z-scored within target)")
    ax.set_ylabel("OLS fitted value, full model (z-scored within target)")
    ax.set_title(f"Confirmatory OLS: observed vs fitted\n"
                 f"(age + {len(feature_cols)} {set_label} predictors)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    png = out_dir / f"ols_scatter_{set_label}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"  OLS scatter saved: {png}")
    return str(png)


def compare_iaps_baselines(out_dir, targets, primary_set, sensitivity_set):
    """Baseline-robustness check for the IAPS features.

    Reads the per-target coef_inference.csv that the OLS loop just wrote for the
    primary (within-file baseline) and sensitivity (Eyes_Open baseline) IAPS
    feature sets, aligns the 8 emotion features by emotion+measure (Hv, Ha, ...),
    and reports whether their importance RANKING and effect DIRECTION agree:
      - spearman_abs_t : Spearman rank corr of |t| across the 8 features
      - sign_agreement : fraction of features with matching beta sign
      - beta_pearson_r : Pearson corr of the betas (magnitude + direction)
    "consistent" if spearman_abs_t >= 0.5 AND sign_agreement >= 0.75.
    """
    def _key(c):
        return str(c).replace("iaps_eo_", "").replace("iaps_", "")

    rows, summary = [], []
    for t in targets:
        p_csv = out_dir / t / primary_set / "coef_inference.csv"
        s_csv = out_dir / t / sensitivity_set / "coef_inference.csv"
        if not (p_csv.exists() and s_csv.exists()):
            continue
        dp = pd.read_csv(p_csv); ds = pd.read_csv(s_csv)
        dp["key"] = dp["composite"].map(_key); ds["key"] = ds["composite"].map(_key)
        m = dp.merge(ds, on="key", suffixes=("_within", "_eo"))
        if len(m) < 2:
            continue
        for _, r in m.iterrows():
            rows.append({
                "target": t, "feature": r["key"],
                "beta_within": r["beta_within"], "t_within": r["t_within"],
                "beta_eo": r["beta_eo"], "t_eo": r["t_eo"],
                "sign_match": bool(np.sign(r["beta_within"]) == np.sign(r["beta_eo"])),
            })
        aw, ae = m["t_within"].abs(), m["t_eo"].abs()
        if len(m) >= 3 and aw.std() > 0 and ae.std() > 0:
            rho, pval = stats.spearmanr(aw, ae)
        else:
            rho, pval = float("nan"), float("nan")
        sign_agree = float(np.mean(
            np.sign(m["beta_within"]) == np.sign(m["beta_eo"])))
        beta_r = (float(np.corrcoef(m["beta_within"], m["beta_eo"])[0, 1])
                  if len(m) >= 3 else float("nan"))
        summary.append({
            "target": t, "n_features": int(len(m)),
            "spearman_abs_t": round(float(rho), 4) if np.isfinite(rho) else None,
            "spearman_abs_t_p": round(float(pval), 4) if np.isfinite(pval) else None,
            "sign_agreement": round(sign_agree, 4),
            "beta_pearson_r": round(beta_r, 4) if np.isfinite(beta_r) else None,
            "consistent": bool(np.isfinite(rho) and rho >= 0.5
                               and sign_agree >= 0.75),
        })

    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / "iaps_baseline_consistency.csv", index=False)
    with open(out_dir / "iaps_baseline_consistency.json", "w") as f:
        json.dump({"primary_set": primary_set,
                   "sensitivity_set": sensitivity_set,
                   "per_target": summary}, f, indent=2, default=str)
    print("\n[IAPS baseline consistency: within (primary) vs EO (sensitivity)]")
    if not summary:
        print("  (no target had both IAPS feature sets fit — nothing to compare)")
    for s in summary:
        print(f"  {s['target']:12s} n={s['n_features']} "
              f"spearman|t|={s['spearman_abs_t']} "
              f"sign_agree={s['sign_agreement']} beta_r={s['beta_pearson_r']} "
              f"-> {'CONSISTENT' if s['consistent'] else 'divergent'}")
    return summary


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
    # EF-vs-IAPS toggle: restrict the OLS loop to named feature sets.
    fs_filter = p.get("feature_sets_to_run")
    if fs_filter:
        unknown = [fs for fs in fs_filter if fs not in feature_sets]
        if unknown:
            print(f"WARN: feature_sets_to_run names unknown sets {unknown}")
        feature_sets = {k: v for k, v in feature_sets.items() if k in fs_filter}
    print(f"Feature sets to run: {list(feature_sets.keys())}")
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
            # Validate coverage before fitting. IAPS columns are NaN for
            # subjects without a viable emotion window; this guards against an
            # EF and an IAPS run silently collapsing to different samples.
            cov = full[[target] + cols].apply(
                pd.to_numeric, errors="coerce").dropna()
            need = len(cols) + len(covariates) + 3
            if len(cov) < need:
                print(f"\n  SKIP [{target}/{fs_name}]: only {len(cov)} complete "
                      f"rows for {len(cols)} predictors + {len(covariates)} cov "
                      f"(need >= {need}).")
                headline.append({"target": target, "feature_set": fs_name,
                                 "skipped": "insufficient_complete_rows",
                                 "n_complete": int(len(cov))})
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

    # ─── IAPS baseline-robustness: within (primary) vs EO (sensitivity) ─
    ibc_cfg = p.get("iaps_baseline_consistency") or {}
    if ibc_cfg.get("enable", False):
        prim = ibc_cfg.get("primary_set", "iaps_va")
        sens = ibc_cfg.get("sensitivity_set", "iaps_va_eo")
        if prim in feature_sets and sens in feature_sets:
            compare_iaps_baselines(out_dir, targets, prim, sens)
        else:
            print(f"\nSKIP iaps_baseline_consistency: both '{prim}' and '{sens}' "
                  f"must be in feature_sets_to_run.")

    # ─── Univariate scan (one predictor + covariates at a time) ──
    uni_cfg = p.get("univariate_scan") or {}
    if uni_cfg.get("enable", False):
        uni_targets = uni_cfg.get("targets") or targets
        all_feature_sets = p["feature_sets"]   # unfiltered dict (not feature_sets_to_run)
        uni_sets = uni_cfg.get("feature_sets") or list(feature_sets.keys())
        uni_corr = str(uni_cfg.get("correction", correction)).lower()
        uni_out = out_dir / "univariate"
        for ut in uni_targets:
            if ut not in full.columns or full[ut].isna().all():
                print(f"\nSKIP univariate scan target '{ut}': missing/all NaN")
                continue
            dirs_for_ut = dict(base_directions)
            dirs_for_ut.update(per_target_dirs.get(ut, {}))
            for us in uni_sets:
                us_cols = [c for c in all_feature_sets.get(us, []) if c in full.columns]
                if not us_cols:
                    print(f"\nSKIP univariate scan [{ut}/{us}]: no columns")
                    continue
                run_univariate_scan(full, ut, us, us_cols, covariates,
                                    dirs_for_ut, uni_corr, alpha_level, uni_out)

    # ─── OLS observed-vs-fitted scatter (one PNG per feature set) ─
    for fs_name, fs_cols in feature_sets.items():
        plot_ols_scatter(full, targets, covariates,
                         [c for c in fs_cols if c in full.columns], out_dir,
                         set_label=fs_name)

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

    # ─── EXPLORATORY: ASM affective decoder (within-subject) ─────
    ad_cfg = p.get("affective_decoder") or {}
    ad_run = False
    ad_ef = False
    if ad_cfg.get("enable", False):
        preproc_dir = latest_output(cfg["paths"].get("preprocessing_root",
                                                      "./preprocessing/output"))
        if preproc_dir is None:
            print("\nSKIP affective_decoder: no preprocessing output found.")
        else:
            print(f"\nAffective decoder reading epochs from: {preproc_dir}")
            ad_out = out_dir / "affective_decoder"
            sdf, geom_df = run_affective_decoder(preproc_dir, ad_cfg, ad_out,
                                                 cfg["random_state"])
            ad_run = sdf is not None

            # ─── Step 4: predict Global_EF from the affective-space geometry
            # THROUGH the existing confirmatory OLS path (run_hypothesis_test:
            # age-as-covariate FWL residualization + block F + bootstrap CI).
            # Unit of analysis = SUBJECT (one geometry vector each), so window
            # train/test leakage is structurally impossible here. EXPLORATORY.
            ef_cfg = ad_cfg.get("predict_ef", {}) or {}
            if ef_cfg.get("enable", False) and geom_df is not None:
                ef_target = ef_cfg.get("target", "Global_EF")
                all_asm = [c for c in geom_df.columns
                           if c.startswith("asm_") and geom_df[c].notna().any()]
                # Two distinct decode-derived feature sets, run as SEPARATE OLS
                # (never pooled — 17 features at n~25 would be hopeless):
                #   asm_va       = 8 per-emotion V-A centroids (decode_IAPS Fig 8)
                #   asm_geometry = the aggregate affective-space geometry scalars
                set_cols = {
                    "asm_va": [c for c in all_asm if c.startswith("asm_va_")],
                    "asm_geometry": [c for c in all_asm
                                     if not c.startswith("asm_va_")],
                }
                sets_to_run = ef_cfg.get("feature_sets",
                                         ["asm_va", "asm_geometry"])
                merged = full.merge(geom_df[["subject_id"] + all_asm],
                                    on="subject_id", how="left")
                if ef_target not in merged.columns:
                    print(f"\nSKIP affective EF prediction: target "
                          f"'{ef_target}' not in dataset.")
                else:
                    dirs_ef = dict(base_directions)
                    dirs_ef.update(per_target_dirs.get(ef_target, {}))
                    for set_label in sets_to_run:
                        cols = set_cols.get(set_label, [])
                        print("\n" + "#" * 60)
                        print(f"# EXPLORATORY step 4: {ef_target} ~ age + "
                              f"{set_label} ({len(cols)} feat)")
                        print("#" * 60)
                        if not cols:
                            print(f"  SKIP: no columns for '{set_label}'.")
                            continue
                        set_out = ad_out / "predict_ef" / set_label

                        # One-by-one simple regression (y ~ age + ONE feature):
                        # per-feature complete-case N, exploratory FDR/Bonf across
                        # the scan. Runs even when the block OLS is underpowered.
                        if ef_cfg.get("univariate", True):
                            try:
                                run_univariate_scan(
                                    merged, ef_target, set_label, cols,
                                    covariates, dirs_ef, correction,
                                    alpha_level, set_out / "univariate")
                            except Exception as e:
                                print(f"  univariate ERROR: "
                                      f"{type(e).__name__}: {e}")

                        # Block OLS (all features together) — needs more N.
                        cov = full[[ef_target]].join(merged[cols])
                        n_ok = cov.apply(pd.to_numeric,
                                         errors="coerce").dropna().shape[0]
                        need = len(cols) + len(covariates) + 3
                        if n_ok < need:
                            print(f"  block SKIP: only {n_ok} complete rows for "
                                  f"{len(cols)} features + {len(covariates)} "
                                  f"cov (need >= {need}).")
                            continue
                        try:
                            run_hypothesis_test(
                                merged, ef_target, set_label, cols,
                                covariates, dirs_ef, components_map,
                                alpha_min, correction, alpha_level, bootstrap_n,
                                cfg["random_state"],
                                set_out / ef_target)
                            ad_ef = True
                            print("  (EXPLORATORY — labelled non-confirmatory; "
                                  "decode-derived features are not in the "
                                  "pre-registered a priori union.)")
                        except Exception as e:
                            print(f"  ERROR: {type(e).__name__}: {e}")

    notes = {
        "stage": "analysis",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "feature_engineering_consumed": str(fe_dir),
        "affective_decoder_enabled": bool(ad_cfg.get("enable", False)),
        "affective_decoder_ran": bool(ad_run),
        "affective_geometry_predicts_ef": bool(ad_ef),
        "targets": targets,
        "retracted_targets": p.get("retracted_targets", []),
        "covariates": covariates,
        "feature_sets_run": list(feature_sets.keys()),
        "n_hypothesis_test_combos": len(headline),
        "iaps_baseline_consistency_enabled": bool(
            (p.get("iaps_baseline_consistency") or {}).get("enable", False)),
        "exploratory_ml_enabled": bool(ml_cfg.get("enable", False)),
        "exploratory_ml_n_combos": (int(grid_df.shape[0])
                                     if grid_df is not None else 0),
        "screening_index_enabled": bool(si_cfg.get("enable", False)),
        "screening_index_path": si_path,
        "outputs": ["<target>/<feature_set>/{composite_alpha.csv, coef_inference.csv,"
                    " block_f_test.json, bootstrap_ci.csv, summary.json}",
                    "headline.json",
                    "iaps_baseline_consistency.{csv,json}",
                    "ols_scatter.png",
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
