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
# Main
# ──────────────────────────────────────────────────────────────────

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

    targets = list(p["targets"])
    covariates = list(p["covariates"])
    feature_sets = p["feature_sets"]
    directions = p.get("directions", {}) or {}
    components_map = p.get("composite_components", {}) or {}
    alpha_min = float(p["alpha_min_acceptable"])
    correction = str(p.get("correction", "bonferroni")).lower()
    alpha_level = float(p["alpha_level"])
    bootstrap_n = int(p["bootstrap_n"])

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

        for fs_name, fs_cols in feature_sets.items():
            cols = [c for c in fs_cols if c in full.columns]
            if not cols:
                print(f"\n  SKIP feature_set '{fs_name}': no composite columns in data")
                continue
            try:
                summary = run_hypothesis_test(
                    full, target, fs_name, cols,
                    covariates, directions, components_map,
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

    notes = {
        "stage": "analysis",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "feature_engineering_consumed": str(fe_dir),
        "targets": targets,
        "covariates": covariates,
        "feature_sets_run": list(feature_sets.keys()),
        "n_combos": len(headline),
        "outputs": ["<target>/<feature_set>/{composite_alpha.csv, coef_inference.csv,"
                    " block_f_test.json, bootstrap_ci.csv, summary.json}",
                    "headline.json"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
