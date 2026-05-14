"""
Reliability analysis (standalone diagnostic).
==============================================

Classical-test-theory reliability for the three behavioural measures used
as EF targets in the pilot draft. This sits outside the main pipeline so
it can be re-run independently from any cleaning/feature recompute.

What is reported per measure (driven by the granularity available in the
source workbook — item- vs. summary-level — and the pilot N):

  AUFEI-O   Item-level Likert data (5 items per subscale x 5 subscales).
            Per subscale (WM, IC, CF, P, SF) and pooled Global:
              * Cronbach's alpha with 95% CI (pingouin, normal-theory)
              * McDonald's omega (one-factor congeneric loadings via
                factor_analyzer MINRES; omega_total formula)
              * Mean inter-item correlation
              * Per-item: corrected item-total r and alpha-if-item-deleted

  Digit Span  Summary-only export (FW_Raw, BW_Raw, Total_Raw). Two-half
              Spearman-Brown reliability treating FW and BW as parallel
              halves of the Total composite. Flagged as approximate
              because the two halves are not strictly parallel.

  Flanker     Summary-only export. Cross-condition RT stability
              (rt_congruent vs. rt_incongruent: Pearson + Spearman) as
              evidence that individual differences in RT are consistent
              across conditions, plus a within-subject rt_cv summary.
              Within-task split-half reliability of the Flanker effect
              cannot be computed because trial-level RTs were not
              retained in the source workbook.

Output: scripts/runs/<ts>/reliability/
    aufei_subscale_reliability.csv
    aufei_item_total.csv
    digit_span_reliability.csv
    flanker_reliability.csv
    summary.txt

Usage:
    python scripts/reliability_analysis.py
    python scripts/reliability_analysis.py --out-dir path/to/dir
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

from utils.io import load_config  # noqa: E402

import pingouin as pg  # noqa: E402


AUFEI_SUBSCALES = {
    "WM": ["WM1", "WM2", "WM3", "WM4", "WM5"],
    "IC": ["IC1", "IC2", "IC3", "IC4", "IC5"],
    "CF": ["CF1", "CF2", "CF3", "CF4", "CF5"],
    "P":  ["P1",  "P2",  "P3",  "P4",  "P5"],
    "SF": ["SF1", "SF2", "SF3", "SF4", "SF5"],
}


def _mean_inter_item_r(df: pd.DataFrame) -> float:
    R = df.corr(method="pearson").to_numpy()
    iu = np.triu_indices_from(R, k=1)
    vals = R[iu]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def _mcdonald_omega(df: pd.DataFrame, max_iter: int = 100, tol: float = 1e-6) -> float:
    """One-factor congeneric omega_total (McDonald, 1999).

    omega = (sum lambda)^2 / ((sum lambda)^2 + sum (1 - h^2))

    Implemented via principal-axis factoring on the item correlation
    matrix (single factor): initialise communalities with squared
    multiple correlations, iterate eigendecomposition of the
    reduced correlation matrix until convergence. Avoids the
    factor_analyzer dependency, which is incompatible with sklearn>=1.6.

    Returns nan if the data are degenerate (constant item, fewer than
    2 items, or non-PSD reduced matrix).
    """
    X_full = df.dropna().to_numpy(dtype=float)
    if X_full.shape[0] < 3 or X_full.shape[1] < 2:
        return np.nan
    # Constant items have zero correlation with everything else and would
    # make corrcoef NaN. Drop them: they contribute zero loading and zero
    # uniqueness to the omega numerator/denominator anyway.
    keep_mask = X_full.std(axis=0, ddof=1) > 0
    if keep_mask.sum() < 2:
        return np.nan
    X = X_full[:, keep_mask]
    n, k = X.shape

    R = np.corrcoef(X, rowvar=False)

    # Initial communalities = squared multiple correlations (SMC).
    # SMC_i = 1 - 1 / diag(R^{-1}). Fall back to max |r_ij| if R is singular.
    try:
        Rinv = np.linalg.inv(R)
        smc = 1.0 - 1.0 / np.diag(Rinv)
        # Numerical noise can push SMC slightly out of [0, 1].
        smc = np.clip(smc, 1e-4, 1.0 - 1e-4)
    except np.linalg.LinAlgError:
        Rabs = np.abs(R.copy())
        np.fill_diagonal(Rabs, 0.0)
        smc = np.clip(Rabs.max(axis=1), 1e-4, 1.0 - 1e-4)

    h2 = smc.copy()
    loadings = np.zeros(k)
    for _ in range(max_iter):
        R_reduced = R.copy()
        np.fill_diagonal(R_reduced, h2)
        # Symmetric eigendecomposition.
        eigvals, eigvecs = np.linalg.eigh(R_reduced)
        # Largest eigenvalue is the last entry.
        lam1 = eigvals[-1]
        v1 = eigvecs[:, -1]
        if lam1 <= 0:
            return np.nan
        new_loadings = np.sqrt(lam1) * v1
        # PAF sign indeterminacy: orient so most loadings are positive.
        if np.sum(new_loadings) < 0:
            new_loadings = -new_loadings
        new_h2 = np.clip(new_loadings ** 2, 0.0, 1.0)
        if np.max(np.abs(new_h2 - h2)) < tol:
            loadings = new_loadings
            h2 = new_h2
            break
        h2 = new_h2
        loadings = new_loadings

    sum_l = loadings.sum()
    uniq = 1.0 - h2
    denom = (sum_l ** 2) + uniq.sum()
    if denom <= 0:
        return np.nan
    return float((sum_l ** 2) / denom)


def _alpha_with_ci(df: pd.DataFrame, ci: float = 0.95):
    """Wrapper around pingouin.cronbach_alpha that tolerates NaN rows."""
    sub = df.dropna()
    if sub.shape[0] < 3 or sub.shape[1] < 2:
        return np.nan, (np.nan, np.nan), sub.shape[0]
    alpha, conf = pg.cronbach_alpha(data=sub, ci=ci)
    return float(alpha), (float(conf[0]), float(conf[1])), int(sub.shape[0])


def _item_total_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For each item: corrected item-total Pearson r and alpha-if-deleted."""
    sub = df.dropna()
    rows = []
    items = list(sub.columns)
    total = sub.sum(axis=1)
    for it in items:
        rest = total - sub[it]
        r = stats.pearsonr(sub[it], rest)[0] if sub[it].std(ddof=1) > 0 and rest.std(ddof=1) > 0 else np.nan
        remaining = [c for c in items if c != it]
        if len(remaining) >= 2:
            try:
                alpha_del, _ = pg.cronbach_alpha(data=sub[remaining])
                alpha_del = float(alpha_del)
            except Exception:
                alpha_del = np.nan
        else:
            alpha_del = np.nan
        rows.append({
            "item": it,
            "r_item_total_corrected": None if pd.isna(r) else round(float(r), 4),
            "alpha_if_deleted": None if pd.isna(alpha_del) else round(alpha_del, 4),
            "item_mean": round(float(sub[it].mean()), 3),
            "item_sd": round(float(sub[it].std(ddof=1)), 3),
        })
    return pd.DataFrame(rows)


def analyse_aufei(aufei_path: str):
    """Returns (subscale_df, item_total_df)."""
    df = pd.read_excel(aufei_path)
    subscale_rows = []
    item_rows = []

    for label, items in AUFEI_SUBSCALES.items():
        present = [c for c in items if c in df.columns]
        if len(present) < 2:
            continue
        sub = df[present]
        alpha, (lo, hi), n_obs = _alpha_with_ci(sub)
        omega = _mcdonald_omega(sub)
        mir = _mean_inter_item_r(sub.dropna())
        subscale_rows.append({
            "subscale": label,
            "n_items": len(present),
            "n_obs": n_obs,
            "cronbach_alpha": None if pd.isna(alpha) else round(alpha, 4),
            "alpha_ci_lo": None if pd.isna(lo) else round(lo, 4),
            "alpha_ci_hi": None if pd.isna(hi) else round(hi, 4),
            "mcdonald_omega": None if pd.isna(omega) else round(omega, 4),
            "mean_inter_item_r": None if pd.isna(mir) else round(mir, 4),
        })
        it_df = _item_total_stats(sub)
        it_df.insert(0, "subscale", label)
        item_rows.append(it_df)

    # Pooled "Global" composite over all 25 items.
    all_items = [c for items in AUFEI_SUBSCALES.values() for c in items if c in df.columns]
    if len(all_items) >= 2:
        sub = df[all_items]
        alpha, (lo, hi), n_obs = _alpha_with_ci(sub)
        omega = _mcdonald_omega(sub)
        mir = _mean_inter_item_r(sub.dropna())
        subscale_rows.append({
            "subscale": "Global (all items)",
            "n_items": len(all_items),
            "n_obs": n_obs,
            "cronbach_alpha": None if pd.isna(alpha) else round(alpha, 4),
            "alpha_ci_lo": None if pd.isna(lo) else round(lo, 4),
            "alpha_ci_hi": None if pd.isna(hi) else round(hi, 4),
            "mcdonald_omega": None if pd.isna(omega) else round(omega, 4),
            "mean_inter_item_r": None if pd.isna(mir) else round(mir, 4),
        })

    return pd.DataFrame(subscale_rows), pd.concat(item_rows, ignore_index=True) if item_rows else pd.DataFrame()


def analyse_digit_span(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    sub = df[["FW_Raw", "BW_Raw"]].dropna()
    if sub.shape[0] < 3:
        return pd.DataFrame()
    r_fw_bw, p_fw_bw = stats.pearsonr(sub["FW_Raw"], sub["BW_Raw"])
    rs_fw_bw, ps_fw_bw = stats.spearmanr(sub["FW_Raw"], sub["BW_Raw"])
    r_sb = 2 * r_fw_bw / (1 + r_fw_bw) if (1 + r_fw_bw) != 0 else np.nan
    return pd.DataFrame([{
        "measure": "Digit Span (FW vs BW as parallel halves)",
        "n_obs": int(sub.shape[0]),
        "pearson_r_FW_BW": round(float(r_fw_bw), 4),
        "pearson_p": round(float(p_fw_bw), 4),
        "spearman_r_FW_BW": round(float(rs_fw_bw), 4),
        "spearman_p": round(float(ps_fw_bw), 4),
        "spearman_brown_2half": round(float(r_sb), 4),
        "note": "Approximate: FW and BW are not strictly parallel halves; "
                "no trial-level data available.",
    }])


def analyse_flanker(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    rows = []

    sub = df[["rt_congruent", "rt_incongruent"]].dropna()
    if sub.shape[0] >= 3:
        r, p = stats.pearsonr(sub["rt_congruent"], sub["rt_incongruent"])
        rs, ps = stats.spearmanr(sub["rt_congruent"], sub["rt_incongruent"])
        rows.append({
            "measure": "RT cong vs RT incong (cross-condition stability)",
            "n_obs": int(sub.shape[0]),
            "pearson_r": round(float(r), 4),
            "pearson_p": round(float(p), 4),
            "spearman_r": round(float(rs), 4),
            "spearman_p": round(float(ps), 4),
        })

    if "rt_cv" in df.columns:
        cv = df["rt_cv"].dropna()
        if cv.size:
            rows.append({
                "measure": "rt_cv (within-subject RT variability)",
                "n_obs": int(cv.size),
                "mean": round(float(cv.mean()), 4),
                "sd": round(float(cv.std(ddof=1)), 4),
                "min": round(float(cv.min()), 4),
                "max": round(float(cv.max()), 4),
            })

    if "n_trials" in df.columns:
        nt = df["n_trials"].dropna()
        if nt.size:
            rows.append({
                "measure": "n_trials per subject",
                "n_obs": int(nt.size),
                "mean": round(float(nt.mean()), 2),
                "sd": round(float(nt.std(ddof=1)), 2),
                "min": int(nt.min()),
                "max": int(nt.max()),
            })

    rows.append({
        "measure": "NOTE",
        "n_obs": "",
        "note": "Within-task split-half reliability of Flanker effect not "
                "computable: trial-level RTs were not retained in the source "
                "export. Cross-condition RT correlation reported instead.",
    })
    return pd.DataFrame(rows)


def _fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "    n/a"
    if isinstance(v, float):
        return f"{v:>7.3f}"
    return f"{v!s:>7}"


def write_summary(subscale_df, item_df, ds_df, fl_df, out_dir):
    lines = []
    lines.append("Reliability analysis - pilot draft")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("AUFEI-O internal consistency")
    lines.append("=" * 78)
    if subscale_df.empty:
        lines.append("(no AUFEI data)")
    else:
        header = f"{'subscale':<22} {'k':>3} {'N':>3}   alpha   95% CI            omega   mean r"
        lines.append(header)
        lines.append("-" * len(header))
        for _, r in subscale_df.iterrows():
            ci = (f"[{r['alpha_ci_lo']:+.2f}, {r['alpha_ci_hi']:+.2f}]"
                  if r["alpha_ci_lo"] is not None else "[n/a]")
            lines.append(
                f"{r['subscale']:<22} {int(r['n_items']):>3} {int(r['n_obs']):>3}  "
                f"{_fmt(r['cronbach_alpha'])}  {ci:<18} "
                f"{_fmt(r['mcdonald_omega'])} {_fmt(r['mean_inter_item_r'])}"
            )
    lines.append("")
    lines.append("Convention: alpha/omega >= .70 acceptable, >= .80 good, >= .90 excellent.")
    lines.append("Negative or near-zero alpha indicates items are not measuring a")
    lines.append("common construct (e.g., reverse-coded items not yet recoded, or")
    lines.append("heterogeneous content).")
    lines.append("")

    if not item_df.empty:
        lines.append("=" * 78)
        lines.append("AUFEI-O item diagnostics (corrected item-total r, alpha-if-deleted)")
        lines.append("=" * 78)
        header = f"{'subscale':<5} {'item':<6} {'mean':>6} {'sd':>6}  r_it-tot  alpha_drop"
        lines.append(header)
        lines.append("-" * len(header))
        for _, r in item_df.iterrows():
            lines.append(
                f"{r['subscale']:<5} {r['item']:<6} {_fmt(r['item_mean'])} "
                f"{_fmt(r['item_sd'])}  {_fmt(r['r_item_total_corrected'])}  "
                f"{_fmt(r['alpha_if_deleted'])}"
            )
        lines.append("")

    lines.append("=" * 78)
    lines.append("Digit Span")
    lines.append("=" * 78)
    if ds_df.empty:
        lines.append("(no Digit Span data)")
    else:
        for _, r in ds_df.iterrows():
            for k, v in r.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

    lines.append("=" * 78)
    lines.append("Flanker")
    lines.append("=" * 78)
    if fl_df.empty:
        lines.append("(no Flanker data)")
    else:
        for _, r in fl_df.iterrows():
            for k, v in r.items():
                if pd.isna(v) or v == "":
                    continue
                lines.append(f"  {k}: {v}")
            lines.append("")

    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(PIPELINE_ROOT, "configs", "config.yaml"))
    parser.add_argument("--out-dir", default=None,
                        help="Output directory; default scripts/runs/<ts>/reliability/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Matches the layout used by stages/engineering/engineering.py:
    # behavioral_dir points at ./data and the workbooks live in ./data/Behavioral/.
    beh_dir = os.path.join(cfg["paths"]["behavioral_dir"], "Behavioral")
    aufei_path = os.path.join(beh_dir, "AUFEI-O_Cleaned.xlsx")
    ds_path = os.path.join(beh_dir, "Digit_Span.xlsx")
    fl_path = os.path.join(beh_dir, "Flanker_Test_Pilot.xlsx")

    for p in (aufei_path, ds_path, fl_path):
        if not os.path.exists(p):
            sys.exit(f"Missing behavioural file: {p}")

    out_dir = args.out_dir or os.path.join(
        PIPELINE_ROOT, "scripts", "runs",
        datetime.now().strftime("%Y-%m-%d_%H%M%S"), "reliability",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing outputs to {out_dir}")

    subscale_df, item_df = analyse_aufei(aufei_path)
    ds_df = analyse_digit_span(ds_path)
    fl_df = analyse_flanker(fl_path)

    subscale_df.to_csv(os.path.join(out_dir, "aufei_subscale_reliability.csv"), index=False)
    item_df.to_csv(os.path.join(out_dir, "aufei_item_total.csv"), index=False)
    ds_df.to_csv(os.path.join(out_dir, "digit_span_reliability.csv"), index=False)
    fl_df.to_csv(os.path.join(out_dir, "flanker_reliability.csv"), index=False)
    summary_path = write_summary(subscale_df, item_df, ds_df, fl_df, out_dir)

    print(f"Wrote {summary_path}")
    print()
    with open(summary_path, "r", encoding="utf-8") as f:
        sys.stdout.write(f.read())


if __name__ == "__main__":
    main()
