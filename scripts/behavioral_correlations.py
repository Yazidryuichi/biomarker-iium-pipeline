"""
Standalone behavioral correlation analysis (outside the main pipeline).

Computes pairwise Spearman + Pearson correlations among Digit Span,
Flanker, Global EF and EF-aspect parameters in the latest engineering
output, with FDR-corrected p-values for the Spearman matrix. Saves
CSVs and a heatmap PNG.

Run: python scripts/behavioral_correlations.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

from utils.io import latest_stage_dir, load_config  # noqa: E402

# Domain groupings — drives the heatmap colour bands and FDR scope.
DOMAINS = {
    "Digit Span": [
        "FW_Span", "BW_Span", "Total_Span", "FW_Raw", "BW_Raw",
    ],
    "Flanker": [
        "acc_overall", "acc_incongruent",
        "flanker_effect", "rt_mean", "rt_congruent", "rt_incongruent",
        "ddm_v", "ddm_a", "ddm_t", "ddm_delta_v",
        "ddm_v_congruent", "ddm_a_congruent", "ddm_t0_congruent",
        "ddm_v_incongruent", "ddm_a_incongruent", "ddm_t0_incongruent",
    ],
    "Global EF": ["Global_EF"],
    "EF aspects": ["WM_score", "IC_score", "CF_score", "P_score", "SF_score"],
}


def _ordered_columns(df):
    """Return (col_names, group_labels) restricted to columns present in df,
    preserving the declared domain order."""
    cols, groups = [], []
    for grp, names in DOMAINS.items():
        for n in names:
            if n in df.columns and df[n].notna().sum() >= 3:
                cols.append(n)
                groups.append(grp)
    return cols, groups


def _spearman_matrix(df, cols):
    n = len(cols)
    R = np.eye(n)
    P = np.ones((n, n))
    N = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            x = pd.to_numeric(df[cols[i]], errors="coerce")
            y = pd.to_numeric(df[cols[j]], errors="coerce")
            mask = x.notna() & y.notna()
            n_pair = int(mask.sum())
            N[i, j] = N[j, i] = n_pair
            if n_pair < 4:
                R[i, j] = R[j, i] = np.nan
                P[i, j] = P[j, i] = np.nan
                continue
            r, p = stats.spearmanr(x[mask], y[mask])
            R[i, j] = R[j, i] = r
            P[i, j] = P[j, i] = p
    return R, P, N


def _pearson_matrix(df, cols):
    n = len(cols)
    R = np.eye(n)
    P = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            x = pd.to_numeric(df[cols[i]], errors="coerce")
            y = pd.to_numeric(df[cols[j]], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 4:
                R[i, j] = R[j, i] = np.nan
                P[i, j] = P[j, i] = np.nan
                continue
            r, p = stats.pearsonr(x[mask], y[mask])
            R[i, j] = R[j, i] = r
            P[i, j] = P[j, i] = p
    return R, P


def _flatten_matrix(R, P, N, cols, method_label, p_fdr=None):
    rows = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            row = {
                "x":      cols[i],
                "y":      cols[j],
                "method": method_label,
                "n":      int(N[i, j]) if N is not None else "",
                "r":      None if np.isnan(R[i, j]) else round(float(R[i, j]), 4),
                "p":      None if np.isnan(P[i, j]) else round(float(P[i, j]), 4),
            }
            if p_fdr is not None:
                row["p_fdr"] = None if np.isnan(p_fdr[i, j]) else round(float(p_fdr[i, j]), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def _fdr_correct_offdiag(P, cols):
    """Apply BH-FDR across the upper triangle and reflect into a full matrix."""
    n = len(cols)
    iu = np.triu_indices(n, k=1)
    raw = P[iu]
    valid_mask = ~np.isnan(raw)
    p_fdr = np.full_like(raw, np.nan)
    if valid_mask.any():
        _, p_corr, _, _ = multipletests(raw[valid_mask], method="fdr_bh")
        p_fdr[valid_mask] = p_corr
    full = np.full((n, n), np.nan)
    full[iu] = p_fdr
    il = np.tril_indices(n, k=-1)
    full[il] = full.T[il]
    np.fill_diagonal(full, np.nan)
    return full


def _heatmap(R, P_fdr, cols, groups, out_path, alpha=0.05):
    """Heatmap of |r| with FDR-significant cells outlined."""
    n = len(cols)
    side = max(8.0, 0.30 * n + 4)
    fig, ax = plt.subplots(figsize=(side, side))

    im = ax.imshow(np.where(np.isnan(R), 0, R), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_val = R[i, j]
            if np.isnan(r_val):
                continue
            text_color = "white" if abs(r_val) > 0.5 else "black"
            ax.text(j, i, f"{r_val:+.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)
            if not np.isnan(P_fdr[i, j]) and P_fdr[i, j] < alpha:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="black",
                                            linewidth=1.2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)

    # Domain band markers along the axes.
    boundaries = []
    prev = None
    for k, g in enumerate(groups):
        if g != prev:
            boundaries.append((k, g))
            prev = g
    for (k, _g) in boundaries[1:]:
        ax.axvline(k - 0.5, color="black", linewidth=0.8)
        ax.axhline(k - 0.5, color="black", linewidth=0.8)

    ax.set_title("Behavioral correlations — Spearman r "
                 f"(black border = FDR-BH p < {alpha})")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering-dir", type=str, default=None,
                        help="Path to a stages/engineering/runs/<ts>/ dir; "
                             "auto-resolves the latest if omitted.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory; default scripts/runs/<ts>/")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    cfg = load_config(os.path.join(PIPELINE_ROOT, "configs", "config.yaml"))
    eng_dir = args.engineering_dir or latest_stage_dir(cfg, "engineering")
    if eng_dir is None:
        sys.exit("No engineering output found. Run --engineering first.")
    full_path = os.path.join(eng_dir, "full_dataset.csv")
    df = pd.read_csv(full_path)
    print(f"Loaded {full_path}  (shape {df.shape})")

    cols, groups = _ordered_columns(df)
    print(f"Columns to correlate: {len(cols)}")
    if len(cols) < 2:
        sys.exit("Need at least 2 columns with non-NaN data.")

    out_dir = args.out_dir or os.path.join(
        PIPELINE_ROOT, "scripts", "runs",
        datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing outputs to {out_dir}")

    # Spearman matrix + FDR.
    R_s, P_s, N_s = _spearman_matrix(df, cols)
    P_s_fdr = _fdr_correct_offdiag(P_s, cols)
    long_s = _flatten_matrix(R_s, P_s, N_s, cols, "Spearman", p_fdr=P_s_fdr)

    # Pearson matrix (no FDR — supplementary).
    R_p, P_p = _pearson_matrix(df, cols)
    long_p = _flatten_matrix(R_p, P_p, N_s, cols, "Pearson")

    # Save matrices.
    pd.DataFrame(R_s, index=cols, columns=cols).to_csv(
        os.path.join(out_dir, "spearman_matrix.csv"))
    pd.DataFrame(P_s, index=cols, columns=cols).to_csv(
        os.path.join(out_dir, "spearman_pvalues.csv"))
    pd.DataFrame(P_s_fdr, index=cols, columns=cols).to_csv(
        os.path.join(out_dir, "spearman_pvalues_fdr.csv"))
    pd.DataFrame(R_p, index=cols, columns=cols).to_csv(
        os.path.join(out_dir, "pearson_matrix.csv"))

    # Long format with FDR — easiest for downstream filtering.
    long_all = pd.concat([long_s, long_p], ignore_index=True)
    long_all.to_csv(os.path.join(out_dir, "correlation_long.csv"), index=False)

    # Significant Spearman pairs after FDR.
    sig = long_s.dropna(subset=["p_fdr"]).query(f"p_fdr < {args.alpha}").copy()
    sig = sig.sort_values("p_fdr")
    sig.to_csv(os.path.join(out_dir, "significant_spearman_fdr.csv"), index=False)

    # Heatmap.
    _heatmap(R_s, P_s_fdr, cols, groups,
             os.path.join(out_dir, "behavioral_correlation_heatmap.png"),
             alpha=args.alpha)

    print(f"\nN columns analysed: {len(cols)}")
    print(f"Pairs with FDR p<{args.alpha}: {len(sig)}")
    if not sig.empty:
        print("\nTop 15 FDR-significant Spearman pairs:")
        print(sig.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
