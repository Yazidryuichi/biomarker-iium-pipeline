"""
Generate publication-quality figures for the README and manuscript.

Usage:
    python generate_figures.py                # From Stage 5 fair-comparison JSON + Stage 4 SHAP CSV
    python generate_figures.py --from-pilot   # SHAP fallback only (hardcoded pilot N=28)

Outputs saved to docs/figures/ (tracked in git for README rendering).

Figure inventory:
    model_comparison.png     — 2x2 fair comparison (feature_set x model_class), Phase 1 honest-headline.
    quantum_vs_classical.png — Matched-model contrast: DM vs Classical within each model class.
    shap_top15.png           — Top 15 biomarker candidates by mean |SHAP| (Stage 4 conventional QEEG).
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "docs", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_stage5_fair_comparison():
    """Load Phase 1 fair-comparison results (Stage 5)."""
    path = os.path.join(RESULTS_DIR, "stage5_fair_comparison.json")
    with open(path) as f:
        return json.load(f)


def load_shap_from_csv():
    """Load SHAP ranking from Stage 4 output."""
    import pandas as pd
    path = os.path.join(RESULTS_DIR, "shap_importance.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Expected columns: feature, mean_abs_shap
    return list(zip(df["feature"][:15], df["mean_abs_shap"][:15]))


def get_pilot_shap():
    """Fallback SHAP top-15 from pilot N=28 run (2026-04-07)."""
    return [
        ("coh_beta_F3_P3", 0.150),
        ("psd_abs_beta_Pz", 0.098),
        ("alpha_react_global", 0.057),
        ("psd_rel_beta_F4", 0.051),
        ("tbr_Cz", 0.036),
        ("psd_abs_beta_O1", 0.035),
        ("coh_alpha_Fz_Pz", 0.031),
        ("psd_rel_theta_Fz", 0.028),
        ("faa_F4_F3", 0.025),
        ("coh_delta_Fz_Pz", 0.023),
        ("alpha_react_O1", 0.021),
        ("tbr_F3", 0.019),
        ("psd_abs_alpha_O2", 0.017),
        ("coh_theta_F3_P3", 0.015),
        ("psd_rel_delta_Cz", 0.013),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — 2×2 fair comparison (replaces old model_comparison.png)
# ─────────────────────────────────────────────────────────────────────────────

def fig_fair_comparison_2x2(stage5_data, out_dir):
    """2×2 fair comparison: feature_set × model_class. Subject-level LOSO AUC
    with 95% subject-bootstrap CIs + permutation p (per-fold BAcc) per cell.
    """
    cells_by_key = {c["cell"]: c for c in stage5_data["cells"]}
    perm_by_cell = {p["cell"]: p["permutation_p"] for p in stage5_data["permutation_tests"]}

    # Display order: DM-SVM, Classical-SVM, DM-RF, Classical-RF
    order = [
        ("density_matrix+svm_linear",  "Density-Matrix\nSVM-linear",  "#C44E52"),
        ("classical+svm_linear",       "Classical QEEG\nSVM-linear",  "#4C72B0"),
        ("density_matrix+rf_shallow",  "Density-Matrix\nRF-shallow",  "#DD8452"),
        ("classical+rf_shallow",       "Classical QEEG\nRF-shallow",  "#55A868"),
    ]

    aucs = [cells_by_key[k]["loso_auc_mean"] for k, _, _ in order]
    ci_lo = [cells_by_key[k]["loso_auc_ci_lo"] for k, _, _ in order]
    ci_hi = [cells_by_key[k]["loso_auc_ci_hi"] for k, _, _ in order]
    labels = [lbl for _, lbl, _ in order]
    colors = [c for _, _, c in order]
    perm_p = [perm_by_cell[k] for k, _, _ in order]

    yerr = np.array([
        [a - lo for a, lo in zip(aucs, ci_lo)],
        [hi - a for a, hi in zip(aucs, ci_hi)],
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    bars = ax.bar(
        x, aucs, yerr=yerr, capsize=5,
        color=colors, edgecolor="white", linewidth=0.8, width=0.6,
        error_kw={"linewidth": 1.0, "alpha": 0.7},
    )

    # AUC label above bar (cleared of CI band)
    for bar, auc, hi in zip(bars, aucs, ci_hi):
        ax.text(
            bar.get_x() + bar.get_width() / 2, hi + 0.025,
            f"AUC = {auc:.2f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    # Permutation p near the base of each bar (italic, gray)
    for bar, p in zip(bars, perm_p):
        p_lbl = f"perm p = {p:.3f}" if p >= 0.001 else "perm p < 0.001"
        ax.text(
            bar.get_x() + bar.get_width() / 2, 0.035,
            p_lbl, ha="center", va="bottom",
            fontsize=8, color="#444", style="italic",
        )

    # Chance line
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(x) - 0.5, 0.515, "chance (AUC=0.5)", fontsize=8, color="gray", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Subject-level LOSO AUC")
    ax.set_title(
        "Fair 2×2 Comparison: Feature Set × Model Class (N=28 pilot)\n"
        "Error bars = subject-bootstrap 95% CI; perm p = label-permutation on per-fold BAcc",
        fontsize=11,
    )
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Matched-model contrast (replaces old quantum_vs_classical.png)
# ─────────────────────────────────────────────────────────────────────────────

def fig_matched_model_contrast(stage5_data, out_dir):
    """DM vs Classical features, matched within model class.
    Two paired groups (SVM-linear, RF-shallow) with subject-level LOSO AUC,
    95% subject-bootstrap CIs, permutation p per bar, and DeLong p per pair.
    """
    cells_by_key = {c["cell"]: c for c in stage5_data["cells"]}
    perm_by_cell = {p["cell"]: p["permutation_p"] for p in stage5_data["permutation_tests"]}

    delong_index = {}
    for d in stage5_data["pairwise_delong"]:
        delong_index[(d["cell_1"], d["cell_2"])] = d

    pairs = [
        ("svm_linear", "Matched: SVM-linear"),
        ("rf_shallow", "Matched: RF-shallow"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    width = 0.35
    x = np.arange(len(pairs))

    dm_color = "#C44E52"
    cl_color = "#4C72B0"

    for i, (model, _) in enumerate(pairs):
        dm_key = f"density_matrix+{model}"
        cl_key = f"classical+{model}"
        dm = cells_by_key[dm_key]
        cl = cells_by_key[cl_key]

        dm_err = np.array([[dm["loso_auc_mean"] - dm["loso_auc_ci_lo"]],
                           [dm["loso_auc_ci_hi"] - dm["loso_auc_mean"]]])
        cl_err = np.array([[cl["loso_auc_mean"] - cl["loso_auc_ci_lo"]],
                           [cl["loso_auc_ci_hi"] - cl["loso_auc_mean"]]])

        ax.bar(
            i - width / 2, dm["loso_auc_mean"], width,
            yerr=dm_err, capsize=4,
            color=dm_color, edgecolor="white", linewidth=0.8,
            label="Density-Matrix" if i == 0 else None,
            error_kw={"linewidth": 1.0, "alpha": 0.7},
        )
        ax.bar(
            i + width / 2, cl["loso_auc_mean"], width,
            yerr=cl_err, capsize=4,
            color=cl_color, edgecolor="white", linewidth=0.8,
            label="Classical QEEG" if i == 0 else None,
            error_kw={"linewidth": 1.0, "alpha": 0.7},
        )

        # AUC labels above bars
        ax.text(
            i - width / 2, dm["loso_auc_ci_hi"] + 0.025,
            f"{dm['loso_auc_mean']:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        ax.text(
            i + width / 2, cl["loso_auc_ci_hi"] + 0.025,
            f"{cl['loso_auc_mean']:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

        # Permutation p near base
        p_dm = perm_by_cell[dm_key]
        p_cl = perm_by_cell[cl_key]
        p_dm_lbl = f"perm p={p_dm:.3f}" if p_dm >= 0.001 else "perm p<0.001"
        p_cl_lbl = f"perm p={p_cl:.3f}" if p_cl >= 0.001 else "perm p<0.001"
        ax.text(i - width / 2, 0.035, p_dm_lbl, ha="center", va="bottom",
                fontsize=7.5, color="#444", style="italic")
        ax.text(i + width / 2, 0.035, p_cl_lbl, ha="center", va="bottom",
                fontsize=7.5, color="#444", style="italic")

        # DeLong p above the pair
        d = delong_index.get((dm_key, cl_key)) or delong_index.get((cl_key, dm_key))
        if d is not None:
            p_d = d["delong_p_two_sided"]
            stars = (
                "***" if p_d < 0.001
                else "**" if p_d < 0.01
                else "*"  if p_d < 0.05
                else "ns"
            )
            delta = d["delta_auc"]
            # Direction of delta depends on pair order; force DM−CL sign
            if d["cell_1"] == cl_key:
                delta = -delta
            y_top = max(dm["loso_auc_ci_hi"], cl["loso_auc_ci_hi"]) + 0.10
            ax.text(
                i, y_top,
                f"ΔAUC = {delta:+.2f}  |  DeLong p = {p_d:.3f}  {stars}",
                ha="center", va="bottom", fontsize=8.5,
                color="black", fontweight="bold",
            )
            # Bracket linking the pair
            bracket_y = y_top - 0.015
            ax.plot(
                [i - width / 2, i - width / 2, i + width / 2, i + width / 2],
                [bracket_y - 0.02, bracket_y, bracket_y, bracket_y - 0.02],
                color="black", linewidth=0.8,
            )

    # Chance line
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(pairs) - 0.5, 0.515, "chance (AUC=0.5)", fontsize=8, color="gray", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pairs], fontsize=10)
    ax.set_ylabel("Subject-level LOSO AUC")
    ax.set_title(
        "Density-Matrix vs Classical QEEG — Matched-Model Contrast (N=28 pilot)\n"
        "DM > Classical effect is model-class-dependent: significant under SVM, tie under RF",
        fontsize=10.5,
    )
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(out_dir, "quantum_vs_classical.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — SHAP top 15 (unchanged from pre-Phase-1)
# ─────────────────────────────────────────────────────────────────────────────

def fig_shap_importance(shap_results, out_dir):
    """Horizontal bar chart of SHAP feature importance."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    features = [r[0] for r in shap_results][::-1]
    values = [r[1] for r in shap_results][::-1]

    def get_color(name):
        if "alpha_react" in name:
            return "#C44E52"
        elif "coh_" in name:
            return "#4C72B0"
        elif "tbr_" in name:
            return "#DD8452"
        elif "faa_" in name:
            return "#55A868"
        else:
            return "#8172B2"

    colors = [get_color(f) for f in features]

    ax.barh(range(len(features)), values, color=colors, edgecolor="white",
            linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Biomarker Candidates (SHAP Feature Importance, Stage 4)")

    legend_patches = [
        mpatches.Patch(color="#C44E52", label="Alpha Reactivity"),
        mpatches.Patch(color="#4C72B0", label="Coherence"),
        mpatches.Patch(color="#DD8452", label="Theta/Beta Ratio"),
        mpatches.Patch(color="#55A868", label="Frontal Alpha Asymmetry"),
        mpatches.Patch(color="#8172B2", label="Power Spectral Density"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(out_dir, "shap_top15.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate README figures")
    parser.add_argument(
        "--from-pilot", action="store_true",
        help="Use hardcoded SHAP fallback (N=28). Stage 5 JSON still required for AUC figures.",
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating figures...")

    # Stage 5 fair-comparison JSON drives Figure 1 + Figure 2
    try:
        stage5 = load_stage5_fair_comparison()
    except FileNotFoundError:
        print(f"  ERROR: results/stage5_fair_comparison.json not found. "
              f"Run `python -m stages.stage5_fair_comparison` first.")
        sys.exit(1)

    fig_fair_comparison_2x2(stage5, OUT_DIR)
    fig_matched_model_contrast(stage5, OUT_DIR)

    # SHAP — prefer Stage 4 CSV, fall back to pilot hardcoded
    shap_data = None if args.from_pilot else load_shap_from_csv()
    if shap_data is None:
        shap_data = get_pilot_shap()
        print("  (SHAP using pilot fallback)")
    fig_shap_importance(shap_data, OUT_DIR)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
