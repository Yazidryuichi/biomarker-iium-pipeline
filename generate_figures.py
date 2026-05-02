"""
Generate publication-quality figures for the README and manuscript.

Usage:
    python generate_figures.py                # From results/ CSVs (after pipeline run)
    python generate_figures.py --from-pilot   # From pilot results (N=28, hardcoded)

Outputs saved to docs/figures/ (tracked in git for README rendering).
"""

import argparse
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


def load_results_from_csv():
    """
    Load results from the latest analysis stage output. Quantum-vs-classical
    comparison is now part of ml_results.csv (when features.include_quantum
    was enabled at extraction time): rows with feature_set in
    {classical_only, quantum_only, classical_plus_quantum}.

    For multi-target runs, reads from the first target subdirectory.
    """
    import pandas as pd
    sys.path.insert(0, os.path.dirname(__file__))
    from utils.io import load_stage_config, latest_stage_dir, get_targets

    config = load_stage_config(
        "analysis",
        globals_path=os.path.join(os.path.dirname(__file__), "configs/config.yaml"),
        create_output_dir=False,
    )
    targets = get_targets(config)
    multi = len(targets) > 1
    primary = targets[0]

    analysis_dir = latest_stage_dir(config, "analysis")
    if analysis_dir is None:
        raise FileNotFoundError("No analysis output found. Run `python pipeline.py --analysis` first.")
    target_dir = os.path.join(analysis_dir, primary) if multi else analysis_dir

    ml = pd.read_csv(os.path.join(target_dir, "ml_results.csv"))
    shap = pd.read_csv(os.path.join(target_dir, "shap_importance.csv"))

    # Quantum-vs-classical: synthesise the legacy DataFrame shape from ml_results
    # (feature_set in {classical, quantum, combined}) when quantum sets exist.
    quantum = None
    fs_present = set(ml["feature_set"].unique())
    if {"all_features", "quantum_only", "classical_plus_quantum"} & fs_present:
        rename = {
            "all_features":           "classical",
            "conventional_qeeg":      "classical",
            "quantum_only":           "quantum",
            "classical_plus_quantum": "combined",
        }
        sub = ml[ml["feature_set"].isin(rename)].copy()
        sub["feature_set"] = sub["feature_set"].map(rename)
        # For each renamed bucket, keep the best-performing model row
        quantum = (sub.sort_values("balanced_accuracy", ascending=False)
                       .drop_duplicates("feature_set")
                       .reset_index(drop=True))

    return ml, shap, quantum


def get_pilot_results():
    """Pilot results from N=28 run (2026-04-07)."""

    # Best model per feature set (Stage 4, 8-model comparison)
    ml_results = [
        ("Conventional\nQEEG", "XGBoost", 0.663, 0.09),
        ("Conv. +\nAdvanced", "XGBoost", 0.558, 0.12),
        ("Covariance", "XGBoost", 0.633, 0.10),
        ("All\nFeatures", "RF", 0.567, 0.13),
    ]

    # SHAP top 15 (conventional QEEG set)
    shap_results = [
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

    # Quantum vs classical comparison (Stage 5)
    quantum_results = {
        "classical": 0.585,
        "quantum": 0.657,
        "combined": 0.643,
    }

    return ml_results, shap_results, quantum_results


def fig_model_comparison(ml_results, out_dir):
    """Bar chart comparing best model per feature set."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    labels = [r[0] for r in ml_results]
    models = [r[1] for r in ml_results]
    accs = [r[2] for r in ml_results]
    stds = [r[3] for r in ml_results]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(labels, accs, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.8, width=0.6)

    # Add model name labels on bars
    for bar, model, acc in zip(bars, models, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{model}\n{acc:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    # Chance line
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(labels) - 0.5, 0.505, "chance", fontsize=8, color="gray",
            ha="right")

    # H4 target line
    ax.axhline(y=0.75, color="#C44E52", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(len(labels) - 0.5, 0.755, "H4 target (0.75)", fontsize=8,
            color="#C44E52", ha="right")

    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("ML Classification by Feature Set (N=28 pilot, 5-fold CV x 10 repeats)")
    ax.set_ylim(0, 0.9)

    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_shap_importance(shap_results, out_dir):
    """Horizontal bar chart of SHAP feature importance."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    features = [r[0] for r in shap_results][::-1]
    values = [r[1] for r in shap_results][::-1]

    # Color by feature category
    def get_color(name):
        if "alpha_react" in name:
            return "#C44E52"  # red — reactivity
        elif "coh_" in name:
            return "#4C72B0"  # blue — coherence
        elif "tbr_" in name:
            return "#DD8452"  # orange — TBR
        elif "faa_" in name:
            return "#55A868"  # green — asymmetry
        else:
            return "#8172B2"  # purple — PSD

    colors = [get_color(f) for f in features]

    bars = ax.barh(range(len(features)), values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Biomarker Candidates (SHAP Feature Importance)")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#C44E52", label="Alpha Reactivity"),
        mpatches.Patch(color="#4C72B0", label="Coherence"),
        mpatches.Patch(color="#DD8452", label="Theta/Beta Ratio"),
        mpatches.Patch(color="#55A868", label="Frontal Alpha Asymmetry"),
        mpatches.Patch(color="#8172B2", label="Power Spectral Density"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(out_dir, "shap_top15.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_quantum_vs_classical(quantum_results, out_dir):
    """Grouped bar chart: quantum vs classical feature performance."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    labels = ["Classical\nQEEG", "Quantum-\nInspired", "Combined"]
    values = [quantum_results["classical"], quantum_results["quantum"],
              quantum_results["combined"]]

    colors = ["#4C72B0", "#C44E52", "#55A868"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2.3, 0.505, "chance", fontsize=8, color="gray")

    # Lift annotation
    lift = quantum_results["quantum"] - quantum_results["classical"]
    ax.annotate(
        f"+{lift:.3f}",
        xy=(0.5, max(quantum_results["classical"], quantum_results["quantum"])),
        xytext=(0.5, 0.72),
        fontsize=10, fontweight="bold", color="#C44E52", ha="center",
        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5),
    )

    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Quantum-Inspired vs Classical Features (N=28 pilot)")
    ax.set_ylim(0, 0.85)

    plt.tight_layout()
    path = os.path.join(out_dir, "quantum_vs_classical.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate README figures")
    parser.add_argument("--from-pilot", action="store_true",
                        help="Use hardcoded pilot results (N=28)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating figures...")

    if args.from_pilot:
        ml, shap, quantum = get_pilot_results()
    else:
        try:
            ml_df, shap_df, quantum_df = load_results_from_csv()
            # Convert DataFrames to the format expected by figure functions
            # Group by feature set, pick best model per set
            best = ml_df.loc[ml_df.groupby("feature_set")["balanced_accuracy"].idxmax()]
            ml = [(row["feature_set"].replace("_", "\n"), row["model"],
                   row["balanced_accuracy"], row.get("bal_acc_std", 0.1))
                  for _, row in best.iterrows()]
            shap = list(zip(shap_df["feature"], shap_df["mean_abs_shap"]))
            if quantum_df is not None:
                quantum = {
                    "classical": quantum_df.loc[
                        quantum_df["feature_set"] == "classical", "balanced_accuracy"
                    ].values[0],
                    "quantum": quantum_df.loc[
                        quantum_df["feature_set"] == "quantum", "balanced_accuracy"
                    ].values[0],
                    "combined": quantum_df.loc[
                        quantum_df["feature_set"] == "combined", "balanced_accuracy"
                    ].values[0],
                }
            else:
                quantum = None
        except FileNotFoundError:
            print("  Results CSVs not found. Use --from-pilot for pilot data.")
            sys.exit(1)

    fig_model_comparison(ml, OUT_DIR)
    fig_shap_importance(shap, OUT_DIR)
    if quantum:
        fig_quantum_vs_classical(quantum, OUT_DIR)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
