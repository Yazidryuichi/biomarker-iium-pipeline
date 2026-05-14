"""
Quantum-Probability Binning Sensitivity Analysis (standalone diagnostic)
=========================================================================

How does the n_bins hyperparameter in the quantum-probability feature
extractor (qi_*) affect the resulting interference statistics? This is
a sanity check, not part of the main pipeline.

Runs on the latest cleaning output. Picks the first subject and computes
mean |interference| at n_bins ∈ {5, 8, 10, 15, 20}. Saves a CSV next to
the script invocation.

Usage:
    python scripts/quantum_binning_sensitivity.py
    python scripts/quantum_binning_sensitivity.py --output diagnostics/qbs.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

from utils.io import load_config, latest_stage_dir
from stages.cleaning import load_cleaned_epochs
from stages._quantum_features import compute_qi_with_nbins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="diagnostics/quantum_binning_sensitivity.csv")
    parser.add_argument("--n-bins", nargs="+", type=int, default=[5, 8, 10, 15, 20])
    args = parser.parse_args()

    os.chdir(PIPELINE_ROOT)
    config = load_config(args.config)
    bands = config["features"]["bands"]

    cleaning_dir = latest_stage_dir(config, "cleaning")
    if cleaning_dir is None:
        sys.exit("No cleaning output found. Run `python pipeline.py --cleaning` first.")
    epochs_dict = load_cleaned_epochs(config, stage_dir=cleaning_dir)

    subjects = {}
    for (sub, cond), epochs in epochs_dict.items():
        subjects.setdefault(sub, {})[cond] = epochs

    if not subjects:
        sys.exit("No subjects found in cleaning output.")

    test_sub = sorted(subjects)[0]
    test_epochs = subjects[test_sub].get("Eyes_Open",
                  list(subjects[test_sub].values())[0])

    print(f"Subject: {test_sub}, condition: Eyes_Open (or first available)")
    print(f"Bands: {list(bands)}")
    print(f"Testing n_bins: {args.n_bins}\n")

    results = []
    for n in args.n_bins:
        feats = compute_qi_with_nbins(test_epochs, bands, n)
        finite_vals = [v for v in feats.values() if np.isfinite(v)]
        mean_interf = np.mean(finite_vals) if finite_vals else float("nan")
        results.append({
            "n_bins": n,
            "n_features": len(feats),
            "mean_interference": round(mean_interf, 6),
        })
        print(f"  n_bins={n:2d}: {len(feats)} features, "
              f"mean |interference|={mean_interf:.6f}")

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
