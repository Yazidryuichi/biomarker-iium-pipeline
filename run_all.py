"""
Biomarker_IIUM Analysis Pipeline — Main Orchestrator
======================================================

Runs the full analysis pipeline from raw EDF files to results.

Usage:
    # Full pipeline
    python run_all.py

    # Single stage
    python run_all.py --stage 1
    python run_all.py --stage 2
    python run_all.py --stage 3
    python run_all.py --stage 4

    # Include emotional conditions (in addition to EO/EC)
    python run_all.py --include-emotional

    # Single subject (for testing)
    python run_all.py --subject D0000795

Dependencies:
    pip install mne autoreject pywavelets scikit-learn xgboost lightgbm
    pip install shap pandas openpyxl pyyaml scipy statsmodels antropy
    pip install coffeine pyriemann  # optional: Riemannian classifiers
"""

import argparse
import os
import sys
import time

# Ensure pipeline root is on path
PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_ROOT)

from utils.io import load_config, discover_subjects
from stages.stage1_cleaning import run_stage1
from stages.stage2_features import run_stage2
from stages.stage3_merge import run_stage3
from stages.stage4_analysis import run_stage4
from stages.exploratory_quantum import run_quantum_exploration


def main():
    parser = argparse.ArgumentParser(
        description="Biomarker_IIUM EEG Analysis Pipeline"
    )
    parser.add_argument(
        "--stage", type=int, default=None,
        help="Run specific stage (1-5). 5=quantum exploration. Default: run all."
    )
    parser.add_argument(
        "--include-emotional", action="store_true",
        help="Include emotional conditions (Happy, Calm, Sad, Scare)"
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Process single subject (e.g. D0000795)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Change to pipeline directory
    os.chdir(PIPELINE_ROOT)

    # Load config
    config = load_config(args.config)
    print("=" * 60)
    print("BIOMARKER_IIUM ANALYSIS PIPELINE")
    print("=" * 60)

    # Discover subjects
    subjects = discover_subjects(config["paths"]["edf_dir"])

    if args.subject:
        if args.subject in subjects:
            subjects = {args.subject: subjects[args.subject]}
        else:
            print(f"ERROR: Subject {args.subject} not found")
            sys.exit(1)

    print(f"Subjects: {len(subjects)}")

    # Determine conditions
    conditions = config["recording"]["conditions"]["primary"]
    if args.include_emotional:
        conditions += config["recording"]["conditions"]["emotional"]
    print(f"Conditions: {conditions}")

    start_time = time.time()

    # ── Stage 1: Cleaning ──
    if args.stage is None or args.stage == 1:
        print("\n" + "=" * 60)
        print("STAGE 1: EEG Cleaning")
        print("=" * 60)
        all_epochs, qc_report = run_stage1(config, subjects, conditions)

    # ── Stage 2: Feature Extraction ──
    if args.stage is None or args.stage == 2:
        # Load cleaned epochs if not from Stage 1
        if args.stage == 2:
            all_epochs = load_cleaned_epochs(config)

        features_df, cov_matrices = run_stage2(config, all_epochs)

    # ── Stage 3: Behavioral Merge ──
    if args.stage is None or args.stage == 3:
        if args.stage == 3:
            import pandas as pd
            features_df = pd.read_csv(
                os.path.join(config["paths"]["output_dir"], "features.csv")
            )

        full_df = run_stage3(config, features_df)

    # ── Stage 4: Analysis ──
    if args.stage is None or args.stage == 4:
        if args.stage == 4:
            import pandas as pd
            full_df = pd.read_csv(
                os.path.join(config["paths"]["output_dir"], "full_dataset.csv")
            )

        results = run_stage4(config, full_df)

    # ── Stage 5: Quantum-Inspired Exploration ──
    if args.stage is None or args.stage == 5:
        if args.stage == 5:
            all_epochs = load_cleaned_epochs(config)
            import pandas as pd
            full_df = pd.read_csv(
                os.path.join(config["paths"]["output_dir"], "full_dataset.csv")
            )

        quantum_df = run_quantum_exploration(config, all_epochs, full_df)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


def load_cleaned_epochs(config):
    """Load previously saved cleaned epochs."""
    import mne

    epoch_dir = os.path.join(config["paths"]["output_dir"], "cleaned_epochs")
    all_epochs = {}

    for f in sorted(os.listdir(epoch_dir)):
        if f.endswith("-epo.fif"):
            parts = f.replace("-epo.fif", "").split("_", 1)
            subject_id = parts[0]
            condition = parts[1] if len(parts) > 1 else "unknown"
            epochs = mne.read_epochs(
                os.path.join(epoch_dir, f), verbose=False
            )
            all_epochs[(subject_id, condition)] = epochs

    print(f"Loaded {len(all_epochs)} epoch files from {epoch_dir}")
    return all_epochs


if __name__ == "__main__":
    main()
