"""
Biomarker_IIUM Analysis Pipeline — Main Orchestrator
======================================================

Runs the full analysis pipeline from raw EDF files to results.

Stages (post-Phase-1 numbering):
    1. Cleaning (HAPPE-compliant preprocessing)
    2. Feature extraction (~920 classical QEEG features)
    3. Behavioural merge
    4. Statistical analysis + ML + SHAP
    5. Fair 2x2 comparison (feature set x model class), subject-level LOSO + DeLong
    6. Density-matrix feature extraction (explicit rho, 900 features at N=15)

DAG ordering note: Stage 5 (fair comparison) consumes Stage 6 (DM feature)
outputs. In the default "run all stages" flow, the orchestrator runs Stage 6
BEFORE Stage 5 — the stage number is a stable identifier, not a strict
execution order.

Usage:
    # Full pipeline (runs 1, 2, 3, 4, 6, 5 in that order)
    python run_all.py

    # Single stage
    python run_all.py --stage 1
    python run_all.py --stage 2
    python run_all.py --stage 3
    python run_all.py --stage 4
    python run_all.py --stage 5    # fair comparison (requires stage 6 outputs)
    python run_all.py --stage 6    # density-matrix feature extraction

    # Legacy quantum-cognition exploration (moves to quantum-exploration/ branch)
    python run_all.py --exploratory-quantum

    # Include emotional conditions (in addition to EO/EC)
    python run_all.py --include-emotional

    # Single subject (for testing)
    python run_all.py --subject D0000795

Dependencies:
    pip install -r requirements.lock  # exact pins via uv pip compile
"""

import argparse
import logging
import os
import subprocess
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

try:
    from stages.exploratory_quantum import run_quantum_exploration
    HAS_QUANTUM_LEGACY = True
except ImportError:
    HAS_QUANTUM_LEGACY = False

try:
    from stages.stage6_density_matrix import run_stage6
    HAS_DENSITY_MATRIX = True
except ImportError:
    HAS_DENSITY_MATRIX = False


def run_stage5_fair_comparison(config):
    """Stage 5 invokes the fair-comparison analysis script as a subprocess.
    The script reads from results/ (Stage 4 + Stage 6 outputs) and writes
    results/stage5_fair_comparison.json.
    """
    results_dir = config["paths"]["output_dir"]
    cmd = [
        sys.executable, "-m", "stages.stage5_fair_comparison",
        "--results-dir", results_dir,
        "--out-json", os.path.join(results_dir, "stage5_fair_comparison.json"),
    ]
    subprocess.run(cmd, check=True, cwd=PIPELINE_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Biomarker_IIUM EEG Analysis Pipeline"
    )
    parser.add_argument(
        "--stage", type=int, default=None,
        help="Run specific stage (1-6). 5=fair comparison (post-pipeline analysis), "
             "6=explicit density-matrix features. Default: run all stages."
    )
    parser.add_argument(
        "--exploratory-quantum", action="store_true",
        help="Run the legacy quantum-cognition exploration (QEPP, von Neumann "
             "entropy on PCA-compressed features). Moves to quantum-exploration/ "
             "branch in Phase 3. Off by default."
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

    # Configure logging
    log_dir = os.path.join(PIPELINE_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, "pipeline.log"), mode="a"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("biomarker_iium")

    # Load config
    config = load_config(args.config)
    logger.info("Pipeline started")
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

    # ── Optional: Legacy quantum-cognition exploration (Stage 5 pre-Phase-1) ──
    # Off the main pipeline. Use --exploratory-quantum to invoke. This module
    # moves to the quantum-exploration/ branch in Phase 3.
    if args.exploratory_quantum and HAS_QUANTUM_LEGACY:
        import pandas as pd
        if "all_epochs" not in locals():
            all_epochs = load_cleaned_epochs(config)
        if "full_df" not in locals():
            full_df = pd.read_csv(
                os.path.join(config["paths"]["output_dir"], "full_dataset.csv")
            )
        quantum_df = run_quantum_exploration(config, all_epochs, full_df)

    # ── Stage 6: Explicit Density-Matrix Features ──
    # Executes BEFORE Stage 5 in the default flow because Stage 5
    # (fair comparison) consumes Stage 6 outputs.
    if (args.stage is None or args.stage == 6) and HAS_DENSITY_MATRIX:
        import pandas as pd
        if args.stage == 6:
            all_epochs = load_cleaned_epochs(config)
            full_df = pd.read_csv(
                os.path.join(config["paths"]["output_dir"], "full_dataset.csv")
            )
            quantum_path = os.path.join(
                config["paths"]["output_dir"], "quantum_features.csv"
            )
            quantum_df = (
                pd.read_csv(quantum_path) if os.path.exists(quantum_path) else None
            )
        else:
            quantum_df = locals().get("quantum_df", None)

        run_stage6(config, all_epochs, full_df, quantum_df=quantum_df)

    # ── Stage 5: Fair 2x2 comparison (feature set x model class) ──
    # Post-pipeline analysis. Reads results/ml_results.csv (Stage 4) and the
    # density-matrix feature outputs (Stage 6) and writes
    # results/stage5_fair_comparison.json. Subject-level LOSO + paired DeLong
    # + subject-bootstrap CIs + label-permutation p per cell.
    #
    # SKIP_STAGE5=1: bypass for CI / smoke-test runs. Stage 5 is a post-hoc
    # statistical analysis (4 cells × 100-fold CV × 1000-permutation × 10000-
    # bootstrap), not part of pipeline correctness. CI verifies stages 1-4 + 6;
    # Stage 5 is omitted there because its compute budget (~1+ hour at N=20)
    # exceeds the smoke-test runner budget and adds no signal for "does the
    # pipeline run?". Real-data runs always run Stage 5.
    if os.environ.get("SKIP_STAGE5") == "1":
        print("\n" + "=" * 60)
        print("STAGE 5: SKIPPED (SKIP_STAGE5=1)")
        print("=" * 60)
    elif args.stage is None or args.stage == 5:
        print("\n" + "=" * 60)
        print("STAGE 5: Fair 2x2 Comparison (feature set x model class)")
        print("=" * 60)
        run_stage5_fair_comparison(config)

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
