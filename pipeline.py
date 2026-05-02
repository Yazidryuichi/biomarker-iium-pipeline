"""
Biomarker_IIUM Pipeline Orchestrator
====================================

Lightweight dispatcher around four stages, each living in its own
``stages/<stage>/`` folder with its own ``config.yaml``. A stage's
config declares its predecessor (``input.from``) and output label
(``output.to``). At launch, ``utils.io.load_stage_config(stage_name)``
merges globals (``configs/config.yaml``) with the per-stage params,
auto-resolves ``input_dir`` from the latest predecessor run, and
creates a fresh timestamped ``output_dir``.

Usage:
    python pipeline.py                           # full pipeline (cleaning → features → engineering → analysis)
    python pipeline.py --cleaning                # stage 1 only
    python pipeline.py --features                # stage 2 only
    python pipeline.py --engineering             # stage 3 only
    python pipeline.py --analysis                # stage 4 only
    python pipeline.py --include-emotional       # add Happy/Calm/Sad/Scare conditions
    python pipeline.py --subject D0000795        # single-subject debug run
    python pipeline.py --config configs/other.yaml   # override globals path

Quantum-inspired models / features are opt-in via stage configs:
    stages/features/config.yaml   params.include_quantum: true   → adds qepp_*, qi_*, tn_*
    stages/analysis/config.yaml   params.include_qsvm:    true   → registers QSVM_* models

Dependencies:
    pip install mne autoreject pywavelets scikit-learn xgboost lightgbm
    pip install shap pandas openpyxl pyyaml scipy statsmodels antropy
    pip install coffeine pyriemann  # optional: Riemannian classifiers
"""

import argparse
import logging
import os
import random
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_ROOT)

from utils.io import load_config, load_stage_config, discover_subjects


def main():
    parser = argparse.ArgumentParser(
        description="Biomarker_IIUM EEG Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Each stage owns stages/<stage>/config.yaml and writes to\n"
            "results/<output.to>/<timestamp>/. No shared run dir."
        ),
    )

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--cleaning",    action="store_true", help="Run EEG cleaning only")
    g.add_argument("--features",    action="store_true", help="Run feature extraction only")
    g.add_argument("--engineering", action="store_true",
                   help="Run feature engineering + behavioral merge only")
    g.add_argument("--analysis",    action="store_true", help="Run analysis & ML only")

    parser.add_argument("--include-emotional", action="store_true",
                        help="Include emotional conditions (Happy, Calm, Sad, Scare)")
    parser.add_argument("--subject", type=str, default=None,
                        help="Process single subject (e.g. D0000795)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to globals config file")
    args = parser.parse_args()

    os.chdir(PIPELINE_ROOT)
    globals_cfg = load_config(args.config)

    seed = globals_cfg.get("random_state", 42)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    log_dir = os.path.join(PIPELINE_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "pipeline.log"), mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    run_all = not any([args.cleaning, args.features, args.engineering,
                       args.analysis])
    do = {
        "cleaning":    run_all or args.cleaning,
        "features":    run_all or args.features,
        "engineering": run_all or args.engineering,
        "analysis":    run_all or args.analysis,
    }

    print("=" * 60)
    print("BIOMARKER_IIUM ANALYSIS PIPELINE")
    print(f"Stages : {[s for s, on in do.items() if on]}")
    print(f"Globals: {os.path.abspath(args.config)}")
    print("=" * 60)

    # In-memory passing within a single invocation; standalone runs auto-load
    # from the latest predecessor output_dir resolved at stage-load time.
    all_epochs = None
    features_df = None
    full_df = None

    if do["cleaning"]:
        from stages.cleaning import run as run_cleaning
        cfg = load_stage_config("cleaning", globals_path=args.config)
        subjects = discover_subjects(cfg["paths"]["edf_dir"])
        if args.subject:
            if args.subject not in subjects:
                sys.exit(f"ERROR: Subject {args.subject} not found")
            subjects = {args.subject: subjects[args.subject]}
        conditions = list(cfg["recording"]["conditions"]["primary"])
        if args.include_emotional:
            conditions += cfg["recording"]["conditions"]["emotional"]
        print(f"Subjects: {len(subjects)}  Conditions: {conditions}")
        all_epochs, _ = run_cleaning(cfg, subjects, conditions)

    if do["features"]:
        from stages.features import run as run_features
        cfg = load_stage_config("features", globals_path=args.config)
        features_df, _ = run_features(cfg, epochs_dict=all_epochs)

    if do["engineering"]:
        from stages.engineering import run as run_engineering
        cfg = load_stage_config("engineering", globals_path=args.config)
        full_df = run_engineering(cfg, features_df=features_df)

    if do["analysis"]:
        from stages.analysis import run as run_analysis
        cfg = load_stage_config("analysis", globals_path=args.config)
        run_analysis(cfg, full_df=full_df)

    print(f"\n{'=' * 60}\nPipeline complete.\n{'=' * 60}")


if __name__ == "__main__":
    main()
