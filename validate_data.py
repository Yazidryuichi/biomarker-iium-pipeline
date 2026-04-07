"""
Data Validation Script
======================
Run this after setting up data to verify everything is in place
before running the pipeline.

Usage:
    python validate_data.py
"""

import os
import sys

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_ROOT)


def validate():
    from utils.io import load_config, discover_subjects

    os.chdir(PIPELINE_ROOT)
    config = load_config()

    print("=" * 50)
    print("DATA VALIDATION")
    print("=" * 50)

    errors = 0
    warnings = 0

    # 1. Check EDF directory
    edf_dir = config["paths"]["edf_dir"]
    if os.path.exists(edf_dir):
        subjects = discover_subjects(edf_dir)
        print(f"\n[OK] EDF directory found: {edf_dir}")
        print(f"     Subjects: {len(subjects)}")

        # Check conditions per subject
        conditions = config["recording"]["conditions"]["primary"]
        for sub_id, conds in subjects.items():
            missing = [c for c in conditions if c not in conds]
            if missing:
                print(f"     [WARN] {sub_id}: missing {missing}")
                warnings += 1

        # Verify one EDF file is readable
        first_sub = list(subjects.keys())[0]
        first_file = list(subjects[first_sub].values())[0]
        try:
            import struct
            with open(first_file, "rb") as f:
                header = f.read(256)
                n_channels = int(header[252:256].decode().strip())
                sfreq_check = float(header[244:252].decode().strip())
            print(f"     [OK] Sample EDF readable: {n_channels} channels")
        except Exception as e:
            print(f"     [ERROR] Cannot read EDF: {e}")
            errors += 1
    else:
        print(f"\n[ERROR] EDF directory not found: {edf_dir}")
        print(f"        Create it and copy EDF files. See data/README.md")
        errors += 1

    # 2. Check behavioral files
    beh_dir = config["paths"]["behavioral_dir"]
    beh_files = {
        "AUFEI-O": os.path.join(beh_dir, "AUFEI-O", "AUFEI-O_Cleaned.xlsx"),
        "Flanker": os.path.join(beh_dir, "Flanker_Test_Pilot.xlsx"),
        "Digit Span": os.path.join(beh_dir, "Digit_Span.xlsx"),
    }

    for name, path in beh_files.items():
        if os.path.exists(path):
            import pandas as pd
            try:
                df = pd.read_excel(path)
                print(f"[OK] {name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"[ERROR] {name}: file exists but cannot read: {e}")
                errors += 1
        else:
            print(f"[ERROR] {name} not found: {path}")
            errors += 1

    # 3. Check dependencies
    print("\n--- Dependencies ---")
    deps = ["mne", "autoreject", "pywt", "sklearn", "xgboost", "shap",
            "scipy", "statsmodels", "yaml", "pandas", "matplotlib"]
    for dep in deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "?")
            print(f"[OK] {dep} {ver}")
        except ImportError:
            print(f"[MISSING] {dep} -- pip install {dep}")
            errors += 1

    # Summary
    print(f"\n{'=' * 50}")
    if errors == 0:
        print(f"VALIDATION PASSED ({warnings} warnings)")
        print("You can run: python run_all.py")
    else:
        print(f"VALIDATION FAILED: {errors} errors, {warnings} warnings")
        print("Fix the errors above before running the pipeline.")
    print(f"{'=' * 50}")

    return errors == 0


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
