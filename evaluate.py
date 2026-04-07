"""
Pipeline Evaluation Harness (IMMUTABLE)
========================================
Autoresearch-inspired scoring: this file must NOT be modified
during optimization iterations. It provides a fixed evaluation
contract against which pipeline changes are measured.

Scores 6 aspects on a 1-10 scale:
  1. Scientific credibility
  2. Mathematical rigor
  3. Data processing quality
  4. Explainability
  5. Reproducibility
  6. AI transparency

Usage:
    python evaluate.py                    # Score current pipeline
    python evaluate.py --json             # Machine-readable output
    python evaluate.py --log results.tsv  # Append to experiment log
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def check_file_exists(path, description):
    """Check if a file exists and return (pass, message)."""
    if os.path.exists(path):
        return True, f"PASS: {description} exists"
    return False, f"FAIL: {description} missing ({path})"


def score_scientific_credibility(pipeline_root):
    """Score scientific credibility (0-10)."""
    score = 0
    details = []

    # Check for hypothesis testing with FDR correction
    s4 = Path(pipeline_root) / "stages" / "stage4_analysis.py"
    if s4.exists():
        content = s4.read_text()
        if "multipletests" in content:
            score += 1
            details.append("FDR correction: YES")
        if "permutation_test_score" in content:
            score += 2
            details.append("Permutation test: YES")
        else:
            details.append("Permutation test: MISSING (-2)")
        if "effect_size" in content or "cohens_d" in content:
            score += 1
            details.append("Effect sizes: YES")
        if "ci_lower" in content and "ci_upper" in content:
            score += 1
            details.append("Confidence intervals: YES")
        if "Spearman" in content and "Pearson" in content:
            score += 1
            details.append("Both correlation methods: YES")
        if "class_weight" in content:
            score += 1
            details.append("Class imbalance handling: YES")
        if "bootstrap" in content.lower() or "boot_means" in content:
            score += 1
            details.append("Bootstrap CI: YES")
        # Check for median-split-inside-CV (no target leakage)
        if "y_continuous" in content or "median_threshold" in content:
            score += 1
            details.append("Median split inside CV: YES")
        else:
            details.append("Median split leakage: PRESENT (-1)")

    # Check for pre-specified hypotheses
    if "H1:" in content and "H2:" in content and "H3:" in content:
        score += 1
        details.append("Pre-specified hypotheses: YES")

    return min(score, 10), details


def score_mathematical_rigor(pipeline_root):
    """Score mathematical rigor (0-10)."""
    score = 0
    details = []

    s2 = Path(pipeline_root) / "stages" / "stage2_features.py"
    if s2.exists():
        content = s2.read_text()
        if "np.trapz" in content:
            score += 2
            details.append("Proper spectral integration (trapz): YES")
        else:
            details.append("Using np.sum instead of np.trapz: WRONG (-2)")
        if "per epoch" in content.lower() or "epoch_cohs" in content:
            score += 2
            details.append("Per-epoch coherence: YES")
        else:
            details.append("Concatenated epoch coherence: WRONG (-2)")
        if "Hjorth" in content:
            score += 1
            details.append("Hjorth parameters: YES")
        if "cwt" in content or "wavelet" in content.lower():
            score += 1
            details.append("Wavelet features: YES")
        if "pac" in content.lower():
            score += 1
            details.append("Phase-amplitude coupling: YES")

    s4 = Path(pipeline_root) / "stages" / "stage4_analysis.py"
    if s4.exists():
        content = s4.read_text()
        if "nested" in content.lower() or "inner_cv" in content:
            score += 2
            details.append("Nested CV for tuning: YES")
        if "SimpleImputer" in content:
            score += 1
            details.append("Proper imputation (not fillna(0)): YES")

    return min(score, 10), details


def score_data_processing(pipeline_root):
    """Score data processing quality (0-10)."""
    score = 0
    details = []

    s1 = Path(pipeline_root) / "stages" / "stage1_cleaning.py"
    if s1.exists():
        content = s1.read_text()
        if "raw_for_ica" in content or "1.0" in content:
            score += 1
            details.append("1Hz high-pass for ICA: YES")
        if "interpolate" in content and "after ICA" in content.lower():
            score += 2
            details.append("Interpolation after ICA: YES")
        elif "interpolate_bads" in content:
            # Check order: is interpolation after ICA?
            interp_pos = content.find("interpolate_bads")
            ica_pos = content.find("run_ica")
            if interp_pos > ica_pos:
                score += 2
                details.append("Interpolation after ICA: YES (by position)")
            else:
                details.append("Interpolation before ICA: WRONG (-2)")
        if "resample" in content:
            score += 1
            details.append("Resampling enforcement: YES")
        if "crop" in content or "trim" in content.lower():
            score += 1
            details.append("Edge trimming after filter: YES")
        if "autoreject" in content.lower():
            score += 1
            details.append("AutoReject for epoch rejection: YES")
        if "qc_report" in content or "qc" in content:
            score += 1
            details.append("QC reporting: YES")

    # Check for config file
    config = Path(pipeline_root) / "configs" / "config.yaml"
    if config.exists():
        score += 1
        details.append("Config-driven pipeline: YES")

    # Check for validate_data.py
    if (Path(pipeline_root) / "validate_data.py").exists():
        score += 1
        details.append("Data validation script: YES")

    # Check for BIDS structure description
    readme = Path(pipeline_root) / "README.md"
    if readme.exists() and "BIDS" in readme.read_text():
        score += 1
        details.append("BIDS awareness: YES")

    return min(score, 10), details


def score_explainability(pipeline_root):
    """Score explainability (0-10)."""
    score = 0
    details = []

    s4 = Path(pipeline_root) / "stages" / "stage4_analysis.py"
    if s4.exists():
        content = s4.read_text()
        if "shap" in content.lower():
            score += 2
            details.append("SHAP analysis: YES")
        if "cv_shap" in content or "all_shap_values" in content:
            score += 2
            details.append("SHAP per-CV-fold: YES")
        else:
            details.append("SHAP on full data only: WEAK (-2)")
        if "shap_std" in content or "shap_cv" in content:
            score += 2
            details.append("SHAP stability analysis: YES")
        if "stable_mask" in content or "feature_votes" in content:
            score += 1
            details.append("CV-stable feature selection: YES")
        if "summary_plot" in content:
            score += 1
            details.append("SHAP summary plot: YES")

    # Check for figures directory
    figs = Path(pipeline_root) / "figures"
    if figs.exists():
        png_count = len(list(figs.glob("*.png")))
        if png_count >= 3:
            score += 2
            details.append(f"Publication figures: {png_count}")
        elif png_count >= 1:
            score += 1

    return min(score, 10), details


def score_reproducibility(pipeline_root):
    """Score reproducibility (0-10)."""
    score = 0
    details = []

    root = Path(pipeline_root)

    checks = [
        (root / "Dockerfile", "Dockerfile"),
        (root / "Makefile", "Makefile"),
        (root / "requirements.txt", "requirements.txt"),
        (root / "configs" / "config.yaml", "Config file"),
        (root / ".gitignore", ".gitignore"),
    ]
    for path, desc in checks:
        if path.exists():
            score += 1
            details.append(f"{desc}: YES")
        else:
            details.append(f"{desc}: MISSING")

    # Check for random seed consistency
    py_files = list(root.glob("stages/*.py")) + [root / "run_all.py"]
    has_seed = any("random_state" in f.read_text() for f in py_files if f.exists())
    if has_seed:
        score += 1
        details.append("Random seeds: YES")

    # Check for logging
    has_logging = any("import logging" in f.read_text() for f in py_files if f.exists())
    if has_logging:
        score += 1
        details.append("Logging framework: YES")
    else:
        details.append("Logging: print() only")

    # Check for evaluate.py
    if (root / "evaluate.py").exists():
        score += 1
        details.append("Evaluation harness: YES")

    # Check for LICENSE
    if (root / "LICENSE").exists():
        score += 1
        details.append("LICENSE: YES")

    # Check git
    if (root / ".git").exists():
        score += 1
        details.append("Git versioning: YES")

    return min(score, 10), details


def score_ai_transparency(pipeline_root):
    """Score AI transparency (0-10)."""
    score = 0
    details = []

    root = Path(pipeline_root)

    # Check README for AI disclosure
    readme = root / "README.md"
    if readme.exists():
        content = readme.read_text().lower()
        if "ai" in content and ("disclosure" in content or "transparency" in content
                                 or "generative" in content or "claude" in content):
            score += 3
            details.append("AI disclosure in README: YES")
        if "audit" in content or "trail" in content:
            score += 1
            details.append("Audit trail mentioned: YES")

    # Check for AI_TRANSPARENCY.md
    ai_file = root / "AI_TRANSPARENCY.md"
    if ai_file.exists():
        score += 3
        details.append("AI_TRANSPARENCY.md: YES")
    else:
        details.append("AI_TRANSPARENCY.md: MISSING")

    # Check for audit trail in code
    py_files = list(root.glob("**/*.py"))
    has_audit = any("audit" in f.read_text().lower() for f in py_files if f.exists())
    if has_audit:
        score += 2
        details.append("Audit trail in code: YES")

    # Check for human verification statements
    has_verify = any("manually verified" in f.read_text().lower() or
                     "human review" in f.read_text().lower()
                     for f in py_files if f.exists())
    if has_verify:
        score += 1
        details.append("Human verification statements: YES")

    return min(score, 10), details


def run_evaluation(pipeline_root, output_json=False, log_file=None):
    """Run full evaluation and report scores."""
    scorers = [
        ("Scientific credibility", score_scientific_credibility),
        ("Mathematical rigor", score_mathematical_rigor),
        ("Data processing", score_data_processing),
        ("Explainability", score_explainability),
        ("Reproducibility", score_reproducibility),
        ("AI transparency", score_ai_transparency),
    ]

    results = {}
    total = 0

    print("=" * 60)
    print("PIPELINE EVALUATION REPORT")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Root: {pipeline_root}")
    print("=" * 60)

    for name, scorer in scorers:
        score, details = scorer(pipeline_root)
        results[name] = {"score": score, "details": details}
        total += score

        status = "PASS" if score >= 8 else "WARN" if score >= 5 else "FAIL"
        print(f"\n{name}: {score}/10 [{status}]")
        for d in details:
            print(f"  {d}")

    avg = total / len(scorers)
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {avg:.1f}/10 (total: {total}/{len(scorers)*10})")
    target_met = all(r["score"] >= 8 for r in results.values())
    print(f"Target (8/10 all aspects): {'MET' if target_met else 'NOT MET'}")
    print(f"{'=' * 60}")

    if output_json:
        print(json.dumps(results, indent=2))

    if log_file:
        # Append to TSV log (autoresearch-style)
        import subprocess
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=pipeline_root, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            commit = "unknown"

        header = "timestamp\tcommit\tavg_score\t" + "\t".join(
            name.replace(" ", "_") for name, _ in scorers
        ) + "\tstatus\n"
        row = f"{datetime.now().isoformat()}\t{commit}\t{avg:.1f}\t" + "\t".join(
            str(results[name]["score"]) for name, _ in scorers
        ) + f"\t{'target_met' if target_met else 'iterating'}\n"

        write_header = not os.path.exists(log_file)
        with open(log_file, "a") as f:
            if write_header:
                f.write(header)
            f.write(row)
        print(f"\nLogged to: {log_file}")

    return results, avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Evaluation Harness")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--log", type=str, default=None, help="Append to TSV log")
    parser.add_argument("--root", type=str, default=".", help="Pipeline root directory")
    args = parser.parse_args()

    run_evaluation(args.root, output_json=args.json, log_file=args.log)
