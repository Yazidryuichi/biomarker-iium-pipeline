"""
Stage 3: Behavioral Data Merge
================================
Merges QEEG features with behavioral measures (AUFEI, Flanker, Digit Span)
and creates classification targets (high/low EF groups).
"""

import os

import numpy as np
import pandas as pd


def compute_age(dob_series, assessment_date="2026-03-09"):
    """Compute age in years from date of birth."""
    dob = pd.to_datetime(dob_series)
    ref = pd.Timestamp(assessment_date)
    return ((ref - dob).dt.days / 365.25).round(1)


def create_ef_groups(scores, method="median"):
    """
    Split subjects into high/low EF groups for classification.
    Uses strict > for ties (ties go to low group) to prevent
    class imbalance with integer-scale questionnaire scores.
    """
    if method == "median":
        threshold = scores.median()
        groups = (scores > threshold).astype(int)
        # Check balance — warn if worse than 40/60
        balance = groups.mean()
        if balance < 0.35 or balance > 0.65:
            print(f"    WARNING: Class imbalance {1-balance:.0%}/{balance:.0%}. "
                  f"Consider tertile split.")
        return groups
    elif method == "tertile":
        low_t = scores.quantile(0.33)
        high_t = scores.quantile(0.67)
        groups = pd.Series(np.nan, index=scores.index)
        groups[scores <= low_t] = 0
        groups[scores >= high_t] = 1
        return groups
    else:
        raise ValueError(f"Unknown method: {method}")


def run_stage3(config, features_df):
    """
    Merge EEG features with behavioral data.

    Returns:
        full_df: merged DataFrame with features + labels
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Behavioral Data Merge")
    print("=" * 60)

    beh_dir = config["paths"]["behavioral_dir"]

    # Load behavioral data
    from utils.io import load_aufei, load_flanker, load_digit_span

    aufei_path = os.path.join(beh_dir, "AUFEI-O", "AUFEI-O_Cleaned.xlsx")
    flanker_path = os.path.join(beh_dir, "Flanker_Test_Pilot.xlsx")
    digit_path = os.path.join(beh_dir, "Digit_Span.xlsx")

    aufei = load_aufei(aufei_path)
    flanker = load_flanker(flanker_path)
    digit = load_digit_span(digit_path)

    print(f"  AUFEI: {len(aufei)} subjects")
    print(f"  Flanker: {len(flanker)} subjects")
    print(f"  Digit Span: {len(digit)} subjects")

    # Compute age
    aufei["age_years"] = compute_age(aufei["DoB"])

    # Merge behavioral measures
    behavioral = aufei[["ID", "Sex", "age_years", "WM_score", "IC_score", "Global_EF"]].copy()

    # Add Flanker
    flanker_cols = ["ID", "flanker_effect", "acc_overall", "rt_mean"]
    flanker_subset = flanker[[c for c in flanker_cols if c in flanker.columns]].copy()
    behavioral = behavioral.merge(flanker_subset, on="ID", how="left")

    # Add Digit Span
    digit_cols = ["ID", "FW_Span", "BW_Span", "Total_Span", "FW_Raw", "BW_Raw"]
    digit_subset = digit[[c for c in digit_cols if c in digit.columns]].copy()
    behavioral = behavioral.merge(digit_subset, on="ID", how="left")

    print(f"  Behavioral merged: {behavioral.shape}")

    # Merge with EEG features
    full_df = features_df.merge(
        behavioral,
        left_on="subject_id",
        right_on="ID",
        how="inner",
    )

    print(f"  Full dataset after merge: {full_df.shape}")
    print(f"  Subjects matched: {len(full_df)}")

    if len(full_df) < len(features_df):
        missing = set(features_df["subject_id"]) - set(full_df["subject_id"])
        print(f"  WARNING: {len(missing)} subjects could not be matched: {missing}")

    # Create classification targets
    full_df["ef_group_global"] = create_ef_groups(full_df["Global_EF"])
    full_df["ef_group_wm"] = create_ef_groups(full_df["WM_score"])
    full_df["ef_group_ic"] = create_ef_groups(full_df["IC_score"])

    print(f"\n  Class balance (Global EF):")
    print(f"    Low:  {(full_df['ef_group_global'] == 0).sum()}")
    print(f"    High: {(full_df['ef_group_global'] == 1).sum()}")

    # Save
    output_dir = config["paths"]["output_dir"]
    full_path = os.path.join(output_dir, "full_dataset.csv")
    full_df.to_csv(full_path, index=False)
    print(f"\n  Full dataset saved: {full_path}")

    return full_df
