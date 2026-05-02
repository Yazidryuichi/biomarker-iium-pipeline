"""
Stage 3: Feature Engineering + Behavioral Merge
================================================
Derives composite features from the raw primitives produced by Stage 2
(features.py), merges with behavioral measures, and creates per-target
binary group columns.

Engineered features (math-derived from raw band powers):
  - tbr_<ch>           : theta / beta power ratio per channel
  - tbr_frontal_mean   : mean TBR across frontal/central channels
  - faa_F4_F3          : ln(alpha_F4) - ln(alpha_F3) — frontal alpha asymmetry
  - alpha_reactivity_global : (alpha_EC - alpha_EO) / alpha_EC, mean across channels
  - alpha_reactivity_<ch>   : same per posterior channel (O1, O2, Pz)

These are computed post-hoc from features.csv (no re-reading raw EEG)
because Stage 2 already integrated band powers via np.trapz. Computing
ratios directly from `psd_abs_<band>_<ch>` is mathematically identical
to integrating bandpassed PSDs from the raw signal.
"""

import os

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────
# Engineered features — math-derived from raw band powers
# ──────────────────────────────────────────────────────────────────

def _safe_log(x):
    return np.log(np.asarray(x, dtype=float) + 1e-10)


def add_tbr(df, tbr_channels, conditions=("eo", "ec")):
    """
    Theta/Beta Ratio per channel per condition + frontal-mean.
    Reads pre-integrated band powers (psd_abs_theta_<ch>, psd_abs_beta_<ch>).
    """
    for cond in conditions:
        ch_tbrs = []
        for ch in tbr_channels:
            theta_col = f"{cond}_psd_abs_theta_{ch}"
            beta_col  = f"{cond}_psd_abs_beta_{ch}"
            if theta_col not in df.columns or beta_col not in df.columns:
                continue
            tbr = df[theta_col] / df[beta_col].replace(0, np.nan)
            df[f"{cond}_tbr_{ch}"] = tbr
            ch_tbrs.append(tbr)
        if ch_tbrs:
            df[f"{cond}_tbr_frontal_mean"] = pd.concat(ch_tbrs, axis=1).mean(axis=1)
    return df


def add_faa(df, faa_left, faa_right, conditions=("eo", "ec")):
    """Frontal Alpha Asymmetry: ln(alpha_right) - ln(alpha_left)."""
    for cond in conditions:
        l_col = f"{cond}_psd_abs_alpha_{faa_left}"
        r_col = f"{cond}_psd_abs_alpha_{faa_right}"
        if l_col in df.columns and r_col in df.columns:
            df[f"{cond}_faa_{faa_right}_{faa_left}"] = (
                _safe_log(df[r_col]) - _safe_log(df[l_col])
            )
    return df


def add_alpha_reactivity(df, posterior_channels=("O1", "O2", "Pz")):
    """
    Alpha reactivity: change in alpha power from Eyes-Closed to Eyes-Open.
    Positive value = normal desynchronization (alpha drops on eyes open).
    """
    # Global: mean alpha across all available channels per condition
    eo_alpha_cols = [c for c in df.columns if c.startswith("eo_psd_abs_alpha_")]
    ec_alpha_cols = [c for c in df.columns if c.startswith("ec_psd_abs_alpha_")]

    if eo_alpha_cols and ec_alpha_cols:
        eo_mean = df[eo_alpha_cols].mean(axis=1)
        ec_mean = df[ec_alpha_cols].mean(axis=1)
        df["alpha_reactivity_global"] = (ec_mean - eo_mean) / (ec_mean + 1e-10)

    # Per-posterior-channel reactivity
    for ch in posterior_channels:
        eo_col = f"eo_psd_abs_alpha_{ch}"
        ec_col = f"ec_psd_abs_alpha_{ch}"
        if eo_col in df.columns and ec_col in df.columns:
            df[f"alpha_reactivity_{ch}"] = (
                (df[ec_col] - df[eo_col]) / (df[ec_col] + 1e-10)
            )

    return df


def add_engineered_features(df, config):
    """Run all engineered-feature computations on the features dataframe."""
    eng = config["engineering"]
    df = add_tbr(df, eng["tbr_channels"])
    df = add_faa(df, eng["faa_left"], eng["faa_right"])
    df = add_alpha_reactivity(df, posterior_channels=tuple(
        eng.get("posterior_alpha_channels", ("O1", "O2", "Pz"))
    ))
    return df


# ──────────────────────────────────────────────────────────────────
# Behavioral merge + binarization
# ──────────────────────────────────────────────────────────────────

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


def merge_behavioral(features_df, config):
    """Merge raw QEEG features with behavioral measures."""
    from utils.io import load_aufei, load_flanker, load_digit_span

    beh_dir = config["paths"]["behavioral_dir"]
    aufei_path = os.path.join(beh_dir, "Behavioral", "AUFEI-O_Cleaned.xlsx")
    flanker_path = os.path.join(beh_dir, "Behavioral", "Flanker_Test_Pilot.xlsx")
    digit_path = os.path.join(beh_dir, "Behavioral", "Digit_Span.xlsx")

    aufei = load_aufei(aufei_path)
    flanker = load_flanker(flanker_path)
    digit = load_digit_span(digit_path)

    print(f"  AUFEI: {len(aufei)} subjects")
    print(f"  Flanker: {len(flanker)} subjects")
    print(f"  Digit Span: {len(digit)} subjects")

    assessment_date = config["engineering"].get("assessment_date", "2026-03-09")
    aufei["age_years"] = compute_age(aufei["DoB"], assessment_date=assessment_date)

    aufei_keep = ["ID", "Sex", "age_years",
                  "WM_score", "IC_score", "CF_score", "P_score", "SF_score", "Global_EF"]
    behavioral = aufei[[c for c in aufei_keep if c in aufei.columns]].copy()

    flanker_cols = ["ID", "flanker_effect", "acc_overall", "rt_mean",
                    "ddm_v", "ddm_a", "ddm_t", "ddm_delta_v"]
    flanker_subset = flanker[[c for c in flanker_cols if c in flanker.columns]].copy()
    behavioral = behavioral.merge(flanker_subset, on="ID", how="left")

    digit_cols = ["ID", "FW_Span", "BW_Span", "Total_Span", "FW_Raw", "BW_Raw"]
    digit_subset = digit[[c for c in digit_cols if c in digit.columns]].copy()
    behavioral = behavioral.merge(digit_subset, on="ID", how="left")

    print(f"  Behavioral merged: {behavioral.shape}")

    full_df = features_df.merge(
        behavioral, left_on="subject_id", right_on="ID", how="inner",
    )

    print(f"  Full dataset after merge: {full_df.shape}")
    print(f"  Subjects matched: {len(full_df)}")

    if len(full_df) < len(features_df):
        missing = set(features_df["subject_id"]) - set(full_df["subject_id"])
        print(f"  WARNING: {len(missing)} subjects could not be matched: {missing}")

    return full_df


def binarize_targets(df):
    """
    Materialize <target>_group for every known continuous behavioral measure.
    Downstream stages pick a target via ml.targets without re-running this stage.
    """
    target_candidates = [
        "Global_EF", "WM_score", "IC_score", "CF_score", "P_score", "SF_score",
        "flanker_effect", "acc_overall", "rt_mean",
        "ddm_v", "ddm_a", "ddm_t", "ddm_delta_v",
        "FW_Span", "BW_Span", "Total_Span",
    ]
    print(f"\n  Class balances (median split):")
    for col in target_candidates:
        if col not in df.columns:
            continue
        scores = pd.to_numeric(df[col], errors="coerce")
        if scores.notna().sum() < 4:
            continue
        df[f"{col}_group"] = create_ef_groups(scores)
        low = int((df[f"{col}_group"] == 0).sum())
        high = int((df[f"{col}_group"] == 1).sum())
        print(f"    {col:18s}  Low={low:3d}  High={high:3d}")

    # Legacy aliases (notebooks still use ef_group_global/wm/ic)
    if "Global_EF_group" in df.columns:
        df["ef_group_global"] = df["Global_EF_group"]
    if "WM_score_group" in df.columns:
        df["ef_group_wm"] = df["WM_score_group"]
    if "IC_score_group" in df.columns:
        df["ef_group_ic"] = df["IC_score_group"]

    return df


# ──────────────────────────────────────────────────────────────────
# Loader + orchestrator
# ──────────────────────────────────────────────────────────────────

def load_full_dataset(config, stage_dir=None):
    """
    Load saved full_dataset.csv. Auto-resolves the latest engineering
    directory if ``stage_dir`` is not given.
    """
    from utils.io import latest_stage_dir

    if stage_dir is None:
        stage_dir = latest_stage_dir(config, "engineering")
        if stage_dir is None:
            raise FileNotFoundError(
                "No engineering output found. "
                "Run `python pipeline.py --engineering` first."
            )
    path = os.path.join(stage_dir, "full_dataset.csv")
    df = pd.read_csv(path)
    print(f"  Loaded full dataset: {df.shape} from {path}")
    return df


def run(config, features_df=None):
    """
    Run feature engineering + behavioral merge + target binarization.

    Writes full_dataset.csv + run_notes.json into ``config['output_dir']``
    (created by ``load_stage_config('engineering')``). Auto-loads features
    from ``config['input_dir']`` (latest features run) when ``features_df``
    is None.

    Returns:
        full_df: features (raw + engineered) merged with behavioral labels.
    """
    from utils.io import write_stage_notes
    from stages.features import load_features

    print("\n" + "=" * 60)
    print("STAGE 3: Feature Engineering + Behavioral Merge")
    print("=" * 60)

    features_input = config.get("input_dir")
    if features_df is None:
        if features_input is None:
            raise FileNotFoundError(
                "No features output found. Run `python pipeline.py --features` first."
            )
        features_df = load_features(config, stage_dir=features_input)

    # 1. Add engineered features (TBR, FAA, alpha reactivity)
    n_before = features_df.shape[1]
    features_df = add_engineered_features(features_df, config)
    n_added = features_df.shape[1] - n_before
    print(f"  Engineered features added: {n_added}")

    # 2. Merge with behavioral measures
    full_df = merge_behavioral(features_df, config)

    # 3. Binarize targets
    full_df = binarize_targets(full_df)

    # 4. Save
    stage_dir = config["output_dir"]
    print(f"  Engineering output dir: {stage_dir}")
    full_path = os.path.join(stage_dir, "full_dataset.csv")
    full_df.to_csv(full_path, index=False)
    print(f"\n  Full dataset saved: {full_path}")

    write_stage_notes(stage_dir, {
        "stage": "engineering",
        "input_features_dir": features_input,
        "n_subjects": int(full_df.shape[0]),
        "n_columns": int(full_df.shape[1]),
        "n_engineered_features": int(n_added),
        "outputs": ["full_dataset.csv"],
    })

    return full_df
