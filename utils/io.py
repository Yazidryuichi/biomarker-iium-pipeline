"""
I/O utilities for loading EDF files, behavioral data, and config.
"""

import os
import re
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path="configs/config.yaml"):
    """Load pipeline configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_subjects(edf_dir):
    """
    Discover all subjects and their EDF files.
    Returns dict: {subject_id: {condition: filepath}}
    """
    subjects = {}
    for d in sorted(os.listdir(edf_dir)):
        dpath = os.path.join(edf_dir, d)
        if not os.path.isdir(dpath) or not d.startswith("D"):
            continue

        subject_id = d
        subjects[subject_id] = {}

        for f in sorted(os.listdir(dpath)):
            if not f.endswith(".edf"):
                continue

            # Extract condition from filename
            # Pattern: X_M_X_Name_IGS_Eyes_Open.edf or X_M_X_Name_IGS_1_Happy.edf
            # Handle case variants: IGS vs igs, 2_IGS prefix
            fname = f.replace(".edf", "")

            # Normalize: find the condition part after IGS_ or igs_
            match = re.search(r"[Ii][Gg][Ss]_(.+)$", fname)
            if not match:
                continue

            condition_raw = match.group(1)

            # Normalize condition names
            # Handle "2_IGS_" prefix case (subject D0000813 re-recording)
            condition_raw = re.sub(r"^2_IGS_", "", condition_raw)

            # Map to standard condition names
            condition_map = {
                "Eyes_Open": "Eyes_Open",
                "Eyes_Closed": "Eyes_Closed",
                "1_Happy": "1_Happy",
                "2_Calm": "2_Calm",
                "3_Sad": "3_Sad",
                "4_Scare": "4_Scare",
                "5_Wash": "5_Wash",
            }

            condition = condition_map.get(condition_raw, condition_raw)
            subjects[subject_id][condition] = os.path.join(dpath, f)

    return subjects


def load_aufei(filepath):
    """
    Load and score AUFEI-O data.
    Returns DataFrame with subject ID, domain scores, and Global Score.
    """
    df = pd.read_excel(filepath)

    # Identify columns
    wm_cols = [c for c in df.columns if c.startswith("WM")]
    ic_cols = [c for c in df.columns if c.startswith("IC")]
    cf_cols = [c for c in df.columns if c.startswith("CF")]
    p_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    sf_cols = [c for c in df.columns if c.startswith("SF")]

    result = pd.DataFrame()
    result["ID"] = df["ID"]
    result["Sex"] = df["Sex"]
    result["DoB"] = pd.to_datetime(df["DoB"])

    # Domain scores (mean of items)
    result["WM_score"] = df[wm_cols].mean(axis=1)
    result["IC_score"] = df[ic_cols].mean(axis=1)
    result["CF_score"] = df[cf_cols].mean(axis=1) if cf_cols else np.nan
    result["P_score"] = df[p_cols].mean(axis=1) if p_cols else np.nan
    result["SF_score"] = df[sf_cols].mean(axis=1) if sf_cols else np.nan

    # Global Score: mean of WM + IC (core EF per Diamond 2013)
    result["Global_EF"] = result[["WM_score", "IC_score"]].mean(axis=1)

    return result


def load_flanker(filepath):
    """Load Flanker Test pilot data."""
    df = pd.read_excel(filepath)
    # Key columns: ID, flanker_effect, acc_overall, rt_mean, ddm_delta_v
    # Values appear to be multiplied by 10000 — normalize
    cols_to_normalize = [
        "acc_overall", "acc_incongruent", "flanker_effect",
        "rt_mean", "rt_incongruent", "rt_congruent", "rt_cv",
    ]
    for col in cols_to_normalize:
        if col in df.columns:
            # Check if values are in unusual scale
            if df[col].abs().max() > 100000:
                # Likely milliseconds * 10000 or similar encoding
                # Keep raw for now, normalize in analysis
                pass
    return df


def load_digit_span(filepath):
    """Load Digit Span data."""
    return pd.read_excel(filepath)


def get_subject_sex(filename):
    """Extract sex from filename: X_M_X or X_F_X pattern."""
    match = re.search(r"_([MF])_", filename)
    return match.group(1) if match else None
