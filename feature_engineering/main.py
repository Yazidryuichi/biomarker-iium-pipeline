"""
Stage 4: Feature Engineering + Behavioural Merge
================================================
Reads latest feature_building/output and the behavioural workbooks; derives
composites from primitives and merges with subject-level behavioural scores.

Math-derived composites:
  tbr_<cond>_<ch>             theta / beta per channel per condition
  <cond>_tbr_frontal_mean     mean TBR over tbr_channels
  <cond>_faa_<right>_<left>   ln(alpha_right) - ln(alpha_left)
  alpha_reactivity_global     (alpha_EC - alpha_EO) / alpha_EC, mean across ch
  alpha_reactivity_<ch>       same per posterior channel

A priori Tier 1 composites (theory-driven, hypothesis-test set):
  fm_theta_eo            mean z(eo_psd_abs_theta_<ch>) over frontal midline
  posterior_alpha_ec     mean z(ec_psd_abs_alpha_<ch>) over parieto-occipital
  tbr_frontal_eo_log     log(mean theta_frontal / mean beta_frontal) EO

When feature_building emitted psd_periodic_*, periodic-flavoured versions of the
Tier 1 composites are also built (fm_theta_eo_periodic, etc.) — sensitivity
analysis for the age-confound critique.

Behavioural merge: AUFEI (subscale scoring + age from DoB), Flanker (Features
sheet — pre-computed DDM preferred), Digit Span (FW/BW/Total).

Output: output/<ts>/{full_dataset.csv, run_notes.json}
"""
from __future__ import annotations

import json
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

STAGE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = STAGE_DIR.parent
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def load_config():
    with open(STAGE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def make_output_dir():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = STAGE_DIR / "output" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"


def latest_output(root):
    root = resolve(root)
    if not root.is_dir():
        return None
    runs = sorted([d for d in root.iterdir()
                   if d.is_dir() and TS_RE.match(d.name)], reverse=True)
    return runs[0] if runs else None


def _safe_log(x):
    return np.log(np.asarray(x, dtype=float) + 1e-10)


# ──────────────────────────────────────────────────────────────────
# Math-derived composites
# ──────────────────────────────────────────────────────────────────

def add_tbr(df, channels, conditions=("eo", "ec")):
    for cond in conditions:
        chs = []
        for ch in channels:
            theta_col = f"{cond}_psd_abs_theta_{ch}"
            beta_col  = f"{cond}_psd_abs_beta_{ch}"
            if theta_col in df.columns and beta_col in df.columns:
                df[f"{cond}_tbr_{ch}"] = df[theta_col] / df[beta_col].replace(0, np.nan)
                chs.append(df[f"{cond}_tbr_{ch}"])
        if chs:
            df[f"{cond}_tbr_frontal_mean"] = pd.concat(chs, axis=1).mean(axis=1)
    return df


def add_faa(df, left, right, conditions=("eo", "ec")):
    for cond in conditions:
        l = f"{cond}_psd_abs_alpha_{left}"
        r = f"{cond}_psd_abs_alpha_{right}"
        if l in df.columns and r in df.columns:
            df[f"{cond}_faa_{right}_{left}"] = _safe_log(df[r]) - _safe_log(df[l])
    return df


def add_alpha_reactivity(df, posterior_channels):
    eo_cols = [c for c in df.columns if c.startswith("eo_psd_abs_alpha_")]
    ec_cols = [c for c in df.columns if c.startswith("ec_psd_abs_alpha_")]
    if eo_cols and ec_cols:
        eo_mean = df[eo_cols].mean(axis=1)
        ec_mean = df[ec_cols].mean(axis=1)
        df["alpha_reactivity_global"] = (ec_mean - eo_mean) / (ec_mean + 1e-10)
    for ch in posterior_channels:
        eo = f"eo_psd_abs_alpha_{ch}"
        ec = f"ec_psd_abs_alpha_{ch}"
        if eo in df.columns and ec in df.columns:
            df[f"alpha_reactivity_{ch}"] = (df[ec] - df[eo]) / (df[ec] + 1e-10)
    return df


def _zmean(df, cols):
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    sub = df[present]
    sd = sub.std(ddof=0).replace(0, np.nan)
    z = (sub - sub.mean()) / sd
    return z.mean(axis=1)


def add_apriori_tier1(df, fm_chs, pa_chs, tbr_chs, source="abs"):
    """source='abs' uses psd_abs_*; source='periodic' uses psd_periodic_*."""
    suffix = "" if source == "abs" else "_periodic"
    band_key = "psd_abs" if source == "abs" else "psd_periodic"

    fm = _zmean(df, [f"eo_{band_key}_theta_{ch}" for ch in fm_chs])
    if fm is not None:
        df[f"fm_theta_eo{suffix}"] = fm
    pa = _zmean(df, [f"ec_{band_key}_alpha_{ch}" for ch in pa_chs])
    if pa is not None:
        df[f"posterior_alpha_ec{suffix}"] = pa
    theta_cols = [f"eo_{band_key}_theta_{ch}" for ch in tbr_chs
                  if f"eo_{band_key}_theta_{ch}" in df.columns]
    beta_cols  = [f"eo_{band_key}_beta_{ch}" for ch in tbr_chs
                  if f"eo_{band_key}_beta_{ch}"  in df.columns]
    if theta_cols and beta_cols:
        t_mean = df[theta_cols].mean(axis=1)
        b_mean = df[beta_cols].mean(axis=1).replace(0, np.nan)
        df[f"tbr_frontal_eo_log{suffix}"] = _safe_log(t_mean / b_mean)
    return df


def add_all_engineered(df, p):
    df = add_tbr(df, p["tbr_channels"])
    df = add_faa(df, p["faa_left"], p["faa_right"])
    df = add_alpha_reactivity(df, p["posterior_alpha_channels"])
    df = add_apriori_tier1(df, p["apriori_fm_theta_channels"],
                           p["apriori_posterior_alpha_channels"],
                           p["apriori_tbr_frontal_channels"], source="abs")
    has_periodic = any(c.startswith(("eo_psd_periodic_", "ec_psd_periodic_"))
                       for c in df.columns)
    if p.get("build_periodic_composites", True) and has_periodic:
        df = add_apriori_tier1(df, p["apriori_fm_theta_channels"],
                               p["apriori_posterior_alpha_channels"],
                               p["apriori_tbr_frontal_channels"], source="periodic")
    return df, has_periodic


# ──────────────────────────────────────────────────────────────────
# Behavioural loaders + age
# ──────────────────────────────────────────────────────────────────

def compute_age(dob_series, assessment_date):
    dob = pd.to_datetime(dob_series, dayfirst=True)
    days = (pd.Timestamp(assessment_date) - dob).dt.days
    return pd.DataFrame({
        "age_years":  (days / 365.25).round(1),
        "age_months": (days / 30.4375).round(1),
    })


def load_aufei(path, subscales, assessment_date):
    df = pd.read_excel(path)
    out = pd.DataFrame({"ID": df["ID"].astype(str).str.strip()})
    if "Sex" in df.columns:
        out["Sex"] = df["Sex"]
    if "DoB" in df.columns:
        ages = compute_age(df["DoB"], assessment_date)
        out["age_years"] = ages["age_years"]
        out["age_months"] = ages["age_months"]
    subscale_cols = []
    for name, items in subscales.items():
        present = [c for c in items if c in df.columns]
        if present:
            out[f"{name}_score"] = df[present].mean(axis=1)
            subscale_cols.append(f"{name}_score")
    if subscale_cols:
        out["Global_EF"] = out[subscale_cols].mean(axis=1)
    return out


def load_flanker(path):
    """Prefer the Features sheet (pre-computed per-subject metrics including DDM)."""
    xl = pd.ExcelFile(path)
    sheet = "Features" if "Features" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    if "ID" not in df.columns and "participant_id" in df.columns:
        df = df.rename(columns={"participant_id": "ID"})
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str).str.strip()
    # ms→s if needed
    for c in ("rt_mean", "rt_congruent", "rt_incongruent"):
        if c in df.columns and df[c].abs().max() > 1000:
            df[c] = df[c] / 1000.0
    keep = ["ID", "acc_overall", "acc_incongruent",
            "flanker_effect", "rt_mean", "rt_congruent", "rt_incongruent",
            "rt_cv", "rt_iqr", "ies_congruent", "ies_incongruent",
            "ddm_v_congruent", "ddm_a_congruent", "ddm_t0_congruent",
            "ddm_v_incongruent", "ddm_a_incongruent", "ddm_t0_incongruent",
            "ddm_delta_v"]
    return df[[c for c in keep if c in df.columns]].copy()


def load_digit_span(path):
    df = pd.read_excel(path)
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str).str.strip()
    keep = ["ID", "FW_Span", "BW_Span", "Total_Span", "FW_Raw", "BW_Raw", "Total_Raw"]
    return df[[c for c in keep if c in df.columns]].copy()


def merge_behavioural(features_df, beh_dir, p):
    aufei = load_aufei(beh_dir / p["aufei_filename"],
                       p["aufei_subscales"], p["assessment_date"])
    flanker = load_flanker(beh_dir / p["flanker_filename"])
    digit = load_digit_span(beh_dir / p["digit_span_filename"])
    print(f"  AUFEI: {len(aufei)} subjects | Flanker: {len(flanker)} | Digit: {len(digit)}")

    behav = aufei.merge(flanker, on="ID", how="left").merge(digit, on="ID", how="left")

    # Normalise both sides' subject keys to plain strings
    for d, col in [(features_df, "subject_id"), (behav, "ID")]:
        s = pd.to_numeric(d[col], errors="coerce")
        d[col] = np.where(s.notna() & (s == s.round()),
                          s.astype("Int64").astype(str),
                          d[col].astype(str)).astype(str)
        d[col] = d[col].str.strip()

    merged = features_df.merge(behav, left_on="subject_id", right_on="ID", how="inner")
    print(f"  Merged: {merged.shape}, matched subjects: {len(merged)}")

    if len(merged) < int(p["min_matched_n"]):
        raise RuntimeError(
            f"merge_behavioural: only {len(merged)} matched (< min_matched_n="
            f"{p['min_matched_n']}). Check ID formats.")
    missing = set(features_df["subject_id"]) - set(merged["subject_id"])
    if missing:
        print(f"  WARN: features without behavioural match: {sorted(missing)}")
    extra = set(behav["ID"]) - set(merged["subject_id"])
    if extra:
        print(f"  WARN: behavioural without feature match: {sorted(extra)}")
    return merged


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    p = cfg["params"]

    fb_dir = latest_output(cfg["paths"]["feature_building_root"])
    if fb_dir is None:
        raise FileNotFoundError(
            f"No feature_building output under {cfg['paths']['feature_building_root']}. "
            f"Run `python feature_building/main.py` first.")
    print(f"Reading features from: {fb_dir}")

    out_dir = make_output_dir()
    print(f"Output: {out_dir}")

    df = pd.read_csv(fb_dir / "features.csv")
    print(f"  features.csv: {df.shape}")

    n_before = df.shape[1]
    df, has_periodic = add_all_engineered(df, p)
    n_added = df.shape[1] - n_before
    print(f"  Engineered features added: {n_added} "
          f"(periodic versions: {'yes' if has_periodic else 'no'})")

    beh_dir = resolve(cfg["paths"]["behavioral_dir"])
    full = merge_behavioural(df, beh_dir, p)

    full.to_csv(out_dir / "full_dataset.csv", index=False)
    print(f"\n  full_dataset.csv: {full.shape}")

    notes = {
        "stage": "feature_engineering",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "feature_building_consumed": str(fb_dir),
        "behavioral_dir": str(beh_dir),
        "n_subjects_merged": int(full.shape[0]),
        "n_columns_total": int(full.shape[1]),
        "n_engineered_added": int(n_added),
        "periodic_composites_built": bool(has_periodic),
        "outputs": ["full_dataset.csv"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)


if __name__ == "__main__":
    main()
