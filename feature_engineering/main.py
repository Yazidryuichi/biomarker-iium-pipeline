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


def add_apriori_union(df, fc_channels=("Fz", "Cz"),
                      po_channels=("O1", "O2", "Pz")):
    """Build the 9 region-collapsed a priori union composites.

    Channel collapsing rationale (15-channel Mitsar montage):
      FC = {Fz, Cz}              FCz absent in the file
      PO = {O1, O2, Pz}          POz absent in the file

    Composites produced (raw region-mean; not z-scored, since predictors are
    standardised inside the model pipeline at the analysis stage):

      rel_theta_FC                cognitive control (FMtheta proxy), EO
      rel_beta_FC                 cognitive control, EO
      rel_alpha_PO                arousal / attention, EO
      aperiodic_exponent_FC       E/I balance, maturation index, EO
      aperiodic_exponent_PO       (same, posterior)
      aperiodic_offset_FC         broadband offset, EO
      aperiodic_offset_PO         (same, posterior)
      paf_PO                      processing speed / WM index, EC
      alpha_reactivity_PO         arousal regulation (EC -> EO normalised)

    Intentionally excluded from Tier-1 union (per design spec):
      TBR   collinear with rel_theta + rel_beta; correlate-only in tables
      FAA   affective / approach-withdrawal; off-construct for cold-EF rest
      Coh   fragile at short avg-ref epochs; deferred to method-refinement
    """
    def _mean(cols):
        present = [c for c in cols if c in df.columns]
        if not present:
            return None
        return df[present].mean(axis=1)

    # Relative band power (EO)
    for band in ("theta", "beta"):
        s = _mean([f"eo_psd_rel_{band}_{ch}" for ch in fc_channels])
        if s is not None:
            df[f"rel_{band}_FC"] = s
    s = _mean([f"eo_psd_rel_alpha_{ch}" for ch in po_channels])
    if s is not None:
        df["rel_alpha_PO"] = s

    # Aperiodic exponent + offset, FC and PO (EO)
    for region, chs in (("FC", fc_channels), ("PO", po_channels)):
        for param in ("exponent", "offset"):
            s = _mean([f"eo_aperiodic_{param}_{ch}" for ch in chs])
            if s is not None:
                df[f"aperiodic_{param}_{region}"] = s

    # PAF posterior, EC
    s = _mean([f"ec_paf_{ch}" for ch in po_channels])
    if s is not None:
        df["paf_PO"] = s

    # Alpha reactivity PO (EC normalised by EC mean)
    eo_cols = [f"eo_psd_abs_alpha_{ch}" for ch in po_channels
               if f"eo_psd_abs_alpha_{ch}" in df.columns]
    ec_cols = [f"ec_psd_abs_alpha_{ch}" for ch in po_channels
               if f"ec_psd_abs_alpha_{ch}" in df.columns]
    if eo_cols and ec_cols:
        eo_mean = df[eo_cols].mean(axis=1)
        ec_mean = df[ec_cols].mean(axis=1)
        df["alpha_reactivity_PO"] = (ec_mean - eo_mean) / (ec_mean + 1e-10)

    return df


APRIORI_UNION_COLS = [
    "rel_theta_FC", "rel_beta_FC", "rel_alpha_PO",
    "aperiodic_exponent_FC", "aperiodic_exponent_PO",
    "aperiodic_offset_FC", "aperiodic_offset_PO",
    "paf_PO", "alpha_reactivity_PO",
]


def curate_features(df, variance_threshold=1e-6, collinearity_threshold=0.95,
                    protect_cols=None):
    """Unsupervised feature curation. Safe pre-CV (no target info used).

    Operates only on EEG feature columns (prefix eo_/ec_ or a known
    composite name). Behavioural / demographic columns are not touched.

    Drops in two passes:
      1. Near-constant: var < variance_threshold
      2. Collinear: pairwise |r| > collinearity_threshold; first occurrence
         in the column order is kept, later ones dropped (greedy).

    `protect_cols` is a list of column names that MUST survive curation.
    The a priori union composites (APRIORI_UNION_COLS) are always
    protected — they are theory-driven and pre-registered; dropping them
    silently for collinearity would defeat the confirmatory analysis.
    They are also placed at the FRONT of the feature column order before
    the collinearity pass, so when they happen to be collinear with a
    non-protected feature, the non-protected feature is the one dropped.

    Returns (df_after, keep_features, report_dict).
    """
    known_composites = set(APRIORI_UNION_COLS) | {
        "alpha_reactivity_global", "eo_tbr_frontal_mean", "ec_tbr_frontal_mean",
        "eo_faa_F4_F3", "ec_faa_F4_F3",
    }
    feature_cols = [c for c in df.columns
                    if c.startswith(("eo_", "ec_")) or c in known_composites
                    or c.startswith(("alpha_reactivity_",
                                     "rel_", "aperiodic_", "paf_"))]
    protected = set(protect_cols or []) | set(APRIORI_UNION_COLS)
    protected = {c for c in protected if c in feature_cols}

    # Place protected columns first so the greedy collinearity pass keeps
    # them and drops the non-protected siblings instead.
    ordered = [c for c in feature_cols if c in protected] + \
              [c for c in feature_cols if c not in protected]

    X = df[ordered].apply(pd.to_numeric, errors="coerce")

    var = X.var(ddof=0)
    low_var = [c for c in var[(var < variance_threshold) | var.isna()].index
               if c not in protected]

    X_filt = X.drop(columns=low_var)
    corr = X_filt.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = []
    for c in upper.columns:
        if c in protected:
            continue
        if any(upper[c] > collinearity_threshold):
            high_corr.append(c)

    drop_set = set(low_var) | set(high_corr)
    keep_features = [c for c in ordered if c not in drop_set]
    non_features = [c for c in df.columns if c not in feature_cols]

    df_after = df[keep_features + non_features].copy()
    report = {
        "n_features_before": len(feature_cols),
        "n_features_after": len(keep_features),
        "variance_threshold": variance_threshold,
        "collinearity_threshold": collinearity_threshold,
        "n_protected": len(protected),
        "protected_columns_kept": sorted(protected),
        "dropped_low_variance": low_var,
        "dropped_collinear": high_corr,
    }
    return df_after, keep_features, report


def add_all_engineered(df, p):
    df = add_tbr(df, p["tbr_channels"])
    df = add_faa(df, p["faa_left"], p["faa_right"])
    df = add_alpha_reactivity(df, p["posterior_alpha_channels"])

    # A priori UNION (9 region-collapsed composites, primary feature set)
    df = add_apriori_union(
        df,
        fc_channels=tuple(p.get("apriori_union_fc_channels", ("Fz", "Cz"))),
        po_channels=tuple(p.get("apriori_union_po_channels",
                                ("O1", "O2", "Pz"))),
    )

    # Legacy 3-feature Tier-1 (kept as comparable feature set for sensitivity)
    df = add_apriori_tier1(df, p["apriori_fm_theta_channels"],
                           p["apriori_posterior_alpha_channels"],
                           p["apriori_tbr_frontal_channels"], source="abs")
    has_periodic = any(c.startswith(("eo_psd_periodic_", "ec_psd_periodic_"))
                       for c in df.columns)
    if p.get("build_periodic_composites", True) and has_periodic:
        df = add_apriori_tier1(df, p["apriori_fm_theta_channels"],
                               p["apriori_posterior_alpha_channels"],
                               p["apriori_tbr_frontal_channels"],
                               source="periodic")
    return df, has_periodic


def build_theory_mapping(targets):
    """Per-target directional prior table for the 9 union composites.

    Emitted as an annotation CSV alongside full_dataset.csv. The analysis
    stage uses this to set one-sided directional p-values where applicable
    in the confirmatory OLS path.
    """
    # Per-composite construct + sign predictions. Sign convention:
    #   for rt_cv (lower = better attentional stability): predictor expected
    #     to correlate negative if it indexes BETTER cognitive function
    #   for ddm_v_incongruent (higher = better): positive if it indexes
    #     better cognitive function
    #   for BW_Span (higher = better WM): positive for cognitive predictors
    #   for Global_EF (higher = better EF): positive for cognitive predictors
    rows = []
    rules = [
        # composite,            construct,                                 v_inc,    BW,       Global_EF, rt_cv
        ("rel_theta_FC",        "frontal theta (cognitive control)",       "positive","positive","positive","negative"),
        ("rel_beta_FC",         "frontal beta (vigilance / readiness)",    "positive","positive","positive","negative"),
        ("rel_alpha_PO",        "posterior alpha (attentional reserve)",   "positive","positive","positive","negative"),
        ("aperiodic_exponent_FC","frontal 1/f slope (E/I; maturation)",     "positive","positive","positive","negative"),
        ("aperiodic_exponent_PO","posterior 1/f slope (E/I; maturation)",   "positive","positive","positive","negative"),
        ("aperiodic_offset_FC", "frontal broadband offset",                 "two_sided","two_sided","two_sided","two_sided"),
        ("aperiodic_offset_PO", "posterior broadband offset",               "two_sided","two_sided","two_sided","two_sided"),
        ("paf_PO",              "individual alpha frequency (processing)",  "positive","positive","positive","negative"),
        ("alpha_reactivity_PO", "alpha system reactivity",                  "positive","positive","positive","negative"),
    ]
    for comp, construct, vi, bw, gef, rcv in rules:
        rows.append({
            "composite": comp, "construct": construct,
            "dir_ddm_v_incongruent": vi,
            "dir_BW_Span": bw,
            "dir_Global_EF": gef,
            "dir_rt_cv": rcv,
        })
    return pd.DataFrame(rows)


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

    # Verify union composites built
    union_built = [c for c in APRIORI_UNION_COLS if c in df.columns]
    print(f"  A priori union composites built: {len(union_built)}/9 "
          f"({union_built})")

    beh_dir = resolve(cfg["paths"]["behavioral_dir"])
    full = merge_behavioural(df, beh_dir, p)

    # Unsupervised feature curation (safe pre-CV: no target info used)
    curation_cfg = p.get("feature_selection_schemes", {}).get(
        "unsupervised_curation", {})
    curation_enable = bool(curation_cfg.get("enable", True))
    if curation_enable:
        full, curated_features, curation_report = curate_features(
            full,
            variance_threshold=float(curation_cfg.get(
                "variance_threshold", 1e-6)),
            collinearity_threshold=float(curation_cfg.get(
                "collinearity_threshold", 0.95)),
        )
        with open(out_dir / "feature_curation_report.json", "w") as f:
            json.dump(curation_report, f, indent=2, default=str)
        print(f"  Curation: {curation_report['n_features_before']} -> "
              f"{curation_report['n_features_after']} features kept "
              f"(dropped {len(curation_report['dropped_low_variance'])} "
              f"low-var + {len(curation_report['dropped_collinear'])} "
              f"collinear)")
    else:
        curated_features = [c for c in full.columns
                            if c.startswith(("eo_", "ec_"))
                            or c in APRIORI_UNION_COLS]
        curation_report = {"enable": False}

    # Resolve the three feature_selection_schemes' column lists
    theory_members = [c for c in APRIORI_UNION_COLS if c in full.columns]
    data_driven_pool = list(curated_features)
    hybrid_pool = [c for c in data_driven_pool if c not in set(theory_members)]
    schemes_resolved = {
        "theory": {
            "members": theory_members,
            "selector_in_analysis": "direct_fit",
            "n_features": len(theory_members),
        },
        "data_driven": {
            "members": data_driven_pool,
            "selector_in_analysis": "l1_embedded_or_kbest",
            "n_features": len(data_driven_pool),
        },
        "hybrid": {
            "apriori_members": theory_members,
            "rest_pool": hybrid_pool,
            "selector_in_analysis":
                "apriori_direct + l1_on_rest_with_kbest_cap",
            "n_apriori": len(theory_members),
            "n_rest_pool": len(hybrid_pool),
        },
    }
    with open(out_dir / "feature_selection_schemes_resolved.json", "w") as f:
        json.dump(schemes_resolved, f, indent=2, default=str)

    # Theoretical mapping annotation (for paper / report)
    targets_for_mapping = (p.get("targets_for_theory_mapping")
                           or ["rt_cv", "ddm_v_incongruent",
                               "BW_Span", "Global_EF"])
    theory_map = build_theory_mapping(targets_for_mapping)
    theory_map.to_csv(out_dir / "apriori_theory_mapping.csv", index=False)

    full.to_csv(out_dir / "full_dataset.csv", index=False)
    print(f"\n  full_dataset.csv: {full.shape}")
    print(f"  Schemes: theory={len(theory_members)} | "
          f"data_driven_pool={len(data_driven_pool)} | "
          f"hybrid_rest_pool={len(hybrid_pool)}")

    notes = {
        "stage": "feature_engineering",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "feature_building_consumed": str(fb_dir),
        "behavioral_dir": str(beh_dir),
        "n_subjects_merged": int(full.shape[0]),
        "n_columns_total": int(full.shape[1]),
        "n_engineered_added": int(n_added),
        "n_apriori_union_built": len(theory_members),
        "periodic_composites_built": bool(has_periodic),
        "curation_enabled": bool(curation_enable),
        "curation_n_after": (curation_report.get("n_features_after")
                             if curation_enable else None),
        "schemes_n": {k: (v.get("n_features") or v.get("n_apriori"))
                      for k, v in schemes_resolved.items()},
        "outputs": ["full_dataset.csv",
                    "feature_curation_report.json",
                    "feature_selection_schemes_resolved.json",
                    "apriori_theory_mapping.csv"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)


if __name__ == "__main__":
    main()
