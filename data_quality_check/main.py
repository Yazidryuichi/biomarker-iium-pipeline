"""
Data Quality Check (pilot reporting)
====================================
Consolidates QC findings across preprocessing + validation + behavioural raw
into a headline-first report. The report leads with construct-validity and
estimability findings before showing per-measure descriptives, because at
this pilot N the construct-level problems dominate any psychometric detail.

Reads:
  preprocessing/output/<latest>/qc.json
  validation/output/<latest>/{aufei, flanker, digit_span}_reliability.csv +
                              reliability_summary.json +
                              preprocessing_pass_summary.json
  feature_engineering/output/<latest>/full_dataset.csv  (merged N, age, sex)
  analysis/output/<latest>/<target>/<feature_set>/summary.json  (analysis N)
  data/Behavioral/{AUFEI-O_Cleaned, Flanker_Test_Pilot, Digit_Span}.xlsx

Writes:
  output/<ts>/report.md                       headline-first consolidated report
  output/<ts>/n_reconciliation.csv            behavioural -> analysis pipeline N
  output/<ts>/eeg_quality_per_recording.csv
  output/<ts>/eeg_quality_summary.csv
  output/<ts>/aufei_descriptives.csv          mean/SD/range + variance-restriction flag
  output/<ts>/aufei_item_ceiling_floor.csv
  output/<ts>/flanker_descriptives.csv
  output/<ts>/flanker_construct_validity.csv  ceiling, sub-chance, effect-size flags
  output/<ts>/flanker_subject_flags.csv       per-subject exclusion candidates
  output/<ts>/flanker_reliability_audit.csv   reliability + recommendation per metric
  output/<ts>/digit_span_descriptives.csv
  output/<ts>/sample_demographics.csv
  output/<ts>/run_notes.json
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


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _maybe_csv(path):
    return pd.read_csv(path) if path and Path(path).exists() else None


def _maybe_json(path):
    if path and Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────────
# EEG quality from preprocessing/qc.json
# ──────────────────────────────────────────────────────────────────

def eeg_quality(preproc_dir):
    qc = json.load(open(preproc_dir / "qc.json", "r"))
    rows = []
    for q in qc:
        rows.append({
            "subject": q.get("subject"),
            "condition": q.get("condition"),
            "status": q.get("status"),
            "duration_after_crop_sec": q.get("duration_after_crop_sec"),
            "n_epochs_before_reject": q.get("n_epochs_before_reject"),
            "n_epochs_after_reject": q.get("n_epochs_after_reject"),
            "pct_epochs_dropped": q.get("pct_epochs_dropped"),
            "n_bad_channels": len(q.get("bad_channels", []) or []),
            "bad_channels": ",".join(q.get("bad_channels", []) or []),
            "n_ica_excluded": len(q.get("ica_excluded_idx", []) or []),
            "ica_labels": ",".join(q.get("ica_excluded_labels", []) or []),
            "autoreject_used": q.get("autoreject_used"),
            "ar_threshold_uv_median": q.get("threshold_uv_median"),
        })
    per = pd.DataFrame(rows)
    ok = per[per["status"] == "OK"].copy()
    rows = []
    for cond, g in ok.groupby("condition"):
        rows.append({
            "condition": cond,
            "n_ok": len(g),
            "n_dropped_low_epoch": int(((per["condition"] == cond)
                                       & (per["status"] == "LOW_EPOCH_COUNT")).sum()),
            "n_errors": int(((per["condition"] == cond)
                            & per["status"].astype(str).str.startswith("ERROR")).sum()),
            "epochs_kept_mean": float(g["n_epochs_after_reject"].mean()),
            "epochs_kept_sd":   float(g["n_epochs_after_reject"].std(ddof=1)),
            "epochs_kept_min":  int(g["n_epochs_after_reject"].min()),
            "epochs_kept_max":  int(g["n_epochs_after_reject"].max()),
            "pct_dropped_mean": float(g["pct_epochs_dropped"].mean()),
            "pct_dropped_max":  float(g["pct_epochs_dropped"].max()),
            "bad_ch_mean":      float(g["n_bad_channels"].mean()),
            "bad_ch_max":       int(g["n_bad_channels"].max()),
            "ica_excl_mean":    float(g["n_ica_excluded"].mean()),
            "ica_excl_max":     int(g["n_ica_excluded"].max()),
            "ar_threshold_uv_median_mean":
                float(g["ar_threshold_uv_median"].dropna().mean())
                if g["ar_threshold_uv_median"].notna().any() else float("nan"),
        })
    return per, pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# AUFEI
# ──────────────────────────────────────────────────────────────────

def aufei_descriptives(beh_dir, p):
    df = pd.read_excel(beh_dir / p["aufei_filename"])
    subscales = p["aufei_subscales"]
    L_min, L_max = int(p["aufei_likert_min"]), int(p["aufei_likert_max"])
    L_range = L_max - L_min
    sd_floor = float(p["aufei_sd_fraction_of_range_floor"])

    desc_rows = []
    for name, items in subscales.items():
        present = [c for c in items if c in df.columns]
        if not present:
            continue
        score = df[present].mean(axis=1)
        sd = float(score.std(ddof=1))
        var_restricted = sd < sd_floor * L_range
        desc_rows.append({
            "subscale": name,
            "k_items": len(present),
            "n_obs": int(score.notna().sum()),
            "mean": float(score.mean()),
            "sd": sd,
            "sd_pct_of_range": round(100 * sd / L_range, 1) if L_range else None,
            "variance_restricted_flag": bool(var_restricted),
            "min": float(score.min()),
            "max": float(score.max()),
            "median": float(score.median()),
            "ceiling_subjects_at_max":
                int((score >= L_max - 1e-9).sum()),
            "floor_subjects_at_min":
                int((score <= L_min + 1e-9).sum()),
        })
    subscale_desc = pd.DataFrame(desc_rows)

    item_rows = []
    for name, items in subscales.items():
        for c in items:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            n = int(s.notna().sum())
            ceil_count = int((s >= L_max - 1e-9).sum())
            floor_count = int((s <= L_min + 1e-9).sum())
            item_rows.append({
                "subscale": name, "item": c, "n_obs": n,
                "mean": float(s.mean()),
                "sd": float(s.std(ddof=1)),
                "min": int(s.min()) if n else None,
                "max": int(s.max()) if n else None,
                "pct_at_ceiling": round(100 * ceil_count / n, 1) if n else None,
                "pct_at_floor":   round(100 * floor_count / n, 1) if n else None,
                "flag": ("at-ceiling (zero variance)" if ceil_count == n and n > 0
                         else "heavy ceiling >=80%" if n and ceil_count / n >= 0.8
                         else "heavy floor >=80%" if n and floor_count / n >= 0.8
                         else ""),
            })
    return subscale_desc, pd.DataFrame(item_rows)


# ──────────────────────────────────────────────────────────────────
# Flanker — descriptives + construct validity + subject flags + reliability audit
# ──────────────────────────────────────────────────────────────────

def flanker_descriptives(beh_dir, p):
    path = beh_dir / p["flanker_filename"]
    df = pd.read_excel(path, sheet_name="Features"
                       if "Features" in pd.ExcelFile(path).sheet_names else 0)
    cols = ["acc_overall", "acc_incongruent", "rt_mean",
            "rt_congruent", "rt_incongruent", "flanker_effect",
            "rt_cv", "rt_iqr",
            "ddm_v_congruent", "ddm_v_incongruent", "ddm_delta_v",
            "ddm_a_congruent", "ddm_a_incongruent",
            "ddm_t0_congruent", "ddm_t0_incongruent"]
    cols = [c for c in cols if c in df.columns]
    rt_cols = [c for c in cols if c.startswith("rt_") and c != "rt_cv"]
    for c in rt_cols:
        if df[c].abs().max() > 1000:
            df[c] = df[c] / 1000.0

    desc = df[cols].describe().T
    desc = desc.rename(columns={"50%": "median", "25%": "q1", "75%": "q3"})
    desc["measure"] = desc.index
    desc = desc[["measure", "count", "mean", "std", "min",
                 "q1", "median", "q3", "max"]]
    return df, desc


def flanker_construct_validity(df, p):
    """Construct-validity audit. The Flanker is meaningful only if it induces
    a congruency effect. Mean and median RT effect compared to literature."""
    typical_floor_ms = float(p["flanker_effect_typical_min_ms"])
    rows = []
    if "flanker_effect" in df.columns:
        fe = df["flanker_effect"].dropna()
        fe_ms = fe * 1000.0
        rows.append({
            "indicator": "flanker_effect_mean_ms",
            "value": float(fe_ms.mean()),
            "reference": f"pediatric literature ~30-80 ms (this pipeline floor {typical_floor_ms} ms)",
            "interpretation": ("FAIL: no detectable congruency effect"
                               if abs(float(fe_ms.mean())) < typical_floor_ms
                               else "passes literature floor"),
        })
        rows.append({
            "indicator": "flanker_effect_median_ms",
            "value": float(fe_ms.median()),
            "reference": "median should be positive (incongruent slower than congruent)",
            "interpretation": ("FAIL: median ~0" if abs(float(fe_ms.median())) < typical_floor_ms
                               else ""),
        })
    if {"rt_congruent", "rt_incongruent"}.issubset(df.columns):
        rc = float(df["rt_congruent"].mean()) * 1000
        ri = float(df["rt_incongruent"].mean()) * 1000
        rows.append({
            "indicator": "rt_congruent_mean_ms",
            "value": rc, "reference": "—",
            "interpretation": "",
        })
        rows.append({
            "indicator": "rt_incongruent_mean_ms",
            "value": ri, "reference": "—",
            "interpretation": "",
        })
        rows.append({
            "indicator": "rt_inc_minus_rt_con_ms",
            "value": ri - rc,
            "reference": f"expect > {typical_floor_ms} ms",
            "interpretation": ("FAIL: conflict not induced"
                               if abs(ri - rc) < typical_floor_ms
                               else "induced congruency effect"),
        })
    ceil = float(p["flanker_acc_ceiling"])
    if "acc_overall" in df.columns:
        n_ceiling = int((df["acc_overall"] >= ceil).sum())
        n_total = int(df["acc_overall"].notna().sum())
        rows.append({
            "indicator": f"pct_subjects_acc_geq_{ceil}",
            "value": round(100 * n_ceiling / n_total, 1) if n_total else None,
            "reference": "expect ~10-20% in tasks calibrated for the age range",
            "interpretation": ("WARN: >50% at ceiling — task is too easy"
                               if n_total and n_ceiling / n_total > 0.5
                               else ""),
        })
    if "acc_overall" in df.columns:
        n_perfect = int((df["acc_overall"] >= 0.9999).sum())
        n_total = int(df["acc_overall"].notna().sum())
        rows.append({
            "indicator": "pct_at_perfect_acc",
            "value": round(100 * n_perfect / n_total, 1) if n_total else None,
            "reference": "EZ-DDM undefined at acc=1.0 -> these subjects drop from reliability",
            "interpretation": ("WARN: large degenerate subsample"
                               if n_total and n_perfect / n_total > 0.2
                               else ""),
        })
    return pd.DataFrame(rows)


def flanker_subject_flags(df, p):
    """Per-subject exclusion candidates. Cell flags for ceiling / sub-chance /
    implausibly-fast RT / DDM-undefined."""
    ceil = float(p["flanker_acc_ceiling"])
    sub = float(p["flanker_subchance_threshold"])
    rt_floor = float(p["flanker_rt_floor_sec"])
    rows = []
    for _, r in df.iterrows():
        sid = r.get("ID") or r.get("participant_id")
        ac  = r.get("acc_overall", float("nan"))
        ai  = r.get("acc_incongruent", float("nan"))
        rt  = r.get("rt_mean", float("nan"))
        flag_ceiling   = bool(np.isfinite(ac) and ac >= ceil)
        flag_subchance = bool(np.isfinite(ai) and ai < sub)
        flag_rt_fast   = bool(np.isfinite(rt) and rt < rt_floor)
        flag_ddm_undef = bool(np.isfinite(ac) and (ac >= 0.9999 or ac <= 1e-4))
        flag_subject_exclude_candidate = flag_subchance  # the strongest case
        if flag_ceiling or flag_subchance or flag_rt_fast or flag_ddm_undef:
            rows.append({
                "ID": sid,
                "acc_overall": float(ac) if np.isfinite(ac) else None,
                "acc_incongruent": float(ai) if np.isfinite(ai) else None,
                "rt_mean": float(rt) if np.isfinite(rt) else None,
                "at_ceiling_accuracy": flag_ceiling,
                "sub_chance_incongruent": flag_subchance,
                "implausibly_fast_rt": flag_rt_fast,
                "ddm_undefined_at_acc_extreme": flag_ddm_undef,
                "exclude_candidate": flag_subject_exclude_candidate,
            })
    return pd.DataFrame(rows)


def flanker_reliability_audit(flanker_rel_csv, p):
    """Map each measured reliability to a recommendation. Difference scores
    are flagged structurally; small-n estimates flagged as uninterpretable."""
    if flanker_rel_csv is None or flanker_rel_csv.empty:
        return pd.DataFrame()
    sb_floor = float(p["difference_score_sb_floor"])
    difference_scores = {"flanker_effect", "ddm_delta_v"}
    aud = []
    for _, r in flanker_rel_csv.iterrows():
        m = r["metric"]
        n = int(r["n"]) if pd.notna(r["n"]) else 0
        sb = r.get("spearman_brown")
        if sb is None or (isinstance(sb, float) and not np.isfinite(sb)):
            sb_val = None
        else:
            sb_val = float(sb)

        kind = "difference_score" if m in difference_scores else "single_measure"
        flags = []
        if n < 4:
            flags.append(f"n={n}: uninterpretable")
        elif n < 10:
            flags.append(f"n={n}: small subsample")
        if kind == "difference_score":
            flags.append("difference score: reliability typically << components")
            if sb_val is None or sb_val < sb_floor:
                flags.append(f"SB<{sb_floor} or undefined: unusable as target")
        recommend = ""
        if kind == "difference_score":
            recommend = "DO NOT USE as target"
        elif sb_val is None:
            recommend = "reliability not estimable"
        elif sb_val >= 0.80 and n >= 10:
            recommend = "candidate target (verify replication)"
        elif sb_val >= 0.80 and n < 10:
            recommend = f"high SB but n={n}; replicate in larger sample"
        elif sb_val >= 0.50:
            recommend = "marginal; not preferred target"
        else:
            recommend = "unreliable"
        aud.append({
            "metric": m, "kind": kind, "n": n,
            "r_halves": r.get("r_halves"),
            "spearman_brown": sb_val,
            "flags": "; ".join(flags),
            "recommendation": recommend,
        })
    return pd.DataFrame(aud)


# ──────────────────────────────────────────────────────────────────
# Digit Span
# ──────────────────────────────────────────────────────────────────

def digit_span_descriptives(beh_dir, p):
    df = pd.read_excel(beh_dir / p["digit_span_filename"])
    cols = [c for c in ["FW_Span", "BW_Span", "Total_Span",
                        "FW_Raw", "BW_Raw", "Total_Raw"] if c in df.columns]
    desc = df[cols].describe().T
    desc = desc.rename(columns={"50%": "median", "25%": "q1", "75%": "q3"})
    desc["measure"] = desc.index
    desc = desc[["measure", "count", "mean", "std", "min",
                 "q1", "median", "q3", "max"]]
    flags = {"n_total": int(len(df))}
    if "FW_Span" in df.columns:
        flags["fw_at_administrative_max"] = int(
            (df["FW_Span"] >= int(p["digit_span_fw_max_plausible"])).sum())
    if "BW_Span" in df.columns:
        flags["bw_at_administrative_max"] = int(
            (df["BW_Span"] >= int(p["digit_span_bw_max_plausible"])).sum())
    return desc, flags


# ──────────────────────────────────────────────────────────────────
# N reconciliation
# ──────────────────────────────────────────────────────────────────

def n_reconciliation(beh_dir, p, preproc_dir, fe_dir, analysis_dir):
    rows = []

    # Behavioural N
    aufei_n = pd.read_excel(beh_dir / p["aufei_filename"]).shape[0]
    flanker_n = pd.read_excel(beh_dir / p["flanker_filename"],
                              sheet_name="Features").shape[0]
    digit_n = pd.read_excel(beh_dir / p["digit_span_filename"]).shape[0]
    rows.append({
        "layer": "behavioural_aufei", "n": aufei_n,
        "reason_for_loss_vs_above": "—",
    })
    rows.append({
        "layer": "behavioural_flanker", "n": flanker_n,
        "reason_for_loss_vs_above": "—",
    })
    rows.append({
        "layer": "behavioural_digit_span", "n": digit_n,
        "reason_for_loss_vs_above": "—",
    })

    # EEG layers
    qc = json.load(open(preproc_dir / "qc.json", "r"))
    eeg_records = len(qc)
    eeg_ok = sum(1 for q in qc if q.get("status") == "OK")
    eeg_drop = sum(1 for q in qc if q.get("status") == "LOW_EPOCH_COUNT")
    dropped_recs = [(q.get("subject"), q.get("condition"),
                     q.get("n_epochs_after_reject"))
                    for q in qc if q.get("status") == "LOW_EPOCH_COUNT"]

    subj_conds = {}
    for q in qc:
        if q.get("status") == "OK":
            subj_conds.setdefault(q.get("subject"), set()).add(q.get("condition"))
    subj_both = sum(1 for c in subj_conds.values() if len(c) >= 2)
    rows.append({
        "layer": "eeg_recordings_attempted", "n": eeg_records,
        "reason_for_loss_vs_above": "—",
    })
    drop_reason = (f"{eeg_drop} dropped below min_epochs: "
                   + "; ".join(f"{sid}/{c} {ep} epochs"
                               for sid, c, ep in dropped_recs))
    rows.append({
        "layer": "eeg_recordings_ok", "n": eeg_ok,
        "reason_for_loss_vs_above": drop_reason if eeg_drop else "—",
    })
    rows.append({
        "layer": "eeg_subjects_with_both_EO_and_EC", "n": subj_both,
        "reason_for_loss_vs_above":
            "subjects retaining only one condition (the missing one was below min_epochs)",
    })

    # Merged behavioural ∩ EEG
    merged_n = None
    unmatched_behav = []
    if fe_dir is not None and (fe_dir / "full_dataset.csv").exists():
        full = pd.read_csv(fe_dir / "full_dataset.csv")
        merged_n = int(full.shape[0])
        feat_ids = set(full["subject_id"].astype(str).str.strip())
        beh_ids = set(pd.read_excel(beh_dir / p["aufei_filename"])["ID"]
                      .astype(str).str.strip())
        unmatched_behav = sorted(beh_ids - feat_ids)
    rows.append({
        "layer": "merged_behavioural_intersect_eeg",
        "n": merged_n,
        "reason_for_loss_vs_above":
            (f"{len(unmatched_behav)} behavioural-only IDs (no EDF / not processed): "
             f"{unmatched_behav}") if unmatched_behav else "—",
    })

    # Analysis (after dropping NaN on covariates + target + composites in feature set)
    analysis_n = None
    analysis_reason = ""
    if analysis_dir is not None:
        head = _maybe_json(analysis_dir / "headline.json")
        if isinstance(head, list) and head and fe_dir is not None:
            analysis_n = head[0].get("n_used")
            n_drop = head[0].get("n_dropped_nan")
            cov = head[0].get("covariates") or []
            tgt = head[0].get("target")
            comps = head[0].get("composites") or []
            # Identify which subjects were dropped and why (read full_dataset
            # and check NaN across the actual OLS column set).
            try:
                full = pd.read_csv(fe_dir / "full_dataset.csv")
                cols = [tgt] + list(cov) + list(comps)
                cols = [c for c in cols if c in full.columns]
                nan_mask = full[cols].isna().any(axis=1)
                dropped_ids = full.loc[nan_mask, "subject_id"].astype(str).tolist()
                # Per-column NaN count among the dropped rows
                nan_by_col = {c: int(full.loc[nan_mask, c].isna().sum())
                              for c in cols
                              if int(full.loc[nan_mask, c].isna().sum()) > 0}
                analysis_reason = (
                    f"OLS dropped {n_drop} rows for NaN across "
                    f"(target={tgt}, covariates={cov}, composites={comps}). "
                    f"Dropped subjects: {dropped_ids}. "
                    f"NaN by column among dropped: {nan_by_col}."
                )
            except Exception:
                analysis_reason = (f"OLS dropped {n_drop} rows for NaN on "
                                   f"target={tgt}, covariates={cov}, or "
                                   f"composites={comps}")
    rows.append({
        "layer": "analysis_full_OLS",
        "n": analysis_n,
        "reason_for_loss_vs_above": analysis_reason or "—",
    })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Sample demographics (from feature_engineering full_dataset.csv)
# ──────────────────────────────────────────────────────────────────

def sample_demographics(fe_dir):
    if fe_dir is None or not (fe_dir / "full_dataset.csv").exists():
        return pd.DataFrame()
    full = pd.read_csv(fe_dir / "full_dataset.csv")
    rows = [{"variable": "n_subjects_merged", "value": int(full.shape[0])}]
    if "age_years" in full.columns:
        s = pd.to_numeric(full["age_years"], errors="coerce")
        rows += [
            {"variable": "age_years_mean", "value": round(float(s.mean()), 2)},
            {"variable": "age_years_sd",   "value": round(float(s.std(ddof=1)), 2)},
            {"variable": "age_years_min",  "value": float(s.min())},
            {"variable": "age_years_max",  "value": float(s.max())},
        ]
    if "Sex" in full.columns:
        for v, c in full["Sex"].value_counts(dropna=False).items():
            rows.append({"variable": f"sex_{v}", "value": int(c)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# Report assembly (Markdown — headline-first)
# ──────────────────────────────────────────────────────────────────

def build_report(ctx):
    L = []
    p = ctx["paths_referenced"]
    L.append("# Pilot Data Quality Report")
    L.append("")
    L.append(f"Generated: {ctx['timestamp']}  |  Git commit: {ctx['git_commit']}")
    L.append("")
    L.append("Sources referenced:")
    L.append(f"- Preprocessing:       `{p['preprocessing']}`")
    L.append(f"- Validation:          `{p['validation']}`")
    L.append(f"- Feature engineering: `{p['feature_engineering']}`")
    L.append(f"- Analysis:            `{p['analysis']}`")
    L.append("")
    L.append("This is an internal QC document, not a pitch deck and not a "
             "deliverable. The findings below should inform main-study "
             "design decisions, not be quoted in isolation.")
    L.append("")

    # ─── Section 0: HEADLINE FINDINGS ──────────────────────────────
    L.append("## 0. Headline findings — read this first")
    L.append("")

    # F-1 Construct-validity failure
    L.append("### F-1. Construct-validity failure: the Flanker did not induce conflict")
    L.append("")
    cv = ctx["flanker_cv"]
    if not cv.empty:
        L.append("| indicator | value | reference | interpretation |")
        L.append("|---|---|---|---|")
        for _, r in cv.iterrows():
            L.append(f"| {r['indicator']} | {fmt(r['value'])} | "
                     f"{r['reference']} | {r['interpretation']} |")
        L.append("")
    L.append("The pediatric Flanker congruency effect typically falls in the "
             "30–80 ms range. Our sample mean is essentially zero, and "
             "`rt_congruent ≈ rt_incongruent` at the sample level. Combined "
             "with the 75% accuracy ceiling, this is most consistent with "
             "the task being too easy to induce a measurable conflict signal.")
    L.append("")
    L.append("**Implication for the main study**: the null observed in the "
             "analysis stage (block F p = 0.83, ΔR² = 0.03 over age) is "
             "downstream of this construct-validity failure, not evidence "
             "that QEEG biomarkers are weak. The Flanker task must be "
             "modified (harder distractors, response deadline, speed "
             "pressure) before the biomarker-target relationship can be "
             "estimated. Until then, no Flanker-derived target carries "
             "interpretable construct meaning, regardless of how reliable "
             "its measurement is.")
    L.append("")

    # F-2 Difference-score targets
    L.append("### F-2. Difference-score targets are not estimable from these data")
    L.append("")
    aud = ctx["flanker_audit"]
    if not aud.empty:
        L.append("| metric | kind | n | r_halves | SB | flags | recommendation |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in aud.iterrows():
            L.append(f"| {r['metric']} | {r['kind']} | {fmt(r['n'])} | "
                     f"{fmt(r['r_halves'])} | {fmt(r['spearman_brown'])} | "
                     f"{r['flags']} | {r['recommendation']} |")
        L.append("")
    L.append("Two targets the previous report recommended must be retracted:")
    L.append("")
    L.append("- **`ddm_delta_v`** — split-half reliability cannot be estimated "
             "(only 2 subjects yield valid per-condition EZ-DDM values in "
             "both halves; the rest hit the acc∈{0,1} degeneracy). The "
             "earlier framing-notes recommendation of `ddm_delta_v` as a "
             "primary target was wrong on two counts: it is a difference "
             "score (component-correlation property), and at this ceiling "
             "we cannot even produce a reliability estimate for it.")
    L.append("- **`flanker_effect`** — SB = 0.13. Difference of two highly "
             "correlated, well-measured components -> reliability << "
             "components' (classic reliability paradox of difference "
             "scores). Unusable as a target.")
    L.append("")
    L.append("**Replacement recommendations** (in order):")
    L.append("")
    L.append("1. **`rt_cv`** — robust, computable on all 28 subjects, no "
             "ceiling failure mode. Use as primary if its biological "
             "interpretation in the EF context is acceptable.")
    L.append("2. **`ddm_v_incongruent`** — SB = 0.99 on n=6 of 28 (non-ceiling "
             "subsample only). Cannot be defended as a primary target at "
             "this N until replicated in a larger non-ceiling sample. "
             "Suitable for a secondary analysis with the caveat documented.")
    L.append("3. **`acc_overall`** / **`acc_incongruent`** — pseudo-reliable "
             "(SB > 0.98) but at ceiling for most subjects; reduces to a "
             "binary perfect-vs-not flag. Not informative.")
    L.append("")

    # F-3 DDM ceiling subsample
    L.append("### F-3. DDM single-condition reliability is non-representative")
    L.append("")
    L.append("`ddm_v` (overall) split-half SB ≈ 0.99 looks excellent — but is "
             "computed on n = 11 of 28 subjects, the strict subsample whose "
             "accuracy is strictly in (0,1) for both split halves. The other "
             "17 are at ceiling in at least one half. The SB estimate "
             "therefore (a) does not generalise to the full sample, and "
             "(b) is likely inflated by the wide range of the non-ceiling "
             "subsample (which includes outliers like the sub-chance "
             "subject — see F-5). Reporting `SB = 0.99` without the "
             "n = 11 / 28 caveat would actively mislead.")
    L.append("")
    L.append("The same caveat applies more strongly to "
             "`ddm_v_congruent` (n = 4) and `ddm_v_incongruent` (n = 6): at "
             "those subsample sizes the SB estimate has CI so wide that the "
             "point estimate is uninterpretable in isolation.")
    L.append("")

    # F-4 Estimator inconsistency clarification
    L.append("### F-4. Estimator clarification (resolving the prior inconsistency)")
    L.append("")
    L.append("The pipeline uses **two different DDM estimators** and they "
             "should not be conflated:")
    L.append("")
    L.append("- **Target column for analysis** (`feature_engineering` -> "
             "`analysis`): `ddm_v_incongruent`, `ddm_delta_v`, etc. are "
             "read **as-is** from the workbook's `Features` sheet. The "
             "estimator used to produce those values is not documented in "
             "the workbook; we observe that they remain finite at "
             "`acc = 1.0`, so they are not pure EZ-DDM. We accept them as "
             "the target column without independent re-estimation.")
    L.append("- **Reliability estimates in this report** (`validation` -> "
             "this stage): computed via **EZ-DDM on each split-half of the "
             "trial-level data** (`Trials` sheet). EZ-DDM is undefined at "
             "`acc ∈ {0, 1}`, which is why the per-condition reliability "
             "estimates collapse to n = 4–6 and `ddm_delta_v` to n = 2.")
    L.append("")
    L.append("The reliability of the workbook's DDM columns is therefore "
             "**unknown** at the full sample size — our EZ-based reliability "
             "estimate is at best an approximate floor for the non-ceiling "
             "subsample and is uninformative about the ceiling subsample. "
             "The earlier framing-note claim that the workbook used a "
             "\"more robust procedure\" was inherited from the pre-rewrite "
             "code and is not independently verified; it is reframed here "
             "as: *unknown estimator that handles ceiling cases differently "
             "from EZ-DDM*.")
    L.append("")

    # F-5 Sub-chance subject
    L.append("### F-5. Pre-registered exclusion candidate: sub-chance subject")
    L.append("")
    if not ctx["flanker_subj_flags"].empty:
        sub = ctx["flanker_subj_flags"][
            ctx["flanker_subj_flags"]["sub_chance_incongruent"] == True]
        if not sub.empty:
            for _, r in sub.iterrows():
                L.append(f"- **{r['ID']}**: acc_overall = {fmt(r['acc_overall'])}, "
                         f"acc_incongruent = {fmt(r['acc_incongruent'])}, "
                         f"rt_mean = {fmt(r['rt_mean'])} s")
            L.append("")
            L.append("Below the 0.50 chance line on incongruent trials in a "
                     "2AFC task -> most likely response-mapping reversed or "
                     "task not understood. Distorts mean DDM v_incongruent "
                     "and delta_v (this subject contributes "
                     "`v_incongruent = -2.13`, `delta_v = -3.96` — the "
                     "min of both distributions). Recommendation: define "
                     "this as a pre-registered exclusion rule in the main "
                     "study analysis plan "
                     "(`acc_incongruent < flanker_subchance_threshold`).")
            L.append("")

    # F-6 AUFEI restricted variance
    L.append("### F-6. AUFEI: low reliability is partly variance restriction, not only item content")
    L.append("")
    sub_desc = ctx["aufei_subscale"]
    rest = sub_desc[sub_desc.get("variance_restricted_flag", False) == True] \
        if not sub_desc.empty else sub_desc
    if not rest.empty:
        L.append("Subscales flagged for variance restriction "
                 f"(SD < {fmt(100 * ctx['p_aufei_sd_fraction'], 0)}% of Likert "
                 f"{ctx['aufei_likert_min']}–{ctx['aufei_likert_max']} range):")
        L.append("")
        L.append("| subscale | mean | sd | sd_pct_of_range | k_items |")
        L.append("|---|---|---|---|---|")
        for _, r in rest.iterrows():
            L.append(f"| {r['subscale']} | {fmt(r['mean'])} | "
                     f"{fmt(r['sd'])} | {fmt(r['sd_pct_of_range'], 1)}% | "
                     f"{r['k_items']} |")
        L.append("")
    L.append("**WM α = −0.13** is invalid as a reliability estimate — items "
             "are anti-correlated in this sample. Combined with subscale "
             "SD ≈ 0.26 on a Likert 1–4 scale (≈ 9% of range), the dominant "
             "explanation is parent-report social-desirability compression "
             "to the top end, not necessarily bad items in absolute terms. "
             "The remediation path is therefore *re-anchoring the response "
             "scale* (e.g. 1-7 with concrete behavioural anchors, or "
             "frequency-based rather than agreement-based items), not "
             "simply replacing items. A CFA on these data without "
             "addressing variance restriction will almost certainly "
             "produce a degenerate solution.")
    L.append("")

    # F-7 AUFEI item-level
    L.append("### F-7. AUFEI item-level: confirmed ceiling problems")
    L.append("")
    severe = ctx["aufei_items"][ctx["aufei_items"]["flag"] != ""]
    if not severe.empty:
        L.append("| subscale | item | mean | sd | pct_at_ceiling | pct_at_floor | flag |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in severe.iterrows():
            L.append(f"| {r['subscale']} | {r['item']} | {fmt(r['mean'])} | "
                     f"{fmt(r['sd'])} | {fmt(r['pct_at_ceiling'], 1)} | "
                     f"{fmt(r['pct_at_floor'], 1)} | {r['flag']} |")
        L.append("")
    L.append("IC3 is at 100% ceiling (zero variance) — already excluded "
             "from the IC subscale composition in `validation/config.yaml` "
             "and `feature_engineering/config.yaml`. CF2 at 89% ceiling is "
             "a new flag from this run; recommend exclusion or rewrite in "
             "the next pilot wave.")
    L.append("")

    # ─── Section 1: N reconciliation ───────────────────────────────
    L.append("## 1. Sample / N reconciliation")
    L.append("")
    L.append("How the sample shrinks as we move from raw behavioural workbooks "
             "to the OLS in the analysis stage.")
    L.append("")
    if not ctx["n_recon"].empty:
        L.append("| layer | n | reason for loss vs. above |")
        L.append("|---|---|---|")
        for _, r in ctx["n_recon"].iterrows():
            L.append(f"| {r['layer']} | {fmt(r['n'])} | "
                     f"{r['reason_for_loss_vs_above']} |")
        L.append("")

    # ─── Section 2: Demographics ──────────────────────────────────
    L.append("## 2. Sample demographics (merged N)")
    L.append("")
    if not ctx["demographics"].empty:
        L.append("| variable | value |")
        L.append("|---|---|")
        for _, r in ctx["demographics"].iterrows():
            L.append(f"| {r['variable']} | {fmt(r['value'])} |")
        L.append("")

    # ─── Section 3: EEG ───────────────────────────────────────────
    L.append("## 3. EEG post-processing quality")
    L.append("")
    if not ctx["eeg_summary"].empty:
        L.append("Per-condition summary (status = OK only). Per-recording "
                 "detail in `eeg_quality_per_recording.csv`.")
        L.append("")
        cols = ["condition", "n_ok", "epochs_kept_mean", "epochs_kept_sd",
                "epochs_kept_min", "epochs_kept_max",
                "pct_dropped_mean", "pct_dropped_max",
                "bad_ch_mean", "ica_excl_mean",
                "ar_threshold_uv_median_mean"]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in ctx["eeg_summary"].iterrows():
            L.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
        L.append("")
    L.append("Detailed per-component ICLabel probabilities and per-channel "
             f"AutoReject thresholds: `{p['preprocessing']}/qc.json`.")
    L.append("")

    # ─── Section 4: AUFEI subscale + reliability ──────────────────
    L.append("## 4. AUFEI-O (parent-report executive function)")
    L.append("")
    if not ctx["aufei_subscale"].empty:
        L.append(f"### Subscale descriptives (Likert "
                 f"{ctx['aufei_likert_min']}–{ctx['aufei_likert_max']})")
        L.append("")
        cols = ["subscale", "k_items", "n_obs", "mean", "sd", "sd_pct_of_range",
                "min", "max", "ceiling_subjects_at_max",
                "variance_restricted_flag"]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in ctx["aufei_subscale"].iterrows():
            L.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
        L.append("")
    if ctx["aufei_rel"] is not None:
        L.append("### Reliability (Cronbach's α, from `validation/`)")
        L.append("")
        L.append("| subscale | k_items | alpha | interpretation |")
        L.append("|---|---|---|---|")
        for _, r in ctx["aufei_rel"].iterrows():
            L.append(f"| {r.get('subscale','')} | {r.get('k_items','')} | "
                     f"{r.get('alpha','')} | {r.get('interpretation','')} |")
        L.append("")
        L.append(f"Item-level item-total diagnostics: "
                 f"`{p['validation']}/aufei_item_total.csv`.")
        L.append("")

    # ─── Section 5: Flanker descriptives ──────────────────────────
    L.append("## 5. Fish Flanker Test")
    L.append("")
    if not ctx["flanker_desc"].empty:
        L.append("### Per-measure descriptives")
        L.append("")
        cols = ["measure", "count", "mean", "std", "min",
                "q1", "median", "q3", "max"]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in ctx["flanker_desc"].iterrows():
            L.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
        L.append("")
    if not ctx["flanker_subj_flags"].empty:
        L.append("### Per-subject flags (ceiling / sub-chance / fast RT / DDM-undefined)")
        L.append("")
        cols = ["ID", "acc_overall", "acc_incongruent", "rt_mean",
                "at_ceiling_accuracy", "sub_chance_incongruent",
                "implausibly_fast_rt", "ddm_undefined_at_acc_extreme",
                "exclude_candidate"]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in ctx["flanker_subj_flags"].iterrows():
            L.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
        L.append("")
    L.append(f"Full reliability audit (per metric, with recommendation): "
             f"`flanker_reliability_audit.csv`.")
    L.append("")

    # ─── Section 6: Digit Span ────────────────────────────────────
    L.append("## 6. Digit Span")
    L.append("")
    if not ctx["ds_desc"].empty:
        L.append("### Descriptives")
        L.append("")
        cols = ["measure", "count", "mean", "std",
                "min", "q1", "median", "q3", "max"]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in ctx["ds_desc"].iterrows():
            L.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
        L.append("")
    ds_f = ctx["ds_flags"]
    if ds_f:
        L.append(f"Subjects at administrative-max span: "
                 f"FW {ds_f.get('fw_at_administrative_max', 'NA')}, "
                 f"BW {ds_f.get('bw_at_administrative_max', 'NA')} "
                 f"(of n = {ds_f.get('n_total','NA')}).")
        L.append("")
    if ctx["ds_rel"] is not None:
        L.append("### Reliability (from `validation/`)")
        L.append("")
        for _, r in ctx["ds_rel"].iterrows():
            L.append(f"- {r.get('measure','')}: r(FW,BW) = "
                     f"{r.get('r_halves','')}, Spearman-Brown = "
                     f"{r.get('spearman_brown','')}.")
            note = r.get('note', '')
            if note:
                L.append(f"  *{note}*")
        L.append("")
    L.append("Item-level Digit Span data is not in the current export. "
             "Without it, no defensible test-reliability estimate is "
             "possible; FW-vs-BW is at best a weak proxy because the two "
             "tap different constructs (passive retention vs. active "
             "manipulation). For the main study, request item-level output "
             "from the testing platform and add a split-half reliability "
             "check to `validation/main.py` mirroring the Flanker pattern.")
    L.append("")

    # ─── Section 7: Pilot framing constraints (kept; no recommendations) ─
    L.append("## 7. Pilot framing constraints")
    L.append("")
    L.append("- At N = 25–28, power to detect r = 0.30 is ≈ 0.35; r = 0.40 ≈ 0.60. "
             "All hypothesis tests in the analysis stage are estimation, not "
             "acceptance/rejection.")
    L.append("- Age accounts for R² ≈ 0.31 of variance in `ddm_v_incongruent` "
             "on its own. The hierarchical OLS handles this via Frisch-Waugh; "
             "any future CV regression path must keep `age_months` in the design "
             "matrix.")
    L.append("- Findings in Section 0 are construct- and instrument-level "
             "problems, not biomarker-level findings. A null biomarker result "
             "with these instruments is uninformative.")
    L.append("")
    return "\n".join(L)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    p = cfg["params"]
    beh_dir = resolve(cfg["paths"]["behavioral_dir"])
    out_dir = make_output_dir()
    print(f"Data-quality output: {out_dir}")

    preproc = latest_output(cfg["paths"]["preprocessing_root"])
    valid   = latest_output(cfg["paths"]["validation_root"])
    fe      = latest_output(cfg["paths"]["feature_engineering_root"])
    analysis_root = STAGE_DIR.parent / "analysis" / "output"
    analysis = latest_output(analysis_root) if analysis_root.exists() else None
    if preproc is None:
        raise FileNotFoundError("no preprocessing output found")
    print(f"  preprocessing:    {preproc}")
    print(f"  validation:       {valid}")
    print(f"  feature_eng:      {fe}")
    print(f"  analysis:         {analysis}")

    # Compute every section
    eeg_per, eeg_summary = eeg_quality(preproc)
    eeg_per.to_csv(out_dir / "eeg_quality_per_recording.csv", index=False)
    eeg_summary.to_csv(out_dir / "eeg_quality_summary.csv", index=False)

    aufei_sub, aufei_items = aufei_descriptives(beh_dir, p)
    aufei_sub.to_csv(out_dir / "aufei_descriptives.csv", index=False)
    aufei_items.to_csv(out_dir / "aufei_item_ceiling_floor.csv", index=False)
    aufei_rel = _maybe_csv(valid / "aufei_subscale_reliability.csv") if valid else None

    flanker_raw, flanker_desc = flanker_descriptives(beh_dir, p)
    flanker_desc.to_csv(out_dir / "flanker_descriptives.csv", index=False)
    flanker_cv = flanker_construct_validity(flanker_raw, p)
    flanker_cv.to_csv(out_dir / "flanker_construct_validity.csv", index=False)
    flanker_subj = flanker_subject_flags(flanker_raw, p)
    flanker_subj.to_csv(out_dir / "flanker_subject_flags.csv", index=False)
    flanker_rel = _maybe_csv(valid / "flanker_reliability.csv") if valid else None
    flanker_audit = flanker_reliability_audit(flanker_rel, p)
    flanker_audit.to_csv(out_dir / "flanker_reliability_audit.csv", index=False)

    ds_desc, ds_flags = digit_span_descriptives(beh_dir, p)
    ds_desc.to_csv(out_dir / "digit_span_descriptives.csv", index=False)
    ds_rel = _maybe_csv(valid / "digit_span_reliability.csv") if valid else None

    demo = sample_demographics(fe)
    demo.to_csv(out_dir / "sample_demographics.csv", index=False)

    n_recon = n_reconciliation(beh_dir, p, preproc, fe, analysis)
    n_recon.to_csv(out_dir / "n_reconciliation.csv", index=False)

    # Console — show only what matters
    print("\n=== Section 0 — Headlines ===")
    print("\n[F-1] Flanker construct validity:")
    print(flanker_cv.to_string(index=False))
    print("\n[F-2] Flanker reliability audit (target recommendations):")
    print(flanker_audit.to_string(index=False))
    print("\n[F-5] Per-subject flags (exclusion candidates):")
    if not flanker_subj.empty:
        print(flanker_subj[["ID", "acc_overall", "acc_incongruent",
                            "sub_chance_incongruent",
                            "exclude_candidate"]].to_string(index=False))
    print("\n[F-6] AUFEI subscales with variance restriction:")
    print(aufei_sub[["subscale", "mean", "sd", "sd_pct_of_range",
                     "variance_restricted_flag"]].to_string(index=False))
    print("\n=== Section 1 — N reconciliation ===")
    print(n_recon.to_string(index=False))

    # Build report
    ctx = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "paths_referenced": {
            "preprocessing":       str(preproc),
            "validation":          str(valid) if valid else "(none)",
            "feature_engineering": str(fe) if fe else "(none)",
            "analysis":            str(analysis) if analysis else "(none)",
        },
        "eeg_per": eeg_per,
        "eeg_summary": eeg_summary,
        "aufei_subscale": aufei_sub,
        "aufei_items": aufei_items,
        "aufei_rel": aufei_rel,
        "aufei_likert_min": int(p["aufei_likert_min"]),
        "aufei_likert_max": int(p["aufei_likert_max"]),
        "p_aufei_sd_fraction": float(p["aufei_sd_fraction_of_range_floor"]),
        "flanker_desc": flanker_desc,
        "flanker_cv": flanker_cv,
        "flanker_subj_flags": flanker_subj,
        "flanker_audit": flanker_audit,
        "flanker_rel": flanker_rel,
        "ds_desc": ds_desc,
        "ds_flags": ds_flags,
        "ds_rel": ds_rel,
        "demographics": demo,
        "n_recon": n_recon,
    }
    report = build_report(ctx)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\nReport: {out_dir / 'report.md'}")

    notes = {
        "stage": "data_quality_check",
        "timestamp": ctx["timestamp"],
        "git_commit": ctx["git_commit"],
        "sources": ctx["paths_referenced"],
        "outputs": ["report.md", "n_reconciliation.csv",
                    "eeg_quality_per_recording.csv", "eeg_quality_summary.csv",
                    "aufei_descriptives.csv", "aufei_item_ceiling_floor.csv",
                    "flanker_descriptives.csv", "flanker_construct_validity.csv",
                    "flanker_subject_flags.csv", "flanker_reliability_audit.csv",
                    "digit_span_descriptives.csv", "sample_demographics.csv"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)


if __name__ == "__main__":
    main()
