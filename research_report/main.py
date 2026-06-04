"""
Research report (sidecar)
=========================
Assembles a 5-section pilot research report from upstream stage outputs.
This is the consolidated research document — NOT a feasibility / funding
pitch. Replaces the old feasibility_report. Sections:

  1. Sample & data quality (from data_quality_check)
  2. Pipeline description + flow diagram
  3. Confirmatory a priori OLS (from analysis/<target>/<feature_set>/)
  4. Exploratory ML grid (from analysis/exploratory_ml/) — with explicit
     construct-validity caveat for the Flanker-derived targets
  5. Screening index — illustrative prospect output (from
     analysis/screening_index/) — with explicit non-clinical labelling

Output: output/<ts>/{research_report.md, run_notes.json}
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


def _read_csv(p):
    return pd.read_csv(p) if p and Path(p).exists() else None


def _read_json(p):
    if p and Path(p).exists():
        with open(p, "r") as f:
            return json.load(f)
    return None


def _fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def _md_table(df, columns=None):
    if df is None or (hasattr(df, "empty") and df.empty):
        return ["_no data_", ""]
    if columns is None:
        columns = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |")
    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────
# Fact collection
# ──────────────────────────────────────────────────────────────────

def collect_facts(cfg):
    f = {"paths": {}}

    preproc = latest_output(cfg["paths"]["preprocessing_root"])
    valid = latest_output(cfg["paths"]["validation_root"])
    fb = latest_output(cfg["paths"]["feature_building_root"])
    fe = latest_output(cfg["paths"]["feature_engineering_root"])
    analysis = latest_output(cfg["paths"]["analysis_root"])
    dqc = latest_output(cfg["paths"]["data_quality_check_root"])
    f["paths"] = {
        "preprocessing": str(preproc) if preproc else None,
        "validation": str(valid) if valid else None,
        "feature_building": str(fb) if fb else None,
        "feature_engineering": str(fe) if fe else None,
        "analysis": str(analysis) if analysis else None,
        "data_quality_check": str(dqc) if dqc else None,
    }

    # Section 1 — Sample & data quality (from data_quality_check)
    if dqc:
        f["dqc_demo"] = _read_csv(dqc / "sample_demographics.csv")
        f["dqc_n_recon"] = _read_csv(dqc / "n_reconciliation.csv")
        f["dqc_eeg"] = _read_csv(dqc / "eeg_quality_summary.csv")
        f["dqc_aufei"] = _read_csv(dqc / "aufei_descriptives.csv")
        f["dqc_flanker_cv"] = _read_csv(dqc / "flanker_construct_validity.csv")
        f["dqc_flanker_audit"] = _read_csv(dqc / "flanker_reliability_audit.csv")
        f["dqc_subj"] = _read_csv(dqc / "flanker_subject_flags.csv")
        f["dqc_digit_desc"] = _read_csv(dqc / "digit_span_descriptives.csv")

    # Section 3 — Confirmatory OLS
    if analysis:
        primary_target = cfg["params"]["primary_target"]
        primary_fs = cfg["params"]["primary_feature_set"]
        olsdir = analysis / primary_target / primary_fs
        f["ols_summary"] = _read_json(olsdir / "summary.json")
        f["ols_coef"] = _read_csv(olsdir / "coef_inference.csv")
        f["ols_block"] = _read_json(olsdir / "block_f_test.json")
        f["ols_alpha"] = _read_csv(olsdir / "composite_alpha.csv")
        f["ols_boot"] = _read_csv(olsdir / "bootstrap_ci.csv")
        f["ols_path"] = str(olsdir)

    # Section 4 — Exploratory ML grid
    if analysis:
        ml_dir = analysis / "exploratory_ml"
        f["ml_grid"] = _read_csv(ml_dir / "exploratory_ml_grid.csv")
        f["ml_summary"] = _read_json(ml_dir / "exploratory_ml_summary.json")
        f["ml_path"] = str(ml_dir)

    # Section 5 — Screening index
    if analysis:
        si_dir = analysis / "screening_index"
        # Find the single screening_index_*.csv (one combo only)
        if si_dir.exists():
            csvs = sorted(si_dir.glob("screening_index_*.csv"))
            metas = sorted(si_dir.glob("screening_index_*_meta.json"))
            f["si_df"] = _read_csv(csvs[0]) if csvs else None
            f["si_meta"] = _read_json(metas[0]) if metas else None
            f["si_path"] = str(si_dir)
        else:
            f["si_df"] = None
            f["si_meta"] = None
            f["si_path"] = None

    return f


# ──────────────────────────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────────────────────────

def section_header(cfg, facts):
    L = []
    L.append(f"# {cfg['params']['project_title']}")
    L.append("")
    L.append(f"**Institution**: {cfg['params']['institution']}  ")
    L.append(f"**Collaborating site**: {cfg['params']['collaborating_site']}  ")
    L.append(f"**Ethics**: {cfg['params']['ethics_body']}")
    L.append("")
    L.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}  ")
    L.append(f"Git commit: {git_commit()}")
    L.append("")
    L.append("Upstream evidence in this report comes from the following "
             "timestamped runs:")
    for k, v in facts["paths"].items():
        if v:
            L.append(f"- `{k}`: `{v}`")
    L.append("")
    L.append("---")
    L.append("")
    return L


def section_sample_and_quality(cfg, facts):
    L = []
    L.append("## 1. Sample and data quality")
    L.append("")
    L.append("### 1.1 Demographics (merged sample)")
    L.append("")
    if facts.get("dqc_demo") is not None:
        L += _md_table(facts["dqc_demo"], ["variable", "value"])

    L.append("### 1.2 Sample attrition reconciliation")
    L.append("")
    if facts.get("dqc_n_recon") is not None:
        L += _md_table(facts["dqc_n_recon"],
                       ["layer", "n", "reason_for_loss_vs_above"])

    L.append("### 1.3 EEG signal-quality summary")
    L.append("")
    if facts.get("dqc_eeg") is not None:
        L += _md_table(facts["dqc_eeg"],
                       ["condition", "n_ok", "epochs_kept_mean",
                        "epochs_kept_min", "pct_dropped_mean",
                        "bad_ch_mean", "ica_excl_mean",
                        "ar_threshold_uv_median_mean"])

    L.append("### 1.4 Behavioural construct-validity and reliability")
    L.append("")
    L.append("Fish Flanker construct-validity audit:")
    L.append("")
    if facts.get("dqc_flanker_cv") is not None:
        L += _md_table(facts["dqc_flanker_cv"],
                       ["indicator", "value", "reference", "interpretation"])
    L.append("Trial-level reliability audit (per metric + recommendation "
             "for downstream targeting):")
    L.append("")
    if facts.get("dqc_flanker_audit") is not None:
        L += _md_table(facts["dqc_flanker_audit"],
                       ["metric", "kind", "n", "spearman_brown",
                        "flags", "recommendation"])

    L.append("AUFEI-O parent-report subscale descriptives + "
             "variance-restriction flag:")
    L.append("")
    if facts.get("dqc_aufei") is not None:
        cols = ["subscale", "k_items", "mean", "sd", "sd_pct_of_range",
                "variance_restricted_flag",
                "ceiling_subjects_at_max"]
        cols = [c for c in cols if c in facts["dqc_aufei"].columns]
        L += _md_table(facts["dqc_aufei"], cols)

    L.append("Digit Span descriptives:")
    L.append("")
    if facts.get("dqc_digit_desc") is not None:
        L += _md_table(facts["dqc_digit_desc"],
                       ["measure", "count", "mean", "std", "min",
                        "median", "max"])

    if facts.get("dqc_subj") is not None:
        ex = facts["dqc_subj"][facts["dqc_subj"]["exclude_candidate"] == True]
        if not ex.empty:
            L.append("**Pre-registered exclusion candidates** "
                     "(sub-chance accuracy):")
            L.append("")
            L += _md_table(ex,
                           ["ID", "acc_overall", "acc_incongruent",
                            "rt_mean", "sub_chance_incongruent"])
    return L


def section_pipeline(cfg, facts):
    L = []
    L.append("## 2. Pipeline description")
    L.append("")
    L.append("Five sequential pipeline stages plus two sidecar reporters. "
             "Each stage is self-contained (single `main.py` + `config.yaml` "
             "+ timestamped `output/`); no shared utilities, no orchestrator. "
             "Predecessor outputs are auto-resolved as the lexically-latest "
             "`YYYY-MM-DD_HHMMSS` subdir, and each run records its inputs in "
             "`run_notes.json` so the audit trail is traceable to raw data + "
             "git commit.")
    L.append("")
    if cfg["params"].get("include_flow_diagram", True):
        L.append("### 2.1 Flow diagram")
        L.append("")
        L.append("```")
        L.append("data/EDF + data/Behavioral")
        L.append("   |")
        L.append("   v preprocessing   EDF -> ICLabel + AutoReject -> cleaned epochs")
        L.append("       (15-ch Mitsar 250 Hz; avg-ref; infomax extended ICA;")
        L.append("        ICLabel p>0.7 for {eye/muscle/heart/line/channel};")
        L.append("        AutoReject local; min_epochs=60 floor)")
        L.append("   |")
        L.append("   v validation      trial-level behavioural reliability + QC")
        L.append("       (AUFEI Cronbach + omega; Flanker odd/even split-half")
        L.append("        Spearman-Brown on DDM/RT; Digit Span FW-vs-BW)")
        L.append("   |")
        L.append("   v feature_building  cleaned epochs -> 2,000+ primitives")
        L.append("       (PSD np.trapezoid; coherence per-epoch; CWT; Hjorth;")
        L.append("        spectral entropy; PAC zero-phase; covariance;")
        L.append("        specparam aperiodic exp/offset; PAF centroid)")
        L.append("   |")
        L.append("   v feature_engineering  primitives -> composites + targets")
        L.append("       (9 a priori UNION composites: rel_theta/beta_FC,")
        L.append("        rel_alpha_PO, aperiodic_exp/off_FC/PO, paf_PO,")
        L.append("        alpha_reactivity_PO. Behavioural merge; curation;")
        L.append("        feature_selection_schemes resolved)")
        L.append("   |")
        L.append("   v analysis           confirmatory OLS + exploratory ML")
        L.append("       (OLS: y ~ age + composites; block F; Bonferroni;")
        L.append("        ML: ElasticNetCV + RF + XGB x 3 schemes x 4 targets;")
        L.append("        selection-corrected permutation; screening index)")
        L.append("")
        L.append("   ~ data_quality_check  internal QC audit (sidecar)")
        L.append("   ~ research_report     this document (sidecar)")
        L.append("```")
        L.append("")
    return L


def section_confirmatory_ols(cfg, facts):
    L = []
    L.append("## 3. Confirmatory analysis — a priori hierarchical OLS")
    L.append("")
    target = cfg["params"]["primary_target"]
    fs = cfg["params"]["primary_feature_set"]
    L.append(f"Primary target: `{target}` (per the target-ordering rule: "
             "construct-validity independence first, then reliability — "
             "rt_cv is robust to the Flanker construct failure and has "
             "SB = 0.97 on full n).")
    L.append("")
    L.append(f"Feature set: `{fs}` — the 9-feature a priori UNION of "
             "theory-driven QEEG composites (region-collapsed to "
             "FC = {Fz, Cz}, PO = {O1, O2, Pz} on the 15-channel Mitsar "
             "montage).")
    L.append("")
    L.append("Hierarchical OLS:")
    L.append("- Restricted model: y ~ age_months")
    L.append("- Full model:       y ~ age_months + 9 composites")
    L.append("- Block F-test:     incremental R² of the composite block over age")
    L.append("- Per-composite:    β, SE, t, one-sided directional p "
             "(per `apriori_theory_mapping.csv`), Bonferroni-corrected "
             "across the 9 composites")
    L.append("- Bootstrap:        subject-resample 95% CI on all coefficients")
    L.append("")
    sm = facts.get("ols_summary") or {}
    block = (sm.get("block_f_test") or {}) if isinstance(sm, dict) else {}
    L.append("### 3.1 Headline results")
    L.append("")
    L.append(f"- N used: **{sm.get('n_used', 'NA')}** "
             f"(dropped {sm.get('n_dropped_nan', 'NA')} for NaN)")
    L.append(f"- Restricted R² (age alone): **{_fmt(sm.get('covariate_r2'), 4)}**")
    L.append(f"- Full R² (age + composites): **{_fmt(sm.get('full_r2'), 4)}**")
    L.append(f"- Block F({_fmt(block.get('df_num'), 0)},"
             f"{_fmt(block.get('df_den'), 0)}) = "
             f"**{_fmt(block.get('F'))}**, ΔR² = "
             f"**{_fmt(block.get('delta_r2'), 4)}**, p = "
             f"**{_fmt(block.get('p_value'))}**")
    L.append(f"- Cronbach gate across composites: "
             f"**{'PASS' if sm.get('cronbach_gate_all_pass') else 'see composite_alpha.csv'}**")
    L.append("")
    L.append("### 3.2 Per-composite inference")
    L.append("")
    if facts.get("ols_coef") is not None:
        L += _md_table(facts["ols_coef"],
                       ["composite", "direction", "beta", "se", "t",
                        "p_one_sided", "p_corrected", "significant"])
    L.append("### 3.3 Bootstrap 95% CIs")
    L.append("")
    if facts.get("ols_boot") is not None:
        L += _md_table(facts["ols_boot"],
                       ["predictor", "median", "ci_lo", "ci_hi", "n_iter"])
    L.append("### 3.4 Cronbach's α per composite")
    L.append("")
    if facts.get("ols_alpha") is not None:
        L += _md_table(facts["ols_alpha"],
                       ["composite", "k_items", "n_obs", "alpha",
                        "pass_gate", "note"])
    L.append(f"Full output directory: `{facts.get('ols_path', 'NA')}/`.")
    L.append("")
    return L


def section_exploratory_ml(cfg, facts):
    L = []
    L.append("## 4. Exploratory ML (sensitivity track)")
    L.append("")
    L.append("**Caveat (mandatory framing).** The exploratory ML grid is "
             "reported as a sensitivity check on the confirmatory OLS path, "
             "not as a competing primary analysis. The null results below "
             "are confounded with the Flanker construct-validity failure "
             "identified in Section 1.4 — they reflect partly the absence "
             "of a measurable conflict signal in the source task, not "
             "purely a power limitation. Until the Flanker is revised in "
             "the next instrumentation wave, an exploratory ML null is "
             "uninformative about whether QEEG biomarkers can predict "
             "EF-task performance.")
    L.append("")
    sm = facts.get("ml_summary") or {}
    if sm:
        L.append("### 4.1 Grid configuration")
        L.append("")
        L.append(f"- Models: {sm.get('models', [])}")
        L.append(f"- Schemes: {sm.get('schemes', [])}")
        L.append(f"- Targets run: {sm.get('targets_run', [])}")
        L.append(f"- CV: {sm.get('cv', {}).get('n_splits')}×"
                 f"{sm.get('cv', {}).get('n_repeats')} RepeatedKFold")
        L.append(f"- Permutation: {sm.get('n_perm')} permutations × "
                 f"{sm.get('perm_n_splits')}-fold null (selection-corrected "
                 "max-statistic per target across the scheme × model grid)")
        L.append(f"- k-cap (SelectKBest for non-linear models on wide "
                 f"pools): {sm.get('k_cap_for_kbest')}")
        L.append("")
    L.append("### 4.2 Per-combo results")
    L.append("")
    L.append("All combinations reported (no best-picking). Selection-"
             "corrected p applies to the max statistic across the (scheme × "
             "model) sub-grid within each target; uncorrected p is the "
             "per-combo permutation p shown alongside for diagnostic "
             "reference only.")
    L.append("")
    if facts.get("ml_grid") is not None:
        cols = ["target", "scheme", "model", "n_used", "n_features_input",
                "oof_r2", "oof_mae", "baseline_mae",
                "perm_p_uncorrected",
                "perm_p_selection_corrected_within_target"]
        cols = [c for c in cols if c in facts["ml_grid"].columns]
        L += _md_table(facts["ml_grid"], cols)
    L.append(f"Full grid CSV: `{facts.get('ml_path', 'NA')}/"
             "exploratory_ml_grid.csv`.")
    L.append("")
    return L


def section_screening_index(cfg, facts):
    L = []
    L.append("## 5. Screening index — illustrative prospect output")
    L.append("")
    L.append("**Non-clinical framing (mandatory).** The output below is a "
             "**candidate EF screening index**, not a diagnostic. It is "
             "computed for **one pre-specified combination** — the "
             "confirmatory OLS on the primary target — not cherry-picked "
             "from the exploratory ML grid. Per-subject scores are **LOO "
             "out-of-sample**, then Bayesian-shrunk toward the sample mean, "
             "with a per-subject prediction interval and a "
             "`confidence_flag` tied to whether the full model's LOO R² "
             "beats age alone (incremental F at α = 0.05). Percentile is "
             "**within-sample only** — there is no clinical norm, no "
             "external caseness label, and **no ROC reported**. This index "
             "inherits the construct validity of its target; for any "
             "Flanker-derived target it is illustrative only until the "
             "task is revised.")
    L.append("")
    si_meta = facts.get("si_meta")
    if si_meta:
        L.append("### 5.1 Index metadata")
        L.append("")
        L.append(f"- Target: `{si_meta.get('target')}`")
        L.append(f"- Covariates: `{si_meta.get('covariates')}`")
        L.append(f"- Composites used: `{si_meta.get('composites')}`")
        L.append(f"- N used: **{si_meta.get('n_used')}**")
        L.append(f"- LOO R² (full): **{_fmt(si_meta.get('loo_r2_full'), 4)}**")
        L.append(f"- LOO R² (age-only): **{_fmt(si_meta.get('loo_r2_age_only'), 4)}**")
        L.append(f"- Incremental F: **{_fmt(si_meta.get('incremental_F'))}**, "
                 f"p = **{_fmt(si_meta.get('incremental_p_value'))}**")
        L.append(f"- Shrinkage weight (w toward sample mean): "
                 f"**{_fmt(si_meta.get('shrinkage_weight'), 3)}**")
        L.append(f"- Confidence flag (sample-wide): "
                 f"**{si_meta.get('confidence_flag')}**")
        L.append("")
    L.append("### 5.2 Per-subject score table (first 20 rows)")
    L.append("")
    if facts.get("si_df") is not None:
        head = facts["si_df"].head(20)
        cols = [c for c in head.columns]
        L += _md_table(head, cols)
        L.append(f"Full table: `{facts.get('si_path', 'NA')}/"
                 "screening_index_*.csv`.")
        L.append("")
    L.append("### 5.3 What this is not")
    L.append("")
    L.append("- **Not a clinical diagnostic.** Score does not map to any "
             "validated clinical category.")
    L.append("- **Not anchored to external norms.** Percentile is within "
             "this 26-subject pilot; absolute interpretation requires "
             "BRIEF-2 / Conners-3 (or equivalent) caseness labels not "
             "collected in this pilot.")
    L.append("- **Not a ROC validation.** ROC against a pseudo-label "
             "derived from the same target would be circular. ROC is a "
             "main-study deliverable contingent on an external caseness "
             "reference.")
    L.append("- **Not a tier categorisation.** No tertile or quartile cut "
             "is presented; tiers labelled 'typical/monitor/needs "
             "evaluation' require external anchoring.")
    L.append("- **Not the prettiest member of an exploratory family.** "
             "This is the pre-specified confirmatory-path output; the "
             "exploratory ML grid in Section 4 is not used to choose what "
             "to show here.")
    L.append("")
    return L


def build_report(cfg, facts):
    sections = []
    sections += section_header(cfg, facts)
    sections += section_sample_and_quality(cfg, facts)
    sections += section_pipeline(cfg, facts)
    sections += section_confirmatory_ols(cfg, facts)
    sections += section_exploratory_ml(cfg, facts)
    sections += section_screening_index(cfg, facts)
    return "\n".join(sections)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    out_dir = make_output_dir()
    print(f"Research report output: {out_dir}")

    facts = collect_facts(cfg)
    missing = [k for k, v in facts["paths"].items() if v is None]
    if missing:
        print(f"  WARNING: upstream not found: {missing}")

    report = build_report(cfg, facts)
    (out_dir / "research_report.md").write_text(report, encoding="utf-8")
    print(f"  research_report.md: {len(report)} chars, "
          f"{report.count(chr(10))+1} lines")

    notes = {
        "stage": "research_report",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "sources": facts["paths"],
        "outputs": ["research_report.md"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
