# Data Quality Report — Pilot Cohort (N = 26)

Snapshot of the latest engineered dataset used by Stage 4 analysis. Generated against:

- Engineering output: `stages/engineering/runs/2026-05-09_110131/full_dataset.csv` (commit `2b48a24`)
- Analysis output: `stages/analysis/runs/2026-05-09_113431/` (six targets, regression-first)

## 1. Cohort

| Property | Value |
|---|---|
| Subjects (post-cleaning) | **26** |
| Sex distribution | Laki-laki = 13, Perempuan = 13 |
| Age | 9.80 ± 1.63 years (range 6.8 – 12.8) |
| Conditions used by main pipeline | Eyes-Open + Eyes-Closed (resting state) |
| Emotional conditions | Available (Happy, Calm, Sad, Scare) but not in the default config |
| Recording rate | per `configs/config.yaml:recording.sfreq` |
| Channels | 15-channel pediatric montage (per config) |

The pilot started at N = 28 in the behavioral files; **2 subjects were dropped** because their EEG cleaning failed the `cleaning.params.min_epochs` floor on at least one of the two resting-state conditions, so no `*-epo.fif` was written and downstream stages never see them. This is the protocol-mandated exclusion (see `CLAUDE.md` "Methodological invariants").

## 2. Behavioral data completeness

All 26 subjects in the analysis cohort have **fully populated behavioral data** for the targets listed below, after the 2026-05-09 fix to `utils.io.load_flanker` and `stages/engineering/engineering.py` which preserves the pre-computed per-condition DDM columns. Before that fix, 11/28 at-ceiling subjects were dropped at flanker EZ-DDM re-derivation.

| Domain | Variable | Non-NaN | Notes |
|---|---|---|---|
| AUFEI-O (parent-rated EF) | Global_EF | 26/26 | mean of 5 subscales |
|  | WM_score, IC_score, CF_score, P_score, SF_score | 26/26 | subscale means |
| Flanker accuracy | acc_overall, acc_incongruent | 26/26 | 11/28 had acc_overall = 1.0; 14/28 had acc_incongruent = 1.0 |
| Flanker RT (s) | rt_mean, rt_congruent, rt_incongruent | 26/26 | converted from ms when needed |
| DDM overall (legacy) | ddm_v, ddm_a, ddm_t | **15/26** | EZ-DDM at-ceiling failures retained from legacy column |
| DDM split (preferred) | ddm_v_congruent, ddm_v_incongruent | 26/26 | from source workbook |
|  | ddm_a_congruent, ddm_a_incongruent | 26/26 | from source workbook |
|  | ddm_t0_congruent, ddm_t0_incongruent | 26/26 | from source workbook |
|  | ddm_delta_v | 26/26 | drift-rate difference |
| Digit Span | FW_Span, BW_Span, Total_Span | 26/26 | scored span scores |
|  | FW_Raw, BW_Raw | 26/26 | raw counts |
| Demographics | age_years, age_months | 26/26 | computed from DoB at `assessment_date` |

**Column to prefer for analysis:** for inhibitory-control modelling the
condition-specific drift `ddm_v_incongruent` is the recommended target
because it is fully populated. The legacy overall `ddm_v` (15/26) should
be treated as an alternate measure, not as ground truth.

## 3. EEG data completeness

The features stage reads cleaned epochs and produces 1825 raw EEG features per subject (PSD per band per channel × 2 conditions; coherence per pair per band × 2 conditions; wavelet, Hjorth, spectral entropy, PAC, frequency-band covariance). After Stage 3 engineering this becomes 1841 columns (16 derived: TBR per channel + frontal mean, FAA, alpha reactivity global + per-channel) plus behavioral columns → **1900 columns × 26 subjects**.

A small number of TBR features are 25/26 (one subject's EO recording
failed to produce a valid PSD ratio for the channel in question — already
absorbed by the analysis stage's `SimpleImputer(strategy="median")`).

## 4. Stage-4 unsupervised feature curation (run 2026-05-09_113431)

Curation runs once on the full feature pool **before** the CV split and
applies no target labels, so it cannot leak the outcome into training
folds.

| Step | Threshold | n features |
|---|---|---|
| Input feature pool (union of requested feature_sets) | — | **1840** |
| After `drop_low_variance` | var > 1e-6 | **250** |
| After `drop_collinear_hierarchical` | \|corr\| < 0.95 | **243** |

7 features were collapsed into clusters by collinearity. The full kept-feature list and `cluster_map` (kept feature → list of dropped followers) are recorded in `feature_curation_report.json` for audit.

## 5. Per-target preparation (age residualization)

`age_months` is regressed out of every continuous target with ordinary least squares before CV, and the residual is used for both regression CV and the post-hoc clinical threshold. The clinical threshold is the **bottom tertile** (q ≈ 0.333) of the residualized distribution; the at-risk direction is *lower residual = at-risk*.

| Target | n used | Age model R² | Tertile threshold | At-risk prevalence |
|---|---|---|---|---|
| Global_EF | 26 | 0.097 | −0.054 | 34.6% (9/26) |
| IC_score | 26 | 0.009 | −0.124 | 34.6% (9/26) |
| WM_score | 26 | 0.066 | −0.134 | 34.6% (9/26) |
| BW_Span | 26 | 0.145 | −0.515 | 34.6% (9/26) |
| ddm_v_incongruent | 26 | 0.306 | −0.108 | 34.6% (9/26) |
| ddm_delta_v | 26 | 0.040 | +0.053 | 34.6% (9/26) |

**Interpretation.** Age residualization is most consequential for `ddm_v_incongruent` (R² = 0.306 — 30% of the raw variance is age-trajectory). For the AUFEI parent-rated subscales (Global_EF, IC_score, WM_score) the age effect is small (R² < 0.10), so residualization is largely a no-op there. For `BW_Span` the age effect is moderate (R² = 0.145).

## 6. Known limitations

- **N = 26 is small.** Repeated 5×5-fold CV gives ≈ 5-subject test folds. With ≈ 240 features after curation this is genuinely under-powered for stable feature-importance ranking. Reporting bootstrap 95% CIs and permutation p-values is mandatory.
- **At-ceiling Flanker accuracy.** 11/26 subjects had perfect overall accuracy and 14/26 had perfect incongruent accuracy. EZ-DDM is undefined at p_correct ∈ {0, 1}, which is why the pre-computed split-by-condition columns from the source workbook are preferred. The legacy `ddm_v` column inherits the at-ceiling NaNs and should not be used as the primary outcome.
- **Class imbalance in some `_group` columns.** Median-split binary group columns are kept in the engineering output for the legacy classification path only. In several measures (`acc_incongruent`, `ddm_v` etc.) the imbalance is severe — these columns should not be used as classification targets at this N.

## 7. Files referenced

- `stages/engineering/runs/2026-05-09_110131/full_dataset.csv` — single source of truth for analysis.
- `stages/analysis/runs/2026-05-09_113431/feature_curation_report.json` — exact kept/dropped feature lists.
- `stages/analysis/runs/2026-05-09_113431/<target>/target_preparation_report.json` — per-target residualization diagnostics.
- `scripts/runs/2026-05-09_115105/significant_spearman_fdr.csv` — FDR-significant behavioral cross-correlations (see Section 3 of `docs/manuscript_methods_pipeline.md`).
