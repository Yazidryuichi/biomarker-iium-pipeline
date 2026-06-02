# CLAUDE.md

Operating notes for Claude Code in this repository. Read this before making
changes.

## What this project is

A 5-stage Python pipeline plus two sidecar reporters that turns raw
resting-state EDF recordings into candidate QEEG biomarkers of executive
function in Indonesian children aged 6–12. Pilot N=26 (one subject-condition
dropped by the cleaning floor), target N=100. Full regression only — no
classification, no quantum, no median-split. See `README.md` for the
user-facing summary.

## Repository layout (load-bearing)

```
biomarker-iium-pipeline/
├── README.md
├── CLAUDE.md
├── .gitignore
├── data/                       EDF/ and Behavioral/ (gitignored)
├── preprocessing/              Stage 1  (pipeline)
├── validation/                 Stage 2  (pipeline)
├── feature_building/           Stage 3  (pipeline)
├── feature_engineering/        Stage 4  (pipeline)
├── analysis/                   Stage 5  (pipeline)
├── data_quality_check/         Sidecar reporter — pilot QC document
└── feasibility_report/         Sidecar reporter — pitch / feasibility doc
```

**Hard rules about the root**:
- Only the entries above may live at the top level. No other loose `.md`
  (METHODS/CONTRIBUTING/AI_TRANSPARENCY/etc.), no `.py`, `Dockerfile`,
  `Makefile`, `pyproject.toml`, `requirements.txt`, `utils/`, `scripts/`,
  `tests/`, `docs/`, `configs/`, `figures/`, `logs/`, `results/`,
  `notebooks/`, `pipeline.py`, etc.
- `CLAUDE.md` is the only Claude-instruction file allowed at root; do not
  spawn additional `.md` siblings to it.
- No shared `utils/` or `lib/`. Each stage inlines its own helpers (config
  loader, behavioural loaders, subject discovery). Some duplication of
  trivial helpers is the price for a clean root.
- No top-level orchestrator. Each stage is run explicitly:
  `python <stage>/main.py`.
- If you have something to communicate to the user (decisions, methodology
  notes), say it in chat — don't drop it as a new `.md` in the repo.

**Each stage holds exactly three things**:
```
<stage>/main.py            self-contained CLI entry
<stage>/config.yaml        globals + stage params in one file
<stage>/output/<ts>/       timestamped run output (gitignored)
```

`output/.gitkeep` is tracked so the folder exists in a fresh checkout. Run
artifacts (`<ts>/`) are gitignored.

**Pipeline stages vs sidecar reporters**:
- A *pipeline stage* has one predecessor and one successor (or is the head
  or tail). Re-running upstream invalidates it.
- A *sidecar reporter* reads from multiple upstream stages, produces a
  human-readable report, and is not consumed by anyone downstream. Two
  exist: `data_quality_check/` (pilot QC audit, internal use) and
  `feasibility_report/` (pitch / feasibility document, donor-facing).
  Sidecars follow the same three-thing layout but are run on demand, not
  as part of the run chain.

## Stage I/O contract

Every stage:
1. Loads `<stage>/config.yaml` via `yaml.safe_load` (inlined in `main.py`).
2. Auto-resolves its predecessor by scanning `<prev>/output/` for the
   lexically-latest `YYYY-MM-DD_HHMMSS` subdir.
3. Creates a new `<stage>/output/<YYYY-MM-DD_HHMMSS>/` for this run.
4. Writes its primary outputs + `run_notes.json` (timestamp, git commit,
   input dir consumed, outputs produced) — the audit trail.

There is no shared "run dir," no shared `results/`, no orchestrator.

## Pipeline flow

```
data/EDF + data/Behavioral
   │
   ▼ preprocessing/main.py       EDF → ICLabel + AutoReject → cleaned epochs
       └── output/<ts>/cleaned_epochs/*.fif + qc.json
   │
   ▼ validation/main.py          trial-level behavioural reliability + QC
       └── output/<ts>/{aufei,flanker,digit_span}_reliability.csv
   │
   ▼ feature_building/main.py    cleaned epochs → primitives.csv
       └── output/<ts>/features.csv + cov_matrices.npz
   │
   ▼ feature_engineering/main.py primitives + behavioural → composites
       └── output/<ts>/full_dataset.csv
   │
   ▼ analysis/main.py            hierarchical OLS hypothesis test
       └── output/<ts>/<target>/<feature_set>/{summary.json,
           coef_inference.csv, bootstrap_ci.csv, block_f_test.json,
           composite_alpha.csv}

   ┄ data_quality_check/main.py  (sidecar; runs on demand)
       reads preprocessing + validation + feature_engineering + analysis
       └── output/<ts>/{report.md, eeg_quality_*, aufei_*,
           flanker_*, digit_span_*, n_reconciliation,
           sample_demographics}.csv

   ┄ feasibility_report/main.py  (sidecar; runs on demand)
       reads all five pipeline stages + data_quality_check
       └── output/<ts>/{feasibility_report.md, key_metrics.csv}
```

`validation` reads `preprocessing/output/` for the pass-rate summary but
does not block downstream stages — `feature_building` only depends on
`preprocessing`.

**Sidecars**:
- `data_quality_check` is the internal QC document. Headline-first
  Markdown report plus supporting CSVs. Audience: PI + reviewers. Lead
  with construct-validity findings and target-reliability audit, not
  rosy summaries.
- `feasibility_report` is the donor/funder-facing feasibility pitch. Pulls
  numbers from all upstream stages plus the QC sidecar; templates the
  narrative prose in `main.py` so re-running with new data refreshes
  the document automatically. Budget/timeline/ask are config-driven
  placeholders the PI fills before circulating.

## Methodological invariants (do not regress)

Load-bearing for scientific validity. Do not "simplify" them.

- **Average reference applied BEFORE ICA fit**. Source EDFs are Mitsar
  A1-A2 implicit (no M1/M2 channels in file). We commit to avg-ref so the
  unmixing W and `ica.apply` share a reference frame, and so ICLabel —
  trained on avg-ref data — classifies in the frame it expects.
  `qc.json` records `source_reference` and `output_reference` for
  provenance.

- **ICA fit on a 1 Hz HP copy of the avg-referenced raw**; the unmixing
  matrix is then applied to the 0.5 Hz HP avg-ref raw. ICLabel will emit
  a runtime warning that data is not 1-100 Hz (we are 1-45 Hz because
  of our lowpass) — this is the documented compromise, not a bug.

- **Bad channel interpolation runs AFTER ICA**, not before. Interpolated
  channels inject smoothed data that degrades ICA decomposition.

- **Fp1/Fp2 exempt from variance-based bad-channel flagging**. Pediatric
  blinks make them naturally high-variance; flagging them pushes the
  blink topography out of the ICA fit, hiding the very component
  ICLabel is supposed to classify.

- **ICLabel + AutoReject local, no lenient retry.** Artifact components
  are flagged when label ∈ `iclabel_exclude_labels` AND probability >
  `iclabel_threshold`. Epoch rejection uses AutoReject local (per-channel
  CV thresholds, per-epoch interpolation). The data-driven lenient retry
  was removed because it gave noisy recordings looser thresholds, biasing
  comparisons. `min_epochs` is the only gate.

- **`min_epochs` floor**. Subject-condition recordings whose surviving
  epoch count falls below this floor are dropped from disk (no `*-epo.fif`
  saved); downstream stages never see them.

- **`np.trapezoid` for spectral integration**, not `np.sum`. Per-epoch
  coherence and PAC, not on concatenated epochs.

- **Zero-phase filtering for PAC** (`sig.sosfiltfilt`, not `sosfilt`). The
  causal `sosfilt` shifts phase, biasing the modulation index. PAC of the
  old pipeline used causal — this was a bug.

- **Trial-level reliability when trial data exists**. The Flanker workbook
  has a `Trials` sheet (1680 rows). Validation reads it and computes
  odd-even split-half + Spearman-Brown on DDM and RT. Do NOT regress this
  to "summary-statistics surrogates" — that was the prior methodological
  error.

- **Hierarchical OLS with age as a regressor, not pre-residualization**.
  Restricted model `y ~ age_months`, full model `y ~ age_months + comps`.
  By Frisch-Waugh-Lovell this is equivalent to residualizing both y AND
  the features against age; keeps the diagnostics (covariate R², full R²,
  block F) interpretable. Pre-residualizing y and feeding bare features
  loses these diagnostics and risks leakage if the residualization
  parameters are estimated on the full sample.

- **Cronbach's α gate per composite** before the composite is treated as
  a measurement of its construct. Threshold: α ≥ 0.50 minimum, ≥ 0.70
  preferred. Composites failing the gate are still computed but should be
  flagged in interpretation.

- **Multiplicity correction restricted to pre-specified hypotheses**
  (Bonferroni or FDR-BH across composites within a feature set). Do not
  extend correction scope to exploratory sweeps without flagging the
  scope change explicitly.

- **Aperiodic correction is opt-in, not default**. `feature_building`
  produces `psd_periodic_<band>_<ch>` columns when specparam is installed
  AND `params.aperiodic_correction: true`. `feature_engineering`
  auto-builds `*_periodic` composite siblings when periodic primitives
  exist. The analysis stage runs both raw and `_periodic` feature sets as
  sensitivity analysis. Treats the periodic version as the cleaner
  estimate for paper-ready inference, the raw as the legacy comparison.

## Knob locations

```
preprocessing/config.yaml         bandpass, notch, epoch_duration, edge_crop,
                                  ica_method ("infomax"), ica_extended,
                                  ica_n_components, iclabel_threshold,
                                  iclabel_exclude_labels,
                                  bad_channel_threshold,
                                  bad_channel_corr_threshold,
                                  bad_channel_flatline_std,
                                  variance_protect_channels (["Fp1","Fp2"]),
                                  use_autoreject_local, fallback_reject_uv,
                                  max_reject_pct, min_epochs

validation/config.yaml            aufei_subscales (item codes per subscale),
                                  flanker_split (odd_even | random),
                                  flanker_min_trials_per_half,
                                  flanker_metrics,
                                  alpha_min_acceptable, alpha_preferred

feature_building/config.yaml      bands, coherence_pairs, wavelet*,
                                  aperiodic_correction (bool),
                                  specparam_* (freq_range, peak_width_limits,
                                  max_n_peaks, min_peak_height)

feature_engineering/config.yaml   aufei_subscales (must mirror validation),
                                  assessment_date, min_matched_n,
                                  tbr_channels, faa_left, faa_right,
                                  posterior_alpha_channels,
                                  apriori_*_channels (Tier 1 components),
                                  build_periodic_composites (bool)

analysis/config.yaml              targets[], covariates (default age_months),
                                  feature_sets{name: [composite cols]},
                                  directions{composite: positive|negative},
                                  composite_components{composite: [items]}
                                  (drives Cronbach gate),
                                  alpha_level, correction, bootstrap_n,
                                  alpha_min_acceptable

data_quality_check/config.yaml    aufei_subscales (must mirror validation),
                                  aufei_likert_min / aufei_likert_max
                                  (ceiling/floor thresholds for items),
                                  aufei_sd_fraction_of_range_floor
                                  (variance-restriction flag),
                                  flanker_acc_ceiling (DDM-undefined flag
                                  threshold; default 0.95),
                                  flanker_subchance_threshold (default 0.50
                                  for 2AFC; identifies likely
                                  response-mapping reversals),
                                  flanker_effect_typical_min_ms
                                  (construct-validity floor from
                                  literature),
                                  difference_score_sb_floor (auto-flag
                                  unusable difference-score targets),
                                  flanker_rt_floor_sec, digit_span_*_max,
                                  write_report_md

feasibility_report/config.yaml    project_title, institution,
                                  collaborating_site, ethics_body
                                  (header metadata for the pitch);
                                  main_study_n_target,
                                  main_study_primary_target,
                                  main_study_secondary_target,
                                  retracted_targets[] (analysis-plan
                                  outputs from the pilot);
                                  power_assumptions{expected_r,
                                  alpha_two_sided, target_power,
                                  delta_r2_detectable, k_composites};
                                  preregistered_exclusion_rules[];
                                  fixes{flanker_task, aufei_instrument,
                                  digit_span, aperiodic_correction,
                                  target_replacement} (narrative blocks
                                  for §6); budget_placeholder,
                                  timeline_placeholder, ask_placeholder
                                  (PI fills before circulating)
```

To add a new knob: put it in `params:` of the stage's config that needs it.
If two stages need the same knob (e.g. `aufei_subscales`), duplicate it —
do NOT introduce a globals file or shared loader.

## Adding a new feature

Decide whether it's a **primitive** (computed from raw epochs — slow) or a
**composite** (computed from columns already in features.csv — fast).

- **Primitive** → `feature_building/main.py`. Add an `extract_*` function,
  call it inside the per-subject-per-condition loop in `main()`, prefix the
  resulting columns with the condition prefix (`eo_`/`ec_`). Re-running
  feature_building is slow (minutes per subject).

- **Composite** → `feature_engineering/main.py:add_all_engineered`. Build
  from existing `psd_abs_*`/`psd_periodic_*` etc. columns. If the composite
  has multiple constituent items, add the components list to
  `analysis/config.yaml:composite_components` so the Cronbach α gate runs
  on it. Re-running feature_engineering takes seconds.

## Adding a new behavioural measure

1. Add the column to the appropriate loader in
   `feature_engineering/main.py` (`load_aufei` / `load_flanker` /
   `load_digit_span`) and add it to the `keep` list.
2. If you want to use it as a target, add it to
   `analysis/config.yaml:targets`.
3. If it has trial-level data, add a reliability check in
   `validation/main.py` mirroring the Flanker pattern (split into halves,
   compute the metric per half, Spearman-Brown corrected r across
   subjects).

## Running

```bash
# Pipeline (sequential, each consumes the previous stage's latest output)
python preprocessing/main.py        # ~5-10 min for 52 files
python validation/main.py           # <30 sec
python feature_building/main.py     # ~5-10 min
python feature_engineering/main.py  # <30 sec
python analysis/main.py             # ~30 sec, bootstrap is the bottleneck

# Sidecars (on demand)
python data_quality_check/main.py   # <10 sec — internal QC audit
python feasibility_report/main.py   # <5 sec  — donor-facing pitch doc
```

Re-run only what changed. Each stage caches; downstream stages auto-pick
the latest run of their predecessor.

| What changed | Re-run from |
|---|---|
| Raw EDF, ICA / AutoReject / bandpass params | `preprocessing` (all downstream invalid) |
| Behavioural workbook contents | `validation` and `feature_engineering` |
| `bands`, `coherence_pairs`, wavelet, aperiodic flag | `feature_building` |
| Composite definitions, behavioural scoring, age params | `feature_engineering` |
| Targets, feature sets, directions, correction, bootstrap N | `analysis` only |
| Want a fresh pilot quality snapshot | `data_quality_check` (sidecar, doesn't invalidate anything else) |
| Want a refreshed donor / feasibility doc | `feasibility_report` (sidecar; re-pulls all upstream) |

Old `<stage>/output/<ts>/` accumulate; prune by hand.

## Conventions

- Python 3, sklearn-style only where used (analysis is statsmodels OLS).
- File paths inside stages: NEVER reference `utils.io` or sibling stages
  by absolute import — every stage is meant to be runnable standalone.
- Random seeds: read `config["random_state"]` (top-level global). Each
  `main.py` should explicitly seed numpy before any stochastic step.
- Optional dependencies (`specparam`/`fooof`, `factor_analyzer`) imported
  at point of use inside `try/except ImportError`, skipped gracefully.
- Print statements go to stdout; long-running stages should log progress
  per subject.
- Windows shell: forward slashes; `python` resolves to a working
  interpreter on the user's machine.

## Documentation

- `README.md` — user-facing summary (what runs, what each stage does,
  data layout).
- `CLAUDE.md` — this file. Operating notes for Claude Code only.
- `data_quality_check/output/<ts>/report.md` — generated pilot quality
  report. Gitignored (under `output/`); regenerate with the sidecar
  whenever a refreshed snapshot is needed. Do not promote it to a
  committed top-level `.md`.
- `feasibility_report/output/<ts>/feasibility_report.md` — generated
  donor/funder-facing feasibility document. Gitignored. Regenerate
  whenever upstream stages produce new evidence the pitch should reflect.
  Do not promote to a committed top-level `.md`.

Do not create any other `.md` files at the root or in stage folders unless
explicitly requested. Methodology notes go in docstrings + this file.

## Pilot quality findings to keep in mind

These are baselines observed in the 2026-05-29 end-to-end run + the data
quality audit at `data_quality_check/output/2026-05-29_165405/report.md`.
Re-run `data_quality_check` for fresh numbers; the qualitative problems
below are likely to persist until the measurement instruments are revised.

- **Flanker is construct-broken in the pilot.** Mean `flanker_effect` =
  1.5 ms vs literature 30–80 ms (pediatric); `rt_congruent` ≈
  `rt_incongruent` at the sample level; 75% accuracy ceiling, 39% at
  exact 1.0. The task did not induce a conflict signal — any null
  observed in `analysis/` is downstream of this construct-validity
  failure, not weak biomarkers. Do not interpret a Flanker-derived null
  result as biomarker-level. The main study requires task modification
  (harder distractors, response deadline, or task replacement).
- **Do not propose any difference score as a primary target.** Specifically
  not `flanker_effect` (SB = 0.13) and not `ddm_delta_v` (SB not
  estimable: only n = 2 valid split-half pairs after EZ-DDM degeneracy
  from the ceiling). This is structural — difference of two highly
  correlated reliable measures has low or undefined reliability. See
  [[feedback-difference-score-targets]].
- **Recommended Flanker target ordering** (replacing the prior bad
  recommendation): primary = `rt_cv` (SB = 0.97 on n = 28); secondary =
  `ddm_v_incongruent` (SB = 0.99 but on n = 6 non-ceiling subsample —
  caveat in any report); not `acc_*` (pseudo-reliable at ceiling but
  uninformative); never `ddm_delta_v` or `flanker_effect`.
- **D0000816 is a pre-registered exclusion candidate.** Subject scored
  `acc_incongruent = 0.07` (below 2AFC chance 0.5). Most likely
  response-mapping reversed. Drives the min of v_incongruent (−2.13) and
  delta_v (−3.96) distributions. Recommend adding
  `acc_incongruent < flanker_subchance_threshold` (default 0.50) as an
  explicit exclusion rule in the main study analysis plan.
- **Two DDM estimators in play — do not conflate them.** The analysis
  target column (`ddm_v_incongruent`, `ddm_delta_v`, etc.) is read
  **as-is** from the workbook's `Features` sheet; the estimator is
  unknown to this pipeline but visibly handles `acc = 1.0` (not EZ-DDM).
  The reliability estimates reported by `validation/` and
  `data_quality_check/` use **EZ-DDM** on split halves of the trial-level
  data, which collapses to n = 4–11 because EZ is undefined at
  `acc ∈ {0, 1}`. The reliability of the workbook's DDM column at the
  full sample size is therefore *unknown*. Do not claim the workbook
  uses "a more robust estimator" without independent evidence; the prior
  framing-note that did so was inherited from pre-rewrite code and is
  reframed as: *unknown estimator that handles ceiling cases
  differently from EZ*.
- **AUFEI ceiling/floor.** IC3 has zero variance (every pilot subject
  scored 4); CF2 is at 89% ceiling. IC3 is already excluded from the IC
  subscale composition; CF2 is a new flag — recommend exclusion or
  rewrite in the next pilot wave.
- **AUFEI low reliability is partly variance restriction, not only item
  content.** WM subscale SD = 0.26 on a 1–4 Likert scale (~9% of range)
  — this is parent-report social-desirability compression to the top
  end. The fix is *re-anchor the response scale* (e.g. 1–7 with concrete
  behavioural anchors, or frequency-based items), not just item rewrite.
  A CFA on these data without addressing variance restriction will
  produce a degenerate solution. AUFEI subscales should be treated as
  construct-invalid until the instrument revision.
- **Digit Span has no item-level data in the current export.** FW vs BW
  is the only available reliability proxy (SB = 0.46) and the
  parallel-halves assumption is invalid (FW = passive retention,
  BW = active manipulation). Report as approximate, not definitive. For
  the main study, request item-level output from the testing platform
  and mirror the Flanker trial-level pattern in `validation/main.py`.
- **Age is a strong confound for the DDM drift target.** Restricted-model
  R² (`ddm_v_incongruent ~ age_months`) was 0.31 in the pilot. The
  hierarchical OLS handles this natively via Frisch-Waugh — keep it; do
  not switch to a CV regression path that drops `age_months` from the
  design matrix.
- **N reconciliation.** 28 behavioural → 51/52 EEG recordings OK
  (D0000798/Eyes_Closed dropped at 53/105 epochs below `min_epochs`) →
  25 subjects with both EO and EC → 26 merged behavioural ∩ EEG
  (D0000796 and D0000823 have behavioural but no EDF files) → 25 in the
  OLS (1 dropped for `age_months` NaN). State these explicitly in any
  report that quotes N.
