# CLAUDE.md

Operating notes for Claude Code in this repository. Read this before making
changes.

**The behavioural side lives in `instruments/CLAUDE.md`** — the three measures,
their construct-validity findings, which targets are usable, and how to add a
new behavioural measure. This file covers the EEG pipeline: layout, stages,
preprocessing invariants, and the analysis path.

## What this project is

A 5-stage Python pipeline plus two sidecar reporters that turns raw
resting-state EDF recordings into candidate QEEG biomarkers of executive
function in Indonesian children aged 6–12. Target N=100; see the pilot N
reconciliation at the end of this file.

**Primary inferential mode: confirmatory hierarchical OLS** on a 9-feature
a priori union of theory-driven QEEG composites, with age as a regressor
(Frisch-Waugh handling of the maturation confound). **Sensitivity track:
exploratory ML grid** (ElasticNetCV primary; RandomForest / XGBoost
sensitivity) over the same composites plus a data-driven curated pool plus a
hybrid; selection-corrected permutation wraps the entire (target × scheme ×
model) grid. No median-split classification, no quantum, no data-driven
feature search on the confirmatory path.

Exploratory ML nulls in this pilot are partly confounded with the Flanker
construct-validity failure — that caveat must accompany any exploratory result
in a report. `README.md` holds the user-facing summary.

## Repository layout (load-bearing)

```
biomarker-iium-pipeline/
├── README.md
├── CLAUDE.md
├── LICENSE                     MIT (kept from the pre-restructure main branch)
├── .gitignore
├── .github/workflows/ci.yml    CI (kept from main; paths need repointing)
├── requirements.lock           pinned deps (kept from main)
├── data/                       EDF/ and Behavioral/ (gitignored)
├── instruments/                the tasks that produce the behavioural data
│   ├── CLAUDE.md               behavioural-side operating notes
│   ├── flanker/                PsychoPy Eriksen Flanker + cleaning notebooks
│   └── digit_span/             browser Digit Span admin tool + TTS stimuli
├── preprocessing/              Stage 1  (pipeline)
├── validation/                 Stage 2  (pipeline)
├── feature_building/           Stage 3  (pipeline)
├── feature_engineering/        Stage 4  (pipeline)
├── analysis/                   Stage 5  (pipeline)
├── data_quality_check/         Sidecar reporter — pilot QC document
└── research_report/            Sidecar reporter — research report
```

**Hard rules about the root**:
- Only the entries above may live at the top level. No other loose `.md`
  (METHODS/CONTRIBUTING/AI_TRANSPARENCY/etc.), no `.py`, `Dockerfile`,
  `Makefile`, `pyproject.toml`, `requirements.txt`, `utils/`, `scripts/`,
  `tests/`, `docs/`, `configs/`, `figures/`, `logs/`, `results/`,
  `notebooks/`, `pipeline.py`, etc.
- `LICENSE`, `.github/`, and `requirements.lock` are the three deliberate
  exceptions, carried over when this layout replaced the old one on `main`
  (2026-07-27). A public repo without a licence, without CI, and without a
  dependency pin is a regression, and none of the three competes with the
  flat-stage design. Do not delete them as "root clutter".
- `instruments/` is the fourth exception. It is NOT a pipeline stage and does
  not follow the three-thing stage layout — it holds the measurement tools
  themselves. Do not fold them into a stage, and do not add a stage that
  imports from them. `instruments/` publishes the task, never the responses;
  the full data rule is in `instruments/CLAUDE.md`.
- Root `CLAUDE.md` is the only Claude-instruction file allowed at root; do not
  spawn `.md` siblings to it. Nested `CLAUDE.md` under `instruments/` is the
  documented exception (one for the behavioural side, one per instrument).
- No shared `utils/` or `lib/`. Each stage inlines its own helpers (config
  loader, behavioural loaders, subject discovery). Some duplication of trivial
  helpers is the price for a clean root.
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

`output/.gitkeep` is tracked so the folder exists in a fresh checkout.

**Pipeline stages vs sidecar reporters**: a *pipeline stage* has one
predecessor and one successor (or is head or tail), and re-running upstream
invalidates it. A *sidecar reporter* reads from multiple upstream stages,
produces a human-readable report, and is consumed by no one downstream.
Sidecars follow the same three-thing layout but run on demand, not as part of
the chain.

## Stage I/O contract

Every stage:
1. Loads `<stage>/config.yaml` via `yaml.safe_load` (inlined in `main.py`).
2. Auto-resolves its predecessor by scanning `<prev>/output/` for the
   lexically-latest `YYYY-MM-DD_HHMMSS` subdir.
3. Creates a new `<stage>/output/<YYYY-MM-DD_HHMMSS>/` for this run.
4. Writes its primary outputs + `run_notes.json` (timestamp, git commit, input
   dir consumed, outputs produced) — the audit trail.

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
       └── output/<ts>/full_dataset.csv + apriori_theory_mapping.csv
   │
   ▼ analysis/main.py            confirmatory OLS + exploratory ML + screening index
       └── output/<ts>/<target>/<feature_set>/{summary.json,
           coef_inference.csv, bootstrap_ci.csv, block_f_test.json,
           composite_alpha.csv}
       └── output/<ts>/exploratory_ml/{exploratory_ml_grid.csv,
           exploratory_ml_summary.json}
       └── output/<ts>/screening_index/{screening_index_<label>.csv,
           screening_index_<label>_meta.json}

   ┄ data_quality_check/main.py  (sidecar; runs on demand)
       reads preprocessing + validation + feature_engineering + analysis
       └── output/<ts>/{report.md, eeg_quality_*, aufei_*, flanker_*,
           digit_span_*, n_reconciliation, sample_demographics,
           behavioral_correlation_*}.csv
           + behavioral_correlation_heatmap.png

   ┄ research_report/main.py     (sidecar; runs on demand)
       reads all five pipeline stages + data_quality_check
       └── output/<ts>/research_report.md
```

`validation` reads `preprocessing/output/` for the pass-rate summary but does
not block downstream stages — `feature_building` only depends on
`preprocessing`.

**Sidecars**:
- `data_quality_check` — internal QC document, headline-first Markdown plus
  supporting CSVs. Audience: PI + reviewers. Lead with construct-validity
  findings and the target-reliability audit, not rosy summaries.
- `research_report` — consolidated five-section research document: (1) sample
  & data quality, (2) pipeline description + flow diagram, (3) confirmatory
  a priori OLS, (4) exploratory ML grid with the construct-validity caveat,
  (5) screening index — illustrative prospect output (LOO out-of-sample,
  Bayesian-shrunk, with prediction interval; explicit non-clinical labeling).

## Methodological invariants (do not regress)

Load-bearing for scientific validity. Do not "simplify" them.

**Preprocessing**

- **Average reference BEFORE ICA fit.** Source EDFs are Mitsar A1-A2 implicit
  (no M1/M2 in file). Avg-ref means the unmixing W and `ica.apply` share a
  reference frame, and ICLabel — trained on avg-ref data — classifies in the
  frame it expects. `qc.json` records `source_reference` / `output_reference`.
- **15 channels, not 19.** Exported EDFs carry Fp1/Fp2, F7/F3/Fz/F4/F8,
  C3/Cz/C4, P3/Pz/P4, O1/O2; the 4 temporal electrodes (T3/T4/T5/T6) are not
  in the file (`n_channels_raw = 15`). The device is nominally 19-channel and
  the IJP manuscript uses 19-channel framing, but the pipeline only ever
  processes 15. This does not change the confirmatory analysis — the 9 union
  composites use only Fz/Cz (FC) and O1/O2/Pz (PO). **Do not claim 19 channels
  were analysed**; describe composites by their region channels.
- **ICA fit on a 1 Hz HP copy** of the avg-referenced raw; the unmixing matrix
  is then applied to the 0.5 Hz HP avg-ref raw. ICLabel warns that data is not
  1–100 Hz (we are 1–45 Hz because of our lowpass) — documented compromise,
  not a bug.
- **Bad-channel interpolation AFTER ICA**, never before. Interpolated channels
  inject smoothed data that degrades the decomposition.
- **Fp1/Fp2 exempt from variance-based bad-channel flagging.** Pediatric blinks
  make them naturally high-variance; flagging them pushes the blink topography
  out of the ICA fit, hiding the very component ICLabel should classify.
- **ICLabel + AutoReject local, no lenient retry.** Components are flagged when
  label ∈ `iclabel_exclude_labels` AND probability > `iclabel_threshold`. Epoch
  rejection is AutoReject local (per-channel CV thresholds, per-epoch
  interpolation). The data-driven lenient retry was removed because it gave
  noisy recordings looser thresholds, biasing comparisons. `min_epochs` is the
  only gate; subject-conditions below it are dropped from disk (no `*-epo.fif`
  saved) and downstream stages never see them.

**Feature extraction**

- **`np.trapezoid` for spectral integration**, not `np.sum`. Per-epoch
  coherence and PAC, not on concatenated epochs.
- **Zero-phase filtering for PAC** (`sig.sosfiltfilt`, not `sosfilt`). Causal
  `sosfilt` shifts phase and biases the modulation index — that was a bug in
  the old pipeline.
- **PAF = spectral centroid first, `find_peaks` second.** Children's alpha is
  often broad without a sharp peak (Klimesch 1999; Grandy et al. 2013).
  `extract_paf` defaults to the gravity frequency in 7–13 Hz; find_peaks is
  used only when a sharp prominent peak is detected AND lies within 2 Hz of the
  centroid. Do not flip the default without re-validating on the developmental
  sample.
- **Aperiodic correction is on by default when specparam is installed.**
  `feature_building` emits `psd_periodic_<band>_<ch>`,
  `aperiodic_exponent_<ch>`, `aperiodic_offset_<ch>` per channel per condition;
  the union composites include `aperiodic_{exponent,offset}_{FC,PO}` as primary
  features. specparam 2.0 API: `m.get_params("aperiodic")` (returns
  `[offset, exponent]` in fixed mode); spectra via
  `m.results.model.modeled_spectrum` and `m.results.model._ap_fit`. The old
  `m.fooofed_spectrum_` / `m.aperiodic_params_` attributes are gone in 2.0.

**Analysis**

- **Hierarchical OLS with age as a regressor, not pre-residualization.**
  Restricted `y ~ age_months`, full `y ~ age_months + comps`. By
  Frisch-Waugh-Lovell this equals residualizing both y AND the features against
  age, while keeping the diagnostics (covariate R², full R², block F)
  interpretable. Pre-residualizing y and feeding bare features loses these and
  risks leakage if the residualization parameters are estimated on the full
  sample.
- **A priori UNION (9 composites) is the primary feature set.** Region-collapsed
  (FC = Fz+Cz; PO = O1+O2+Pz): relative band powers, aperiodic
  exponent/offset, PAF, alpha reactivity. The legacy 3-feature Tier-1
  (`fm_theta_eo`, `posterior_alpha_ec`, `tbr_frontal_eo_log`) remains a
  sensitivity feature set but is no longer primary. TBR, FAA and coherence are
  intentionally excluded from the union (TBR = collinear with rel_theta +
  rel_beta; FAA = off-construct for cold-EF rest; coherence = fragile at short
  avg-ref epochs). They are computed and available as correlate-table rows but
  never enter the OLS or ML grid.
- **Per-target directional priors** are loaded from `apriori_theory_mapping.csv`
  emitted by `feature_engineering` — the single source of truth for which
  composites are predicted positive or negative per target. The `directions:`
  block in `analysis/config.yaml` covers only the legacy Tier-1.
- **Cronbach's α gate per composite** before it is treated as a measurement of
  its construct. α ≥ 0.50 minimum, ≥ 0.70 preferred. Failing composites are
  still computed but must be flagged in interpretation.
- **Multiplicity correction restricted to pre-specified hypotheses**
  (Bonferroni or FDR-BH across composites within a feature set). Do not extend
  correction scope to exploratory sweeps without flagging the scope change.
- **Exploratory ML grid is sensitivity, not primary.** ElasticNetCV is primary
  (linear, regularized, l1_ratio path for collinearity). RandomForest
  (depth=3, min_samples_leaf=3) and XGBoost (when installed, depth=3) are
  non-linearity / model-class checks. No nested hyperparameter tuning beyond
  ElasticNetCV's internal path (Vabalas 2019 — outer tuning at N≈26 inflates
  the estimate). Three schemes: theory (9 union direct), data_driven (full
  curated pool; L1 self-selects for ElasticNet, in-CV SelectKBest k=10 for
  trees), hybrid (union + L1 on rest pool).
- **Selection-corrected permutation wraps the full ML grid**, not the best
  post-hoc combo: per-target max-statistic permutation across the
  (scheme × model) sub-grid. Per-combo uncorrected p is also reported but
  flagged as such. Do not pick the best combo and re-permute on it alone —
  that gives an anti-conservative p.
- **Screening index runs for ONE pre-specified combo**, not the whole grid
  (`feedback_no_manufactured_validation` Rule 7). Pre-specified in config
  (`screening_index.target` + `feature_set`) as the confirmatory OLS combo:
  primary target `rt_cv`, feature set `a_priori_union`. LOO predictions +
  Bayesian shrinkage + per-subject prediction interval + `confidence_flag` tied
  to incremental F over age-only. NOT a diagnostic. NOT ROC against any
  pseudo-label (no external caseness label exists in the pilot). Percentile
  within-sample only — no tier labels.
- **Two live OLS targets, construct-validity independence first.** Primary
  `rt_cv` (construct-valid even when the Flanker is broken; SB = 0.97 on
  n = 28). Secondary `Global_EF` (parent-report EF, single item-level
  composite; biomarker-null in the pilot). `ddm_v_incongruent` and `BW_Span`
  are retired from the target list — still computed as columns, not entered
  into the OLS. `ddm_delta_v` and `flanker_effect` remain retracted. Rationale
  and the full target ranking: `instruments/CLAUDE.md`.
- **`Global_EF` is a single item-level composite** — mean of all 25 AUFEI-O
  items, built by `feature_engineering.load_aufei` minus `aufei_drop_items`
  (default `[]`). Item screening was evaluated and abandoned: dropping 7 items
  moved α only 0.81 → 0.80. Report the global α / ω only, never per-subscale on
  the confirmatory path.

**IAPS affective track (optional)**

IAPS emotion features are optional and dual-baseline (within primary, EO
sensitivity). The affective-viewing EDFs (`IGS_1_Happy` / `2_Calm` / `3_Sad` /
`4_Scare`; Scare = Fear) are separate per-emotion files, NOT annotations in one
recording; each runs marker-to-marker (emotion onset ~0 s, next-emotion marker
~1 s before EOF, then `BAD_ACQ_SKIP`). There is NO annotated fixation, so onset
detection falls back to 0 s.

When `preprocessing.params.iaps.enable`, preprocessing cleans each emotion file
ONCE through the resting ICA/AutoReject path (ICA on the full recording) and
crops TWO non-overlapping 15 s windows: `response` = first 15 s after onset
(transient-safe, `[5,20] s`) saved as `{sid}_{emotion}-epo.fif`;
`within_baseline` = last 15 s of the block saved as
`{sid}_{emotion}_base-epo.fif`. Low `min_epochs` floor (4). Viability for both
windows goes to `qc.json`.

`feature_building` is UNTOUCHED (still `eo`/`ec` only). `feature_engineering`
reads those epochs and emits 16 columns: valence = ln(α_F4) − ln(α_F3) (frontal
alpha asymmetry), arousal = mean(β)/mean(α) over F3/Fz/F4, as response MINUS
baseline under two baselines — **within-file (PRIMARY)** `iaps_{Hv,Ha,Cv,Ca,
Sv,Sa,Fv,Fa}` and **Eyes_Open (SENSITIVITY)** `iaps_eo_{...}`. Both predict the
SAME EF targets via feature sets `iaps_va` (primary) and `iaps_va_eo`
(sensitivity); `analysis.params.feature_sets_to_run` is the EF-vs-IAPS toggle.
`analysis` also emits `iaps_baseline_consistency.{csv,json}` — Spearman |t|
rank agreement + beta sign agreement across the two baselines. It lives in
analysis, not validation, because validation runs before the features exist.
Emotion files have the same 15 channels as resting.

## Config knobs

Each stage's knobs live in `params:` of its own `config.yaml` — open the file;
it is the source of truth and is commented. To add a knob, put it in the stage
that needs it. If two stages need the same knob (e.g. `aufei_subscales`),
**duplicate it** — do NOT introduce a globals file or shared loader.

Only the non-obvious ones are listed here:

| knob | stage | why it is not obvious |
|---|---|---|
| `variance_protect_channels` | preprocessing | `["Fp1","Fp2"]` — the blink-topography exemption above |
| `min_epochs` | preprocessing | 60 for resting; the `iaps:` block overrides it to 4 (short windows) and 20 (decoder blocks) |
| `paf_low_hz` / `paf_high_hz` | feature_building | the PAF centroid band (7–13 Hz), not a filter |
| `aufei_drop_items` | feature_engineering | `[]` = keep all 25 items in `Global_EF`; screening was abandoned |
| `aufei_subscales` | validation, feature_engineering, data_quality_check | duplicated in three configs by design; keep them mirrored |
| `targets_for_theory_mapping` | feature_engineering | drives `apriori_theory_mapping.csv`, which overrides `directions:` |
| `directions` | analysis | legacy Tier-1 only — union directions come from the mapping CSV |
| `retracted_targets` | analysis | documented but deliberately not run |
| `screening_index.{target,feature_set}` | analysis | the Rule-7 pre-specification; changing it post-hoc invalidates the index |
| `feature_selection_schemes` | feature_engineering | theory / data_driven / hybrid pools for the ML grid |
| `flanker_*` thresholds | data_quality_check | construct-validity floors from literature, not data-driven cutoffs |

## Adding a new feature

Decide whether it is a **primitive** (computed from raw epochs — slow) or a
**composite** (computed from columns already in `features.csv` — fast).

- **Primitive** → `feature_building/main.py`. Add an `extract_*` function, call
  it inside the per-subject-per-condition loop in `main()`, prefix the
  resulting columns with the condition prefix (`eo_`/`ec_`). Re-running is slow
  (minutes per subject).
- **Composite** → `feature_engineering/main.py:add_all_engineered`. Build from
  existing `psd_abs_*` / `psd_periodic_*` columns. If it has multiple
  constituent items, add the components list to
  `analysis/config.yaml:composite_components` so the Cronbach α gate runs on
  it. Re-running takes seconds.

For a new *behavioural* measure, see `instruments/CLAUDE.md`.

## Running

```bash
# Pipeline (sequential, each consumes the previous stage's latest output)
python preprocessing/main.py        # ~5-10 min for 52 files
python validation/main.py           # <30 sec
python feature_building/main.py     # ~5-10 min
python feature_engineering/main.py  # <30 sec
python analysis/main.py             # ~5-30 min with exploratory_ml.enable=true
                                    # (permutation grid is the bottleneck);
                                    # ~30 sec for confirmatory OLS only

# Sidecars (on demand)
python data_quality_check/main.py   # <10 sec — internal QC audit
python research_report/main.py      # <10 sec — research report
```

Re-run only what changed; each stage auto-picks the latest run of its
predecessor.

| What changed | Re-run from |
|---|---|
| Raw EDF, ICA / AutoReject / bandpass params | `preprocessing` (all downstream invalid) |
| Behavioural workbook contents | `validation` and `feature_engineering` |
| `bands`, `coherence_pairs`, wavelet, aperiodic flag | `feature_building` |
| Composite definitions, behavioural scoring, age params | `feature_engineering` |
| Targets, feature sets, directions, correction, bootstrap N | `analysis` only |
| Want a fresh pilot quality snapshot | `data_quality_check` (sidecar, invalidates nothing) |
| Want a refreshed research report | `research_report` (sidecar; re-pulls all upstream) |

Old `<stage>/output/<ts>/` accumulate; prune by hand.

## Conventions

- Python 3, sklearn-style only where used (analysis is statsmodels OLS).
- File paths inside stages: NEVER reference `utils.io` or sibling stages by
  absolute import — every stage must be runnable standalone.
- Random seeds: read `config["random_state"]` (top-level global). Each
  `main.py` seeds numpy explicitly before any stochastic step.
- Optional dependencies (`specparam`/`fooof`, `factor_analyzer`, matplotlib /
  seaborn for the QC heatmap) imported at point of use inside
  `try/except ImportError`, skipped gracefully.
- Print statements go to stdout; long-running stages log progress per subject.
- Windows shell: forward slashes; `python` resolves to a working interpreter on
  the user's machine.

## Documentation

- `README.md` — user-facing summary.
- `CLAUDE.md` — this file (EEG pipeline). `instruments/CLAUDE.md` — behavioural
  side. `instruments/<tool>/CLAUDE.md` — per-instrument detail.
- `data_quality_check/output/<ts>/report.md` and
  `research_report/output/<ts>/research_report.md` are generated and
  gitignored. Regenerate with the sidecar when a fresh snapshot is needed; do
  not promote either to a committed top-level `.md`.

Do not create any other `.md` files at the root or in stage folders unless
explicitly requested. Methodology notes go in docstrings + this file.

## Pilot N reconciliation

From the 2026-08-08 QC run (preprocessing `2026-07-07_145424`, analysis
`2026-07-28_110315`, `min_epochs = 60`). Re-run `data_quality_check` for fresh
numbers — **these shift whenever `min_epochs` or the preprocessing params
change**, so quote the run, not the number.

| layer | n | loss vs. above |
|---|---|---|
| behavioural (AUFEI / Flanker / Digit Span) | 28 | — |
| EEG recordings attempted | 52 | — |
| EEG recordings OK | 47 | 5 below `min_epochs`: D0000798/EC 55, D0000813/EC 51, D0000820/EO 43, D0000822/EO 51, D0000822/EC 58 |
| subjects with both EO and EC | 22 | lost the subjects retaining only one condition |
| merged behavioural ∩ EEG | 25 | 3 behavioural-only: D0000796, D0000822, D0000823 |
| analysis full OLS | 22 | 3 dropped for NaN (target `rt_cv`): D0000798, D0000813, D0000820 — mostly `alpha_reactivity_PO`, which needs both conditions |

D0000796 and D0000823 withdrew during data collection and have no EDF at all.
D0000822 is different — it was recorded but both conditions fell below
`min_epochs`, so it is a cleaning-floor loss, not a withdrawal. Do not merge
the two categories when reporting attrition.

**The IJP manuscript frames the sample as 28 enrolled, 2 withdrew, 26
analysed.** That predates the current `min_epochs = 60` floor and no longer
matches any live run — reconcile it against a named run before submission
rather than quoting it as-is.

Age is a strong confound for DDM drift targets: restricted-model R²
(`ddm_v_incongruent ~ age_months`) was 0.31 in the pilot. The hierarchical OLS
handles this natively via Frisch-Waugh — do not switch to a CV regression path
that drops `age_months` from the design matrix.

Behavioural / instrument quality findings — the Flanker construct failure,
difference-score targets, AUFEI ceiling and variance restriction, Digit Span
item-level gap — are in `instruments/CLAUDE.md`.
