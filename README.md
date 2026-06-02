# biomarker-iium-pipeline

Resting-state EEG → candidate QEEG biomarkers of executive function in
Indonesian children aged 6–12. Pilot N=26 (one subject-condition dropped by
the cleaning floor), target N=100.

## Pipeline

Five self-contained stages. Run each explicitly — there is no orchestrator.

```
data/                                 EDF/ and Behavioral/  (gitignored)
preprocessing/                        Stage 1  EDF → cleaned epochs
validation/                           Stage 2  behavioural reliability + QC
feature_building/                     Stage 3  primitives (PSD, coherence, …)
feature_engineering/                  Stage 4  composites + behavioural merge
analysis/                             Stage 5  hierarchical OLS regression
```

Each stage holds exactly three things:

```
<stage>/main.py            self-contained CLI entry
<stage>/config.yaml        globals + stage params
<stage>/output/<ts>/       timestamped run output (gitignored)
```

Run:

```bash
python preprocessing/main.py
python validation/main.py
python feature_building/main.py
python feature_engineering/main.py
python analysis/main.py
```

Each stage auto-resolves its predecessor as the latest timestamped subdir
under `<prev_stage>/output/`. To re-run a downstream stage with new params,
edit its `config.yaml` and rerun just that stage; cached upstream outputs
are reused.

## What each stage does

**preprocessing** — Reads EDF, sets 10-20 montage, filters (0.5–45 Hz +
50 Hz notch), detects bad channels, applies average reference, fits infomax
extended ICA on a 1 Hz HP copy, classifies components with ICLabel
(probability > 0.7 in artifact labels = excluded), interpolates bad
channels post-ICA, epochs (2 s, no overlap), AutoReject local.
Subject-conditions with surviving epochs < `min_epochs` (default 60) are
dropped from disk.

**validation** — Reliability of the behavioural targets before EEG features
are extracted. AUFEI item-level Cronbach's α and McDonald's ω per subscale;
Flanker trial-level split-half (odd/even) with Spearman-Brown correction on
DDM + RT metrics (reads the `Trials` sheet of the workbook, not just the
summary); Digit Span FW-vs-BW approximate split-half. Also summarises
Stage 1 pass rate (subjects retaining both Eyes_Open and Eyes_Closed).

**feature_building** — Per subject per condition: PSD per band per channel
via `np.trapezoid`, coherence per pair per band (per-epoch then averaged),
multi-scale CWT, Hjorth, spectral entropy, theta-beta PAC (zero-phase
filtering), frequency-band covariance. Optional: aperiodic-corrected band
power via specparam to remove 1/f maturation confound (`psd_periodic_*`).

**feature_engineering** — Math-derived composites from primitives: TBR per
channel + frontal mean, FAA, alpha reactivity (EC vs EO). A priori Tier 1
composites: `fm_theta_eo`, `posterior_alpha_ec`, `tbr_frontal_eo_log` (and
their `_periodic` siblings when periodic primitives are present). Merges
behavioural data: AUFEI subscale scoring inline + age from DoB, Flanker
Features sheet (pre-computed DDM), Digit Span FW/BW/Total.

**analysis** — Hierarchical OLS hypothesis-test per (target, feature_set):
restricted model `y ~ age_months`, full model `y ~ age_months + composites`,
block F-test for incremental R², per-composite β + one-sided directional p
+ Bonferroni or FDR-BH correction across composites, subject-resample
bootstrap 95% CI. Cronbach's α gate per composite (≥0.50 minimum). By
Frisch-Waugh, putting age in the OLS as a regressor is equivalent to
residualizing both y AND the features against age — handles the
age-confound critique without target leakage.

## Methodological invariants

- Average reference applied **before** ICA. Source EDFs are Mitsar A1-A2
  implicit; we commit to avg-ref so the unmixing matrix and ICLabel share a
  reference frame.
- ICA fit on a 1 Hz HP copy; unmixing applied to the 0.5 Hz HP raw.
- Fp1/Fp2 exempt from variance-based bad-channel flagging (pediatric blinks
  are naturally high-variance there).
- `np.trapezoid` for spectral integration, per-epoch coherence (not
  concatenated).
- `min_epochs` floor is the only gate — no data-driven lenient retry on
  AutoReject thresholds (that would give noisy recordings looser bounds).
- Cronbach's α ≥ 0.50 minimum (preferred ≥ 0.70) gate per composite before
  it enters the OLS as a measurement of its construct.
- Bonferroni or FDR-BH correction restricted to pre-specified hypotheses,
  not exploratory sweeps.

## Requirements

```
mne                       EDF I/O, ICA, epoching
mne-icalabel              ICLabel artifact classification
autoreject                local epoch rejection
onnxruntime OR torch      backend for ICLabel
numpy pandas scipy pyyaml
pywavelets                CWT
statsmodels               OLS hypothesis test
factor_analyzer           McDonald's ω (validation stage; optional)
specparam OR fooof        aperiodic correction (feature_building; optional)
openpyxl                  Excel reading (behavioural workbooks)
```

## Data layout

```
data/
├── EDF/
│   └── D0000xxx/              one folder per subject
│       ├── *_IGS_Eyes_Open.edf
│       └── *_IGS_Eyes_Closed.edf
└── Behavioral/
    ├── AUFEI-O_Cleaned.xlsx     single-sheet, item-level Likert
    ├── Flanker_Test_Pilot.xlsx  three sheets: Features, Sheet1, Trials
    └── Digit_Span.xlsx          summary FW/BW/Total only
```
