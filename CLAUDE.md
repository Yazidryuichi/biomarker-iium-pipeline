# CLAUDE.md

Operating notes for Claude Code in this repository. Read this before making changes.

## What this project is

A reproducible 4-stage Python pipeline that turns raw resting-state EDF recordings into candidate QEEG biomarkers of executive function (EF) in Indonesian children aged 6-12. Pilot N=26 (after one severely-degraded subject-condition dropped by the cleaning floor), target N=100. Quantum-inspired feature extraction and quantum-kernel models are opt-in additions to Stages 2 and 4 respectively. See `README.md` for the scientific framing and `METHODS.md` for methodological detail.

## Repository layout

```
pipeline.py                Slim orchestrator. CLI flags select stages.
configs/config.yaml        Globals only: paths, recording, random_state. NOT a single source of truth.
stages/
  cleaning/
    config.yaml            Stage-local: input.from, output.to, params (bandpass, ICA, ICLabel, AutoReject, min_epochs)
    cleaning.py            Stage 1: EDF -> cleaned epochs (infomax-ICA + ICLabel + AutoReject local; avg ref end-to-end)
    verify_qc.py           Diff two cleaning runs' qc.json and flag QC regressions
    runs/<ts>/             Per-invocation outputs (gitignored, co-located with code)
    __init__.py            re-exports `run`, `load_cleaned_epochs`
  features/
    config.yaml            Stage-local: bands, coherence_pairs, wavelet, include_quantum
    features.py            Stage 2: epochs -> raw primitive features (PSD, coherence, wavelet, Hjorth, entropy, PAC, cov)
                                    + optional quantum-inspired features (QEPP/QI/tensor)
    _quantum.py            Helper — QEPP/QI/tensor-network extractors (used by features.py)
    runs/<ts>/             Per-invocation outputs (gitignored)
    __init__.py            re-exports `run`, `load_features`
  engineering/
    config.yaml            Stage-local: tbr_channels, faa_left/right, posterior_alpha_channels, assessment_date
    engineering.py         Stage 3: math-derived composites (TBR, FAA, alpha reactivity)
                                    + behavioural merge + <target>_group binarization
    runs/<ts>/             Per-invocation outputs (gitignored)
    __init__.py            re-exports `run`, `load_full_dataset`
  analysis/
    config.yaml            Stage-local: cv_folds, cv_repeats, models, feature_sets, scoring, targets, include_qsvm
    analysis.py            Stage 4: descriptives, correlations (H1-H3), classification (H4), SHAP
    qsvm_classifier.py     Optional QSVM (pennylane) classifier (used by analysis.py)
    runs/<ts>/             Per-invocation outputs (gitignored)
    __init__.py            re-exports `run`
scripts/
  quantum_binning_sensitivity.py   Standalone diagnostic: how does QI n_bins affect interference values
utils/
  io.py                    load_config (globals), load_stage_config (merged), discover_subjects,
                           behavioural loaders (load_aufei/load_flanker/load_digit_span with EZ-DDM),
                           multi-target helpers (get_targets, target_group_col),
                           per-stage I/O helpers (latest_stage_dir, make_stage_dir, write_stage_notes)
  bio_interpretation.py    Maps SHAP-ranked features to biological/cognitive descriptions
data/
  EDF/                     Per-subject directories (Dxxxxxxx) of EDF files
  Behavioral/              AUFEI-O, Flanker, Digit Span Excel files
evaluate.py                IMMUTABLE scoring harness; do not edit
generate_figures.py        Builds README/manuscript figures from latest analysis output
```

`evaluate.py` still grep-references the pre-refactor filenames (`stage1_cleaning.py`, etc.) — that file is intentionally immutable, do not "fix" it. `run_all.py` has been replaced by `pipeline.py`.

## Stage I/O model (important — this is the architecture)

Each stage owns a folder `stages/<stage>/` with its own `config.yaml` declaring `input.from` (predecessor stage, `raw_edf`, or `behavioral`) and `output.to` (the stage label whose `runs/` dir receives the new timestamped subdir — usually equal to the stage's own name). Stages are **independent**: code, config, and outputs are co-located inside `stages/<stage>/`. There is no shared "run dir" and no shared `results/` parent.

Layout per stage:
```
stages/cleaning/runs/<YYYY-MM-DD_HHMMSS>/cleaned_epochs/, qc.json, run_notes.json
stages/features/runs/<YYYY-MM-DD_HHMMSS>/features.csv, cov_matrices.npz, run_notes.json
stages/engineering/runs/<YYYY-MM-DD_HHMMSS>/full_dataset.csv, run_notes.json
stages/analysis/runs/<YYYY-MM-DD_HHMMSS>/correlations.csv, run_notes.json,
                                         <target>/{ml_results.csv, shap_importance.csv, shap_annotated.csv, figures/}
```

`stages/*/runs/` is gitignored. There is no top-level `results_dir` global anymore — `paths` only holds `edf_dir` and `behavioral_dir`.

Stage config schema (`stages/<stage>/config.yaml`):
```yaml
input:  { from: <stage_name | "raw_edf" | "behavioral"> }
output: { to:   <stage_name> }
params: { ... stage-specific keys ... }
```

Rules:
- Every stage call creates a **new timestamped subdirectory** under `stages/<output.to>/runs/`. No overwriting. The dir is created eagerly by `load_stage_config`.
- Every stage **auto-resolves its input** at load time, and **only consumes the single most recent predecessor run** — `utils.io.latest_stage_dir` sorts subdirs lexicographically (= chronologically for the ISO timestamp format) and returns just the newest. Older `runs/<ts>/` dirs are kept on disk as an archive but are never re-read by downstream stages. `utils.io.load_stage_config("<stage>")` returns a merged dict containing `paths`, `recording`, `random_state` (globals), `<stage>: <params>` (stage-local), `input_dir` (resolved predecessor `<latest ts>/`), and `output_dir` (newly created). Stage code reads these directly — no `latest_stage_dir`/`make_stage_dir` calls inside stage modules.
- Every stage writes a `run_notes.json` recording timestamp, git commit, the input dirs it consumed, and outputs produced. This is the audit trail.
- `pipeline.py` (full run) runs all 4 stages back-to-back, each loading its own merged config and getting its own timestamp. Within one full-run invocation, stages also pass artifacts in-memory to skip the disk round-trip — but the saved outputs still land in their respective timestamped dirs under each stage folder.
- Old runs accumulate in `stages/<stage>/runs/`; prune by hand (`rm -rf` specific timestamps) or wipe everything for a stage with `make clean`. There is no automatic retention policy.

**When you add a new stage or output:**
- Create `stages/<new_stage>/{config.yaml, <new_stage>.py, __init__.py}`. Wire it into `pipeline.py` via `from stages.<new_stage> import run`.
- Read params from `config["<stage_name>"]`, globals (`paths`, `recording`, `random_state`) from the top level. `input_dir` and `output_dir` are pre-resolved.
- Use `write_stage_notes(config["output_dir"], payload)` for the audit trail.
- Do NOT add new keys like `paths.<stage>_dir` to globals. The only global path keys are `edf_dir` and `behavioral_dir`. Each stage's run dir is derived from its own folder location (`stages/<stage>/runs/`), not from a global path.

For multi-target output layouts inside `analysis/`, see "Multi-target architecture" below.

## Feature taxonomy: primitives vs composites

- **Primitives** (Stage 2, `stages/features/features.py`): direct signal-processing outputs from cleaned epochs. PSD per band per channel, raw coherence per pair per band, wavelet, Hjorth, spectral entropy, PAC, frequency-band covariance. These are expensive to compute and stable across experiments — caching them in `features/<ts>/features.csv` is the bottleneck-relieving move.
- **Composites** (Stage 3, `stages/engineering/engineering.py`): math-derived from primitives. TBR (theta/beta), FAA (log alpha asymmetry), alpha reactivity (EC vs EO), `tbr_frontal_mean`. Cheap to compute from cached primitives — this is where you experiment with new ratios, asymmetry indices, normalisations.

When you want a new derived feature (ratio, normalised by age, cross-band coupling, etc.), add it to `stages/engineering/engineering.py:add_engineered_features`, not the features stage. Re-running engineering on cached features.csv is fast; re-running features on cleaned epochs is slow.

## Quantum-inspired features and models (opt-in)

There is no separate "Stage 5". Quantum-inspired feature extraction and quantum-kernel models are folded into the main pipeline as opt-in additions:

- **`stages/features/config.yaml` → `params.include_quantum: true`** — Stage 2 extracts QEPP, QI, tensor-network features (slow, ~30s/subject extra) from the primary Eyes-Open epoch. Columns are unprefixed (`qepp_*`, `qi_*`, `tn_*`).
- **`stages/analysis/config.yaml` → `params.include_qsvm: true`** — Stage 4 registers `QSVM_4q_ZZ`, `QSVM_6q_ZZ`, `QSVM_6q_prod` models alongside classical ones. Requires `pennylane`.

When quantum feature columns are present, Stage 4's `get_feature_sets` automatically adds `quantum_only` and `classical_plus_quantum` feature sets — no further config needed. The classical-vs-quantum comparison is then visible in `ml_results.csv` rows with those `feature_set` values.

QSVM bypasses the standard sklearn Pipeline (it does its own PCA + scaling internally). The CV loop in `run_classification` has a small branch for QSVM that only imputes NaNs before fitting.

For the `_quantum_binning_sensitivity` diagnostic, see `scripts/quantum_binning_sensitivity.py` — standalone, not part of the main pipeline.

## Multi-target architecture

`stages/analysis/config.yaml` → `params.targets` is a list. Stage engineering produces a `<target>_group` binary column for every known continuous behavioural measure (median split via `create_ef_groups`). Stage 4 loops over `get_targets(config)`. Output layout depends on count:

- 1 target: `analysis/<ts>/ml_results.csv` (flat)
- N>1 targets: `analysis/<ts>/<target>/ml_results.csv`

Correlations (target-agnostic, hypothesis-driven across many y's) stay at the stage root. Legacy `ef_group_global/wm/ic` aliases are still written by engineering for backward compatibility but new code should use `<target>_group`. Use `utils.io.get_targets(config)` and `utils.io.target_group_col(target)` rather than reading `analysis.target` directly — the resolver handles both string and list forms (and the legacy `ml.targets` location).

If you add a new behavioural measure, extend `target_candidates` in `stages/engineering/engineering.py:binarize_targets` so a `<col>_group` column gets materialized; otherwise downstream stages skip it with a "binary label missing" message.

## Methodological invariants (do not regress)

These are load-bearing for scientific validity. Do not "simplify" them.

- **Median split inside the CV fold**, never globally. The binarization threshold for `<target>_group` columns at merge time is for descriptive/quantum use only. Stage 4's classification recomputes the threshold on the training fold. Touching this risks target leakage.
- **`np.trapz` for spectral integration**, not `np.sum`. Per-epoch coherence, not concatenated. Hjorth + wavelet + PAC features must remain present — `evaluate.py` checks for them and the score will drop.
- **Imputation, scaling, feature selection all inside the sklearn Pipeline**. Never fit on the full dataset before splitting.
- **`n_jobs=1` for `permutation_test_score` and `RandomizedSearchCV`**. macOS hits OOM with parallel workers cloning the full feature matrix across many models. The comment explaining this is in `analysis.py` near the permutation test — keep it.
- **SHAP per-CV-fold averaging** (`all_shap_values`), not SHAP on a model fit to the full cohort. The full-data SHAP is computed only for the summary plot.
- **`cleaning.params.min_epochs` floor**. Subject-condition recordings whose surviving epoch count falls below this floor are dropped from disk (no `*-epo.fif` saved); downstream stages then never see them. Severely degraded recordings (e.g. one EO file with 29/144 epochs surviving in the pilot) bias PSD/coherence estimates more than they help — better excluded than diluted.
- **FDR (Benjamini-Hochberg)** restricted to pre-specified hypotheses (H1-H3 + planned secondary tests). Do not extend the FDR scope to exploratory feature sweeps without flagging the change explicitly.
- **Average reference applied before ICA fit**, not after `ica.apply`. Source EDFs are Mitsar A1-A2 (linked-mastoid, no M1/M2 channels in file); we commit to average reference for the rest of Stage 1 so the unmixing matrix W and `ica.apply` share a reference frame, and so ICLabel — which was trained on avg-ref data — classifies in the frame it expects. `qc.json` records `source_reference` and `output_reference` for provenance.
- **ICA fit on a 1 Hz high-pass copy of the avg-referenced raw**; the unmixing matrix is then applied to the 0.5 Hz bandpassed avg-ref raw. Bad channel interpolation runs **after** ICA, not before. Edge trimming uses `filter_edge_crop_sec` (default 5 s; the previous 0.5 s was too short for a 0.5 Hz HP FIR transition).
- **ICLabel + AutoReject local — no lenient retry.** Artifact components are flagged by `mne-icalabel` (infomax extended ICA, label in `iclabel_exclude_labels` AND probability > `iclabel_threshold`); the legacy `find_bads_eog`/Fp1-Fp2 correlation path is gone. Epoch rejection uses AutoReject local (per-channel CV thresholds, per-epoch interpolation). The data-driven lenient retry was removed because it gave noisy recordings looser thresholds, biasing comparisons; `min_epochs` is the only gate. Fp1/Fp2 are exempt from variance-based bad-channel flagging (pediatric blinks make them naturally high-variance; flagging them pushed the blink topography out of the ICA fit).

## Conventions

- Python 3, sklearn-style pipelines. Use the existing `_build_models(random_state)` helper rather than instantiating models inline.
- **Required runtime deps for Stage 1** (cleaning): `mne`, `mne-icalabel`, `autoreject`, and an ICLabel backend (`onnxruntime` or `torch`). ICLabel will raise `ImportError` at point of use if no backend is present.
- Optional dependencies (xgboost, lightgbm, catboost, torch, pennylane, shap, coffeine, pyriemann) are imported in `try/except ImportError` at point of use and skipped gracefully — preserve this pattern.
- Print statements go to stdout via the configured logging handler in `pipeline.py`. Long-running stages should log progress; do not silence them.
- Random seeds: read `config["random_state"]` (top-level global). `pipeline.py` seeds `random`, `numpy`, and `PYTHONHASHSEED` globally. New stochastic code must thread the seed through.
- File paths inside stages: never read `config["paths"]["<stage>_dir"]` (those keys are gone). Use `config["input_dir"]` and `config["output_dir"]` resolved by `load_stage_config`. Never hardcode `./results/...` or `./stages/<stage>/runs/...`.
- Windows shell: use forward slashes and Unix-style commands; `python` resolves to a working interpreter on the user's machine.

## Running the pipeline

```bash
python pipeline.py                            # full pipeline (each stage gets its own timestamp)
python pipeline.py --cleaning                 # stage 1 only
python pipeline.py --features                 # stage 2 only — auto-loads latest cleaning output
python pipeline.py --engineering              # stage 3 only — auto-loads latest features output
python pipeline.py --analysis                 # stage 4 only — auto-loads latest engineering output
python pipeline.py --include-emotional        # add Happy/Calm/Sad/Scare conditions
python pipeline.py --subject D0000795         # single-subject debug run
python pipeline.py --config configs/other.yaml   # override globals path
```

After editing `stages/analysis/config.yaml:params.targets` (or adding/changing engineered features), re-run `--engineering` once so the new columns appear in `full_dataset.csv` before `--analysis`.

## Iteration workflow — which stages to re-run

The cache discipline is: cleaning is the slow one, features is moderate, engineering is fast, analysis is fast. Always start from the cheapest stage that covers the change.

| What changed | Run from |
|---|---|
| Raw EDF data, ICA / AutoReject / bandpass params (`stages/cleaning/config.yaml`) | `--cleaning` (then everything downstream) |
| Bands, coherence pairs, wavelet, `include_quantum` (`stages/features/config.yaml`) | `--features` (cleaning cached) |
| TBR / FAA / alpha-reactivity definitions, `assessment_date`, behavioural files (`stages/engineering/config.yaml` or `data/Behavioral/`) | `--engineering` (cleaning + features cached) |
| `targets` list, models, `feature_sets`, `cv_folds`, `cv_repeats`, scoring, `include_qsvm` (`stages/analysis/config.yaml`) | `--analysis` only |
| Globals (`paths`, `recording`, `random_state` in `configs/config.yaml`) | full `python pipeline.py` (recording params would invalidate cleaning) |

Each invocation creates a new `stages/<stage>/runs/<ts>/` and the next stage will auto-pick it. No manual flag plumbing.

## Configuration map (where each knob lives)

```
configs/config.yaml                  paths (edf_dir, behavioral_dir),
                                     recording (sfreq, channels, conditions),
                                     random_state
stages/cleaning/config.yaml          bandpass, notch, epoch_duration/overlap,
                                     filter_edge_crop_sec,
                                     ica_method ("infomax"), ica_fit_params (extended),
                                     ica_n_components,
                                     iclabel_threshold, iclabel_exclude_labels,
                                     bad_channel_threshold (MAD-z),
                                     bad_channel_corr_threshold,
                                     bad_channel_flatline_std,
                                     variance_protect_channels (["Fp1","Fp2"]),
                                     use_autoreject_local, fallback_reject_uv,
                                     max_reject_pct (warning only; no retry),
                                     min_epochs
stages/features/config.yaml          bands, coherence_pairs, wavelet*,
                                     include_quantum
stages/engineering/config.yaml       tbr_channels, faa_left/right,
                                     posterior_alpha_channels, assessment_date
stages/analysis/config.yaml          cv_folds, cv_repeats, test_size,
                                     targets[], models[], feature_sets[],
                                     scoring[], include_qsvm
```

To add a knob to a stage, put it in `params:` of that stage's config. To add a knob shared by multiple stages, put it at the global config root and read via `config["<key>"]`.

## Documentation files

- `README.md` — scientific framing, study design, hypotheses, citations
- `METHODS.md` — methodological detail
- `AI_TRANSPARENCY.md` — AI assistance disclosure (load-bearing for `evaluate.py`'s AI-transparency score)
- `CONTRIBUTING.md` — contribution guide
- `PIPELINE_STATUS_REPORT.md` and `docs/progress_report_talenta_*.md` — status snapshots; do not edit unless asked

Do not create new top-level `.md` files unless explicitly requested. Update existing docs in place.
