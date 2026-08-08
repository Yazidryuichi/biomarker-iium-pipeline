# CLAUDE.md — instruments/

Operating notes for the behavioural side of the project: the three measures
that produce the target and covariate columns the pipeline regresses onto EEG
features. The EEG pipeline itself is documented in the repo-root `CLAUDE.md`;
read that one for stage layout, preprocessing invariants, and the OLS/ML path.

Per-instrument detail lives one level down:
- `flanker/CLAUDE.md` — PsychoPy Eriksen Flanker task, launcher, cleaning notebooks
- `digit_span/CLAUDE.md` — browser Digit Span admin tool, TTS stimuli, audio convention

## What these are and why they are not a stage

`instruments/` holds the measurement tools themselves. It is a deliberate
exception to the flat-stage layout: it is NOT a pipeline stage, does not follow
the three-thing (`main.py` / `config.yaml` / `output/<ts>/`) contract, and
nothing in the pipeline imports from it. The pipeline consumes the instruments'
*exported workbooks* under `data/Behavioral/`, never their source folders.

| instrument | what it measures | export the pipeline reads |
|---|---|---|
| AUFEI-O | parent-report EF, 25 items, 1–4 Likert | `data/Behavioral/AUFEI-O_Cleaned.xlsx` |
| Flanker (`flanker/`) | conflict / inhibitory control, RT + accuracy | `data/Behavioral/Flanker_Test_Pilot.xlsx` (`Features`, `Trials`) |
| Digit Span (`digit_span/`) | working memory, forward + backward span | `data/Behavioral/Digit_Span.xlsx` |

AUFEI-O has no folder here — it is a paper/form questionnaire, not a software
instrument. Its quality findings are documented below all the same, because
they are instrument problems, not pipeline problems.

## Publish the task, never the responses

Every instrument folder holds participant data next to its code, and all of it
is gitignored: Flanker `data/`, `experiment/data/`, `cleaned/`, `psychopy_env/`;
Digit Span `output/` and `saved_page/`.

This is not tidiness. Flanker filenames and their `participant` column carry
children's given names; the cleaned exports key rows by name instead of by
code; the Digit Span workbooks carry the `Kode -> Nama` mapping that
de-anonymises the whole EEG dataset; `saved_page/` was saved mid-session with
seven names rendered into it.

Before committing anything new under `instruments/`, run `git add -An
instruments` and grep the resulting file list for participant names. Notebooks
are covered by the repo-wide `*.ipynb` rule because they store names in cell
outputs — do not add an exception without stripping outputs first.

**New Flanker runs land in `flanker/experiment/data/`, but the cleaning
notebooks read `flanker/data/`.** Move new session output across or the
analysis silently misses it. Both paths are gitignored.

## Adding a new behavioural measure

1. Add the column to the appropriate loader in `feature_engineering/main.py`
   (`load_aufei` / `load_flanker` / `load_digit_span`) and to its `keep` list.
2. To use it as a target, add it to `analysis/config.yaml:targets`.
3. If it has trial-level data, add a reliability check in `validation/main.py`
   mirroring the Flanker pattern: split into halves, compute the metric per
   half, Spearman-Brown corrected r across subjects.

**Trial-level reliability whenever trial data exists.** The Flanker workbook
has a `Trials` sheet (1680 rows) and validation uses it. Do NOT regress this to
"summary-statistics surrogates" — that was the prior methodological error.

## Instrument quality findings

Baselines from the data quality audit; the qualitative problems will persist
until the instruments are revised. Re-run `python data_quality_check/main.py`
for fresh numbers.

### Flanker is construct-broken in the pilot

Mean `flanker_effect` = 1.5 ms against a pediatric literature range of
30–80 ms; `rt_congruent` (812 ms) ≈ `rt_incongruent` (814 ms) at sample level;
75% of subjects at the accuracy ceiling, 39% at exactly 1.0.

The task did not induce a conflict signal. **Any null in `analysis/` on a
Flanker-derived target is downstream of this, not a biomarker-level finding**
(pilot block F p = 0.83, ΔR² = 0.03 over age). The main study needs task
modification — harder distractors, response deadline, speed pressure — or task
replacement. Until then no Flanker-derived target carries interpretable
construct meaning regardless of how reliably it is measured.

### Never propose a difference score as a target

`flanker_effect` SB = 0.13. `ddm_delta_v` SB not estimable — only n = 2 valid
split-half pairs survive EZ-DDM degeneracy at the ceiling. This is structural,
not bad luck: the difference of two highly correlated reliable measures has low
or undefined reliability. Both are in `analysis/config.yaml:retracted_targets`.
See [[feedback-difference-score-targets]].

### Flanker target ordering

Primary `rt_cv` — SB = 0.97 on all 28 subjects, no ceiling failure mode,
construct-valid even while the task itself is broken. Ranked second if ever
revived: `ddm_v_incongruent`, SB = 0.99 but on the n = 6 non-ceiling subsample,
so it cannot be defended as primary at this N (currently retired from the live
target list). Not `acc_*` — pseudo-reliable at ceiling but reduces to a
perfect-vs-not flag. Never `ddm_delta_v` or `flanker_effect`.

### Two DDM estimators — do not conflate them

- **Target columns** (`ddm_v_incongruent`, `ddm_delta_v`, …) are read **as-is**
  from the workbook's `Features` sheet. The estimator is undocumented; it stays
  finite at `acc = 1.0`, so it is not pure EZ-DDM. We accept it without
  re-estimation.
- **Reliability estimates** in `validation/` and `data_quality_check/` use
  **EZ-DDM on split halves** of the trial-level data. EZ is undefined at
  `acc ∈ {0,1}`, which is why per-condition n collapses to 4–6.

The reliability of the workbook's DDM columns at full sample size is therefore
*unknown*. Do not claim the workbook uses "a more robust estimator" — that
framing was inherited from pre-rewrite code and is not independently verified.
Correct framing: *unknown estimator that handles ceiling cases differently
from EZ*.

### D0000816 is a pre-registered exclusion candidate

`acc_overall` = 0.54, `acc_incongruent` = 0.07 — below the 0.50 chance line of
a 2AFC task, so most likely the response mapping was reversed. Drives the min
of both `v_incongruent` (−2.13) and `delta_v` (−3.96). Recommend
`acc_incongruent < flanker_subchance_threshold` (default 0.50) as an explicit
exclusion rule in the main-study analysis plan.

### AUFEI-O ceiling, floor, and variance restriction

IC3 has zero variance (every pilot subject scored 4) and is already excluded
from the IC subscale. CF2 sits at 89% ceiling — recommend exclusion or rewrite
in the next wave.

Low subscale reliability is **partly variance restriction, not only item
content**: WM SD = 0.26 on a 1–4 scale, ~9% of range, which is parent-report
social-desirability compression to the top end. The fix is re-anchoring the
response scale (1–7 with concrete behavioural anchors, or frequency-based
items), not just rewriting items. A CFA on these data without addressing that
will produce a degenerate solution.

Treat AUFEI subscales as construct-invalid until the instrument is revised.
The confirmatory path therefore reports only the single item-level `Global_EF`
composite (all 25 items, α ≈ 0.81, ω ≈ 0.82, n = 28) and never the subscales;
per-subscale scores are still emitted for the QC sidecars. The high α is
item-count-driven, not evidence of strong unidimensionality.

### Digit Span has no item-level data in the current export

FW-vs-BW is the only available reliability proxy (SB = 0.46) and the
parallel-halves assumption behind it is invalid — FW is passive retention, BW
is active manipulation. Report as approximate, never definitive. For the main
study, request item-level output from the testing platform and mirror the
Flanker trial-level pattern in `validation/main.py`.
