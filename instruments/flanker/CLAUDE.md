# Flanker Task (instruments/flanker)

## What this project is

An **Eriksen Flanker task** (a classic attention / inhibitory-control paradigm) built in **PsychoPy** and run in the browser, plus the data-cleaning and feature-extraction pipeline for the collected responses. Participants respond to the direction of a central arrow flanked by congruent or incongruent arrows; the key measures are reaction time and accuracy, and the congruency effect (incongruent − congruent).

Data was collected around March 2026 (plus a couple of June 2026 sessions).

## Layout

```
instruments/flanker/
├── CLAUDE.md
├── run_flanker.bat          # one-click launcher; --pilot for a windowed test run
├── experiment/              # the PsychoPy task (all files reference each other as siblings)
│   ├── index.html           # web entry point; loads psychopy.js
│   ├── psychopy.js, psychopy-legacy-browsers.js
│   ├── psychopy.psyexp      # PsychoPy Builder source
│   ├── psychopy_lastrun.py  # compiled/exported Python of the experiment
│   ├── conditions.xlsx      # trial conditions (congruent/incongruent × left/right)
│   ├── conditions.zip
│   ├── *_left.png / *_right.png   # arrow stimuli (congruent/incongruent)
│   └── data/                # where NEW runs land — NOT where the notebooks read
├── data/                    # THE raw participant output (PsychoPy .csv/.log/.psydat), 32 participants
├── cleaned/                 # analysis OUTPUT (features + summary + trial-level)
├── cleaning_csv.ipynb       # alternate cleaning pass (reads data/ → cleaned_data/) — BROKEN, see caveat
├── flanker_cleaning.ipynb   # cleaning pipeline (reads data/ → cleaned/)
├── flanker_features.ipynb   # feature extraction (reads data/ → cleaned/)
└── psychopy_env/            # local PsychoPy virtualenv — DO NOT edit/commit; regenerable
```

## Analysis notebooks

Three notebooks, run independently (not a strict chain). **All three now read `data/`** and use paths **relative to this folder** — run them with `flanker_test/` as the working directory.

- **flanker_features.ipynb** — writes `cleaned/flanker_features.xlsx` (per-participant features). Working.
- **flanker_cleaning.ipynb** — writes `cleaned/flanker_trials_clean.csv` and `cleaned/flanker_summary.csv`. Working.
- **cleaning_csv.ipynb** — alternate cleaning pass, writes to `cleaned_data/` (created on run). **Does not run** — see caveat. Its output dir is deliberately separate from `cleaned/` because its filenames collide with `flanker_cleaning.ipynb`.

Do not reintroduce absolute paths. This folder has already moved twice (`Project\FlankerTest` → `Project\flanker_test` → `qeeg_risk_screening\flanker_test`) and the absolute paths silently broke each time. The stored cell outputs still show the old `FlankerTest` paths — those are historical logs, not live config.

## Running the task

```
run_flanker.bat            # fullscreen
run_flanker.bat --pilot    # windowed test run
```

Every path in the wrapper is `%~dp0`-relative. This folder has moved twice and
absolute paths broke silently each time — do not reintroduce them.

**All `.exe` shims in `psychopy_env\Scripts\` are broken.** pip bakes an
absolute interpreter path into each one and the venv has been relocated, so
`psychopy.exe`, `pip.exe` and the rest fail *silently* (`psychopy.exe` is a
`gui_script`, so it exits with no console output at all). `python.exe` and
`pythonw.exe` are the real interpreter and work fine:

- install with `python.exe -m pip install ...`, never `pip.exe`
- launch the GUI with
  `pythonw.exe "…\site-packages\psychopy\app\psychopyApp.py" --runner`
- to debug a silent GUI failure, re-run through `python.exe` so the traceback
  appears

`psychopy_lastrun.py` does `os.chdir(_thisDir)`, so the working directory is
always `experiment/` regardless of where you launch from — which is why new
output lands in `experiment/data/`.

## Caveats

- **New runs do not land where the notebooks read.** The task writes to
  `experiment/data/`; the cleaning and feature notebooks read `data/`. Move new
  session output across or the analysis silently misses it. Both are gitignored.
- **`cleaning_csv.ipynb` is broken.** `parse_flanker_csv()` has no `return` statement, so it returns `None` and the batch loop dies on the first file with `TypeError: 'NoneType' object is not subscriptable`. Its last stored run produced nothing. Either add the missing `return clean` (and re-verify against `flanker_cleaning.ipynb`, which does the same job correctly) or delete the notebook — right now it is dead weight that duplicates working code.
- The `experiment/` files were grouped from the project root during tidy-up; they only reference each other by relative name, so keeping them together is required.
- `psychopy_env/` is a large local virtual environment (~32k files, 1.25 GB — 95% of this folder); ignore it in any git repo, search, or backup. Kept deliberately: reinstalling PsychoPy on Windows is painful.

## Cleanup 2026-07-27

- `data_copy/` deleted — verified a **full subset** of `data/` (all 90 files present there; `data/` additionally holds 2 June-2026 participants). All notebooks repointed to `data/`.
- `PILOT/flanker_test/` deleted — was a byte-for-byte duplicate of 21 files already in `data/`, and duplicated them a second time internally under `day_1/`.
- Two empty stray dirs removed from inside `data/` (`cleaned_data/`, `flanker_clean/`).
