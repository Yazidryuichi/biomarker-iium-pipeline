# Contributing to the QEEG Biomarker Pipeline

## For team members

### Getting started

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/Yazidryuichi/biomarker-iium-pipeline.git
   cd biomarker-iium-pipeline
   pip install -r requirements.txt
   ```

2. Get the data files from the shared drive (ask Dandy or Yazid for access).

3. Place data in `data/` directory following the structure in README.

4. Test on one subject:
   ```bash
   python pipeline.py --subject D0000795
   ```

### Workflow

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** and test them locally.

3. **Commit** with a descriptive message:
   ```bash
   git add -A
   git commit -m "Add: description of what you changed"
   ```

4. **Push** and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

### Branching conventions

- `main` - stable, reviewed code
- `feature/*` - new features or analyses
- `fix/*` - bug fixes
- `experiment/*` - exploratory analyses (may not be merged)

### Commit message format

```
Add: new feature description
Fix: bug description
Update: what was changed and why
Refactor: what was restructured
```

## Data safety rules

**These rules are non-negotiable:**

1. **NEVER commit EDF files or behavioral data** (.edf, .xlsx, .csv with participant info)
2. **NEVER commit results that contain identifiable information** (participant IDs are OK if anonymized)
3. Check `git status` before every commit to verify no data files are staged
4. If you accidentally commit data, notify the team immediately so we can rewrite history

The `.gitignore` is configured to block common data files, but always verify.

## Adding a new analysis

If you want to add a new analysis (e.g., a new feature extraction method):

1. Add the function to the appropriate stage module (`stages/features/features.py` for raw signal-processing primitives, `stages/engineering/engineering.py` for math-derived composites)
2. Integrate it into the extraction or engineering pipeline (call it from `extract_all_features()` or `add_engineered_features()`)
3. Add any new dependencies to `requirements.txt`
4. Test on a single subject (`python pipeline.py --subject D0000795`) before running full pipeline
5. Document what the feature measures and cite the method
6. Re-run **only the cheapest stage that covers your change**: engineering composites are derivable from cached features.csv (fast); raw primitives require re-running the features stage on cleaned epochs (moderate); preprocessing changes require re-running cleaning (slow).

## Pipeline configuration

Each stage owns a config file. Globals (`paths`, `recording`, `random_state`) live at the repository root in `configs/config.yaml`. Stage-specific knobs live with the stage code:

| Stage | Config | Typical knobs |
|---|---|---|
| 1 cleaning | `stages/cleaning/config.yaml` | bandpass, notch, ICA, AutoReject, `min_epochs` floor |
| 2 features | `stages/features/config.yaml` | frequency bands, coherence_pairs, wavelet, `include_quantum` |
| 3 engineering | `stages/engineering/config.yaml` | tbr_channels, faa_left/right, posterior_alpha_channels, assessment_date |
| 4 analysis | `stages/analysis/config.yaml` | targets, models, feature_sets, cv_folds, cv_repeats, `include_qsvm` |

Edit the relevant stage config rather than hardcoding values in the scripts. Re-running a stage creates a new timestamped subdir under `stages/<stage>/runs/<ts>/` — outputs are never overwritten.

## Questions?

Contact Yazid (yazid.habiburahman@skema.edu) or Dandy for pipeline questions.
Contact Dr. Dewi for research design questions.
