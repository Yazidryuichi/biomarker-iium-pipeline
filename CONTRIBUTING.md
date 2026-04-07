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
   python run_all.py --subject D0000795
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

1. Add the function to the appropriate stage file (`stages/stage2_features.py` for features)
2. Integrate it into the extraction pipeline (call it from `extract_all_features()`)
3. Add any new dependencies to `requirements.txt`
4. Test on a single subject before running full pipeline
5. Document what the feature measures and cite the method

## Pipeline configuration

All parameters are in `configs/config.yaml`. If you need to change:
- Frequency bands
- Filter settings
- ML model hyperparameters
- Channel selections

Edit the config file rather than hardcoding values in the scripts.

## Questions?

Contact Yazid (yazid.habiburahman@skema.edu) or Dandy for pipeline questions.
Contact Dr. Dewi for research design questions.
