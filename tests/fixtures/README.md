# Synthetic Fixture

A deterministic, PHI-free synthetic dataset that mirrors the real pipeline
inputs in shape, schema, and value ranges, but contains no real subjects.

## Purpose

- **Reviewer trust**: anyone clicking the repo can reproduce the pipeline
  end-to-end without access to the real (IRB-protected) dataset.
- **CI**: GitHub Actions runs `pytest` + `make figures` against this fixture
  on every push.
- **Regression**: catch breakage in stages 1–6 before it reaches the real
  data run.

## Generate

```bash
python tests/generate_synthetic_fixture.py \
    --out tests/fixtures/synthetic \
    --n-subjects 28 \
    --duration 30 \
    --seed 42
```

Adds `--include-emotional` to also generate the four emotional-condition
EDF files per subject. CI uses `--duration 10` to keep Stage 6
density-matrix compute under the runner timeout; reviewer / real-data
runs use 30+.

## What the fixture contains

| Path | Description |
|---|---|
| `EF_Biomarker/EDF_Files/D000000{0..27}/*.edf` | 28 subjects × 2 (or 6) conditions of 30 s × 15 channels × 250 Hz EEG. Gaussian noise + ~10 Hz alpha sine on posterior channels + slow drift + occasional blink-like artefacts on Fp1/Fp2 to exercise ICA. |
| `data_root/AUFEI-O/AUFEI-O_Cleaned.xlsx` | 28 rows, ID + Sex + DoB + WM1-5 + IC1-5 + CF1-5 + P1-5 + SF1-5 (Likert 1-5). |
| `data_root/Flanker_Test_Pilot.xlsx` | 28 rows, ID + acc/RT/flanker_effect/DDM columns. |
| `data_root/Digit_Span.xlsx` | 28 rows, ID + DS_Forward + DS_Backward. |
| `config.fixture.yaml` | Copy of `configs/config.yaml` with paths rewritten to point at the fixture. |

## Run pipeline against fixture

```bash
python run_all.py --config tests/fixtures/synthetic/config.fixture.yaml
```

This executes stages 1 → 2 → 3 → 4 → 6 → 5 against the synthetic data.

## Caveats

- The "EF level" in the AUFEI scores is just `even subject-id last digit → high`
  (mean 4.0) vs odd → low (mean 2.0). This is enough variance for the binary
  classification target derivation to work; it is **not** a meaningful EF
  signal and the classifier should not be expected to perform meaningfully
  above chance.
- The alpha amplitude is correlated with a per-subject seed but **not** with
  the AUFEI bias. There is no expected feature-level signal beyond the
  pipeline running without error.
- All names and identifiers are placeholders.

## Why not commit the EDF files?

- 28 × 2 conditions × 30 s × 15 channels × 250 Hz × 2 bytes ≈ ~12 MB per condition
  set; the EDF format adds ~30% overhead. Committing pre-generated files would
  add ~30 MB to the repo. Reviewers can regenerate in ~30 s.
- The `.gitignore` here excludes the generated output directories so only the
  generator script and this README are tracked.
