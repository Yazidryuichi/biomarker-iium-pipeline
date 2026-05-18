# Quantitative EEG Biomarkers of Executive Function in Children

[![CI](https://github.com/Yazidryuichi/biomarker-iium-pipeline/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Yazidryuichi/biomarker-iium-pipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**A Computational Pipeline for Identifying Neural Markers of Executive Function Dysfunction Using Resting-State EEG and Machine Learning**

## Background and Motivation

Executive function (EF) — the set of cognitive processes enabling goal-directed behaviour, including working memory, inhibitory control, and cognitive flexibility (Diamond, 2013; Miyake et al., 2000) — is a strong predictor of academic achievement, social competence, and mental health outcomes across the lifespan (Moffitt et al., 2011). Early identification of EF difficulties in children is critical for timely intervention, yet current assessment relies almost exclusively on behavioural questionnaires and neuropsychological testing, which are time-consuming, culturally biased, and difficult to scale.

Quantitative EEG (QEEG) offers a promising objective alternative. The theta/beta ratio (TBR) at frontal sites has been the most studied QEEG marker of attention and EF (Arns et al., 2013), though its clinical utility remains debated (Zhang et al., 2017). More recently, connectivity-based measures (coherence, phase-amplitude coupling) and machine learning approaches have shown potential for multivariate biomarker discovery from resting-state EEG (Bomatter et al., 2024).

This project develops a complete, reproducible computational pipeline that moves from raw EEG recordings to candidate biomarker identification, with the long-term goal of building scalable, objective screening tools for EF assessment in low-resource settings.

## Study Design

| Parameter | Value |
|-----------|-------|
| Population | Indonesian children aged 6-12 (typically developing) |
| Sample | N = 28 (pilot); target N = 100 |
| Design | Cross-sectional, observational |
| EEG | 15-channel (10-20 system), 250 Hz, resting-state (eyes open + eyes closed) |
| Behavioural measures | AUFEI-O (executive function questionnaire; Dewi et al., 2025), Fish Flanker Test (inhibitory control), Digit Span (working memory) |
| Ethics | Approved by RS Soeharto Heerdjan Ethics Committee |
| Data collection site | Talenta Center, Jakarta |

### Pre-specified Hypotheses

Derived from the literature review prior to data analysis:

| ID | Hypothesis | Rationale |
|----|-----------|-----------|
| H1 | Frontal TBR correlates negatively with Global EF score | Elevated TBR reflects cortical hypoarousal associated with prefrontal hypofunction (Arns et al., 2013) |
| H2 | Frontal theta power correlates negatively with Global EF score | Excessive resting theta indicates cortical immaturity in children (Barry et al., 2003) |
| H3 | Frontal TBR correlates positively with Flanker Effect | Higher TBR should predict poorer inhibitory control (Zhang et al., 2017) |
| H4 | ML classification of EF level from QEEG achieves balanced accuracy >= 75% | Threshold based on comparable paediatric EEG-ML studies |

## Pipeline Architecture

The pipeline follows a modular, five-stage architecture with clearly separated data ingestion, preprocessing, feature engineering, modelling, and interpretation layers:

```
Raw EDF files
    |
    v
STAGE 1: Preprocessing (HAPPE-compliant)
    Resampling (250 Hz) -> Bandpass 0.5-45 Hz -> Notch 50 Hz -> Edge trimming
    -> Bad channel detection (MAD z-score + correlation)
    -> ICA artefact removal (FastICA on 1 Hz high-pass copy, applied to 0.5 Hz)
    -> Bad channel interpolation (after ICA) -> Average reference
    -> 2-second epochs -> AutoReject artefact rejection
    |
    v
STAGE 2: Feature Extraction (922 conventional features per subject per condition)
    Conventional QEEG: PSD (Welch, np.trapz integration), TBR, FAA, coherence
    Advanced: wavelet CWT, Hjorth parameters, spectral entropy, phase-amplitude coupling
    Covariance: frequency-band covariance matrices (Riemannian-compatible)
    [Full feature-count breakdown in METHODS.md §"Feature taxonomy"]
    |
    v
STAGE 3: Behavioural Data Merge
    AUFEI-O domain scores + Flanker Effect + Digit Span + demographics
    -> Classification target via fold-internal median split
    |
    v
STAGE 4: Statistical Analysis and Machine Learning
    4A. Hypothesis testing: Spearman correlations, FDR correction, effect sizes
    4B. Classification: 4 feature sets x 7 models, nested CV, permutation testing
    4C. Feature importance: per-fold SHAP with stability analysis + biological annotation
    |
    v
STAGE 5: Fair Comparison (feature set x model class, 2x2)
    LOSO predicted probabilities, subject-bootstrap CIs, paired DeLong tests
    |
    v
STAGE 6: Density-matrix Feature Extraction (exploratory, main branch)
    Channel-covariance density matrices, von Neumann entropy,
    Hilbert-Schmidt similarity. Quantum-cognition interpretive
    layer kept on the `quantum-exploration/` branch.
```

## Methodological Safeguards

| Concern | How it is addressed |
|---------|-------------------|
| Data leakage | Median split threshold computed on training fold only; imputation, scaling, and feature selection all inside the CV pipeline |
| Multiple comparisons | FDR (Benjamini-Hochberg) across pre-specified tests; separate scope for exploratory analyses |
| Chance-level classification | Permutation test (200 permutations) with reported p-value |
| Overfitting to small N | Bootstrap 95% CI; nested CV for hyperparameter tuning; SHAP stability across folds |
| Class imbalance | Balanced class weights on applicable models (RF, SVM) |
| Preprocessing order | ICA fitted on 1 Hz high-pass copy (prevents slow-drift contamination); bad channels interpolated after ICA, not before (preserves ICA source separation quality) |
| Biological validity | Every top SHAP feature is mapped to a neural system, expected direction, EF relevance, and supporting literature (see `utils/bio_interpretation.py`) |
| Mathematical transparency | All equations, loss functions, and feature definitions documented in [METHODS.md](METHODS.md) |
| Reproducibility | Dockerfile, Makefile, pinned dependency versions, deterministic seeds, structured logging |
| AI-assisted development | Full disclosure following COPE guidelines in [AI_TRANSPARENCY.md](AI_TRANSPARENCY.md) |

## Preliminary Results (N = 28 pilot)

### Limitations to bear in mind at this N

Before reading the numbers below, four points apply to everything reported in this section:

1. **No correlation survives FDR correction at N = 28.** The strongest single correlation (TBR frontal mean vs Global EF, Spearman rho = -0.489 at N = 25 after artefact rejection on 3 frontal-midline subjects) has p = 0.013 uncorrected but does not survive Benjamini-Hochberg across the pre-specified test family (FDR p = 0.09). This is expected — the study is powered for N = 100, not the pilot.
2. **No classifier is significantly above chance at N = 28** under permutation testing. The best cells are marginal (p ≈ 0.08-0.12, see "Fair comparison" subsection below). The classification tables presented here are for transparency and method development; they are not claims of biomarker discovery.
3. **The originally reported 95% bootstrap CIs were too tight.** The 7-classifier × 4-feature-set table (moved to appendix at the end of this section) uses per-fold bootstrap CIs (N=100 folds, only 28 actual subjects) that under-represent variance. Subject-level CIs from the Stage 5 fair-comparison rerun are 2-3× wider and should be preferred. The headline below leads with that fair comparison.
4. **All claims here are pilot-scale.** We expect to revisit at the target N. Where a finding cuts against canonical practice (e.g., the dominance of posterior-parietal relative beta over the frontal TBR), the right reading is "the pre-specified TBR hypothesis did not find support at the pilot N," not "TBR has been refuted."

### Correlations

TBR frontal mean (aggregated Fz/F3/F4) shows the strongest pre-specified association with Global EF: Spearman rho = -0.489 at N = 25 (after artefact rejection on 3 subjects with insufficient frontal-midline data), raw p = 0.013, FDR-corrected p = 0.09. TBR at Cz on the same N = 25 subset reaches rho = -0.455 (raw p = 0.022). Effect direction is consistent with prior literature (higher TBR corresponds to lower executive function). The conversion to Cohen's d for the frontal-mean correlation (d = 2·rho / √(1-rho²) = -1.12) is reported for comparability with prior QEEG-EF work but is non-standard for correlation effect sizes and should be read as a rough magnitude indicator only. **No correlations survive FDR correction at this sample size.**

### Fair comparison (Stage 5) — density-matrix features vs classical QEEG features, matched 2×2 design

The original Stage 5 comparison contrasted density-matrix features under linear SVM against classical features under shallow Random Forest — different feature sets AND different model classes, conflating feature-set effect with model-class effect. The fair comparison runs a 2×2 design under identical CV, with **leave-one-subject-out** predicted probabilities (unbiased AUC, no fold-level non-independence) and **subject-bootstrap** CIs (N_BOOT=10000 resamples of subjects, not folds).

| Feature set | Model | LOSO AUC [subject-bootstrap 95% CI] | LOSO BAcc [CI] | Permutation p (per-fold BAcc) |
|---|---|---|---|---|
| Density matrix | Linear SVM | 0.785 [0.583 - 0.947] | 0.752 [0.585 - 0.896] | 0.078 |
| Density matrix | Shallow RF | 0.714 [0.492 - 0.896] | 0.537 [0.354 - 0.717] | 0.098 |
| Classical QEEG | Linear SVM | 0.320 [0.111 - 0.544] | 0.428 [0.254 - 0.610] | 0.569 |
| Classical QEEG | Shallow RF | 0.617 [0.391 - 0.831] | 0.642 [0.459 - 0.818] | 0.118 |

**Paired DeLong tests on subject-level AUC** (the standard test for paired ROC curves; replaces the per-fold Wilcoxon, which treated correlated folds as independent):

| Comparison | ΔAUC | DeLong z | DeLong p (two-sided) |
|---|---|---|---|
| DM-SVM vs Classical-SVM (matched model) | +0.464 | 3.47 | **0.001** |
| DM-RF vs Classical-RF (matched model) | +0.097 | 0.66 | 0.507 |
| DM-SVM vs Classical-RF (the original portfolio comparison) | +0.168 | 1.12 | 0.261 |
| DM-SVM vs DM-RF (within DM features) | +0.071 | 1.20 | 0.229 |
| Classical-SVM vs Classical-RF (within classical features) | -0.296 | -2.80 | 0.005 |

**The honest reading:**

- **Under matched linear-SVM**, density-matrix features beat classical features substantially and significantly (ΔAUC +0.464, DeLong p = 0.001). However, classical-SVM in this design performs *below chance* (AUC 0.32) — this is consistent with high-dimensional overfitting (622 classical features, k=10 ANOVA selection, N=28) rather than a genuine ceiling on classical features. The classical-SVM cell sets a low bar.
- **Under matched shallow-RF**, density-matrix features and classical features perform **similarly** (ΔAUC +0.097, DeLong p = 0.507). This is the toughest test for the DM-features claim and the result a methods-sensitive reader would weight most.
- **The original portfolio comparison** (DM-SVM vs Classical-RF, the apples-to-oranges baseline) shows ΔAUC = +0.168, but with subject-level CIs that overlap and DeLong p = 0.261. The numerical gap survives the rerun; the *statistical significance* does not under proper paired testing.
- **No cell reaches significance against chance** under permutation testing at N = 28 (all p ≥ 0.078). The classifier-vs-chance question awaits the target N.

**Net:** the density-matrix feature representation appears to carry information that survives explicit feature extraction in a way classical spectral and coherence summaries do not — but the strength of that claim is model-dependent and CIs are wide. The replication arm at N = 100 (and on ds004284) is the test that matters.

![Fair 2×2 Comparison](docs/figures/model_comparison.png)

*2×2 fair comparison (feature set × model class) at N=28. Error bars are subject-bootstrap 95% CIs (resamples of subjects, not folds). Permutation p (label-permuted per-fold BAcc) shown at the base of each bar. Density-matrix features under linear SVM are the only cell whose CI lower bound clears chance.*

See [`results/stage5_fair_comparison.json`](results/stage5_fair_comparison.json) for the full numerical artifact, [`stages/stage5_fair_comparison.py`](stages/stage5_fair_comparison.py) for the rerun script, and `results/stage5_per_subject.csv` for per-subject LOSO probabilities.

#### Sensitivity analysis: does L1 regularisation lift Classical+SVM out of below chance?

The classical+svm_linear cell's LOSO AUC = 0.32 is consistent with L2-SVC overfitting on 10 ANOVA-selected features at N=28. To test whether the matched-SVM DM-vs-Classical DeLong gap (p=0.001) is robust to regularisation choice, the same Stage 5 script now accepts `--include-l1-sensitivity`. With that flag, three additional cells are added on the classical feature set under the same SelectKBest(k=10) + StandardScaler pre-filter:

- `classical+svm_l1` — LinearSVC with L1 penalty, calibrated for predict_proba via 3-fold sigmoid
- `classical+lr_l1` — Logistic regression with L1 penalty (liblinear solver)
- `classical+lr_elasticnet` — Logistic regression with elasticnet (l1_ratio=0.5, saga solver)

Run:

```bash
# 1. Run Stage 5 with the L1 sensitivity cells appended:
python -m stages.stage5_fair_comparison \
    --results-dir results \
    --out-json results/stage5_fair_comparison.json \
    --include-l1-sensitivity

# 2. Auto-fill the table below from the new JSON:
python scripts/update_l1_table.py
```

The helper script ([`scripts/update_l1_table.py`](scripts/update_l1_table.py)) is idempotent: rerunning produces the same output. Use `--dry-run` to preview the proposed table without modifying the README.

Expected outcomes (placeholder — `scripts/update_l1_table.py` rewrites this table once Stage 5 has produced an `--include-l1-sensitivity` run):

| Cell | LOSO AUC [95% CI] | DeLong p vs DM+svm_linear | Reading |
|---|---|---|---|
| classical+svm_linear (baseline, below chance) | 0.320 [0.111, 0.544] | 0.001 ✓ | overfitting artifact |
| classical+svm_l1 | TBD | TBD | does L1 prune classical's noise? |
| classical+lr_l1 | TBD | TBD | parallel test under LR family |
| classical+lr_elasticnet | TBD | TBD | hybrid regularisation |

If L1 lifts classical above chance, the matched-SVM DM-advantage narrative weakens. If L1 stays below chance, the DM advantage is robust to regularisation choice and the original DeLong p=0.001 reads as a real feature-representation gap.

#### Appendix: original 7-model × 4-feature-set sweep (per-fold reporting — superseded; kept for method-development transparency)

The original Stage 4 sweep below uses **per-fold bootstrap CIs (100 folds, 28 actual subjects)** which under-represent variance by a factor of √(100/28) ≈ 1.9. The Stage 5 fair-comparison subject-level CIs above (~0.20–0.35 wide) are the honest version. This appendix is retained because the per-fold mean BAccs informed the original method development and the SHAP rankings below still consume them.

| Feature Set | Best Model | Balanced Accuracy [per-fold 95% CI — too tight] | AUC |
|-------------|-----------|-------------------------------------|-----|
| Conventional QEEG | XGBoost | 0.636 [0.587 - 0.687] | 0.677 |
| Conventional + Advanced | MLP | 0.636 [0.590 - 0.683] | 0.637 |
| Covariance | RandomForest | 0.618 [0.568 - 0.667] | 0.625 |
| All features | RandomForest | 0.587 [0.545 - 0.629] | 0.590 |

*Per-fold balanced accuracy across 7 classifiers and 4 feature sets. XGBoost on conventional QEEG features achieves 0.636. The pilot-stage target of 0.75 is not met at N = 28 and was a power-mismatched target to begin with.*

**Permutation test:** p = 0.149 for the best per-fold mean BAcc — not statistically significant at N = 28. The best model performs above the permutation mean (0.494), but the permutation p-value is the headline; the SHAP-based biomarker rankings below should be read as candidate features for the target-N rerun, not as established biomarkers.

### Biomarker Candidates (SHAP Feature Importance — exploratory, conditional on the underlying classifier)

The SHAP rankings below come from the 7-model × 4-feature-set sweep above. Read them as **candidate features for re-evaluation at the target N**, not as established biomarkers — the underlying classifier they explain is not significantly above chance at N = 28.

![SHAP Feature Importance](docs/figures/shap_top15.png)

| Rank | Feature | SHAP | Stability | Neural System |
|------|---------|------|-----------|---------------|
| 1 | Relative beta power Pz | 0.0654 | Stable | Posterior-parietal cortical activation |
| 2 | Relative beta power P3 | 0.0524 | Stable | Left posterior-parietal cortical activation |
| 3 | Relative beta power O1 | 0.0426 | Stable | Left occipital cortical activation |
| 4 | Relative beta power F8 | 0.0398 | Stable | Right inferior frontal cortical activation |
| 5 | Relative beta power F4 | 0.0394 | Stable | Right frontal cortical activation |
| 6 | Relative beta power F3 | 0.0258 | Stable | Left frontal cortical activation |
| 7 | Relative beta power P4 | 0.0255 | Stable | Right posterior-parietal cortical activation |
| 8 | Relative beta power Fp1 | 0.0229 | Stable | Left frontopolar cortical activation |
| 9 | Relative beta power Fp2 | 0.0197 | Stable | Right frontopolar cortical activation |
| 10 | Relative beta power Cz | 0.0171 | Stable | Central cortical activation |

The dominance of posterior-parietal relative beta power (Pz, P3, O1) — rather than the frontal theta/beta ratio that Arns, Conners & Kraemer (2013) foreground — is noteworthy at the pilot N. The textbook theta/beta ratio at Cz did not appear in the top ten; neither did inter-regional coherence features. This is reported transparently as a pilot-scale observation, not as a refutation of TBR.

### Non-linear feature transforms (covariance density matrix, von Neumann entropy) — exploratory

This subsection was previously titled "Quantum-Inspired Features." The features themselves are unchanged; the framing has been corrected. Von Neumann entropy of an EEG covariance matrix and density-matrix Hilbert-Schmidt similarity are mathematically well-defined non-linear transforms of the multichannel signal; whether the "quantum" framing adds anything beyond non-linear feature engineering is an open theoretical question that this pilot is not designed to resolve. A separate `quantum-exploration/` branch retains the quantum-framing analysis for future strengthening at N = 100.

| Feature set | Best Model | Balanced Accuracy | AUC |
|---|---|---|---|
| Density-matrix only (linear SVM) | LinearSVC | 0.657 (per-fold) / 0.752 (LOSO) | 0.780 (per-fold) / 0.785 (LOSO) |
| Classical QEEG only (RF) | RandomForest | 0.618 (per-fold) / 0.642 (LOSO) | 0.615 (per-fold) / 0.617 (LOSO) |
| Combined | Logistic Regression | 0.608 | 0.634 |

Density-matrix features outperform classical QEEG features under matched linear-SVM (DeLong p = 0.001, Stage 5 fair comparison above), but the comparison is sensitive to the classical-feature model choice: under matched RF, the gap is not significant (DeLong p = 0.507). The classical-SVM cell underperforms chance, which is consistent with high-dim overfitting on 622 features at N = 28 rather than a genuine result. **Read these comparisons as: the density-matrix representation appears to compress signal more usefully than the conventional QEEG menu, at this scale and on this cohort, under the model classes tested.** The kernel routes to the same density matrix (Hilbert-Schmidt SVM, parameterised quantum-circuit SVM on Stage 5 features) underperform the direct-feature route — what's gained by the explicit density-matrix features is lost when the same information is presented as a similarity kernel.

![Matched-Model Contrast](docs/figures/quantum_vs_classical.png)

*Matched-model contrast at N=28: density-matrix features vs classical QEEG features, holding model class fixed. Under linear-SVM the gap is large and significant by paired DeLong (ΔAUC=+0.46, p=0.001). Under shallow-RF the gap collapses (ΔAUC=+0.10, p=0.51). Annotated p-values per bar are label-permutation tests on per-fold BAcc; the DeLong p between bars in each pair is the paired-ROC test on subject-level AUCs.*

For theoretical grounding of the non-linear feature-transform framing, see Schuld (2021) on the kernel-equivalence of supervised quantum models. The "quantum-cognition" interpretive layer (Busemeyer & Bruza 2012, Khrennikov & Yamada 2025, Alotaibi et al. 2026) lives in the `quantum-exploration/` branch and is intentionally deferred to N = 100.

## Current Status and Next Steps

| Phase | Status | Details |
|-------|--------|---------|
| Ethics approval | Complete | RS Soeharto Heerdjan Ethics Committee |
| Pilot data collection (N = 28) | Complete | EEG + AUFEI + Flanker + Digit Span |
| Preprocessing pipeline | Complete | HAPPE-compliant, validated on all 28 subjects |
| Feature extraction | Complete | 920 conventional + 258 density-matrix-derived features per subject |
| ML classification | Complete | 7 models, nested CV, permutation test, bootstrap CI |
| SHAP + biological interpretation | Complete | Per-fold SHAP with stability metrics |
| Density-matrix feature extraction (stage 6) | Complete | 3 feature families, sensitivity analysis. Quantum-cognition framing on `quantum-exploration/` branch. |
| Fair 2x2 comparison (stage 5) | Complete | Subject-level LOSO, paired DeLong, subject-bootstrap CIs |
| Full data collection (N = 100) | Planned | Required for confirmatory analysis |
| External validation | Planned | Independent cohort or public dataset |
| Manuscript preparation | In progress | Target: journal submission after N = 100 |

### Immediate priorities

1. Complete data collection to N = 100 (statistical power for H1-H4)
2. Re-run pipeline on full sample to obtain confirmatory results
3. Validate density-matrix features on independent dataset (ds004284 replication queued)
4. Prepare manuscript for submission

## Repository Structure

```
biomarker-iium-pipeline/
  run_all.py                 # Pipeline orchestrator (stages 1-6)
  evaluate.py                # Pipeline quality evaluation (6 dimensions)
  validate_data.py           # Data integrity checks
  convert_to_bids.py         # BIDS format conversion (mne-bids)
  generate_figures.py        # Publication figure generation
  configs/
    config.yaml              # All analysis parameters
  stages/
    stage1_cleaning.py       # EEG preprocessing
    stage2_features.py       # Feature extraction (922 conventional features)
    stage3_merge.py          # Behavioural data integration
    stage4_analysis.py       # Statistics, ML, SHAP
    stage5_fair_comparison.py # 2x2 fair comparison (feature x model), LOSO + DeLong
    stage6_density_matrix.py  # Explicit density-matrix feature extraction
    exploratory_quantum.py   # Quantum-cognition framing (see quantum-exploration/ branch)
  utils/
    io.py                    # Data loading utilities
    bio_interpretation.py    # SHAP-to-neuroscience mapping
  METHODS.md                 # Mathematical formulations
  AI_TRANSPARENCY.md         # AI disclosure (COPE-compliant)
  Dockerfile                 # Containerised reproducibility
  Makefile                   # Build targets
  pyproject.toml             # Package metadata
  requirements.txt           # Pinned dependencies
```

## Reproducibility

```bash
# Option 1: Local installation
pip install -r requirements.txt
python run_all.py

# Option 2: Docker
make docker

# Option 3: Individual stages
make stage1    # preprocessing
make stage4    # analysis only
make evaluate  # pipeline quality check
make bids      # convert to BIDS format
```

Data are not included in this repository (participant privacy). See the data setup section in the codebase for file placement instructions.

### Stage execution order vs stage numbering

The stage numbers are stable identifiers, **not strictly the execution order**. In a full `python run_all.py` run, the orchestrator executes stages 1 → 2 → 3 → 4 → 6 → 5 (because Stage 5's fair comparison consumes Stage 6's density-matrix feature outputs). Reading the code, the numbers reflect the logical stage of analysis; the runtime DAG resolves the data dependency at execution time. See [`run_all.py`](run_all.py) lines 15-18 for the full DAG note.

### Environment variables

The orchestrator and individual stages respect these environment variables for skipping or accelerating expensive sub-steps. Real-data research runs should leave them all unset.

| Env var | What it does | When to use |
|---|---|---|
| `SKIP_SHAP=1` | Skip SHAP tree explainer in Stage 4C (can hang ≥40 min on synthetic fixtures with shap≥0.49.1) | CI smoke tests; not for real-data runs |
| `SKIP_STAGE5=1` | Skip the Stage 5 fair-comparison subprocess entirely. Stage 5 is post-hoc statistical inference (100-fold × 1000-perm × 10000-bootstrap ≈ 1+ hour at N=20-28), not part of pipeline correctness | CI smoke tests when `CI_FAST` isn't viable; not for real-data runs |
| `CI_FAST=1` | Read `stage5.ci_fast_n_perm` and `stage5.ci_fast_n_boot` overrides from `configs/config.yaml` (defaults: n_perm=10, n_boot=200). Lets Stage 5 actually run in CI in ~2 min instead of being skipped wholesale | CI integration tests where you want Stage 5 *correctness* validated but not statistical power |
| `RUN_CNN_LSTM=1` | Enable CNN-LSTM in Stage 4's classifier sweep. **Disabled by default because PyTorch + joblib's `fork` start method hang on macOS** (see Stage 4 status note). Linux runs are reportedly stable | Linux-only; you want the deep-learning baseline included |

### Stage 5 hyperparameters via config

Stage 5's CV and inference knobs are externalised in [`configs/config.yaml`](configs/config.yaml) under the `stage5:` section: `n_select`, `n_splits`, `n_repeats`, `n_boot`, `n_perm`, `random_state`. Override via the YAML file, or pass `--config /path/to/config.yaml` to `stages/stage5_fair_comparison.py`, or override `n_perm` at the CLI with `--n-perm`. Priority: CLI > `CI_FAST` env > config file > module defaults.

## References

- Arns, M., Conners, C.K., & Kraemer, H.C. (2013). A decade of EEG theta/beta ratio research in ADHD: A meta-analysis. *Journal of Attention Disorders*, 17(5), 374-383.
- Alotaibi, A., et al. (2026). Quantum-inspired feature engineering for EEG signal classification. *Scientific Reports*, 16.
- Barry, R.J., et al. (2003). EEG differences between eyes-closed and eyes-open resting conditions. *Clinical Neurophysiology*, 114(12), 2166-2174.
- Bomatter, P., et al. (2024). Machine learning of brain-specific biomarkers from EEG. *NeuroImage*, 289, 119156.
- Busemeyer, J.R., & Bruza, P.D. (2012). *Quantum Models of Cognition and Decision*. Cambridge University Press.
- Dewi, S.Y., et al. (2025). Development of executive function assessment for Indonesian children (AUFEI). *[In preparation]*.
- Diamond, A. (2013). Executive functions. *Annual Review of Psychology*, 64, 135-168.
- Gabard-Durnam, L.J., et al. (2018). The Harvard Automated Processing Pipeline for EEG (HAPPE). *Frontiers in Neuroscience*, 12, 97.
- Khrennikov, A., & Yamada, M. (2025). Quantum-like representation of neuronal networks' activity. *Frontiers in Human Neuroscience*, 19.
- Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Miyake, A., et al. (2000). The unity and diversity of executive functions. *Cognitive Psychology*, 41(1), 49-100.
- Moffitt, T.E., et al. (2011). A gradient of childhood self-control predicts health, wealth, and public safety. *PNAS*, 108(7), 2693-2698.
- Sauseng, P., et al. (2005). A shift of visual spatial attention is selectively associated with human EEG alpha activity. *European Journal of Neuroscience*, 22(11), 2917-2926.
- Zhang, D.W., et al. (2017). Theta/beta ratio and executive function in ADHD. *Clinical Neurophysiology*, 128(8), 1436-1443.

## Team

| Name | Role |
|------|------|
| Dr. S.Y. Dewi | Principal Investigator, AUFEI Development |
| Yazid R. Habiburahman | Co-Investigator, Computational Pipeline |
| Dandy Aulya | Co-Investigator, Data Collection |

## License

MIT License. See [LICENSE](LICENSE) for details.
