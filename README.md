# Quantitative EEG Biomarkers of Executive Function in Children

**A Computational Pipeline for Identifying Neural Markers of Executive Function Dysfunction Using Resting-State EEG and Machine Learning**

## Background and Motivation

Executive function (EF) — the set of cognitive processes enabling goal-directed behaviour, including working memory, inhibitory control, and cognitive flexibility (Diamond, 2013; Miyake et al., 2000) — is a strong predictor of academic achievement, social competence, and mental health outcomes across the lifespan (Moffitt et al., 2011). Early identification of EF difficulties in children is critical for timely intervention, yet current assessment relies almost exclusively on behavioural questionnaires and neuropsychological testing, which are time-consuming, culturally biased, and difficult to scale.

Quantitative EEG (QEEG) offers a promising objective alternative. The theta/beta ratio (TBR) at frontal sites has been the most studied QEEG marker of attention and EF (Arns et al., 2013), though its clinical utility remains debated (Zhang et al., 2017). More recently, connectivity-based measures (coherence, phase-amplitude coupling) and machine learning approaches have shown potential for multivariate biomarker discovery from resting-state EEG (Bomatter et al., 2024).

This project develops a complete, reproducible computational pipeline that moves from raw EEG recordings to candidate biomarker identification, with the long-term goal of building scalable, objective screening tools for EF assessment in low-resource settings.

## Study Design

| Parameter | Value |
|-----------|-------|
| Population | Indonesian children aged 6-12 (typically developing) |
| Sample | N = 26 analysed (28 enrolled; 2 lost to behavioural-data attrition); target N = 100 |
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
| H4 | ML classification of EF level from QEEG achieves balanced accuracy >= 75 % | Threshold based on comparable paediatric EEG-ML studies; not yet met at N = 26 (best 0.665 with permutation p = 0.035) |

## Pipeline Architecture

The pipeline follows a modular, four-stage architecture with clearly separated data ingestion, preprocessing, feature engineering, and modelling layers. Quantum-inspired feature extraction and quantum-kernel models are opt-in additions to Stages 2 and 4 respectively (no separate "Stage 5"):

```
Raw EDF files
    |
    v
STAGE 1: Preprocessing — stages/cleaning/cleaning.py (HAPPE-compliant)
    Resampling (250 Hz) -> Bandpass 0.5-45 Hz -> Notch 50 Hz -> Edge trimming
    -> Bad channel detection (MAD z-score + correlation)
    -> ICA artefact removal (FastICA on 1 Hz high-pass copy, applied to 0.5 Hz)
    -> Bad channel interpolation (after ICA) -> Average reference
    -> 2-second epochs -> AutoReject artefact rejection
    -> min_epochs floor: drop subject-condition recordings below threshold
    |
    v
STAGE 2: Feature Extraction — stages/features/features.py
    Primitives only (~1825 features per subject across EO + EC):
      Conventional QEEG: PSD (Welch, np.trapz), coherence (per-epoch)
      Advanced: wavelet CWT, Hjorth, spectral entropy, phase-amplitude coupling
      Covariance: frequency-band covariance matrices (Riemannian-compatible)
    Optional (`include_quantum: true`): QEPP, QI, tensor-network features
    |
    v
STAGE 3: Engineering & Behavioural Merge — stages/engineering/engineering.py
    Engineered composites: TBR, FAA, alpha reactivity (math-derived from primitives)
    Merge with AUFEI-O + Flanker + Digit Span + demographics
    Materialise <target>_group binary columns (median split, descriptive only)
    |
    v
STAGE 4: Statistical Analysis & ML — stages/analysis/analysis.py
    4A. Hypothesis tests: Spearman correlations, FDR-BH, effect sizes (H1-H3)
    4B. Classification (H4): config-selected feature sets x models, fold-internal
        median split, RepeatedStratifiedKFold, permutation test, bootstrap CI,
        nested-CV hyperparameter tuning on best feature set
    4C. Per-fold SHAP averaging + biological annotation
    Optional (`include_qsvm: true`): QSVM_4q_ZZ / 6q_ZZ / 6q_prod models
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

## Preliminary Results (N = 26 pilot, target = Global EF)

The pilot results below correspond to the slim, paper-ready configuration: 4 feature sets x 2 classical models (RandomForest + SVM, both with `class_weight="balanced"`), 5x5 RepeatedStratifiedKFold, fold-internal median split, permutation test on the best model. Class balance for `Global_EF_group` after median split is 15 / 11 (58 % / 42 %).

### Correlations (target-agnostic)

Pre-specified hypothesis tests are reported in `correlations.csv` after Benjamini–Hochberg FDR across the eight pre-specified pairs. At N = 26 no pair survives FDR; the strongest negative association of frontal TBR with Global EF is in the expected direction, consistent with Arns et al. (2013).

### Classification (Global EF)

| Feature Set | Model | Balanced Accuracy [95 % CI] | F1 | AUC |
|---|---|---|---|---|
| conventional_qeeg (EO + EC) | RandomForest | 0.497 [0.42 - 0.57] | 0.38 | 0.53 |
| conventional_qeeg (EO + EC) | SVM | 0.502 [0.44 - 0.58] | 0.36 | 0.42 |
| conventional_qeeg_ec | RandomForest | 0.533 [0.47 - 0.60] | 0.42 | 0.62 |
| conventional_qeeg_ec | SVM | 0.545 [0.47 - 0.63] | 0.45 | 0.39 |
| all_features (everything) | RandomForest | 0.482 [0.41 - 0.55] | 0.39 | 0.53 |
| all_features (everything) | SVM | 0.498 [0.44 - 0.56] | 0.33 | 0.49 |
| **covariance_only_ec** | **RandomForest** | **0.665 [0.58 - 0.75]** | **0.62** | **0.73** |
| covariance_only_ec | SVM | 0.653 [0.58 - 0.72] | 0.58 | 0.69 |

**Permutation test** (RandomForest on `covariance_only_ec`): real score 0.683, permutation mean 0.498, **p = 0.035** — above chance.

**Hyperparameter tuning** (nested CV on `covariance_only_ec`): RandomForest_tuned 0.700, SVM_tuned 0.657. The H4 target of 0.75 is not yet met at N = 26 — expected, given the study is powered for N = 100.

The pattern that emerges:

- **Spatial covariance (eyes-closed) outperforms conventional spectral features.** The dominance of `covariance_only_ec` over `conventional_qeeg` (TBR / FAA / coherence) suggests that whole-montage co-variation, not single-channel band power, is where the EF signal sits at this sample size.
- **Eyes-closed > eyes-open.** EC variants outperform EO variants for every feature family. This is consistent with the proposal's pre-registered preference for EC as the primary resting-state condition for paediatric data.
- **Conventional QEEG is at chance for Global EF.** Classifying parent-rated EF from PSD / TBR / FAA / coherence alone is no better than chance at N = 26 — a useful null finding that argues for connectivity-based features in future work.

### Biomarker Candidates (SHAP, per-fold averaged)

Top features identified by SHAP (TreeExplainer averaged over 5 stratified folds, then `mean(|SHAP|)` ranked) on the best feature set / model combination, written to `shap_importance.csv` and `shap_annotated.csv`:

| Rank | Feature | mean abs SHAP | Stability |
|---|---|---|---|
| 1 | ec_cov_theta_C3-P4 | 0.160 | stable (CV 0.32) |
| 2 | ec_cov_delta_Fp1-F3 | 0.108 | stable (CV 0.31) |
| 3 | ec_cov_delta_F3-Pz | 0.068 | stable (CV 0.37) |
| 4 | ec_cov_delta_F7-Pz | 0.050 | stable (CV 0.33) |
| 5 | ec_cov_delta_Fp1-P4 | 0.040 | stable (CV 0.42) |

All top-ranked features are eyes-closed covariance entries linking frontal to centro-parietal sites (delta and theta bands), consistent with network-level theories of executive control (Sauseng et al., 2005) and reinforcing that fronto-parietal connectivity carries more EF signal than single-channel spectral ratios at this sample size.

### Quantum-Inspired Features (Exploratory, opt-in)

When `stages/features/config.yaml: include_quantum: true`, Stage 2 also extracts QEPP interference patterns, quantum probability interactions, and von Neumann entropy from the primary Eyes-Open epoch per subject, and Stage 4 automatically adds `quantum_only` and `classical_plus_quantum` feature sets to the comparison table. Optional `include_qsvm: true` registers QSVM_4q_ZZ / 6q_ZZ / 6q_prod kernel-SVM models. These results are exploratory and require validation at N = 100. For theoretical grounding, see Busemeyer & Bruza (2012), Khrennikov & Yamada (2025), and Alotaibi et al. (2026).

## Current Status and Next Steps

| Phase | Status | Details |
|-------|--------|---------|
| Ethics approval | Complete | RS Soeharto Heerdjan Ethics Committee |
| Pilot data collection (N = 28) | Complete | EEG + AUFEI + Flanker + Digit Span |
| Preprocessing pipeline | Complete | HAPPE-compliant; min_epochs floor drops severely-degraded recordings (1 of 52 file-conditions excluded in pilot) |
| Feature extraction | Complete | ~1825 conventional features per subject; quantum-inspired features opt-in |
| ML classification | Complete | Slim lineup (RF + SVM); permutation test, bootstrap CI, nested-CV tuning |
| SHAP + biological interpretation | Complete | Per-fold SHAP with stability metrics |
| Quantum-inspired exploration | Complete | 3 feature families, sensitivity analysis |
| Full data collection (N = 100) | Planned | Required for confirmatory analysis |
| External validation | Planned | Independent cohort or public dataset |
| Manuscript preparation | In progress | Target: journal submission after N = 100 |

### Immediate priorities

1. Complete data collection to N = 100 (statistical power for H1-H4)
2. Re-run pipeline on full sample to obtain confirmatory results
3. Validate quantum-inspired features on independent dataset
4. Prepare manuscript for submission

## Repository Structure

```
biomarker-iium-pipeline/
  pipeline.py                # Slim orchestrator. CLI flags select stages.
  evaluate.py                # Immutable pipeline quality evaluation (6 dimensions)
  generate_figures.py        # Publication figure generation
  configs/
    config.yaml              # Globals only: paths, recording, random_state
  stages/                    # Each stage is a self-contained folder
    cleaning/                  Stage 1: EDF -> cleaned epochs
      cleaning.py
      config.yaml              bandpass, ICA, AutoReject, min_epochs
      runs/<ts>/               cleaned_epochs/, qc.json, run_notes.json (gitignored)
    features/                  Stage 2: epochs -> feature primitives
      features.py
      _quantum.py              Optional QEPP / QI / tensor-network helper
      config.yaml              bands, coherence_pairs, wavelet, include_quantum
      runs/<ts>/               features.csv, cov_matrices.npz, run_notes.json
    engineering/               Stage 3: composites + behavioural merge
      engineering.py
      config.yaml              tbr_channels, faa_left/right, posterior_alpha_channels
      runs/<ts>/               full_dataset.csv, run_notes.json
    analysis/                  Stage 4: descriptives, correlations, ML, SHAP
      analysis.py
      qsvm_classifier.py       Optional pennylane QSVM
      config.yaml              cv_folds, cv_repeats, models[], feature_sets[], targets[]
      runs/<ts>/               correlations.csv, ml_results.csv, shap_importance.csv,
                               shap_annotated.csv, figures/, run_notes.json
  utils/
    io.py                    Stage I/O helpers + behavioural loaders
    bio_interpretation.py    SHAP-to-neuroscience mapping
  scripts/                   Standalone diagnostics (not part of main pipeline)
  data/
    EDF/                     Per-subject directories (Dxxxxxxx)
    Behavioral/              AUFEI-O, Flanker, Digit Span
  METHODS.md                 # Mathematical formulations
  AI_TRANSPARENCY.md         # AI disclosure (COPE-compliant)
  CLAUDE.md                  # Operating notes for AI-assisted development
  CONTRIBUTING.md            # Contribution guide
  Dockerfile, Makefile, requirements.txt, pyproject.toml
```

## Reproducibility

```bash
# Option 1: Local installation
pip install -r requirements.txt
python pipeline.py                 # full pipeline
python pipeline.py --analysis      # one stage; auto-loads latest predecessor

# Option 2: Docker
make docker

# Option 3: Individual stages (each gets its own timestamped output dir)
python pipeline.py --cleaning       # stage 1
python pipeline.py --features       # stage 2 (auto-picks latest cleaning run)
python pipeline.py --engineering    # stage 3
python pipeline.py --analysis       # stage 4
python pipeline.py --include-emotional   # also process Happy/Calm/Sad/Scare conditions
```

Data are not included in this repository (participant privacy). See the data setup section in the codebase for file placement instructions.

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
