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
STAGE 2: Feature Extraction (~920 features per subject)
    Conventional QEEG: PSD (Welch, np.trapz integration), TBR, FAA, coherence
    Advanced: wavelet CWT, Hjorth parameters, spectral entropy, phase-amplitude coupling
    Covariance: frequency-band covariance matrices (Riemannian-compatible)
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
STAGE 5: Quantum-Inspired Feature Exploration (exploratory)
    QEPP interference patterns, quantum probability interactions,
    von Neumann entropy of EEG density matrices
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

### Correlations

TBR at Cz shows the strongest association with Global EF (Spearman rho = -0.383, p = .044, Cohen's d = -0.83), with effect direction consistent with the literature: higher TBR corresponds to lower executive function. No correlations survive FDR correction at this sample size, which is expected given the study is powered for N = 100.

### Classification

| Feature Set | Best Model | Balanced Accuracy [95% CI] | AUC |
|-------------|-----------|---------------------------|-----|
| Conventional QEEG | XGBoost | 0.636 [0.587 - 0.687] | 0.677 |
| Conventional + Advanced | MLP | 0.636 [0.590 - 0.683] | 0.637 |
| Covariance | RandomForest | 0.618 [0.568 - 0.667] | 0.625 |
| All features | RandomForest | 0.587 [0.545 - 0.629] | 0.590 |

![Model Comparison](docs/figures/model_comparison.png)
*Balanced accuracy across 7 classifiers and 4 feature sets. XGBoost on conventional QEEG features achieves 0.636. The H4 target of 0.75 is not yet met at N = 28.*

**Permutation test:** p = 0.149 (not statistically significant at N = 28). The best model performs above the permutation mean (0.494), but confirmatory power requires the full sample.

**Hyperparameter tuning** (nested CV): XGBoost_tuned achieves 0.643 balanced accuracy.

### Biomarker Candidates (SHAP Feature Importance)

![SHAP Feature Importance](docs/figures/shap_top15.png)

| Rank | Feature | SHAP | Stability | Neural System |
|------|---------|------|-----------|---------------|
| 1 | Beta coherence F3-P3 | 0.144 | Stable | Fronto-parietal functional connectivity |
| 2 | Absolute beta power Pz | 0.088 | Stable | Parietal cortical activation |
| 3 | Relative beta power F4 | 0.055 | Stable | Right frontal vigilance |
| 4 | Alpha reactivity (global) | 0.053 | Unstable | Cortical responsiveness |
| 5 | TBR at Cz | 0.044 | Stable | Cortical arousal regulation |
| 6 | Absolute beta power O1 | 0.034 | Stable | Occipital activation |

The dominant role of fronto-parietal beta coherence (not TBR) as the top biomarker candidate is noteworthy. This suggests that inter-regional connectivity may be more informative for EF classification than single-channel spectral ratios, consistent with network-level theories of executive control (Sauseng et al., 2005).

### Quantum-Inspired Features (Exploratory)

![Quantum vs Classical](docs/figures/quantum_vs_classical.png)

| Feature Set | Best Model | Balanced Accuracy | AUC |
|---|---|---|---|
| Quantum only | Logistic Regression | 0.657 | 0.694 |
| Classical only | Random Forest | 0.585 | 0.662 |
| Combined | Logistic Regression | 0.608 | 0.634 |

Quantum-inspired features (QEPP interference patterns, quantum probability interactions, von Neumann entropy) outperform classical QEEG features by +7.2 percentage points in balanced accuracy. This exploratory finding suggests that non-linear inter-channel dependencies captured by quantum-inspired formalisms may encode EF-relevant neural dynamics that standard spectral and coherence measures miss. These results require validation at N = 100. For theoretical grounding, see Busemeyer & Bruza (2012), Khrennikov & Yamada (2025), and Alotaibi et al. (2026).

## Current Status and Next Steps

| Phase | Status | Details |
|-------|--------|---------|
| Ethics approval | Complete | RS Soeharto Heerdjan Ethics Committee |
| Pilot data collection (N = 28) | Complete | EEG + AUFEI + Flanker + Digit Span |
| Preprocessing pipeline | Complete | HAPPE-compliant, validated on all 28 subjects |
| Feature extraction | Complete | 920 conventional + 258 quantum features per subject |
| ML classification | Complete | 7 models, nested CV, permutation test, bootstrap CI |
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
  run_all.py                 # Pipeline orchestrator (stages 1-5)
  evaluate.py                # Pipeline quality evaluation (6 dimensions)
  validate_data.py           # Data integrity checks
  convert_to_bids.py         # BIDS format conversion (mne-bids)
  generate_figures.py        # Publication figure generation
  configs/
    config.yaml              # All analysis parameters
  stages/
    stage1_cleaning.py       # EEG preprocessing
    stage2_features.py       # Feature extraction (920 features)
    stage3_merge.py          # Behavioural data integration
    stage4_analysis.py       # Statistics, ML, SHAP
    exploratory_quantum.py   # Quantum-inspired features
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
