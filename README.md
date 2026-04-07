# QEEG Biomarker Pipeline for Executive Function in Indonesian Children

**Early Development of Executive Function Dysfunction Biomarkers Using Quantitative EEG: A Machine Learning Approach in Indonesian Children Aged 6-12**

A Python analysis pipeline for developing QEEG-based biomarkers of executive function (EF) dysfunction, integrating resting-state EEG with behavioral measures (AUFEI, Fish Flanker Test, Digit Span).

## Study overview

| Parameter | Value |
|-----------|-------|
| Sample | N = 28 children (target: 100), ages 6-12 |
| EEG | 15-channel (10-20 system), 250 Hz, resting-state |
| Conditions | Eyes Open, Eyes Closed, Happy, Calm, Sad, Scare |
| Behavioral | AUFEI-O (exec function), Fish Flanker (inhibitory control), Digit Span (working memory) |
| Analysis | QEEG features + Machine Learning + SHAP interpretability |
| Ethics | Approved by Soeharto Heerdjan Hospital |

## Pipeline architecture

```
Raw EDF files
    |
    v
STAGE 1: Cleaning
    Bandpass (0.5-45 Hz) -> Notch (50 Hz) -> Bad channel detection
    -> Interpolation -> Average reference -> ICA -> 2s epochs -> Artifact rejection
    |
    v
STAGE 2: Feature Extraction (922 features per subject)
    2A. Conventional QEEG: PSD, TBR, FAA, alpha reactivity, coherence
    2B. Advanced: wavelet-Fourier CWT, Hjorth parameters, spectral entropy, PAC
    2C. Covariance: frequency-band covariance matrices (for Riemannian classifiers)
    |
    v
STAGE 3: Behavioral Data Merge
    AUFEI-O scores + Flanker Effect + Digit Span + demographics
    -> median-split EF groups (high/low) for classification
    |
    v
STAGE 4: Analysis
    4A. Correlations (H1-H3): TBR vs Global EF, Theta vs EF, TBR vs Flanker
    4B. ML Classification (H4): 4 feature sets x 8 models, 5-fold CV x 10 repeats
        Models: RF, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP, CNN-LSTM
        Metrics: balanced accuracy, sensitivity, specificity, F1, AUC-ROC
        Hyperparameter tuning: RandomizedSearchCV (nested CV)
    4C. SHAP Feature Importance (H5): biomarker candidate identification
    |
    v
STAGE 5: Quantum-Inspired Exploration (exploratory)
    5A. QEPP: Quantum Entangled Particles Pattern (channel-pair interference)
    5B. Quantum Probability: non-classical band-power interactions
    5C. Tensor Network: von Neumann entropy, purity, fronto-parietal entanglement
    -> Compare quantum vs classical vs combined features (7 models)
```

## Quick start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/biomarker-iium-pipeline.git
cd biomarker-iium-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data files (not included in repo - see Data section below)
#    Copy EDF files into data/EDF_Files/
#    Copy behavioral Excel files into data/

# 4. Update config paths
#    Edit configs/config.yaml to match your data location

# 5. Run on single subject (test)
python run_all.py --subject D0000795

# 6. Run full pipeline
python run_all.py

# 7. Include emotional conditions
python run_all.py --include-emotional

# 8. Run individual stages
python run_all.py --stage 1   # cleaning only
python run_all.py --stage 2   # feature extraction
python run_all.py --stage 3   # behavioral merge
python run_all.py --stage 4   # analysis only
```

## Data setup

**EEG and behavioral data are NOT included in this repository** (contains identifiable participant information).

Team members should obtain data files from the shared drive and place them as follows:

```
data/
  EDF_Files/
    D0000795/
      X_M_X_Name_IGS_Eyes_Open.edf
      X_M_X_Name_IGS_Eyes_Closed.edf
      X_M_X_Name_IGS_1_Happy.edf
      ...
    D0000796/
      ...
  AUFEI-O_Cleaned.xlsx
  Flanker_Test_Pilot.xlsx
  Digit_Span.xlsx
```

Then update `configs/config.yaml`:

```yaml
paths:
  edf_dir: "./data/EDF_Files"
  behavioral_dir: "./data"
```

## Hypotheses

| ID | Hypothesis | Test |
|----|-----------|------|
| H1 | Frontal TBR negatively correlates with Global AUFEI score | Pearson/Spearman + FDR |
| H2 | Frontal theta power negatively correlates with Global AUFEI score | Pearson/Spearman + FDR |
| H3 | Frontal TBR positively correlates with Flanker Effect | Pearson/Spearman + FDR |
| H4 | ML model predicts EF level with >= 75% balanced accuracy | 5-fold CV x 10 repeats |
| H5 | SHAP identifies top QEEG biomarker candidates | TreeExplainer |

## Feature sets compared

The pipeline compares four feature extraction approaches:

| Set | N features | Description |
|-----|-----------|-------------|
| Conventional QEEG | ~150 | PSD, TBR, FAA, coherence, alpha reactivity |
| Conventional + Advanced | ~600 | + wavelet-Fourier CWT, Hjorth, entropy, PAC |
| Covariance only | ~500 | Frequency-band covariance matrices |
| All combined | ~920 | Everything above |

## Preliminary results (N=28 pilot)

### Correlations
- TBR at Cz shows strongest association with Global EF (r = -0.40, p = .035 uncorrected)
- Effect in expected direction but underpowered at N=28

### Classification (best per feature set, 8 models)

| Feature Set | Best Model | Balanced Accuracy | AUC |
|-------------|-----------|-------------------|-----|
| Conventional QEEG | XGBoost | 0.663 | 0.703 |
| Conv. + Advanced | XGBoost | 0.558 | 0.606 |
| Covariance only | XGBoost | 0.633 | 0.669 |
| All features | RandomForest | 0.567 | 0.560 |

H4 target (>=0.75) not yet met at N=28. Underpowered — target N=100.

### Top biomarker candidates (SHAP)
1. Beta coherence F3-P3 (fronto-parietal connectivity, SHAP=0.150)
2. Absolute beta power Pz (parietal midline, SHAP=0.098)
3. Alpha reactivity global (SHAP=0.057)
4. Relative beta power F4 (right frontal, SHAP=0.051)
5. TBR at Cz (SHAP=0.036)
6. Absolute beta power O1 (SHAP=0.035)

### Stage 5: Quantum vs Classical (exploratory)

| Feature Set | Best Model | Bal. Acc | AUC |
|---|---|---|---|
| **Quantum only** | **LogReg** | **0.657** | **0.694** |
| Classical only | RF | 0.585 | 0.662 |
| Combined | LogReg | 0.608 | 0.634 |

Key finding: quantum-inspired features (QEPP entanglement patterns, tensor network entropy, quantum probability interactions) outperform classical QEEG for predicting executive function. Combined features do not improve over quantum-only, suggesting partial redundancy. Stage 5 now compares all 7 sklearn-compatible models (upgraded from 2).

**Three quantum feature families:**
- **QEPP** (Alotaibi et al. 2026): interference patterns between channel pairs via Hilbert transform
- **Quantum Probability**: tests whether band-power pairs violate classical independence (Busemeyer & Bruza 2012)
- **Tensor Network**: von Neumann entropy, purity, and fronto-parietal mutual information from EEG covariance density matrices

## Project structure

```
biomarker-iium-pipeline/
  run_all.py              # Main orchestrator (CLI)
  requirements.txt        # Python dependencies
  configs/
    config.yaml           # All pipeline parameters
  stages/
    stage1_cleaning.py    # EDF -> filtered -> ICA -> clean epochs
    stage2_features.py    # Epochs -> 922 QEEG/wavelet/cov features
    stage3_merge.py       # Features + AUFEI + Flanker + Digit Span
    stage4_analysis.py    # Correlations, ML classification (8 models), SHAP, tuning
    exploratory_quantum.py # Quantum-inspired features (QEPP, tensor network, quantum probability)
  utils/
    io.py                 # Data loading, config, file discovery
  data/                   # NOT IN REPO - see Data Setup
  results/                # Generated outputs (gitignored)
  figures/                # Generated plots (gitignored)
```

## Dependencies

- Python >= 3.10
- MNE >= 1.6 (EEG processing)
- AutoReject (artifact rejection)
- PyWavelets (continuous wavelet transform)
- scikit-learn, XGBoost, LightGBM, CatBoost (ML classification)
- SHAP (feature importance)
- SciPy, statsmodels (statistical tests)
- pandas, matplotlib, seaborn (data handling, visualization)

Optional:
- PyTorch (CNN-LSTM classifier)
- coffeine + pyriemann (Riemannian covariance classifiers)
- pylossless (alternative lossless cleaning pipeline)

## References

- Diamond, A. (2013). Executive functions. *Annual Review of Psychology*, 64, 135-168.
- Arns, M., et al. (2013). EEG theta/beta ratio meta-analysis. *J Atten Disord*, 17(5), 374-383.
- Bomatter, P., et al. (2024). Machine learning of EEG predicts cognitive outcomes. *NeuroImage*, 119156.
- Gabard-Durnam, L.J., et al. (2018). HAPPE pipeline for developmental EEG. *Frontiers in Neuroscience*, 12, 97.
- Lundberg, S.M., & Lee, S.I. (2017). SHAP values. *Advances in Neural Information Processing Systems*, 30.

## Team

| Name | Role |
|------|------|
| Dr. S.Y. Dewi | Principal Investigator |
| Yazid R. Habiburahman | Co-Investigator, Pipeline Development |
| Dandy Aulya | Co-Investigator, Data Collection |
| Talenta Center | Host Institution |
| IIUM | Collaboration Partner |

## License

This code is shared for research collaboration purposes. Contact the PI before any external use.
