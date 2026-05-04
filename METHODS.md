# Mathematical and Statistical Methods

This document provides explicit mathematical formulations for all analytical methods used in the pipeline, following the principle that **nothing should be implicit**.

## 1. EEG Preprocessing (Stage 1)

### 1.1 Bandpass Filter
FIR filter with passband [0.5, 45] Hz. Hamming window, automatic filter length. Filter applied via MNE `raw.filter()`.

### 1.2 Bad Channel Detection

**Variance-based:** Modified z-score using Median Absolute Deviation (MAD):

```
z_i = 0.6745 * (var_i - median(var)) / MAD(var)
```

Channel rejected if |z_i| > 3.0 (configurable).

**Correlation-based:** Channel rejected if max correlation with any other channel < 0.3.

### 1.3 ICA for Artifact Removal

FastICA with 14 components (n_channels - 1). ICA fitted on a temporary 1 Hz high-pass filtered copy to prevent slow-drift contamination (HAPPE protocol). Components applied to the original 0.5 Hz filtered data.

EOG component identification via correlation with Fp1/Fp2. Maximum 3 components removed (conservative for 15-channel montage).

### 1.4 Epoch Rejection

AutoReject (Jas et al., 2017) computes data-driven peak-to-peak thresholds. Fallback: 150 uV. Maximum 30% epoch rejection rate enforced; if exceeded, threshold relaxed by 1.5x.

### 1.5 Subject-Condition Floor

After per-epoch rejection, any subject-condition recording whose surviving epoch count falls below `cleaning.params.min_epochs` (default 60) is **dropped from disk** — no `*-epo.fif` is written and downstream stages never see it. Severely degraded recordings (in the pilot, one Eyes-Open recording with 29/144 epochs surviving) bias PSD and coherence estimates more than they help; excluding the affected condition while keeping the same subject's other condition is the conservative choice.

## 2. Feature Extraction (Stage 2)

### 2.1 Power Spectral Density (PSD)

Welch's method with default MNE parameters. Band power computed via numerical integration:

```
P_band = integral(PSD(f), f_min, f_max)  [using np.trapz]
```

**Absolute power:** P_band in V^2/Hz integrated over Hz = V^2

**Relative power:** P_band / P_total

Frequency bands:
- Delta: 1-4 Hz
- Theta: 4-8 Hz
- Alpha: 8-13 Hz
- Beta: 13-30 Hz

### 2.2 Theta/Beta Ratio (TBR)

```
TBR_ch = P_theta(ch) / P_beta(ch)
```

Computed at Fz, F3, F4, Cz. Frontal mean: average of all channel TBRs.

Rationale: TBR is the most studied QEEG biomarker for executive function and attention (Arns et al., 2013; Zhang et al., 2017).

### 2.3 Frontal Alpha Asymmetry (FAA)

```
FAA = ln(alpha_F4) - ln(alpha_F3)
```

Positive FAA = greater left frontal activity = approach motivation.

Alpha power computed via np.trapz (not np.sum) for resolution-independent measurement.

### 2.4 Alpha Reactivity

```
reactivity = (alpha_EC - alpha_EO) / alpha_EC
```

Measures desynchronization from eyes-closed to eyes-open. Normal reactivity > 0.

### 2.5 Coherence

Per-epoch computation using scipy.signal.coherence, then averaged across epochs. This avoids discontinuity artifacts from epoch concatenation.

```
Coh(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))
```

Channel pairs: Fz-Pz, F3-P3, F4-P4 (fronto-parietal connectivity).

### 2.6 Hjorth Parameters

Per-epoch computation, then averaged:

```
Activity = var(x)
Mobility = sqrt(var(dx) / var(x))
Complexity = sqrt(var(ddx) / var(dx)) / Mobility
```

Where dx = first derivative, ddx = second derivative of EEG signal.

### 2.7 Spectral Entropy

```
SE = -sum(p_i * log(p_i))
```

Where p_i is the normalized PSD at frequency bin i. High SE = complex/irregular signal.

### 2.8 Phase-Amplitude Coupling (PAC)

Modulation index (Tort et al., 2010 variant):

```
MI = (log(N) + sum(p_k * log(p_k))) / log(N)
```

Where p_k is the mean beta amplitude in each of N=18 theta phase bins, normalized to a probability distribution.

### 2.9 Covariance Features

Per-band covariance matrices computed as:

```
C_band = (1/N) * sum(cov(X_filtered_epoch))
```

Upper triangle vectorized for ML input. Full matrices saved for Riemannian classifiers.

## 3. Classification (Stage 4)

### 3.1 Target Variable

Continuous Global EF score (mean of AUFEI Working Memory + Inhibitory Control domains) binarized via **fold-internal median split**:

```
For each CV fold:
    threshold = median(Global_EF[train_indices])
    y_train = (Global_EF[train] > threshold).astype(int)
    y_test = (Global_EF[test] > threshold).astype(int)
```

This prevents target leakage: the binarization threshold is never computed using test data.

### 3.2 Pipeline Architecture

```
SimpleImputer(strategy="median")
    -> StandardScaler()
    -> SelectKBest(f_classif, k=adaptive)
    -> Classifier
```

Feature selection k is adaptive: `k = max(5, min(15, n_features // 10))`.

### 3.3 Models

The full set of models implemented in `_build_models()` is listed below; the **active lineup is selected via `stages/analysis/config.yaml: models[]`**, and feature sets via `feature_sets[]`. The current paper-ready default is RandomForest + SVM only — chosen because at N = 26 a tighter lineup is more honestly interpretable than an eight-model sweep. Boosted trees (XGBoost, CatBoost) and others remain available behind a single config edit.

| Model | Key Parameters | Class Weighting |
|-------|---------------|-----------------|
| RandomForest | n_estimators=100, max_depth=3 | balanced |
| SVM (RBF) | C=1.0 | balanced |
| XGBoost | n_estimators=50, max_depth=3, lr=0.1 | via scale_pos_weight |
| LightGBM | n_estimators=100, max_depth=3, lr=0.1 | via is_unbalance |
| CatBoost | iterations=100, depth=3, lr=0.1 | auto |
| KNN | k=5 | N/A |
| MLP | (64, 32), early_stopping | N/A |
| CNN-LSTM | Conv1D(16) -> LSTM(32) -> FC(1) | BCEWithLogitsLoss |
| QSVM_4q_ZZ / 6q_ZZ / 6q_prod | pennylane kernel-SVM | N/A — opt-in |

### 3.4 Cross-Validation

**Main evaluation:** RepeatedStratifiedKFold (`cv_folds=5`, `cv_repeats=5` = 25 evaluations per model in the current configuration). Fold-internal median split of the continuous target prevents threshold leakage.

**Hyperparameter tuning:** Nested CV with RandomizedSearchCV on the best-scoring feature set only (inner: 3 folds, outer: 5 x 5; `n_iter=20`; `n_jobs=1` to avoid joblib OOM under macOS).

### 3.5 Statistical Validation

**Permutation test:** 200 permutations of label shuffling on the best-scoring (feature_set, model) pair to establish a null distribution. Reported p-value tests H0: "model accuracy = chance."

**Bootstrap CI:** 1000 bootstrap resamples of CV balanced accuracy scores. 95% CI reported via percentile method.

**FDR correction:** Benjamini-Hochberg applied across the 8 pre-specified hypothesis tests.

### 3.6 Metrics

- **Primary:** Balanced accuracy (accounts for class imbalance)
- **Secondary:** Sensitivity, specificity, F1-score, AUC-ROC
- **Effect sizes:** Cohen's d via r-to-d conversion: `d = 2r / sqrt(1 - r^2)`

### 3.7 SHAP Analysis

SHAP values computed per CV fold and averaged:

```
For each fold k in 5-fold CV:
    model.fit(X_train_k, y_train_k)
    shap_k = TreeExplainer(model).shap_values(X_test_k)

mean_importance = mean(|shap_k|, across folds)
stability_CV = std(|shap_k|) / mean(|shap_k|)
```

Features with CV < 0.5 labeled "stable"; otherwise "unstable."

## 4. Quantum-Inspired Features (opt-in addition to Stage 2 / Stage 4)

Activated via `stages/features/config.yaml: include_quantum: true` (extraction) and `stages/analysis/config.yaml: include_qsvm: true` (kernel-SVM models). When the quantum feature columns (`qepp_*`, `qi_*`, `tn_*`) are present, Stage 4's `get_feature_sets` automatically adds `quantum_only` and `classical_plus_quantum` feature sets to the comparison. There is no separate "Stage 5".

### 4.1 QEPP (Quantum Entangled Particles Pattern)

Treats bandpass-filtered EEG as analytic signals (Hilbert transform). For channel pair (i, j):

```
interference = |state_i + state_j|^2 - (|state_i|^2 + |state_j|^2)
             = 2 * Re(state_i * conj(state_j))
```

This captures phase-coupling between channels. Note: mathematically equivalent to the real part of coherency, repackaged with quantum-inspired terminology.

### 4.2 Quantum Probability Interactions

Tests for non-classical interactions between band powers:

```
interference_matrix = P_joint(band_a, band_b) - P_marginal(band_a) * P_marginal(band_b)
```

Band powers discretized into n_bins=10 bins (sensitivity analysis: n_bins in {5, 8, 10, 15, 20}).

### 4.3 Tensor Network Features

Covariance matrix treated as density matrix (trace-normalized):

```
rho = C / Tr(C)
```

**Von Neumann entropy:** `S = -sum(lambda_i * log2(lambda_i))`

**Purity:** `gamma = sum(lambda_i^2)`

**Entropy coupling** (NOT mutual information): `S_frontal + S_parietal - S_total`

Note: The submatrix of a covariance matrix is not a proper partial trace over a tensor product Hilbert space. This metric is named "entropy coupling" to avoid conflation with quantum mutual information.

## 5. Hypotheses

Pre-specified, derived from literature review:

- **H1:** Negative correlation between frontal TBR and Global EF score (Arns et al., 2013)
- **H2:** Negative correlation between frontal theta power and Global EF score
- **H3:** Positive correlation between frontal TBR and Flanker Effect (higher TBR = poorer inhibitory control)
- **H4:** ML classification of EF level from QEEG features achieves balanced accuracy >= 75%

## References

- Arns, M., Conners, C.K., & Kraemer, H.C. (2013). A decade of EEG theta/beta ratio research in ADHD. *J Attention Disorders*, 17(5), 374-383.
- Diamond, A. (2013). Executive functions. *Annual Review of Psychology*, 64, 135-168.
- Jas, M., et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage*, 159, 417-429.
- Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Miyake, A., et al. (2000). The unity and diversity of executive functions. *Cognitive Psychology*, 41(1), 49-100.
- Zhang, D.W., et al. (2017). Theta/beta ratio and EEG in ADHD. *Clinical Neurophysiology*, 128(8), 1436-1443.
