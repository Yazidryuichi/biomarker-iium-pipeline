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

## 2. Feature Extraction (Stage 2)

### 2.0 Feature taxonomy (canonical reference)

Stage 2 extracts **922 conventional features per subject per condition**, verified against `results/features.csv` (which has 924 columns: 922 features + `subject_id` + `condition`). The breakdown, by feature-name prefix:

| Prefix | Count | What | Stage 5 `CLASSICAL_PREFIXES` subset |
|---|---|---|---|
| `cov_` | 480 | Frequency-band channel covariance entries | ✓ |
| `cwt_` | 225 | Continuous wavelet transform | — |
| `psd_` | 120 | 8 bands × 15 channels (abs + rel) | ✓ |
| `hjorth_` | 45 | 3 Hjorth params (activity, mobility, complexity) × 15 channels | — |
| `spectral_` | 15 | Spectral entropy (1 per channel) | — |
| `pac_` | 15 | Phase-amplitude coupling (1 per channel) | — |
| `coh_` | 12 | Channel-pair coherence (selected pairs) | ✓ |
| `tbr_` | 5 | Theta/beta ratio (Fz, F3, F4, Cz + frontal mean) | ✓ |
| `alpha_reactivity` | 4 | Eyes-open vs eyes-closed alpha (posterior channels) | ✓ |
| `faa_` | 1 | Frontal alpha asymmetry F4-F3 | ✓ |
| **Total** | **922** | **All Stage 2 conventional features** | **622 used by Stage 5 fair comparison** |

Stage 5's fair 2×2 comparison (§3 of this document and `stages/stage5_fair_comparison.py`) operates on the 622-feature subset matched by `CLASSICAL_PREFIXES = ("psd_", "coh_", "cov_", "tbr_", "faa_", "alpha_reactivity")` — excluding the 300 "advanced" features (cwt_, hjorth_, spectral_, pac_) that are retained in Stage 4's broader 4-feature-set sweep but not in Stage 5's matched comparison. **The 622 vs 922 figures in this document and PIPELINE_STATUS_REPORT.md refer to these two different scopes**; both are correct in their context.

Stage 6 adds a **separate set of 900 density-matrix features per subject per condition** (4 bands × 15² = 900 real entries; see §5). These are stored in `results/density_matrix_features.csv` and used by Stage 5 as the matched-CV alternative to the 622 classical features.

**Grand total per subject per condition: 922 + 900 = 1,822 features**, evaluated under matched preprocessing pipelines.

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

| Model | Key Parameters | Class Weighting |
|-------|---------------|-----------------|
| Random Forest | n_estimators=100, max_depth=3 | balanced |
| XGBoost | n_estimators=50, max_depth=3, lr=0.1 | via scale_pos_weight |
| LightGBM | n_estimators=100, max_depth=3, lr=0.1 | via is_unbalance |
| CatBoost | iterations=100, depth=3, lr=0.1 | auto |
| SVM (RBF) | C=1.0 | balanced |
| KNN | k=5 | N/A |
| MLP | (64, 32), early_stopping | N/A |

### 3.4 Cross-Validation

**Main evaluation:** RepeatedStratifiedKFold (5 folds x 10 repeats = 50 evaluations per model).

**Hyperparameter tuning:** Nested CV with RandomizedSearchCV (inner: 3 folds, outer: 5x5).

### 3.5 Statistical Validation

**Permutation test:** 500 permutations of label shuffling to establish null distribution. Reported p-value tests H0: "model accuracy = chance."

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

## 4. Quantum-Inspired Features (Stage 5, Exploratory)

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

## 4.5 Stage 6: Explicit Density-Matrix Features (Path B)

Stage 5 builds quantum-inspired *summary* features (QEPP, von Neumann entropy, etc.) and then runs a quantum-kernel SVM on the PCA-reduced subset. Stage 6 instead builds the explicit density matrix entry-wise from the multichannel analytic signal and uses every entry as a feature directly. This isolates the empirical question that Stage 5 cannot cleanly answer: how much of the QSVM result is the density matrix itself versus the encoding-circuit + PCA pipeline?

### 4.5.1 Construction

For each subject, condition (Eyes_Open) and band $b \in \{\delta, \theta, \alpha, \beta\}$:

1. Zero-phase Butterworth bandpass (`scipy.signal.sosfiltfilt`, order 4) on each of the $N=15$ channels. `sosfiltfilt` is mandatory: a causal `sosfilt` injects a frequency-dependent group delay that contaminates every off-diagonal phase term in the density matrix.
2. Hilbert transform per channel produces the analytic signal $z_c(t) \in \mathbb{C}$.
3. At each time $t$, stack channels into the column vector $\psi(t) = [z_1(t), \ldots, z_N(t)]^\top \in \mathbb{C}^N$.
4. Normalize to unit norm: $\hat{\psi}(t) = \psi(t) / \|\psi(t)\|_2$. Time samples with $\|\psi(t)\| < 10^{-12}$ are dropped (silent at this band).
5. Average rank-1 projectors over all kept time samples across all epochs:
   $$ \rho_b = \frac{1}{T} \sum_t \hat{\psi}(t) \, \hat{\psi}(t)^{\dagger} $$
6. Symmetrize $\rho_b \leftarrow \tfrac{1}{2}(\rho_b + \rho_b^{\dagger})$ and renormalize so $\mathrm{Tr}(\rho_b) = 1$.

The result is a Hermitian, positive semi-definite, trace-one $N \times N$ complex matrix — a density matrix in the formal quantum-mechanical sense, even though the construction is purely classical.

QC across N=28 subjects × 4 bands: max trace error $4 \times 10^{-16}$, min eigenvalue above $-7 \times 10^{-17}$ (within float64 round-off). Cauchy-Schwarz $|\rho_{ij}|^2 \le \rho_{ii}\rho_{jj}$ holds throughout.

### 4.5.2 Feature Vector

Per band: $N$ real diagonal entries $\rho_{ii}$ (occupation probabilities) plus the $N(N-1)/2$ strictly-upper-triangle complex entries split into real and imaginary parts. Total per band: $N^2 = 225$ real features. Across four bands: 900 real features at $N=15$.

The $(\mathrm{Re}, \mathrm{Im})$ decomposition is mathematically equivalent to $(|\cdot|, \arg)$ but avoids the cyclic-angle pathology that breaks linear and SVM classifiers.

### 4.5.3 Hilbert-Schmidt Kernel

The Hilbert-Schmidt inner product on density matrices defines a positive semi-definite kernel:
$$ K(s, t) = \frac{1}{B} \sum_b \mathrm{Tr}(\rho_b^s \rho_b^t) = \frac{1}{B} \sum_b \langle \mathrm{vec}(\rho_b^s), \mathrm{vec}(\rho_b^t) \rangle_{\mathrm{HS}} $$

This is the *exact* analogue of the QSVM fidelity kernel under the Schuld (2021) equivalence $K_Q(x,y) = |\langle\psi(x)|\psi(y)\rangle|^2 = \mathrm{Tr}(\rho_x \rho_y)$. A linear SVM on the flat features and a precomputed-kernel SVM with $K(s,t)$ are mathematically equivalent up to selection bias from `SelectKBest`; this self-consistency is enforced as a unit test.

### 4.5.4 Matched Cross-Validation

`RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)`. The same 100 train/test fold indices are reused for every model — flat-feature classifiers, the HS-kernel SVM, *and* the QSVM re-evaluated on Stage 5 quantum features. This makes the QSVM-vs-density-matrix difference a paired comparison and licenses Wilcoxon signed-rank testing.

### 4.5.5 Empirical Result (N=28, balanced 14/14 on `ef_group_global`)

The numbers below are reported at two CV levels for transparency about method development:
**per-fold means** from `RepeatedStratifiedKFold(n_splits=10, n_repeats=10)` are kept visible
because they reproduce the original Stage 5 / portfolio headline, but the **subject-level
leave-one-out (LOSO)** AUC and balanced accuracy are the inference the matched 2×2 fair
comparison weights (see `stages/stage5_fair_comparison.py` and `results/stage5_fair_comparison.json`).
At N=28 the LOSO 95% subject-bootstrap CIs are 2-3× wider than the per-fold bootstrap CIs
originally reported; the per-fold bootstrap treated 100 correlated folds as independent and
was too tight.

| Feature set                     | Model class     | Per-fold BAcc | Per-fold AUC | LOSO BAcc | LOSO AUC [95% subject-bootstrap CI] |
| ------------------------------- | --------------- | ------------- | ------------ | --------- | ----------------------------------- |
| Density matrix                  | Linear SVM      | 0.657         | 0.780        | 0.752     | 0.785 [0.583, 0.947]                |
| Density matrix                  | Shallow RF      | 0.642         | 0.755        | 0.537     | 0.714 [0.492, 0.896]                |
| Classical QEEG                  | Linear SVM      | 0.458         | 0.445        | 0.428     | 0.320 [0.111, 0.544]                |
| Classical QEEG                  | Shallow RF      | 0.618         | 0.615        | 0.642     | 0.617 [0.391, 0.831]                |
| Density matrix                  | HS-kernel SVM   | 0.540         | 0.395        | —         | — (per-fold AUCs not retained)      |
| Quantum features (Stage 5)      | QSVM-6q-ZZ      | 0.495         | 0.525        | —         | — (per-fold AUCs not retained)      |

The original "0.657 vs 0.585" headline (QSVM on quantum features vs SVM-RBF on classical) was
recharacterised in Phase 1 as a fair 2×2 design (feature set × model class) under matched CV
with paired DeLong tests on subject-level AUC. The honest verdicts:

- **Under matched linear-SVM**: density-matrix beats classical substantially, ΔAUC = +0.464,
  DeLong p = **0.001**. *However*, the classical+SVM cell sits at AUC = 0.320 — below chance,
  consistent with high-dimensional overfitting (622-feature classical subset, k=10 ANOVA selection at N=28; see §2.0 for the 622 vs 922 distinction) —
  setting a low bar that the SVM-matched contrast doesn't have to clear high to win. Sensitivity
  work with L1-regularised SVM or smaller-k feature selection is queued.
- **Under matched shallow-RF**: density-matrix and classical are statistically indistinguishable
  at this N, ΔAUC = +0.097, DeLong p = 0.507.
- **Under the original cross-model contrast** (DM-SVM vs Classical-RF), the numerical gap
  ΔAUC = +0.168 survives the rerun but the statistical significance does not (DeLong p = 0.261).

The honest reading: density-matrix features appear to compress signal more usefully than the
conventional QEEG menu **under linear SVM**, but the strength of that claim is model-class-dependent
and the subject-level CIs are wide. Replication at N = 100 and on OpenNeuro ds004284 is the test
that matters. See README "Fair comparison (Stage 5)" section and `stages/stage5_fair_comparison.py`
for the full statistical artifact.

The HS-kernel SVM and QSVM both underperform the linear SVM on flat features on this data: paired Wilcoxon $\mathrm{HS} - \mathrm{QSVM} = +0.045$, $p = 0.44$ (n.s.). This means the Schuld (2021) prediction that the encoding-circuit kernel and the explicit density-matrix kernel span the same RKHS does **not** hold here. The likely cause is the lossy compression in Stage 5: 258 quantum-inspired features → PCA to 6 dims → rescale to $[0, \pi]$ → encoded into a 6-qubit circuit. None of these steps preserves the off-diagonal complex structure of the explicit $\rho$. This is a finding, not a bug.

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
