# Mathematical and Statistical Methods

This document provides explicit mathematical formulations for all analytical methods used in the pipeline, following the principle that **nothing should be implicit**.

## 1. EEG Preprocessing (Stage 1)

### 1.1 Bandpass Filter and Edge Crop

FIR filter with passband [0.5, 45] Hz, Hamming window, automatic filter length, applied via MNE `raw.filter()`. A 50 Hz notch removes mains line noise.

After filtering, `filter_edge_crop_sec` seconds are trimmed from each end of the recording to discard the FIR transition band. Default 5 s (the previous 0.5 s was too short for a 0.5 Hz HP FIR, where the transition is typically several seconds).

### 1.2 Reference Scheme

Source EDFs are recorded on a Mitsar amplifier with linked-mastoid (A1-A2) reference. The mastoid channels are **not** present in the EDF — Mitsar exports the 15 scalp channels with A1-A2 already subtracted.

Stage 1 commits to **average reference** for the rest of the pipeline:

1. After filtering and bad-channel detection, `raw.set_eeg_reference("average")` is applied (bad channels are excluded from the average automatically).
2. ICA fit, ICLabel classification, and `ica.apply` all operate on this avg-referenced data, so the unmixing matrix W and the data it is applied to share a reference frame (an explicit MNE requirement that the previous flow violated).
3. After bad-channel interpolation, the average reference is re-applied so the newly-restored channels contribute to the average.

The `qc.json` per file records `source_reference: "A1-A2 (Mitsar implicit)"` and `output_reference: "average"` for provenance.

Rationale: ICLabel (Section 1.4) was trained on average-referenced data and is unreliable on other schemes; the FAA literature (Coan & Allen, 2004) and HAPPE both use average reference; downstream spectral measures (TBR, alpha reactivity) are well-defined in this frame.

### 1.3 Bad Channel Detection

Three independent flags; the union is treated as bad before ICA. All thresholds are configurable in `stages/cleaning/config.yaml: params`.

**Variance-based:** Modified z-score using Median Absolute Deviation (MAD):

```
z_i = 0.6745 * (var_i - median(var)) / MAD(var)
```

Channel rejected if `|z_i| > bad_channel_threshold` (default 3.5). Channels listed in `variance_protect_channels` (default `["Fp1", "Fp2"]`) are exempt: pediatric blinks make frontal electrodes naturally high-variance, and flagging them as bad would push them out of the ICA fit and hide exactly the blink topography ICLabel is meant to classify. They remain eligible for the two methods below if a channel is genuinely dead.

**Correlation-based:** Channel rejected if max absolute correlation with any other channel < `bad_channel_corr_threshold` (default 0.2; previously hardcoded at 0.3).

**Flatline:** Channel rejected if standard deviation < `bad_channel_flatline_std` (default 1.0e-7 V).

QC records the three flag lists separately (`bad_by_variance`, `bad_by_correlation`, `bad_by_flatline`) so downstream review can attribute each rejection.

### 1.4 ICA + ICLabel for Artifact Removal

**Decomposition.** Infomax ICA (extended) via MNE `ICA(method="infomax", fit_params={"extended": True})`. Component count is `min(ica_n_components, n_good_channels - 1)`, floored at 4; the `-1` accounts for the rank loss from average referencing. Infomax + extended is used instead of FastICA because ICLabel (below) was trained on the infomax decomposition.

**Fit / apply discipline.** ICA is fit on a temporary 1 Hz high-pass copy of the average-referenced raw (slow drifts < 1 Hz degrade decomposition; HAPPE recommendation). The unmixing matrix W is then applied to the 0.5 Hz bandpassed average-referenced raw. Both share a reference frame (Section 1.2).

**Component classification.** Each fitted IC is classified by `mne-icalabel` (Pion-Tonachini et al., 2019), which assigns one of seven labels (`brain`, `muscle artifact`, `eye blink`, `heart beat`, `line noise`, `channel noise`, `other`) with a probability. The classifier was trained on a large multi-site EEG corpus (over 5000 components, expert-labeled) using a convolutional neural network on the topography map, power spectrum, and autocorrelation features of each component. Inference runs via the `onnxruntime` backend.

A component is excluded if **both** of:

1. Its label is in `iclabel_exclude_labels` (default: `eye blink`, `muscle artifact`, `heart beat`, `line noise`, `channel noise`)
2. Its probability exceeds `iclabel_threshold` (default 0.7; the ICLabel paper recommends ≥0.7 for low-density montages)

No hardcoded cap on the number of excluded components — ICLabel decides. The previous `find_bads_eog` + Fp1/Fp2 correlation path is removed: it was circular (Fp1/Fp2 are the very channels EOG contaminates) and unable to detect non-ocular artifacts.

In the pilot (N=26 × 2 conditions = 52 recordings), this yields 46 excluded ICs, all labeled `eye blink`. Muscle / line noise / channel noise components were not detected, which is expected because the data is bandpassed to 45 Hz: muscle artifact (typically > 30 Hz) and 50 Hz line noise (also notched) lack the spectral signature ICLabel learned from 1-100 Hz data. We accept this trade-off — the dominant resting-state artifact in pediatric EEG is the eye blink, which is classified accurately.

### 1.5 Epoch Rejection — AutoReject Local

Continuous data is segmented into 2 s non-overlapping epochs, then passed through `autoreject.AutoReject` (local) with:

- `n_interpolate = [1, 2, 3]` — max channels eligible for per-epoch interpolation
- `consensus = [0.2, 0.3, 0.4]` — fraction of channels that must vote "bad" to drop the epoch
- `cv = 5` — internal cross-validation for threshold selection

AutoReject (Jas et al., 2017) learns a per-channel peak-to-peak threshold via cross-validation. For each epoch it then either drops the epoch (if too many channels exceed their threshold) or interpolates the small number of channels that do. QC records the per-channel threshold summary (mean, median, max in µV), the count of dropped epochs, and the mean number of channels interpolated per kept epoch.

**No lenient retry.** The previous flow re-ran rejection at 1.5× the threshold when over `max_reject_pct` of epochs were dropped. This gave noisier recordings a more lenient threshold than cleaner recordings, biasing cross-subject comparisons. The retry is removed; `max_reject_pct` is now a warning-only log line, and the `min_epochs` floor (Section 1.6) is the sole gate.

**Fallback.** If AutoReject fails (e.g. too few epochs for CV), a fixed peak-to-peak threshold of `fallback_reject_uv` µV (default 150) is applied. The fallback is also not retried.

### 1.6 Subject-Condition Floor

After per-epoch rejection, any subject-condition recording whose surviving epoch count falls below `cleaning.params.min_epochs` (default 60) is **dropped from disk** — no `*-epo.fif` is written and downstream stages never see it. Severely degraded recordings (in the pilot, one Eyes-Open recording with 29/144 epochs surviving) bias PSD and coherence estimates more than they help; excluding the affected condition while keeping the same subject's other condition is the conservative choice.

### 1.7 QC Audit Trail

For every input EDF, Stage 1 writes one row to `qc.json` with the fields used by analyses and the `verify_qc.py` regression checker:

- Provenance: `subject`, `condition`, `filepath`, `source_reference`, `output_reference`, `duration_sec`, `duration_after_crop_sec`
- Bad channels: `bad_channels`, `bad_by_variance`, `bad_by_correlation`, `bad_by_flatline`, `n_bad_channels`
- ICA: `ica_excluded` (indices), `ica_labels`, `ica_probabilities` (for excluded), `ica_all_labels`, `ica_all_probabilities` (for the full decomposition), `ica_n_components_fit`
- Rejection: `n_epochs_before_reject`, `n_epochs_after_reject`, `n_epochs_dropped`, `pct_epochs_dropped`, `autoreject_used`, `autoreject_threshold_uv_{mean,median,max}`, `n_channels_interpolated_per_epoch_mean`, `reject_threshold_uv` (null when AR succeeded)
- Outcome: `status` ∈ {`OK`, `LOW_EPOCH_COUNT`}, `mean_amplitude_uv`, `std_amplitude_uv`

`stages/cleaning/verify_qc.py OLD_TS NEW_TS` produces a side-by-side diff of two cleaning runs and reports PASS/FAIL against target thresholds (threshold-variance reduction, IC-exclusion distribution, drop-rate variance, status flips).

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

## 3. Analysis (Stage 4)

### 3.0 Target Variable Preparation

The primary target is treated as a **continuous regression problem** with a **post-hoc clinical screening threshold**, not as a binary classification problem from the outset. Two preprocessing steps are applied to the raw target before modelling:

**(1) Age residualization.** Executive function develops sharply between ages 6 and 12 (Diamond, 2013). A given raw EF score therefore means very different things in a 7-year-old versus a 12-year-old. We model the typical age trajectory with ordinary least squares and analyse the residual ("performance relative to peers of the same age"):

```
y_residual = y - LinearRegression(age_months).predict(age_months)
```

The model R² on the raw target is reported in `target_preparation_report.json:age_model_r2`. Default covariate is `age_months` (auto-derived from `age_years` if absent in `full_dataset.csv`); additional covariates may be added via `analysis.target.primary.residualize_covariates` in `stages/analysis/config.yaml`.

The age model is fit **once on the full cohort**. Because it uses no biomarker features and no fold-specific labels, refitting it inside the CV loop is unnecessary and would only inject noise; the residualization step is best understood as a target transform (analogous to log-transforming a skewed outcome) rather than a learned model.

**(2) Two-stage approach: regression development → post-hoc threshold.** All learning and validation is performed on the continuous residualized target. After cross-validation, a fixed quantile of the residualized distribution is used as a clinical "at-risk" cut-off, and screening metrics (sensitivity, specificity, AUC, PPV, NPV) are computed from the cross-validated continuous predictions:

```
threshold = quantile(y_residual, q=1/3)   # default tertile_bottom
y_binary  = (y_residual <= threshold).astype(int)        # 1 = at-risk
y_pred_bin = (y_pred_oof <= threshold).astype(int)
```

The same threshold (computed once from the residualized distribution) is applied to both true and predicted continuous values, so the binary metrics inherit the regression model's calibration without retraining a separate classifier. Sensitivity analysis additionally reports metrics under `quartile_bottom` (q=0.25) and `median` (q=0.5) thresholds in `clinical_screening_metrics.csv`.

**Tertile rationale.** A bottom-tertile cut-off is the standard developmental-screening convention for "at-risk" identification: it flags the lowest ~33% of children, which matches the typical referral rate in school-based EF screening protocols. A quartile cut-off would be too narrow for a screening (high specificity, low sensitivity); a median cut-off is too liberal (half the cohort is "at-risk"). Reporting all three lets the reader pick the operating point that matches their clinical use case.

### 3.1 Hypothesis Correlations

Pre-specified Spearman correlations (H1-H3 plus secondary tests) are reported with both raw and FDR-corrected p-values. Pearson coefficients are reported as a supplementary check on linearity. Correction is restricted to the pre-specified set, not the full feature×outcome grid (see CLAUDE.md "Methodological invariants"). Correlations run on the raw (unresidualized) measures for backward compatibility with the original hypothesis registration.

### 3.2 Pipeline Architecture

```
[unsupervised pre-CV]
    drop_low_variance(threshold=1e-6)
        -> drop_collinear_hierarchical(corr_threshold=0.95)
[per CV fold]
    SimpleImputer(strategy="median")
        -> StandardScaler()
        -> [optional univariate filter, disabled by default]
        -> Regressor
```

Note: **unsupervised steps (variance and collinearity drops) are applied once on the full feature matrix; no target leakage** because neither step consults `y`. Fitting them once before CV is therefore equivalent to refitting per fold but much cheaper, and saves the diagnostic record in a single JSON. Collinearity clustering uses scipy hierarchical clustering on `1 - |corr|` distance with average linkage; the cut at `t = 1 - 0.95 = 0.05` collapses any feature pair with |corr| ≥ 0.95 into one cluster, and the cluster's highest-variance member is retained. The full cluster_map (kept feature → list of dropped followers) is written to `feature_curation_report.json`.

The optional univariate filter (mutual-information regression by default) is disabled because L1 regularisation in `LassoCV`/`ElasticNetCV` already performs supervised feature selection; enabling it for tree-only lineups is a one-line config change.

### 3.3 Models

The default lineup is regression-first, listed below. The **active set is selected via `stages/analysis/config.yaml: models[]`**, and feature sets via `feature_sets[]`. At N = 26 the tighter lineup (Lasso + ElasticNet + RandomForest) is more honestly interpretable than a wide sweep.

| Model                   | Key Parameters                                         | Selection                |
|-------------------------|--------------------------------------------------------|--------------------------|
| LassoCV                 | cv=5, n_alphas=50, max_iter=5000                       | L1 (built-in)            |
| ElasticNetCV            | cv=5, l1_ratio in {0.1, 0.5, 0.7, 0.9, 0.95, 1.0}      | L1+L2 (built-in)         |
| RandomForestRegressor   | n_estimators=300, max_depth=4, min_samples_leaf=2      | tree-based feature usage |
| SVR (RBF)               | C=1.0, gamma="scale"                                   | external (univariate)    |
| GradientBoostingRegressor | n_estimators=200, max_depth=3, lr=0.05               | tree-based feature usage |

**Legacy classifiers** (RandomForest, SVM, KNN, MLP, XGBoost, LightGBM, CatBoost, CNN-LSTM, QSVM_4q_ZZ / 6q_ZZ / 6q_prod) remain available behind `analysis.legacy_classification.enable: true`. When enabled, the pre-refactor median-split + classifier flow runs alongside the regression flow on the same target and writes `legacy_ml_results.csv`. This is intended only as a backwards-compat sensitivity comparison; the regression pipeline is the primary analysis.

### 3.4 Cross-Validation

**CV-only protocol — no train/test holdout.** All performance estimates and hyperparameter selection are obtained via cross-validation on the full cohort. We do **not** carve out a separate test set. At N = 26 a holdout split would either (a) be too small to give a stable estimate (k-shot test sets at N ≤ 6 have ROC-AUC standard errors of 0.15–0.25, larger than any plausible effect; Varoquaux 2018) or (b) shrink the training set to the point where every fold becomes a different model. Vabalas et al. (2019) further show that at N < 100 a single-split holdout produces **systematically biased and high-variance** accuracy estimates, while repeated k-fold CV is unbiased and tighter. Reporting a held-out test number at this N would create the appearance of independent validation without the statistical reality.

**Main evaluation:** `RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)` = 25 evaluations per (feature_set, model). Stratification is not applicable to regression. Out-of-fold predictions for clinical screening are produced by a single `KFold(n_splits=5)` pass via `cross_val_predict` so that every subject has exactly one held-out prediction. Reporting follows the **single-CV-pass** convention recommended by Varoquaux (2018) — fold-level Pearson r values and their across-fold standard deviation are the headline summary, with bootstrap CIs for inference, rather than collapsing to a single point estimate.

**Hyperparameter selection.** Performed by the regularisation-path search built into `LassoCV` and `ElasticNetCV` (internal 5-fold over a 50-point alpha grid; for ElasticNet, additionally over `l1_ratio in {0.1, 0.5, 0.7, 0.9, 0.95, 1.0}`). For RandomForestRegressor we use **fixed conservative hyperparameters** (`n_estimators=100, max_depth=3, min_samples_leaf=3`) chosen *a priori* for the N=26 regime; no grid search or RandomizedSearchCV. Vabalas et al. (2019, §4.2) show that nested-CV hyperparameter search at small N can paradoxically inflate estimated performance because the search adapts to fold-specific noise, and recommend either fixed defaults or strictly `n_iter=1` hyperparameter search. We chose the fixed-defaults route to keep the pipeline auditable.

**No nested RandomizedSearchCV in the regression path.** The legacy classification flow (Section 3.3 footnote, only active behind `legacy_classification.enable: true`) retains a nested-CV tuning loop for backwards comparison; the primary regression analysis does not.

`n_jobs=1` everywhere — joblib parallelism has caused OOM at this feature count on macOS in prior runs.

### 3.5 Statistical Validation

**Permutation test (regression).** 200 random shuffles of the target on the best-scoring (feature_set, model) pair by mean Pearson r. The p-value tests H0: "the model's Pearson r is consistent with the null distribution of permuted-y r values." Implementation runs `cross_val_predict` once per permutation; results stored in `permutation_results.json`.

**Bootstrap CI.** 1000 bootstrap resamples (with replacement) of the OOF (y_true, y_pred) pair, computing both Pearson r and screening AUC at each resample. 95% percentile CIs are reported per (feature_set, model) and per threshold method in `bootstrap_ci.json`.

**Clinical screening metrics derived from regression predictions.** Sensitivity, specificity, balanced accuracy, F1, PPV, and NPV are computed by applying the pre-computed clinical threshold to OOF predictions. AUC uses the *continuous* OOF predictions directly (signed so that "lower predicted residual" → higher at-risk score), so it is threshold-free and robust to mis-specifying the operating point. No separate classifier is trained.

**FDR correction.** Benjamini-Hochberg (`fdr_bh`) is applied (a) across the pre-specified hypothesis correlations (Section 3.1) and (b) across one-sided t-tests of fold-level Pearson r > 0 across all evaluated (feature_set, model) cells; the latter is reported in `regression_summary.csv` as a coarse screen for which combinations exceed null performance after multiplicity correction.

### 3.6 Metrics

- **Primary (regression):** Pearson r between OOF prediction and true residual (per fold, mean across folds).
- **Secondary (regression):** Spearman rho, R², MAE.
- **Clinical screening (post-hoc):** Sensitivity, specificity, balanced accuracy, F1, PPV, NPV at the pre-specified threshold; AUC from continuous OOF predictions.
- **Effect sizes:** For correlation hypotheses, Cohen's d via r-to-d conversion: `d = 2r / sqrt(1 - r^2)`.

### 3.7 SHAP Analysis

SHAP values are computed for the best regressor on the curated feature matrix. `TreeExplainer` is used for tree-based models (RandomForest, GradientBoosting), `LinearExplainer` for L1/L2-regularised regressors (LassoCV, ElasticNetCV), and `KernelExplainer` (background = 50 samples) as the fallback for SVR. The mean absolute SHAP value per feature is reported in `shap_importance.csv`, biological annotations (when available) in `shap_annotated.csv`, and the standard SHAP summary plot in `figures/shap_summary.png`.

For the legacy classification path (when enabled), the original per-CV-fold SHAP averaging is retained:

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
- Coan, J.A. & Allen, J.J.B. (2004). Frontal EEG asymmetry as a moderator and mediator of emotion. *Biological Psychology*, 67(1-2), 7-49.
- Diamond, A. (2013). Executive functions. *Annual Review of Psychology*, 64, 135-168.
- Jas, M., et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage*, 159, 417-429.
- Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. *NeuroImage*, 198, 181-197. https://doi.org/10.1016/j.neuroimage.2019.05.026
- Miyake, A., et al. (2000). The unity and diversity of executive functions. *Cognitive Psychology*, 41(1), 49-100.
- Vabalas, A., Gowen, E., Poliakoff, E., & Casson, A.J. (2019). Machine learning algorithm validation with a limited sample size. *PLOS ONE*, 14(11), e0224365. https://doi.org/10.1371/journal.pone.0224365
- Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, 180(Pt A), 68-77. https://doi.org/10.1016/j.neuroimage.2017.06.061
- Wagenmakers, E.-J., van der Maas, H.L.J., & Grasman, R.P.P.P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, 14(1), 3-22.
- Zhang, D.W., et al. (2017). Theta/beta ratio and EEG in ADHD. *Clinical Neurophysiology*, 128(8), 1436-1443.
