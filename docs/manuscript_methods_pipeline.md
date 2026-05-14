# Pipeline Specification — Methods Section Draft

A self-contained methods description ready to drop into the manuscript. References to internal config files and scripts are kept so the methods can be reproduced; remove them in the final manuscript copy if a stricter prose style is required.

---

## 1. Overview

We analysed resting-state EEG (Eyes-Open and Eyes-Closed) and behavioral data from 26 typically-developing children aged 6–12 years in a four-stage Python pipeline (`pipeline.py`):

1. **Cleaning.** EDF → cleaned, artifact-rejected epochs.
2. **Feature extraction.** Cleaned epochs → primitive QEEG features (band powers, coherence, wavelet, Hjorth, spectral entropy, phase-amplitude coupling, frequency-band covariance).
3. **Engineering.** Math-derived composites (theta/beta ratio, frontal alpha asymmetry, alpha reactivity) and merge with behavioral measures.
4. **Analysis.** Unsupervised feature curation, age residualization, regression cross-validation, post-hoc clinical screening, permutation/bootstrap inference, SHAP explainability.

Each stage is independent: a per-stage `config.yaml` declares its predecessor input and timestamped output directory; rerunning a downstream stage automatically picks up the latest upstream output. The full lineage of every result file is recorded in a `run_notes.json` audit trail (timestamp, git commit, input directories consumed).

## 2. EEG Preprocessing (Stage 1)

- **Filtering.** FIR bandpass [0.5, 45] Hz, Hamming window. ICA was fit on a temporary 1 Hz high-pass copy (HAPPE protocol) and applied to the 0.5 Hz filtered data.
- **Bad channels.** Variance-based modified z-score (MAD; rejection at |z| > 3) and correlation-based exclusion. Interpolation runs **after** ICA (not before).
- **ICA.** FastICA, n_components = n_channels − 1. EOG components identified by correlation with Fp1/Fp2; ≤ 3 components removed.
- **Epoching and rejection.** AutoReject (Jas et al., 2017) with peak-to-peak thresholds; max 30% epoch rejection enforced.
- **Subject-condition floor.** Recordings whose surviving epoch count fell below `cleaning.params.min_epochs` were dropped from disk. In the present pilot this excluded 2 subjects' recordings; the cohort retained for analysis was N = 26.

## 3. Feature Extraction (Stage 2)

Per condition (Eyes-Open, Eyes-Closed):

- **Power spectral density.** Welch's method, band powers via `np.trapz` numerical integration over delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz). Both absolute and relative power per channel.
- **Coherence.** `scipy.signal.coherence` per epoch then averaged across epochs, for fronto-parietal pairs (Fz–Pz, F3–P3, F4–P4) per band.
- **Wavelet.** Continuous wavelet transform (Morlet); per-band magnitudes per channel.
- **Hjorth parameters.** Activity, mobility, complexity per epoch then averaged per channel.
- **Spectral entropy.** Shannon entropy of normalised PSD per channel.
- **Phase-amplitude coupling.** Tort modulation index (theta phase × beta amplitude, 18 phase bins).
- **Frequency-band covariance.** Per-band covariance matrices per condition (upper triangle vectorised; full matrices retained for Riemannian classifiers).

## 4. Feature Engineering (Stage 3)

Math-derived composites computed from primitives without re-reading raw EEG:

- **TBR per channel** (Fz, F3, F4, Cz) = θ-power / β-power; `tbr_frontal_mean` is the channel-mean.
- **FAA** = ln(α-power F4) − ln(α-power F3).
- **Alpha reactivity** = (α_EC − α_EO) / α_EC, computed globally and per posterior channel (O1, O2, Pz).

Behavioral data (AUFEI-O parent ratings, computerised Flanker including pre-computed EZ-DDM v/a/t for both congruent and incongruent conditions, Digit Span) were merged on subject ID; age in years and months was computed from date of birth at the assessment date.

## 5. Stage 4 Analysis — Pre-modelling

### 5.1 Unsupervised feature curation (target-blind)

Two passes applied **once** on the full feature matrix before cross-validation. Because no target labels are consulted, the steps cannot leak the outcome and are equivalent to per-fold refitting.

1. **Variance threshold.** Drop columns with var ≤ 10⁻⁶.
2. **Hierarchical collinearity merge.** Average-linkage clustering on distance d(i, j) = 1 − |corr(i, j)|, cut at t = 1 − 0.95 so that any pair with |corr| ≥ 0.95 collapses into one cluster. The cluster's highest-variance member is retained; followers are recorded in a `cluster_map`.

Reduction in this run: 1840 → 250 → **243** features. Diagnostics in `feature_curation_report.json`.

### 5.2 Target preparation (regression-first with post-hoc clinical threshold)

We treat each behavioral outcome as a continuous regression target rather than a binary classification target ab initio.

1. **Age residualization.** Each target is residualized against `age_months` via ordinary least squares on the full cohort. The model is fit once on all 26 subjects (no fold-specific re-fit) because it uses no biomarker features and no fold labels; this is a target transform analogous to log-transforming a skewed outcome. The covariate model R² is reported per target and stored in `target_preparation_report.json`.

2. **Post-hoc clinical threshold.** A fixed quantile of the residualized distribution defines the at-risk label. Default q = 1/3 (`tertile_bottom`) — children below the 33rd percentile of residual EF for their age are flagged at-risk. Sensitivity analyses additionally report metrics under q = 0.25 and q = 0.5. Direction: **lower residual = at-risk** for all targets in this study (lower EF, lower span, lower drift rate, lower delta-v all indicate worse executive control).

The threshold is computed once from the residualized distribution of the **true** target and applied identically to true and predicted continuous values, so binary screening metrics inherit the regression model's calibration without retraining a separate classifier.

## 6. Stage 4 Analysis — Modelling

### 6.1 CV-only protocol (no train/test holdout)

All performance estimates and hyperparameter selection are obtained via cross-validation on the full cohort. We do **not** carve out a separate test set. At N = 26, a holdout split would either be too small to give a stable estimate (Varoquaux 2018: ROC-AUC standard errors ≈ 0.15–0.25 for k-shot test sets at N ≤ 6) or shrink the training set so much that every fold becomes a different model. Vabalas et al. (2019) further show that at N < 100, a single-split holdout produces systematically biased and high-variance estimates while repeated k-fold CV is unbiased and tighter. Reporting follows Varoquaux's single-CV-pass convention: fold-level Pearson r values and their across-fold standard deviation as the headline summary, with bootstrap CIs for inference.

### 6.2 Pipeline architecture (per CV fold)

```
SimpleImputer(strategy="median")
    -> StandardScaler()
    -> [optional univariate filter, disabled by default]
    -> Regressor
```

The optional univariate filter (mutual-information regression) is disabled because L1 regularisation in `LassoCV`/`ElasticNetCV` already performs supervised selection.

### 6.3 Models and hyperparameter policy

Default lineup (CV-only; no `train_test_split`, no nested `RandomizedSearchCV` in the regression path):

| Model | Selection | Key parameters |
|---|---|---|
| LassoCV | built-in 5-fold over a 50-point alpha grid | max_iter = 5000 |
| ElasticNetCV | built-in 5-fold over alpha × {l1_ratio} grid | l1_ratio ∈ {0.1, 0.5, 0.7, 0.9, 0.95, 1.0}, max_iter = 5000 |
| RandomForestRegressor | **fixed** (no tuning) | n_estimators = 100, max_depth = 3, min_samples_leaf = 3 |

`LassoCV` and `ElasticNetCV` self-tune via their internal CV. RandomForest uses fixed conservative hyperparameters chosen *a priori* for the N = 26 regime — Vabalas et al. (2019, §4.2) show that nested-CV hyperparameter search at small N can paradoxically inflate estimated performance because the search adapts to fold-specific noise. We chose the fixed-defaults route to keep the pipeline auditable. `n_jobs = 1` everywhere to prevent joblib OOM at this feature count.

### 6.4 Cross-validation

**Main CV.** `RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)` = 25 evaluations per (feature_set, model). Stratification is not applicable to regression.

**Out-of-fold predictions for clinical screening.** A single `KFold(n_splits=5)` pass via `cross_val_predict` so that every subject has exactly one held-out continuous prediction. Screening metrics (sensitivity, specificity, AUC, PPV, NPV, F1, balanced accuracy) are computed by applying the pre-computed clinical threshold to those OOF predictions; AUC uses the continuous OOF values directly so it is threshold-free.

### 6.5 Inference

- **Permutation test.** 200 random shuffles of the target on the best-scoring (feature_set, model) pair by mean Pearson r. The reported p-value tests H₀: model Pearson r is consistent with the null distribution under permuted y. (`permutation_results.json`)
- **Bootstrap CI.** 1000 resamples (with replacement) of the OOF (y_true, y_pred) pair, computing both Pearson r and screening AUC at each resample. 95% percentile CIs per (feature_set, model) and per threshold method. (`bootstrap_ci.json`)
- **FDR.** Benjamini–Hochberg (`fdr_bh`) applied (a) across pre-specified hypothesis correlations (H1–H3 + planned secondary tests) and (b) across one-sided t-tests of fold-level Pearson r > 0 across all evaluated (feature_set, model) cells. Results in `regression_summary.csv`.

### 6.6 Targets analysed

The same pipeline is applied independently to six pre-specified targets, each with its own age residualization, clinical threshold, regression CV, screening, inference, and SHAP:

| Target | Source | Direction (lower = at-risk?) |
|---|---|---|
| Global_EF | AUFEI-O mean of 5 subscales | Yes |
| IC_score | AUFEI-O Inhibitory Control subscale | Yes |
| WM_score | AUFEI-O Working Memory subscale | Yes |
| BW_Span | Digit Span Backward | Yes |
| ddm_v_incongruent | EZ-DDM drift rate, incongruent trials | Yes |
| ddm_delta_v | EZ-DDM (congruent − incongruent) drift difference | Yes |

Per-target outputs land under `stages/analysis/runs/<ts>/<target>/`.

## 7. Behavioral Cross-Correlation (Outside the Main Pipeline)

To confirm convergent and discriminant validity of the behavioral measures, an offline script (`scripts/behavioral_correlations.py`) computes pairwise Spearman and Pearson correlations among Digit Span, Flanker (including all DDM variants), Global EF, and the five EF subscales, with Benjamini–Hochberg FDR across the upper triangle of the Spearman matrix. Outputs (full matrices, FDR-significant pairs, heatmap PNG) are written under `scripts/runs/<ts>/`. This analysis is reported as supplementary in the manuscript.

## 8. SHAP Explainability

For each target, SHAP values are computed for the best regressor on the curated feature matrix. `TreeExplainer` is used for tree-based models (RandomForest, GradientBoosting); `LinearExplainer` for L1/L2-regularised regressors (LassoCV, ElasticNetCV); `KernelExplainer` (background = 50 samples) as the fallback. Mean absolute SHAP per feature is reported in `shap_importance.csv`; biological annotations in `shap_annotated.csv`; the standard SHAP summary plot in `figures/shap_summary.png`.

## 9. Key References

- **Cross-validation policy:**
  Vabalas, A., Gowen, E., Poliakoff, E., & Casson, A.J. (2019). Machine learning algorithm validation with a limited sample size. *PLOS ONE*, 14(11), e0224365.
  Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, 180(Pt A), 68–77.
- **EZ-diffusion model:**
  Wagenmakers, E.-J., van der Maas, H.L.J., & Grasman, R.P.P.P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, 14(1), 3–22.
- **AutoReject:**
  Jas, M., Engemann, D.A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage*, 159, 417–429.
- **SHAP:**
  Lundberg, S.M. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- **Theta/beta ratio:**
  Arns, M., Conners, C.K., & Kraemer, H.C. (2013). A decade of EEG theta/beta ratio research in ADHD. *J. Attention Disorders*, 17(5), 374–383.
  Zhang, D.W., et al. (2017). Theta/beta ratio and EEG in ADHD. *Clin. Neurophysiol.*, 128(8), 1436–1443.

A fuller reference list is maintained in `METHODS.md`.
