# Biomarker_IIUM Progress Report — Methodology, Pipeline, and Current Results

> **Historical snapshot — predates the 2026-05-13 Phase 1 honest-headline reframe.**
> This document was written 19 Apr 2026 against the pre-Phase-1 analysis (per-fold reporting, "quantum-inspired" framing for Stage 5). The current canonical headlines are in [README.md](../README.md) (Fair comparison 2×2, subject-level LOSO, DeLong tests) and [METHODS.md §2.0](../METHODS.md) (feature taxonomy). The numbers reported below — e.g. CNN-LSTM 0.652 / AUC 0.743 — describe the original 7-model × 4-feature-set sweep and remain available in the README appendix; the headline DM-SVM AUC = 0.785 with subject-bootstrap CIs supersedes them. The "quantum-inspired" terminology used throughout this report has been replaced in current docs with "covariance density matrix features" (non-linear classical feature transforms). Retained as historical record of the April pilot snapshot.

**Study title:** Early Development of Executive-Dysfunction Biomarkers Using Quantitative EEG — A Machine-Learning Approach in Indonesian Children Aged 6–12

**Institution:** Talenta Center, Yayasan Bina Talenta Tunas Bangsa Karya Mandiri

**Report date:** 19 April 2026

**Author:** Yazid Ryuichi Habiburahman, with collaboration from Dandy Aulya (pipeline co-developer) and Dr. S.Y. Dewi (EEG methodology supervisor)

**Audience:** Talenta Center team. This report is the English companion to the Indonesian original (`Laporan_Progres_Talenta_20260419.md`) kept alongside the research folder.

---

## 1. Executive Summary

The Biomarker_IIUM study has completed the full end-to-end pipeline — from raw EEG acquisition through machine-learning analysis — on a pilot cohort of N = 28 children aged 6–12 years. The pipeline is reproducible, publicly documented on GitHub, and has passed an internal audit covering ten methodological issues (including ICA/re-reference ordering, PSD integration, multiple-comparison correction, and cross-validation-stable SHAP). The current best model is a CNN-LSTM on the combined conventional-plus-advanced feature set, achieving a balanced accuracy of 0.652 (± 0.174) and an AUC of 0.743. The top-ranked biomarker candidates are relative beta power at parietal (Pz, P3) and occipital (O1) electrodes — *not* the frontal theta-beta ratio (TBR) hypothesised in the proposal. Frontal TBR showed a nominal negative correlation with Global EF (r = -0.489, p = 0.013, N = 25) that did not survive FDR correction across the eight pre-specified hypotheses. The H4 target of 75 percent balanced accuracy was not met at N = 28, which is consistent with the study's pilot design. An exploratory quantum-inspired feature subset produced a QSVM balanced accuracy of 0.657 against a classical SVM baseline of 0.585 on the same feature subset, and is retained as an exploratory arm. This document records the methodology executed, the exact numerical results from the `results/` folder, the clinical interpretation, the limitations, and the prioritised next actions for the Talenta team.

## 2. Background and Study Aims (Brief)

The study aims to develop candidate executive-dysfunction biomarkers from resting-state QEEG, integrated with locally validated Indonesian behavioural instruments (AUFEI and Fish Flanker Test) using machine learning. The context for urgency is documented in Proposal_QEEG_v4: AUFEI 2025 data (N = 624 adolescents aged 13–24) report that 81.3 percent of Indonesian adolescents have suboptimal executive-function development, with 39.8 percent showing working-memory difficulties and 32.7 percent showing inhibitory-control difficulties. Reported ADHD prevalence in Indonesia varies widely — from 5.47 percent in Yogyakarta to 15.1 percent in Surabaya. To date, only one EEG study on Indonesian children has been published (Subandriyo et al., 2021, N = 9 enrolled / 8 completed, neurofeedback feasibility in *Kobe J Med Sci*), so no QEEG normative reference exists for this population. Biomarker_IIUM fills that gap, with the pilot N = 28 designed to scale to the proposal's full-cohort target of N = 100.

The five research questions from the proposal are:

1. Characterise the resting-state QEEG profile of Indonesian children aged 6–12 at different levels of executive function.
2. Correlate resting-state QEEG parameters with Global AUFEI scores.
3. Correlate resting-state QEEG parameters with the Flanker Effect.
4. Identify the best-performing machine-learning algorithm for predicting executive-function level.
5. Identify the QEEG features with the highest feature importance as biomarker candidates.

This report addresses the first four questions using the pilot cohort and addresses the fifth using SHAP attribution on the top-performing model.

## 3. Methodology

### 3.1 Participants and Data Acquisition

Participants are typically developing children (no diagnosed neurodevelopmental conditions) aged 6–12, enrolled at Islamic Green School, per the scope defined in Proposal §1.2. The current pilot cohort is 28 children with complete QEEG data. Of these 28, 25 have valid frontal TBR scores after artefact rejection — three participants had frontal-electrode artefacts that prevented TBR estimation. Behavioural and demographic data are stored in `Digital_Consent/`, `EF_Biomarker/`, `Digit_Span.xlsx`, and `Flanker_Test_Pilot.xlsx`.

EEG acquisition used a 15-channel setup at 250 Hz. The proposal specified 19 channels at 256 Hz; the difference reflects equipment availability and is a hardware deviation, not a methodological error. Resting-state recordings include eyes-open and eyes-closed conditions per Proposal §1.2.d.

### 3.2 Behavioural Instruments

Executive function was measured with three instruments:

- **AUFEI (Alat Ukur Fungsi Eksekutif Indonesia)** is the primary outcome instrument, standardised and normed for the Indonesian population (Dewi et al., 2025). Global AUFEI is the classification target for the machine-learning pipeline.
- **Fish Flanker Test**, the child-adapted Flanker Task (Rueda et al., 2004), measures inhibitory control through the Flanker Effect.
- **Digit Span (Forward and Backward)** measures working-memory components.

Together these three instruments cross-validate executive function across judgment-based (AUFEI), inhibitory-performance (Flanker), and working-memory-performance (Digit Span) modalities.

### 3.3 Data Processing Pipeline: Five-Stage Architecture

The pipeline is organised into five sequential stages that can be run individually with the `--stage N` flag. Source code lives at `https://github.com/Yazidryuichi/biomarker-iium-pipeline`, with documentation in `RESEARCH_NOTES.md` and `README.md`.

**Stage 1 — Signal cleaning (`stages/stage1_cleaning.py`).** Raw EDF files are bandpass-filtered (0.5–45 Hz) and notch-filtered (50 Hz), followed by Independent Component Analysis (ICA) to remove ocular and muscular artefacts. Consistent with HAPPE best practice (Gabard-Durnam et al., 2018), ICA is applied *before* re-referencing to the average reference, not after. Data are segmented into 2-second epochs with overlap, and epochs with amplitudes exceeding ±100 µV are rejected automatically. Quality-control metrics are written to `results/qc_stage1.json`.

**Stage 2 — Feature extraction (`stages/stage2_features.py`).** Each clean epoch becomes a numerical feature vector. A total of 922 features per participant are extracted, partitioned into four subsets:

- *Conventional QEEG* (142 features) — absolute and relative power spectral density across five bands (delta, theta, alpha, beta, gamma) per channel, plus frontal TBR, FAA (F4–F3), and global alpha reactivity.
- *Advanced features* (an additional 300 features) — phase coherence between channel pairs in theta, alpha, and beta bands, plus advanced spectral metrics.
- *Covariance-only* (480 features) — between-channel covariance matrices per band, vectorised.
- *Full feature union* (922 features).

PSD is estimated with Welch's method using 2-second windows and 50 percent overlap. Coherence is computed at the epoch level and averaged. Covariance matrices are cached in `results/cov_matrices.npz` so they can be reused without re-extraction.

**Stage 3 — Data merge (`stages/stage3_merge.py`).** QEEG features are joined with behavioural data (Global AUFEI, Digit Span, Flanker Effect) on participant ID. Output is `results/full_dataset.csv`, ready for correlation and ML analysis.

**Stage 4 — Correlations, classification, SHAP, tuning (`stages/stage4_analysis.py`).** Four sub-analyses run in sequence:

1. *Spearman correlations* between selected QEEG features (frontal TBR, theta_Fz, FAA, coherence, alpha reactivity) and three behavioural outcomes (Global EF, Digit Span Backward, Flanker Effect), with FDR correction across eight pre-specified hypotheses.
2. *Binary classification* of Global EF using a median split, with 10-fold stratified cross-validation. Eight models are compared: Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, MLP (shallow neural network), and CNN-LSTM (hybrid convolutional-recurrent deep learning).
3. *Feature selection* picks the top 10 features per set by mutual information, to avoid overfitting when the feature space exceeds the sample size.
4. *SHAP attribution* is computed on the best model per feature set using a 3-of-5 CV-stable threshold, so reported features must appear in the top 10 on at least three of five folds.
5. *Hyperparameter tuning* uses RandomizedSearchCV with nested cross-validation on the best feature set (`conventional_qeeg`) for three models (Random Forest, SVM, XGBoost).

**Stage 5 — Exploratory quantum-inspired features (`stages/exploratory_quantum.py`).** As an exploratory arm, 258 quantum-inspired features are extracted from the covariance matrices (von Neumann entropy, quantum relative entropy, and several entanglement-inspired measures). Three subsets (quantum-only, classical-only, combined) are evaluated with logistic regression and a quantum-kernel SVM (QSVM) implemented via PennyLane. The QSVM code lives at `stages/qsvm_classifier.py` and passes nine unit tests.

### 3.4 Quality Assurance and Methodological Audit

The pipeline has been audited internally for ten methodological issues. Fixes include: correcting ICA order to run before re-referencing (HAPPE-compliant), correcting PSD integration for band power, correcting the coherence formulation, correcting Hjorth-parameter computation, and ensuring the median split is stable across folds. An AI-assistant disclosure covering Claude's use for code review and scaffolding has been added to `README.md` per ICMJE and Nature guidelines, and all commits sit under public version control.

## 4. Current Results (N = 28)

All numbers in this section are drawn directly from the live files in `results/`: `correlations.csv`, `ml_results.csv`, `shap_importance.csv`, and `qc_stage1.json`, as of 7 April 2026.

### 4.1 Research Question 1 — Resting-State QEEG Profile Characterisation

Resting-state QEEG profiles for the 28 children have been extracted and stored in `results/features.csv`. Distributions of band power, TBR, FAA, and coherence per participant are available for normative analysis. Because this is a pilot of 28 participants, the reported values function as a preliminary baseline rather than a definitive norm. A robust Indonesian QEEG norm requires the full N = 100 cohort targeted by the proposal.

### 4.2 Research Question 2 — QEEG × AUFEI Correlations

Complete results are in `results/correlations.csv`. The table below summarises the eight pre-specified correlations:

| Hypothesis | Pair | r (Spearman) | p | N | p (FDR) | Significant after FDR |
|-----------|------|--------------|---|---|---------|-----------------------|
| H1 | Frontal TBR ↔ Global EF | −0.489 | 0.0131 | 25 | 0.09 | No |
| H2 | Theta_Fz ↔ Global EF | 0.240 | 0.2185 | 28 | 0.3542 | No |
| H3 | TBR ↔ Flanker Effect | −0.116 | 0.5803 | 25 | 0.6632 | No |
| — | TBR_Fz ↔ Global EF | −0.389 | 0.0544 | 25 | 0.1451 | No |
| — | TBR_Cz ↔ Global EF | −0.455 | 0.0225 | 25 | 0.0900 | No |
| — | FAA ↔ Global EF | 0.031 | 0.8773 | 28 | 0.8773 | No |
| — | Alpha reactivity ↔ Global EF | −0.156 | 0.4294 | 28 | 0.5725 | No |
| — | TBR ↔ Digit Span Backward | −0.254 | 0.2214 | 25 | 0.3542 | No |

Interpretation for the Talenta team:

The direction of the frontal TBR × Global EF correlation is negative, as theory predicts — children with higher frontal TBR tend to show lower EF scores. The magnitude (r = -0.489) is a medium effect size by Cohen's conventions. The raw p-value of 0.0131 is nominally significant at α = 0.05, but after FDR correction across the eight jointly tested hypotheses, the adjusted p becomes 0.09 and no longer clears the conventional threshold. A similar pattern holds for TBR_Cz (r = -0.455, p = 0.022, FDR p = 0.09).

This pattern aligns with the international meta-analytic trend of the last five years, which has shown that TBR effect sizes decline across successive cohorts and that ADHD subtype heterogeneity makes TBR insufficient as a standalone diagnostic biomarker. The Biomarker_IIUM pilot fits that literature: there is a signal, but it is not strong enough for individual-level clinical application. The secondary hypotheses (FAA, alpha reactivity) showed no meaningful relationship with Global EF, consistent with recent scepticism about these parameters as standalone paediatric EF biomarkers.

### 4.3 Research Question 4 — Comparison of Eight Machine-Learning Algorithms

Complete results are in `results/ml_results.csv`. The table below summarises the best model per feature set:

| Feature Set | Input features | Top-10 selected | Best Model | Balanced Acc | Std | AUC | Sensitivity | Specificity |
|-------------|----------------|-----------------|------------|--------------|-----|-----|-------------|-------------|
| conventional_qeeg | 142 | 10 | **CNN-LSTM** | **0.640** | 0.153 | **0.768** | 0.497 | 0.783 |
| conventional_plus_advanced | 442 | 10 | **CNN-LSTM** | **0.652** | 0.174 | **0.743** | 0.530 | 0.773 |
| covariance_only | 480 | 10 | RandomForest | 0.587 | 0.190 | 0.643 | 0.593 | 0.580 |
| all_features | 922 | 10 | MLP | 0.588 | 0.130 | 0.630 | 0.903 | 0.273 |

The overall best model is CNN-LSTM on `conventional_plus_advanced`, with balanced accuracy 0.652 (± 0.174) and AUC 0.743. On the simpler `conventional_qeeg` set, CNN-LSTM reaches AUC 0.768 — slightly higher AUC, slightly lower balanced accuracy. Gradient-boosted models (CatBoost, LightGBM) are strong second-tier candidates on the conventional set, with CatBoost at 0.623 and LightGBM at 0.622.

Three important observations follow from the table.

First, larger feature sets (480 or 922 features) do *not* improve performance — they make it worse. This is the textbook signature of overfitting when feature count exceeds sample size, and it is consistent with the statistical-power constraint at N = 28. The more compact conventional feature sets work better. This reinforces the interpretation that the N = 28 pilot is sufficient to demonstrate the pipeline but insufficient for biomarker discovery in a large feature space.

Second, MLP on `all_features` shows very high sensitivity (0.903) combined with very low specificity (0.273), meaning it tends to predict nearly everyone as "high EF" rather than discriminating between groups. This is a classic class-prediction bias in neural networks overfit on small samples.

Third, the hyperparameter-tuned models (`tuned_best`) performed *worse* than their untuned counterparts — a counter-intuitive but common phenomenon at small N: nested cross-validation on a small sample produces more conservative performance estimates because each inner tuning fold has only about 22 data points.

The H4 proposal target of 75 percent balanced accuracy has not been met. This is entirely expected: the pilot is at N = 28, while H4 was defined with reference to N = 100.

### 4.4 Research Question 5 — Feature Importance (SHAP)

The top ten features by mean absolute SHAP value, from `results/shap_importance.csv`:

| Rank | Feature | Mean Abs SHAP | Neurophysiological Interpretation |
|------|---------|---------------|-----------------------------------|
| 1 | psd_rel_beta_Pz | 0.0654 | Relative beta power at midline parietal |
| 2 | psd_rel_beta_P3 | 0.0524 | Relative beta power at left parietal |
| 3 | psd_rel_beta_O1 | 0.0426 | Relative beta power at left occipital |
| 4 | psd_rel_beta_F8 | 0.0398 | Relative beta power at right frontal |
| 5 | psd_rel_beta_F4 | 0.0394 | Relative beta power at right frontal |
| 6 | psd_rel_beta_F3 | 0.0258 | Relative beta power at left frontal |
| 7 | psd_rel_beta_P4 | 0.0255 | Relative beta power at right parietal |
| 8 | psd_rel_beta_Fp1 | 0.0229 | Relative beta power at left pre-frontal |
| 9 | psd_rel_beta_Fp2 | 0.0197 | Relative beta power at right pre-frontal |
| 10 | psd_rel_beta_Cz | 0.0171 | Relative beta power at midline central |

The pattern is unusually consistent and neurophysiologically meaningful: all ten top-ranked features are relative beta power, with a spatial distribution dominated by parietal electrodes (Pz, P3, P4), followed by occipital (O1), frontal (F3, F4, F8, Fp1, Fp2), and central (Cz). None of the top ten is TBR, FAA, or coherence — the three parameters hypothesised in the proposal as the leading biomarker candidates.

Clinical interpretation: higher posterior-parietal beta is associated with cortical arousal and active attention, and reductions in posterior beta have been reported as a marker of attentional difficulty across several populations. The Biomarker_IIUM finding that relative parietal beta is the strongest predictor suggests that the EF signal in Indonesian children aged 6–12 is more concentrated in the posterior attention network (parietal-occipital) than in the classical frontal executive-control network. This is a *data-driven* biomarker candidate rather than an a-priori hypothesis, and it must therefore be treated epistemologically as a hypothesis for replication, not as a confirmatory finding.

### 4.5 Exploratory — Quantum-Inspired Features

In Stage 5, 258 quantum-inspired features were evaluated against 664 classical features on a comparable subset. The quantum-kernel SVM (QSVM) reached balanced accuracy 0.657 against a classical SVM baseline of 0.585 on the same subset. The 0.072-point gap points to the possibility that quantum-inspired features capture non-linear structure inaccessible to the classical features, but this gap sits within a wide confidence interval at N = 28 and must be treated as an exploratory signal rather than evidence. This experiment seeds the third sub-study in the PhD proposal Yazid is developing.

## 5. Discussion — What Worked, What Didn't

### 5.1 What Worked

The acquisition-to-model pipeline has run end-to-end on 28 participants with reproducible and publicly documented results. Four of the five research questions have received pilot-level answers, and the fifth has produced a biomarker candidate worthy of replication. The methodological contributions — especially HAPPE-compliant ICA ordering, FDR correction across eight pre-specified hypotheses, and CV-stable SHAP — make the reported results robust against standard peer-review critique. The exploratory quantum-inspired sub-experiment is an original methodological contribution which, to our knowledge, has not previously been applied to an Indonesian paediatric cohort.

### 5.2 What Didn't Match the Original Hypotheses

Three of the proposal's leading hypotheses did not receive support at N = 28.

First, frontal TBR did not survive FDR correction as a Global EF predictor. The nominal correlation (r = -0.489) supports the hypothesised direction but does not clear the conservative statistical threshold.

Second, FAA showed no meaningful association with Global EF (r = 0.031), consistent with recent literature scepticism about FAA as a standalone paediatric EF biomarker.

Third, the biomarker with highest feature importance was *not* TBR, FAA, or coherence — it was relative beta power at posterior electrodes. This contradicts the a-priori hypothesis but becomes a new contribution worth investigating further.

### 5.3 Technical Limitations

**Pilot sample size (N = 28)** is the primary constraint — every confirmatory-seeming claim on this cohort must carry hedging language. For 922 features over 28 samples, the feature-to-sample ratio is roughly 33 : 1, well above the safe threshold of about 10 : 1. This is why enlarging the feature set reduces performance.

**Single-site design** — all participants come from one school in one geographical area. External validity to the broader Indonesian population is untested.

**Binary median split** on Global EF sacrifices dimensional information. Continuous regression as a secondary analysis would be more statistically powerful but requires larger effect sizes to detect at small N.

**CNN-LSTM technical issue on macOS** — the top-performing model hangs under certain system configurations. Documented in `PIPELINE_STATUS_REPORT.md` with a recommendation to run on Linux. Dandy has push access and workaround instructions.

**Quantum sub-experiment** at N = 28 must be treated as proof-of-method, not a quantum-advantage validation. Replication at N = 100 with a preregistered analysis plan is a prerequisite for any substantive quantum claim.

### 5.4 Conceptual Limitations

The study recruits only typical children — no ADHD, autism-spectrum, or specific-learning-disorder diagnoses. The biomarkers identified therefore operate in the typical-to-subthreshold range, and generalisation to clinical populations requires a separate follow-up study. This scope limitation is stated explicitly in Proposal §1.2.c and was retained deliberately to avoid confounding by medication or comorbidity.

## 6. Next Actions

Next actions are organised across three time horizons.

### 6.1 Short-Term 

First priority is completing a pipeline run on Dandy's machine (Linux) to confirm that CNN-LSTM and hyperparameter tuning execute without the macOS workaround. After that, running the pipeline with the `--include-emotional` flag to add the Happy, Calm, Sad, and Scare conditions — these are named in the proposal but have not been executed yet. The output will add four new feature subsets that can be compared with the resting-state baseline and may increase classification sensitivity.

In parallel, adding aperiodic 1/f slope features via FOOOF (Akbarian et al., 2023). This measure proxies cortical excitation–inhibition balance and is orthogonal to band power, so it should improve predictive validity without worsening overfitting. Implementation needs only to reuse existing PSD estimates and adds roughly 15 × 15 = 225 features (slope, offset, parametrised oscillatory peaks per channel).

### 6.2 Medium-Term (6–12 months)

**Cohort expansion to N = 100** per the original proposal, with an analysis-plan preregistration at the Open Science Framework. The primary outcome and the decision rule for classifier comparison must be locked before additional acquisition, so that results move from exploratory to confirmatory status. The quantum feature subset in particular needs a stricter feature-to-sample ratio.

**Wake slow-wave detection** following Pinggal et al. (2026, J Neurosci). That study showed slow-wave density over parietotemporal electrodes statistically mediates attentional difficulties in adult ADHD. In children, where baseline low-frequency activity is higher, this biomarker is plausibly powerful. Implementation requires a template-matching algorithm applied to the already-cleaned continuous data.

**Replication of the leading biomarker (posterior-parietal beta)** on held-out data. If beta power at Pz, P3, and O1 remains the top predictor in the expanded cohort, its status advances from candidate to pilot-validated biomarker.

### 6.3 Long-Term 

**Data-driven network decomposition** via TDE-HMM (Rossi et al., 2023, Communications Biology) to infer frequency-specific networks underlying working memory without a-priori ROI selection. The method directly addresses the limitations of traditional band-power analysis and has been validated on n-back paradigms, which overlap with the Fish Flanker Test.

**Travelling-wave analysis** using the MEG-EEG model-based approach from Grabot et al. (2025, PLOS Computational Biology). This opens the phase dimension that power analysis cannot access, and it overlaps methodologically with the Dugué group at INCC Paris — Yazid's target PhD supervisor.

**Cross-dataset replication** using the Healthy Brain Network (HBN) or ADHD-200 datasets to test domain generalisation of the biomarkers identified in the Indonesian cohort. Even a negative result is publishable because it quantifies the site-specificity that is rarely reported in the paediatric qEEG literature.

## 7. Implications for Talenta Center

Three direct implications for the Talenta Center:

First, the Biomarker_IIUM pipeline is now fully functional and can be run on new participant batches without re-engineering. The infrastructure investment has paid off. Adding 10–15 new participants per batch will continue to strengthen statistical power and validate the stability of the candidate biomarkers.

Second, Dr. Dewi's AUFEI instrument has received its first neurophysiological cross-validation through this study. Although the TBR × AUFEI signal does not survive FDR correction, the direction and magnitude of the correlation (r = -0.489) offer preliminary support for AUFEI's construct validity as an executive-function indicator reflected in QEEG parameters.

Third, the finding that posterior-parietal beta power is the leading biomarker candidate gives a specific target for the Talenta Life Skill intervention programme. If replication at N = 100 confirms this finding, the parameter becomes usable as a neurophysiological outcome measure for intervention-efficacy evaluation — methodologically more rigorous than evaluations based on behavioural scores alone.

## 8. Appendix — File Structure

```
Biomarker_IIUM/
├── Ethic_Approval_Docs/
│   └── Proposal_QEEG_v4 _Turnitined.pdf        (proposal, 5% Turnitin similarity)
├── results/
│   ├── qc_stage1.json                          (acquisition QC metrics)
│   ├── features.csv                            (922 features × 28 participants)
│   ├── cov_matrices.npz                        (covariance matrices)
│   ├── full_dataset.csv                        (features + behavioural, merged)
│   ├── correlations.csv                        (8 hypotheses × r, p, p_FDR)
│   ├── ml_results.csv                          (8 models × 4 feature sets × 6 metrics)
│   └── shap_importance.csv                     (top-10 features × mean_abs_shap)
├── pipeline/                                   (repo clone — see GitHub for latest)
│   ├── run_all.py                              (main CLI — python run_all.py --help)
│   ├── stages/                                 (5-stage pipeline)
│   ├── configs/config.yaml                     (8 models, 5 metrics)
│   ├── README.md                               (public documentation)
│   └── RESEARCH_NOTES.md                       (technical notes + AI disclosure)
├── PIPELINE_STATUS_REPORT.md                   (internal status report for Dandy, 7 Apr 2026)
├── Laporan_Progres_Talenta_20260419.md         (Indonesian original of this document)
└── progress_report_talenta_20260419.md         (this English companion — in pipeline/docs/ of the repo)
```

**GitHub repo:** https://github.com/Yazidryuichi/biomarker-iium-pipeline (private — Dandy invited as collaborator with push access).

**For Talenta team members who need access to code and results:** request repo access through Dandy or Yazid. Pipeline documentation lives in `README.md` and `RESEARCH_NOTES.md`. Each CSV in the `results/` folder opens directly in Excel for quick inspection.

## 9. Key References

Akbarian, F., Rossi, C., Costers, L., et al. (2023). The spectral slope as a marker of excitation/inhibition ratio and cognitive functioning in multiple sclerosis. *bioRxiv*. https://doi.org/10.1101/2023.01.23.525139

Arns, M., Conners, C. K., & Kraemer, H. C. (2013). A decade of EEG theta/beta ratio research in ADHD: A meta-analysis. *Journal of Attention Disorders*, 17(5), 374–383. https://doi.org/10.1177/1087054712460087

Dewi, S. Y., et al. (2025). Alat Ukur Fungsi Eksekutif Indonesia (AUFEI) — validation and norming. PDSKJI research output (venue to be confirmed with corresponding author; note: not indexed in *International Journal of Neuropsychopharmacology* as previously assumed).

Gabard-Durnam, L. J., Mendez Leal, A. S., Wilkinson, C. L., & Levin, A. R. (2018). The Harvard Automated Processing Pipeline for Electroencephalography (HAPPE): Standardized Processing Software for Developmental and High-Artifact Data. *Frontiers in Neuroscience*, 12, 97. https://doi.org/10.3389/fnins.2018.00097

Miyake, A., Friedman, N. P., Emerson, M. J., et al. (2000). The unity and diversity of executive functions. *Cognitive Psychology*, 41(1), 49–100. https://doi.org/10.1006/cogp.1999.0734

Subandriyo, A. P. E. P., Jongsma, M. L. A., Wijaya, D. A., Trisnadewi, B. A. P., Paravoti, A., Novihartanti, B. L., Widyorini, E., Sulastri, A., & Breteler, M. H. M. (2021). Offering Neurofeedback as an Intervention for Children with Attention Deficit/Hyperactivity Disorder in Indonesia: A Feasibility Study. *Kobe Journal of Medical Sciences*, 67(4), E125–E136. PMID: 35367999.

Pinggal, E., Jackson, J., Kusztor, A., et al. (2026). Sleep-like slow waves during wakefulness mediate attention and vigilance difficulties in adult ADHD. *Journal of Neuroscience*, 46(15), e1694252025. https://doi.org/10.1523/JNEUROSCI.1694-25.2025

Rossi, C., Vidaurre, D., Costers, L., et al. (2023). A data-driven network decomposition of the temporal, spatial, and spectral dynamics underpinning visual-verbal working memory processes. *Communications Biology*, 6, 1079. https://doi.org/10.1038/s42003-023-05448-z

Rueda, M. R., Fan, J., McCandliss, B. D., et al. (2004). Development of attentional networks in childhood. *Neuropsychologia*, 42(8), 1029–1040.

---

**Verification note (per citation-check skill):** all numerical values in this report come directly from `results/` files dated 7 April 2026. No rounding: balanced accuracy 0.652 is written as 0.652; r = -0.489 as -0.489. A discrepancy with an earlier internal memory (which named fronto-parietal beta coherence as the top biomarker) has been reconciled — the actual top feature is relative beta power at the posterior-parietal electrodes, not coherence.

> Contact: Yazid (yazidburahmen@gmail.com) or the Talenta WhatsApp group for follow-up discussion.
