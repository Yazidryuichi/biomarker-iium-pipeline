# AI Transparency Declaration

## Generative AI Usage in This Project

This document declares all uses of generative AI tools in the development of this research pipeline, in accordance with emerging standards for AI-assisted scientific research.

### Code Development

- **Tool used:** Claude Code (Anthropic Claude Opus 4.6)
- **Scope:** Pipeline architecture design, code generation, code review, and bug fixing
- **Human oversight:** All AI-generated code was reviewed, tested, and validated by the research team before inclusion
- **Verification:** Each pipeline stage was tested on pilot data (N=28) with results manually inspected

### Analysis Decisions

- **Feature selection methods:** Chosen based on literature review and domain expertise, not AI recommendation alone
- **Model selection:** 8 algorithms specified in the research proposal prior to AI involvement
- **Hyperparameter ranges:** Based on published recommendations for small-sample EEG classification
- **Statistical thresholds:** Pre-specified (alpha=0.05, FDR correction) per research proposal

### What AI Did NOT Do

- AI did not collect, modify, or have access to raw EEG or behavioral data
- AI did not make decisions about participant inclusion/exclusion
- AI did not interpret clinical significance of results
- AI did not write the research proposal or ethics application
- AI did not select which hypotheses to test

### Bias Mitigation

- Cross-validation with permutation testing ensures reported accuracy is above chance
- SHAP values computed per CV fold to prevent overfitting of feature importance
- All preprocessing decisions follow published protocols (HAPPE, MNE best practices)
- Code review identified and fixed data leakage risks before final analysis

### Reproducibility

- All code is version-controlled (git) with full commit history
- Pipeline is containerized (Docker) for environment reproducibility
- Random seeds are fixed and documented (seed=42 throughout)
- Configuration is externalized in `configs/config.yaml`

### Audit Trail

Each pipeline run generates:
- `results/qc_stage1.json` — preprocessing quality metrics per subject
- `results/ml_results.csv` — all model performance metrics
- `results/correlations.csv` — hypothesis test results with effect sizes
- `results/shap_importance.csv` — feature importance with stability metrics
- `evaluate.py` output — pipeline quality scores across 6 dimensions

### Contact

For questions about AI usage in this project, contact the research team.

---

*This declaration follows recommendations from the Committee on Publication Ethics (COPE) and major journal policies on AI-assisted research (Nature, Science, PNAS).*
