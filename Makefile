.PHONY: all clean stage1 stage2 stage3 stage4 stage5 stage6 \
        quantum-exploration validate figures docker test install lock

# Default: run full pipeline (stages 1, 2, 3, 4, 6, 5 in that order;
# stage 5 = fair comparison runs after stage 6 = DM features because it
# consumes Stage 6 outputs).
all: validate
	python run_all.py

# Individual stages
stage1:
	python run_all.py --stage 1

stage2:
	python run_all.py --stage 2

stage3:
	python run_all.py --stage 3

stage4:
	python run_all.py --stage 4

# Stage 5: post-pipeline fair 2x2 comparison (feature set x model class).
# Subject-level LOSO + paired DeLong + subject-bootstrap CIs. Reads from
# results/ (Stage 4 + Stage 6 outputs). Writes results/stage5_fair_comparison.json.
stage5:
	python -m stages.stage5_fair_comparison \
		--results-dir results \
		--out-json results/stage5_fair_comparison.json

# Stage 6: explicit channel-covariance density-matrix features.
# Produces 900 real features per subject at N=15 channels x 4 bands.
stage6:
	python run_all.py --stage 6

# Legacy: quantum-cognition exploration (QEPP, von Neumann entropy on
# PCA-compressed features). Moves to quantum-exploration/ branch in Phase 3.
quantum-exploration:
	python run_all.py --exploratory-quantum

# Validate data setup before running
validate:
	python validate_data.py

# Generate publication figures (model_comparison.png 2x2, quantum_vs_classical.png
# matched-model, shap_top15.png). Reads results/stage5_fair_comparison.json.
figures:
	python generate_figures.py

# Evaluate pipeline quality (immutable scorer)
evaluate:
	python evaluate.py

# Docker build and run
docker:
	docker build -t biomarker-iium .
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/results:/app/results biomarker-iium

# Convert raw data to BIDS format
bids:
	python convert_to_bids.py

# Clean derived outputs (preserves raw data)
clean:
	rm -rf results/cleaned_epochs results/features.csv results/full_dataset.csv
	rm -rf results/ml_results.csv results/correlations.csv results/shap_importance.csv
	rm -rf results/quantum_features.csv results/quantum_vs_classical.csv
	rm -rf results/stage5_fair_comparison.json results/stage5_per_fold.csv \
		results/stage5_per_subject.csv
	rm -rf figures/*.png logs/

# Install dependencies (use lock for reproducibility, requirements.txt for ranges)
install:
	pip install -r requirements.lock

# Regenerate requirements.lock from requirements.txt (uv pip compile)
lock:
	uv pip compile requirements.txt -o requirements.lock --python-version 3.12

# Single subject test
test:
	python run_all.py --subject D0000795

# Run unit tests
pytest:
	python -m pytest tests/ -v

# Generate the synthetic fixture (28 subjects, gaussian-noise EDF + behavioural xlsx)
synthetic-fixture:
	python tests/generate_synthetic_fixture.py \
		--out tests/fixtures/synthetic \
		--n-subjects 28 \
		--seed 42

# Run the full pipeline against the synthetic fixture
test-pipeline-end-to-end: synthetic-fixture
	python run_all.py --config tests/fixtures/synthetic/config.fixture.yaml
