.PHONY: all clean stage1 stage2 stage3 stage4 stage5 validate figures docker test

# Default: run full pipeline
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

stage5:
	python run_all.py --stage 5

# Validate data setup before running
validate:
	python validate_data.py

# Generate publication figures
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
	rm -rf figures/*.png

# Install dependencies
install:
	pip install -r requirements.txt

# Single subject test
test:
	python run_all.py --subject D0000795
