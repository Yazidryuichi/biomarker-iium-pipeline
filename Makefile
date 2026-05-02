.PHONY: all clean cleaning features engineering analysis validate figures docker test install bids evaluate

# Default: run full pipeline
all: validate
	python pipeline.py

# Individual stages — each writes to results/<stage>/<timestamp>/
cleaning:
	python pipeline.py --cleaning

features:
	python pipeline.py --features

engineering:
	python pipeline.py --engineering

analysis:
	python pipeline.py --analysis

# Validate data setup before running
validate:
	python validate_data.py

# Generate publication figures from latest analysis output
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

# Clean all stage outputs (preserves raw data)
clean:
	rm -rf results/cleaning results/features results/engineering results/analysis
	rm -rf docs/figures/*.png

# Install dependencies
install:
	pip install -r requirements.txt

# Single subject test
test:
	python pipeline.py --subject D0000795
