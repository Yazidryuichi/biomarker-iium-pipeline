FROM python:3.11-slim

WORKDIR /app

# System dependencies for MNE and scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run full pipeline
ENTRYPOINT ["python", "run_all.py"]
