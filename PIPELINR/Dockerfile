# ── Base image ────────────────────────────────────────────────────────────────
# Use slim Python 3.10 – TensorFlow 2.x is fully supported
FROM python:3.10-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy pipeline source ──────────────────────────────────────────────────────
COPY stage1_ingest.py     .
COPY stage2_preprocess.py .
COPY stage3_predict.py    .
COPY pipeline.py          .

# ── Optional: copy local CSV as fallback data source ─────────────────────────
# Comment this out if you always want to pull live data
COPY MSFT.csv             .

# ── Create output directories ─────────────────────────────────────────────────
RUN mkdir -p data models

# ── Default command ───────────────────────────────────────────────────────────
# Override at runtime:  docker run msft-pipeline --source api --retrain
CMD ["python", "pipeline.py", "--source", "auto"]
