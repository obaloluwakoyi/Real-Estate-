# MSFT LSTM Stock Prediction Pipeline

A production-grade CI/CD pipeline for MSFT stock price prediction using an LSTM neural network.

## Architecture

```
Stage 1: Ingest          Stage 2: Preprocess        Stage 3: Predict
─────────────────        ─────────────────────       ──────────────────────
MSFT.csv  ─┐             raw_msft.csv               X_train / X_test
yfinance  ─┴─► get_data() ──► preprocess_data() ──► train_model()
                                  ├─ StandardScaler        ↓
                              models/scaler.pkl     predict_price()
                                                         ↓
                                                  data/predictions.csv
                                                  models/stock_model.h5
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (auto source: tries yfinance, falls back to CSV)
python pipeline.py

# 3. Force live data from yfinance
python pipeline.py --source api

# 4. Force retraining (ignores cached model)
python pipeline.py --retrain
```

## Docker

```bash
# Build the image
docker build -t msft-pipeline .

# Run with live data
docker run msft-pipeline --source api

# Mount a volume to persist models and predictions
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data msft-pipeline
```

## Outputs

| File | Description |
|------|-------------|
| `data/raw_msft.csv` | Raw ingested price data |
| `data/X_train.npy` | Scaled training sequences |
| `data/X_test.npy` | Scaled test sequences |
| `data/predictions.csv` | Actual vs Predicted prices |
| `models/stock_model.h5` | Trained Keras model |
| `models/scaler.pkl` | Fitted StandardScaler (reuse for inference) |

## Loading Saved Model for Inference

```python
from keras.models import load_model
import pickle, numpy as np

model  = load_model("models/stock_model.h5")
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# New data: shape (1, 60, 1) — last 60 days of Close prices
new_sequence = np.load("your_sequence.npy")
scaled       = scaler.transform(new_sequence.reshape(-1, 1)).reshape(1, 60, 1)
pred_scaled  = model.predict(scaled)
price        = scaler.inverse_transform(pred_scaled)
print(f"Predicted next close: ${price[0][0]:.2f}")
```

## GitHub Actions (Automated Scheduler)

The pipeline runs automatically every weekday at **8:00 AM UTC** via `.github/workflows/daily_prediction.yml`.

- Pulls live MSFT data from yfinance
- Skips retraining if the model code hasn't changed (uses GitHub Actions cache)
- Uploads `predictions.csv` as a downloadable artifact (retained 30 days)
- Can be triggered manually from the GitHub Actions UI
