"""
pipeline.py – Orchestrator
---------------------------
Chains Stage 1 → Stage 2 → Stage 3 in sequence.

Usage:
  python pipeline.py                # run full pipeline (auto data source)
  python pipeline.py --source csv   # force local CSV
  python pipeline.py --source api   # force live yfinance
  python pipeline.py --retrain      # force model retraining
"""

import argparse
import logging
import time

from stage1_ingest     import get_data
from stage2_preprocess import preprocess_data
from stage3_predict    import train_model, predict_price
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)

def main():
    parser = argparse.ArgumentParser(description="MSFT LSTM Stock Prediction Pipeline")
    parser.add_argument("--source",  default="auto", choices=["auto", "csv", "api"],
                        help="Data source for Stage 1")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retraining in Stage 3")
    args = parser.parse_args()

    start = time.time()
    logging.info("══════════════════════════════════════")
    logging.info("  MSFT LSTM Pipeline — Starting")
    logging.info("══════════════════════════════════════")

    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    logging.info("▶ Stage 1: Ingestion")
    df = get_data(source=args.source)
    logging.info(f"  ✓ {len(df)} rows ingested")

    # ── Stage 2: Preprocess ───────────────────────────────────────────────────
    logging.info("▶ Stage 2: Preprocessing")
    data = preprocess_data()
    logging.info(f"  ✓ X_train: {data['X_train'].shape}  X_test: {data['X_test'].shape}")

    # ── Stage 3: Train + Predict ──────────────────────────────────────────────
    logging.info("▶ Stage 3: Training & Prediction")
    model   = train_model(data["X_train"], data["y_train"], force_retrain=args.retrain)
    results = predict_price(data["X_test"], data["y_test_raw"], model)

    elapsed = time.time() - start
    mae  = (results["Actual"] - results["Predicted"]).abs().mean()
    rmse = ((results["Actual"] - results["Predicted"]) ** 2).mean() ** 0.5

    logging.info("══════════════════════════════════════")
    logging.info(f"  Pipeline complete in {elapsed:.1f}s")
    logging.info(f"  MAE: ${mae:.2f}   RMSE: ${rmse:.2f}")
    logging.info(f"  Predictions → data/predictions.csv")
    logging.info(f"  Model       → models/stock_model.h5")
    logging.info(f"  Scaler      → models/scaler.pkl")
    logging.info("══════════════════════════════════════")

if __name__ == "__main__":
    main()
