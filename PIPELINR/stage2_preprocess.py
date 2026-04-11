"""
Stage 2: Transformation (The Preprocessor)
-------------------------------------------
Reads raw MSFT data, scales it with StandardScaler, creates LSTM sequences,
and saves the scaler so it can be reused for new data without retraining.

Outputs:
  data/X_train.npy, data/y_train.npy
  data/X_test.npy,  data/y_test_raw.npy  (inverse-transform targets)
  models/scaler.pkl
"""

import os
import math
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PREPROCESS] %(message)s")

RAW_PATH     = "data/raw_msft.csv"
SCALER_PATH  = "models/scaler.pkl"
WINDOW_SIZE  = 60          # look-back window fed to LSTM
TRAIN_RATIO  = 0.80


def preprocess_data(
    raw_path: str = RAW_PATH,
    window: int = WINDOW_SIZE,
    train_ratio: float = TRAIN_RATIO,
) -> dict:
    """
    Load raw CSV → scale → create sequences → save scaler.

    Returns dict with keys:
        X_train, y_train, X_test, y_test_raw,
        training_data_len, scaler
    """
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    logging.info(f"Loading raw data from {raw_path}…")
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    dataset = df[["Close"]].values                     # (N, 1)
    total_rows = len(dataset)
    logging.info(f"Dataset shape: {dataset.shape}")

    # ── 2. Train / test split ─────────────────────────────────────────────────
    training_data_len = math.ceil(total_rows * train_ratio)
    logging.info(f"Train rows: {training_data_len}  |  Test rows: {total_rows - training_data_len}")

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)        # fit only on full dataset
    logging.info(f"Scaler fitted  μ={scaler.mean_[0]:.4f}  σ={scaler.scale_[0]:.4f}")

    # Save scaler so inference never needs to re-fit
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved → {SCALER_PATH}")

    # ── 4. Build training sequences ───────────────────────────────────────────
    train_scaled = scaled_data[:training_data_len]
    X_train, y_train = _make_sequences(train_scaled, window)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    logging.info(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

    # ── 5. Build test sequences ───────────────────────────────────────────────
    test_scaled = scaled_data[training_data_len - window:]
    X_test_seq, _ = _make_sequences(test_scaled, window)
    X_test  = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], 1)
    y_test_raw = dataset[training_data_len:]           # original prices for evaluation
    logging.info(f"X_test:  {X_test.shape}   y_test_raw: {y_test_raw.shape}")

    # ── 6. Persist arrays ────────────────────────────────────────────────────
    np.save("data/X_train.npy",      X_train)
    np.save("data/y_train.npy",      y_train)
    np.save("data/X_test.npy",       X_test)
    np.save("data/y_test_raw.npy",   y_test_raw)
    logging.info("Arrays saved to data/")

    return dict(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test_raw=y_test_raw,
        training_data_len=training_data_len,
        scaler=scaler,
    )


def _make_sequences(scaled_arr: np.ndarray, window: int):
    """Slide a window over scaled_arr to create (X, y) pairs."""
    X, y = [], []
    for i in range(window, len(scaled_arr)):
        X.append(scaled_arr[i - window : i, 0])
        y.append(scaled_arr[i, 0])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    result = preprocess_data()
    print("\nPreprocessing complete.")
    for k, v in result.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")
