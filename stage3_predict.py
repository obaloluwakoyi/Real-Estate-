"""
Stage 3: Inference / Training (The Predictor)
----------------------------------------------
Loads preprocessed arrays, trains the LSTM (or loads a saved model),
runs predictions, and saves:
  models/stock_model.h5   – trained Keras model
  data/predictions.csv    – predicted vs actual closing prices
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model   # FIX: use tensorflow.keras
from tensorflow.keras.layers import Dense, LSTM              # FIX: use tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PREDICT] %(message)s")

MODEL_PATH      = "models/stock_model.h5"
SCALER_PATH     = "models/scaler.pkl"
PREDICTIONS_CSV = "data/predictions.csv"
EPOCHS          = 5
BATCH_SIZE      = 32


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    force_retrain: bool = False,
) -> tf.keras.Model:
    """
    Build and train the LSTM model (or load from disk if already trained).

    Args:
        X_train:       shape (samples, 60, 1)
        y_train:       shape (samples,)
        force_retrain: ignore saved model and retrain from scratch
    Returns:
        Trained Keras model
    """
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH) and not force_retrain:
        logging.info(f"Loading existing model from {MODEL_PATH}")
        return load_model(MODEL_PATH)

    logging.info("Building LSTM model…")
    model = Sequential([
        LSTM(200, return_sequences=True,  input_shape=(X_train.shape[1], 1)),
        LSTM(200, return_sequences=False),
        Dense(100, activation="relu"),
        Dense(50,  activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    callbacks = [
        EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="loss"),
    ]

    logging.info(f"Training for up to {EPOCHS} epochs…")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    logging.info(f"Training complete. Final loss: {history.history['loss'][-1]:.6f}")
    model.save(MODEL_PATH)
    logging.info(f"Model saved → {MODEL_PATH}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_price(
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    model: tf.keras.Model = None,
) -> pd.DataFrame:
    """
    Run predictions on X_test and inverse-transform back to dollar prices.

    Args:
        X_test:     shape (samples, 60, 1)
        y_test_raw: original (unscaled) closing prices for comparison
        model:      optional pre-loaded model; loads from disk if None
    Returns:
        DataFrame with columns: Actual, Predicted
    """
    if model is None:
        logging.info(f"Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    logging.info(f"Running predictions on {len(X_test)} test samples…")
    preds_scaled = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    predictions  = scaler.inverse_transform(preds_scaled)

    results = pd.DataFrame({
        "Actual":    y_test_raw.flatten(),
        "Predicted": predictions.flatten(),
    })

    mae  = (results["Actual"] - results["Predicted"]).abs().mean()
    rmse = ((results["Actual"] - results["Predicted"]) ** 2).mean() ** 0.5
    logging.info(f"MAE:  ${mae:.2f}   RMSE: ${rmse:.2f}")

    results.to_csv(PREDICTIONS_CSV, index=False)
    logging.info(f"Predictions saved → {PREDICTIONS_CSV}")
    return results


# ── Full pipeline entry point ─────────────────────────────────────────────────

def run_pipeline(force_retrain: bool = False):
    """Load preprocessed data → train/load model → predict → save results."""
    logging.info("=== Stage 3: Predictor ===")

    X_train    = np.load("data/X_train.npy")
    y_train    = np.load("data/y_train.npy")
    X_test     = np.load("data/X_test.npy")
    y_test_raw = np.load("data/y_test_raw.npy")

    model   = train_model(X_train, y_train, force_retrain=force_retrain)
    results = predict_price(X_test, y_test_raw, model)

    print("\nSample predictions (last 5 rows):")
    print(results.tail())
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    args = parser.parse_args()
    run_pipeline(force_retrain=args.retrain)
