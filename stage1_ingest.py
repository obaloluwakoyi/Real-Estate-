"""
Stage 1: Ingestion (The Loader)
--------------------------------
Fetches MSFT stock data either from a local CSV or a live API (yfinance).
Outputs a clean raw DataFrame saved to data/raw_msft.csv
"""

import os
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INGEST] %(message)s")

RAW_OUTPUT_PATH = "data/raw_msft.csv"
LOCAL_CSV_PATH  = "MSFT.csv"
TICKER          = "MSFT"
START_DATE      = "2015-01-01"


def get_data(source: str = "auto") -> pd.DataFrame:
    """
    Fetch MSFT data.

    Args:
        source: "csv"  – load from LOCAL_CSV_PATH
                "api"  – pull from yfinance (live)
                "auto" – try API first, fall back to CSV
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    os.makedirs("data", exist_ok=True)

    if source in ("api", "auto"):
        try:
            df = _fetch_from_api()
            df.to_csv(RAW_OUTPUT_PATH, index=False)
            logging.info(f"Live data saved → {RAW_OUTPUT_PATH}  ({len(df)} rows)")
            return df
        except Exception as e:
            if source == "api":
                raise
            logging.warning(f"API fetch failed ({e}). Falling back to CSV.")

    df = _load_from_csv()
    df.to_csv(RAW_OUTPUT_PATH, index=False)
    logging.info(f"CSV data saved → {RAW_OUTPUT_PATH}  ({len(df)} rows)")
    return df


def _fetch_from_api() -> pd.DataFrame:
    import yfinance as yf
    logging.info(f"Fetching {TICKER} from yfinance ({START_DATE} → today)…")
    ticker = yf.Ticker(TICKER)

    # FIX: auto_adjust=False is deprecated in newer yfinance; use auto_adjust=True
    # and rely on the adjusted Close column directly.
    try:
        df = ticker.history(start=START_DATE, auto_adjust=True)
    except TypeError:
        # Fallback for very old yfinance versions that don't support auto_adjust
        df = ticker.history(start=START_DATE)

    df.reset_index(inplace=True)

    # Normalise the Date column (may be tz-aware in newer yfinance)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

    # Select only the columns we need; "Adj Close" becomes "Close" with auto_adjust=True
    available = df.columns.tolist()
    adj_col = "Adj Close" if "Adj Close" in available else "Close"
    df["Adj Close"] = df[adj_col]

    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    if df.empty:
        raise ValueError("yfinance returned an empty DataFrame.")
    return df


def _load_from_csv() -> pd.DataFrame:
    if not os.path.exists(LOCAL_CSV_PATH):
        raise FileNotFoundError(f"Local CSV not found: {LOCAL_CSV_PATH}")
    logging.info(f"Loading data from {LOCAL_CSV_PATH}…")
    df = pd.read_csv(LOCAL_CSV_PATH, parse_dates=["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


if __name__ == "__main__":
    df = get_data(source="auto")
    print(df.tail())
    print(f"\nRows: {len(df)}  |  Date range: {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")
