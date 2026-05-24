"""
app.py – Streamlit Dashboard for MSFT LSTM Stock Prediction Pipeline
----------------------------------------------------------------------
Run with:  streamlit run app.py
"""

import os
import sys
import time
import pickle
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSFT Price Predictor",
    page_icon="📈",
    layout="wide",
)

# Ensure pipeline modules resolve from the same directory as app.py
sys.path.insert(0, os.path.dirname(__file__))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _predictions_exist() -> bool:
    return os.path.exists("data/predictions.csv")

def _model_exists() -> bool:
    return os.path.exists("models/stock_model.h5")

def _raw_data_exists() -> bool:
    return os.path.exists("data/raw_msft.csv")

def _load_predictions() -> pd.DataFrame:
    return pd.read_csv("data/predictions.csv")

def _load_raw() -> pd.DataFrame:
    return pd.read_csv("data/raw_msft.csv", parse_dates=["Date"])

def _metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    diff = df["Actual"] - df["Predicted"]
    mae  = diff.abs().mean()
    rmse = (diff ** 2).mean() ** 0.5
    mape = (diff.abs() / df["Actual"]).mean() * 100
    return mae, rmse, mape

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Pipeline Controls")

    source = st.radio(
        "Data source",
        options=["auto", "api", "csv"],
        index=0,
        help="auto = try yfinance then fall back to local CSV",
    )

    force_retrain = st.checkbox(
        "Force model retraining",
        value=False,
        help="Ignore any cached model and train from scratch",
    )

    run_pipeline = st.button("▶ Run Pipeline", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**Status**")
    model_status  = "✅ Exists" if _model_exists()  else "❌ Not found"
    raw_status    = "✅ Exists" if _raw_data_exists() else "❌ Not found"
    pred_status   = "✅ Exists" if _predictions_exist() else "❌ Not found"
    st.markdown(f"Model: {model_status}")
    st.markdown(f"Raw data: {raw_status}")
    st.markdown(f"Predictions: {pred_status}")

    st.divider()
    st.caption("MSFT LSTM Pipeline · Streamlit UI")

# ── Title ─────────────────────────────────────────────────────────────────────

st.title("📈 MSFT Stock Price Predictor")
st.markdown(
    "LSTM-based next-day closing price prediction pipeline. "
    "Configure the options in the sidebar and click **Run Pipeline** to start."
)

# ── Pipeline execution ────────────────────────────────────────────────────────

if run_pipeline:
    from stage1_ingest     import get_data
    from stage2_preprocess import preprocess_data
    from stage3_predict    import train_model, predict_price

    progress = st.progress(0, text="Initialising…")
    log_box  = st.empty()
    logs: list[str] = []

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        logs.append(f"`{ts}` {msg}")
        log_box.markdown("\n\n".join(logs[-12:]))  # show last 12 lines

    try:
        # Stage 1
        progress.progress(10, text="Stage 1 · Fetching data…")
        log(f"▶ Stage 1: fetching data (source=`{source}`)…")
        df_raw = get_data(source=source)
        log(f"  ✓ {len(df_raw):,} rows ingested  |  {df_raw['Date'].iloc[0]} → {df_raw['Date'].iloc[-1]}")

        # Stage 2
        progress.progress(35, text="Stage 2 · Preprocessing…")
        log("▶ Stage 2: scaling & building LSTM sequences…")
        data = preprocess_data()
        log(f"  ✓ X_train {data['X_train'].shape}   X_test {data['X_test'].shape}")

        # Stage 3 – train
        progress.progress(55, text="Stage 3 · Training model…")
        log(f"▶ Stage 3: {'retraining' if force_retrain else 'train / load'} model…")
        model = train_model(data["X_train"], data["y_train"], force_retrain=force_retrain)
        log("  ✓ Model ready")

        # Stage 3 – predict
        progress.progress(80, text="Stage 3 · Generating predictions…")
        log("▶ Generating predictions…")
        results = predict_price(data["X_test"], data["y_test_raw"], model)

        mae, rmse, mape = _metrics(results)
        log(f"  ✓ MAE ${mae:.2f}  |  RMSE ${rmse:.2f}  |  MAPE {mape:.2f}%")
        progress.progress(100, text="Complete ✅")
        st.success("Pipeline finished successfully!")

    except FileNotFoundError as e:
        st.error(f"**File not found:** {e}\n\nIf using CSV source, make sure `MSFT.csv` is in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"**Pipeline error:** {e}")
        st.stop()

# ── Results display ───────────────────────────────────────────────────────────

if not _predictions_exist():
    st.info("No predictions found yet. Run the pipeline to generate results.")
    st.stop()

results = _load_predictions()
mae, rmse, mape = _metrics(results)

# ── Metrics row ───────────────────────────────────────────────────────────────

st.subheader("📊 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test samples", f"{len(results):,}")
c2.metric("MAE", f"${mae:.2f}")
c3.metric("RMSE", f"${rmse:.2f}")
c4.metric("MAPE", f"{mape:.2f}%")

# ── Prediction chart ──────────────────────────────────────────────────────────

st.subheader("📉 Actual vs Predicted Closing Price")

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=results["Actual"],
    name="Actual",
    line=dict(color="#1f77b4", width=2),
))
fig.add_trace(go.Scatter(
    y=results["Predicted"],
    name="Predicted",
    line=dict(color="#ff7f0e", width=2, dash="dot"),
))
fig.update_layout(
    xaxis_title="Test sample index",
    yaxis_title="Price (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ── Residuals chart ───────────────────────────────────────────────────────────

st.subheader("🔍 Prediction Error (Residuals)")

residuals = results["Actual"] - results["Predicted"]
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Residuals over time", "Residual distribution"))

fig2.add_trace(go.Scatter(
    y=residuals, mode="lines",
    line=dict(color="#2ca02c", width=1.5),
    name="Residual",
), row=1, col=1)
fig2.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

fig2.add_trace(go.Histogram(
    x=residuals, nbinsx=40,
    marker_color="#9467bd",
    name="Distribution",
    showlegend=False,
), row=1, col=2)

fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

# ── Raw price history ─────────────────────────────────────────────────────────

if _raw_data_exists():
    st.subheader("📋 Full Price History (MSFT)")
    df_raw = _load_raw()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_raw["Date"], y=df_raw["Close"],
        fill="tozeroy",
        line=dict(color="#1f77b4", width=1.5),
        name="Close",
    ))
    # Shade the test region
    train_len = int(len(df_raw) * 0.80)
    fig3.add_vrect(
        x0=df_raw["Date"].iloc[train_len],
        x1=df_raw["Date"].iloc[-1],
        fillcolor="orange", opacity=0.08,
        annotation_text="Test period", annotation_position="top left",
    )
    fig3.update_layout(
        xaxis_title="Date",
        yaxis_title="Close (USD)",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────────────────────

with st.expander("🗂 View predictions table"):
    st.dataframe(
        results.assign(Error=results["Actual"] - results["Predicted"]).style.format({
            "Actual": "${:.2f}",
            "Predicted": "${:.2f}",
            "Error": "${:.2f}",
        }),
        use_container_width=True,
        height=350,
    )
    csv_bytes = results.to_csv(index=False).encode()
    st.download_button("⬇ Download predictions.csv", csv_bytes, "predictions.csv", "text/csv")
