from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from logistics_optimization.dashboard.api_client import LogisticsApiClient
from logistics_optimization.dashboard.charts import build_demand_heatmap


st.title("Demand Forecasting")
client = LogisticsApiClient()

try:
    heatmap_points = client.heatmap()
    metrics = client.forecasting_metrics()
except Exception as exc:  # pragma: no cover - UI fallback
    st.error(f"Unable to load forecasting data: {exc}")
    st.stop()

frame = pd.DataFrame(heatmap_points)
zones = sorted(frame["zone_id"].unique().tolist()) if not frame.empty else []

control_col, metric_col = st.columns([2, 1])
with control_col:
    st.plotly_chart(build_demand_heatmap(heatmap_points), use_container_width=True)

with metric_col:
    st.subheader("Model Metrics")
    st.metric("MAE", metrics["mae"])
    st.metric("RMSE", metrics["rmse"])
    st.metric("Training Loss", metrics["latest_training_loss"])

st.subheader("Training Controls")
train_left, train_right, train_button_col = st.columns([1, 1, 1])
with train_left:
    sequence_length = st.slider("Sequence Length", min_value=3, max_value=8, value=4)
with train_right:
    epochs = st.slider("Epochs", min_value=2, max_value=20, value=8)
with train_button_col:
    if st.button("Train Transformer", use_container_width=True):
        try:
            result = client.train_forecaster(sequence_length=sequence_length, epochs=epochs)
            st.success(
                f"Trained {result['model_name']} | MAE={result['validation_mae']} | RMSE={result['rmse']}"
            )
        except Exception as exc:  # pragma: no cover - UI fallback
            st.error(f"Training failed: {exc}")

st.subheader("Predict Next Time Slot Demand")
if not zones:
    st.warning("No zone data is currently available in the database.")
else:
    selected_zone = st.selectbox("Zone", options=zones)

    if st.button("Forecast Next Demand", use_container_width=True):
        try:
            recent_observations = client.recent_zone_observations(zone_id=selected_zone, limit=sequence_length + 2)
            prediction = client.predict_demand(recent_observations)
            st.success(
                f"Zone {selected_zone}: predicted demand = {prediction['predicted_demand']} "
                f"({prediction['model_name']}, {prediction['inference_latency_ms']} ms)"
            )
            st.dataframe(pd.DataFrame(recent_observations), use_container_width=True)
        except Exception as exc:  # pragma: no cover - UI fallback
            st.error(f"Prediction failed: {exc}")
