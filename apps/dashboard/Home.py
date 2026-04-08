from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from logistics_optimization.dashboard.api_client import LogisticsApiClient


st.set_page_config(page_title="Logistics Optimization Dashboard", layout="wide")

client = LogisticsApiClient()

st.title("AI-Based Logistics Optimization System")
st.caption("Transformer demand forecasting and graph-based route optimization for transportation networks.")

left, right = st.columns([2, 1])

with left:
    st.markdown(
        """
        This dashboard exposes the two implemented academic project modules:

        - Transformer-based demand forecasting by zone and time slot
        - Graph-based route optimization with configurable distance and time costs
        """
    )

with right:
    st.subheader("System Health")
    try:
        health = client.health()
        st.success(f"{health['status'].upper()} | {health['environment']}")
    except Exception as exc:  # pragma: no cover - UI fallback
        st.error(f"API unavailable: {exc}")

st.info("Use the sidebar pages to train the forecaster, inspect zone demand heatmaps, and optimize transport routes.")
