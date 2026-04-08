from pathlib import Path
import sys
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from logistics_optimization.dashboard.api_client import LogisticsApiClient
from logistics_optimization.dashboard.charts import build_route_figure


st.title("Route Optimization")
client = LogisticsApiClient()

try:
    network = client.demo_network()
except Exception as exc:  # pragma: no cover - UI fallback
    st.error(f"Unable to load network data: {exc}")
    st.stop()

node_ids = [node["node_id"] for node in network["nodes"]]

left, right = st.columns([2, 1])

with left:
    st.plotly_chart(build_route_figure(network), use_container_width=True)

with right:
    start_node = st.selectbox("Start Node", options=node_ids, index=0)
    end_node = st.selectbox("End Node", options=node_ids, index=min(2, len(node_ids) - 1))
    algorithm = st.selectbox("Algorithm", options=["dijkstra", "astar"])
    distance_weight = st.slider("Distance Weight", min_value=0.0, max_value=1.0, value=0.5)
    time_weight = 1.0 - distance_weight
    st.caption(f"Travel-Time Weight: {time_weight:.2f}")

    if st.button("Optimize Route", use_container_width=True):
        try:
            payload = {
                "request_id": str(uuid4()),
                "nodes": network["nodes"],
                "edges": network["edges"],
                "start_node": start_node,
                "end_node": end_node,
                "algorithm": algorithm,
                "weights": {
                    "distance_weight": distance_weight,
                    "time_weight": time_weight,
                },
            }
            result = client.optimize_route(payload)
            st.success(
                f"Path: {' -> '.join(result['path'])} | "
                f"Distance: {result['total_distance_km']} km | "
                f"Travel Time: {result['total_travel_time_min']} min"
            )
            st.plotly_chart(build_route_figure(network, route_path=result["path"]), use_container_width=True)
            st.json(result)
        except Exception as exc:  # pragma: no cover - UI fallback
            st.error(f"Route optimization failed: {exc}")
