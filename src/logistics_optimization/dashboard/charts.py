from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go


def build_demand_heatmap(points: Sequence[dict]) -> go.Figure:
    frame = pd.DataFrame(points)
    if frame.empty:
        return go.Figure()

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    pivot = frame.pivot_table(index="zone_id", columns="hour_of_day", values="demand", aggfunc="mean", fill_value=0)

    figure = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(column) for column in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar={"title": "Demand"},
        )
    )
    figure.update_layout(
        title="Zone Demand Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Zone",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def build_route_figure(network: dict, route_path: Sequence[str] | None = None) -> go.Figure:
    nodes = {item["node_id"]: item for item in network["nodes"]}
    figure = go.Figure()

    highlighted_pairs = {
        tuple(pair) for pair in zip(route_path or [], (route_path or [])[1:], strict=False)
    }
    highlighted_pairs.update({(target, source) for source, target in highlighted_pairs})

    for edge in network["edges"]:
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        is_active = (edge["source"], edge["target"]) in highlighted_pairs
        figure.add_trace(
            go.Scatter(
                x=[source["x"], target["x"]],
                y=[source["y"], target["y"]],
                mode="lines",
                line={"color": "#d1495b" if is_active else "#90a4ae", "width": 5 if is_active else 2},
                hoverinfo="text",
                text=[f"{edge['source']} → {edge['target']}"],
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[node["x"] for node in nodes.values()],
            y=[node["y"] for node in nodes.values()],
            mode="markers+text",
            text=[node["node_id"] for node in nodes.values()],
            textposition="top center",
            marker={"size": 18, "color": "#00798c"},
            hovertext=[node["label"] for node in nodes.values()],
            showlegend=False,
        )
    )
    figure.update_layout(
        title="Graph-Based Transportation Network",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure

