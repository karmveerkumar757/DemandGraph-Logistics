from logistics_optimization.ml.optimization.algorithms import GraphRoutingEngine
from logistics_optimization.schemas.routing import GraphEdge, GraphNode, OptimizationWeights, RouteOptimizationRequest


def test_dijkstra_selects_reasonable_path() -> None:
    engine = GraphRoutingEngine()
    request = RouteOptimizationRequest(
        request_id="demo-test",
        nodes=[
            GraphNode(node_id="A", label="A", x=0.0, y=0.0),
            GraphNode(node_id="B", label="B", x=1.0, y=0.0),
            GraphNode(node_id="C", label="C", x=2.0, y=0.0),
        ],
        edges=[
            GraphEdge(source="A", target="B", distance_km=2.0, travel_time_min=6.0),
            GraphEdge(source="B", target="C", distance_km=2.0, travel_time_min=6.0),
            GraphEdge(source="A", target="C", distance_km=10.0, travel_time_min=20.0),
        ],
        start_node="A",
        end_node="C",
        algorithm="dijkstra",
        weights=OptimizationWeights(distance_weight=0.5, time_weight=0.5),
    )
    result = engine.optimize(request)
    assert result.path == ["A", "B", "C"]
    assert result.total_distance_km == 4.0

