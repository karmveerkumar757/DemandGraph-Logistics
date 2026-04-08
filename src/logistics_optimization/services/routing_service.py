from sqlalchemy.orm import Session

from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import get_logger
from logistics_optimization.ml.optimization.algorithms import GraphRoutingEngine
from logistics_optimization.repositories.route_repository import RouteRepository
from logistics_optimization.schemas.routing import (
    DemoNetworkResponse,
    GraphEdge,
    GraphNode,
    RouteOptimizationRequest,
    RouteOptimizationResponse,
)


class RoutingService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.logger = get_logger("logistics.routing", level=self.settings.log_level, log_dir=self.settings.log_dir)
        self.repository = RouteRepository(session)
        self.engine = GraphRoutingEngine()

    def optimize(self, payload: RouteOptimizationRequest) -> RouteOptimizationResponse:
        response = self.engine.optimize(payload)
        self.repository.save(response)
        self.logger.optimization_event(
            "Route optimization completed",
            request_id=payload.request_id,
            graph_nodes=len(payload.nodes),
            route_distance=response.total_distance_km,
            avg_travel_time=response.total_travel_time_min / max(len(response.steps), 1),
            optimization_status=response.optimization_status,
        )
        return response

    def latest(self) -> RouteOptimizationResponse | None:
        return self.repository.latest()

    def demo_network(self) -> DemoNetworkResponse:
        nodes = [
            GraphNode(node_id="A", label="Warehouse A", x=0.0, y=0.0),
            GraphNode(node_id="B", label="Hub B", x=2.0, y=1.0),
            GraphNode(node_id="C", label="Retail C", x=4.0, y=0.5),
            GraphNode(node_id="D", label="Depot D", x=1.0, y=3.0),
            GraphNode(node_id="E", label="District E", x=3.0, y=3.2),
            GraphNode(node_id="F", label="Terminal F", x=5.0, y=2.5),
        ]
        edges = [
            GraphEdge(source="A", target="B", distance_km=3.0, travel_time_min=12.0),
            GraphEdge(source="B", target="C", distance_km=2.2, travel_time_min=9.0),
            GraphEdge(source="A", target="D", distance_km=2.5, travel_time_min=10.0),
            GraphEdge(source="D", target="E", distance_km=2.0, travel_time_min=8.0),
            GraphEdge(source="E", target="F", distance_km=2.3, travel_time_min=10.0),
            GraphEdge(source="C", target="F", distance_km=2.1, travel_time_min=7.0),
            GraphEdge(source="B", target="E", distance_km=2.4, travel_time_min=11.0),
            GraphEdge(source="C", target="E", distance_km=2.0, travel_time_min=8.5),
        ]
        return DemoNetworkResponse(nodes=nodes, edges=edges)

