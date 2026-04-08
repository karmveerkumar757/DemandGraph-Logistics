import heapq
from dataclasses import dataclass
from math import hypot

from logistics_optimization.domain.interfaces.routing import RoutingEngine
from logistics_optimization.ml.optimization.graph_builder import WeightedGraph, build_weighted_graph, edge_cost
from logistics_optimization.schemas.routing import RouteOptimizationRequest, RouteOptimizationResponse, RouteStep


@dataclass(slots=True)
class PathResult:
    path: list[str]
    total_cost: float
    steps: list[RouteStep]
    total_distance_km: float
    total_travel_time_min: float


class GraphRoutingEngine(RoutingEngine):
    def optimize(self, request: RouteOptimizationRequest) -> RouteOptimizationResponse:
        graph = build_weighted_graph(request.nodes, request.edges)
        if request.algorithm.lower() == "astar":
            result = self._a_star(graph=graph, request=request)
        else:
            result = self._dijkstra(graph=graph, request=request)

        return RouteOptimizationResponse(
            request_id=request.request_id,
            algorithm=request.algorithm.lower(),
            objective=f"distance={request.weights.distance_weight:.2f},time={request.weights.time_weight:.2f}",
            path=result.path,
            steps=result.steps,
            total_distance_km=round(result.total_distance_km, 3),
            total_travel_time_min=round(result.total_travel_time_min, 3),
            total_cost=round(result.total_cost, 3),
            optimization_status="success",
        )

    def _dijkstra(self, graph: WeightedGraph, request: RouteOptimizationRequest) -> PathResult:
        frontier: list[tuple[float, str]] = [(0.0, request.start_node)]
        distances = {request.start_node: 0.0}
        previous: dict[str, tuple[str, object]] = {}

        while frontier:
            current_cost, current_node = heapq.heappop(frontier)
            if current_node == request.end_node:
                break

            for edge in graph.neighbors(current_node):
                new_cost = current_cost + edge_cost(edge, request.weights)
                if new_cost < distances.get(edge.target, float("inf")):
                    distances[edge.target] = new_cost
                    previous[edge.target] = (current_node, edge)
                    heapq.heappush(frontier, (new_cost, edge.target))

        return self._materialize_path(
            graph=graph,
            previous=previous,
            request=request,
            total_cost=distances.get(request.end_node, float("inf")),
        )

    def _a_star(self, graph: WeightedGraph, request: RouteOptimizationRequest) -> PathResult:
        frontier: list[tuple[float, str]] = [(0.0, request.start_node)]
        g_score = {request.start_node: 0.0}
        previous: dict[str, tuple[str, object]] = {}

        while frontier:
            _, current_node = heapq.heappop(frontier)
            if current_node == request.end_node:
                break

            for edge in graph.neighbors(current_node):
                tentative = g_score[current_node] + edge_cost(edge, request.weights)
                if tentative < g_score.get(edge.target, float("inf")):
                    g_score[edge.target] = tentative
                    previous[edge.target] = (current_node, edge)
                    priority = tentative + self._heuristic(graph, edge.target, request.end_node, request.weights.distance_weight)
                    heapq.heappush(frontier, (priority, edge.target))

        return self._materialize_path(
            graph=graph,
            previous=previous,
            request=request,
            total_cost=g_score.get(request.end_node, float("inf")),
        )

    def _materialize_path(
        self,
        *,
        graph: WeightedGraph,
        previous: dict[str, tuple[str, object]],
        request: RouteOptimizationRequest,
        total_cost: float,
    ) -> PathResult:
        if request.end_node not in graph.nodes or request.start_node not in graph.nodes:
            raise ValueError("Start or end node is missing from the graph.")
        if request.start_node != request.end_node and request.end_node not in previous:
            raise ValueError("No route found between the selected nodes.")

        path = [request.end_node]
        steps: list[RouteStep] = []
        total_distance = 0.0
        total_travel_time = 0.0
        current = request.end_node
        while current != request.start_node:
            parent, edge = previous[current]
            steps.append(
                RouteStep(
                    source=edge.source,
                    target=edge.target,
                    distance_km=edge.distance_km,
                    travel_time_min=edge.travel_time_min,
                )
            )
            total_distance += edge.distance_km
            total_travel_time += edge.travel_time_min
            path.append(parent)
            current = parent

        path.reverse()
        steps.reverse()
        return PathResult(
            path=path,
            total_cost=total_cost,
            steps=steps,
            total_distance_km=total_distance,
            total_travel_time_min=total_travel_time,
        )

    def _heuristic(self, graph: WeightedGraph, current_node: str, target_node: str, distance_weight: float) -> float:
        current = graph.nodes[current_node]
        target = graph.nodes[target_node]
        return hypot(current.x - target.x, current.y - target.y) * max(distance_weight, 0.1)
