from dataclasses import dataclass, field

from logistics_optimization.schemas.routing import GraphEdge, GraphNode, OptimizationWeights


@dataclass(slots=True)
class WeightedGraph:
    nodes: dict[str, GraphNode]
    adjacency: dict[str, list[GraphEdge]] = field(default_factory=dict)

    def neighbors(self, node_id: str) -> list[GraphEdge]:
        return self.adjacency.get(node_id, [])


def build_weighted_graph(nodes: list[GraphNode], edges: list[GraphEdge]) -> WeightedGraph:
    graph = WeightedGraph(nodes={node.node_id: node for node in nodes}, adjacency={})
    for edge in edges:
        graph.adjacency.setdefault(edge.source, []).append(edge)
        graph.adjacency.setdefault(edge.target, []).append(
            GraphEdge(
                source=edge.target,
                target=edge.source,
                distance_km=edge.distance_km,
                travel_time_min=edge.travel_time_min,
            )
        )
    return graph


def edge_cost(edge: GraphEdge, weights: OptimizationWeights) -> float:
    return (edge.distance_km * weights.distance_weight) + (edge.travel_time_min * weights.time_weight)

