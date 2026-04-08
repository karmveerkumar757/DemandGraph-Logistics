from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    node_id: str
    label: str
    x: float
    y: float


class GraphEdge(BaseModel):
    source: str
    target: str
    distance_km: float = Field(gt=0)
    travel_time_min: float = Field(gt=0)


class OptimizationWeights(BaseModel):
    distance_weight: float = 0.5
    time_weight: float = 0.5


class RouteStep(BaseModel):
    source: str
    target: str
    distance_km: float
    travel_time_min: float


class RouteOptimizationRequest(BaseModel):
    request_id: str
    nodes: list[GraphNode] = Field(min_length=2)
    edges: list[GraphEdge] = Field(min_length=1)
    start_node: str
    end_node: str
    algorithm: str = "dijkstra"
    weights: OptimizationWeights = Field(default_factory=OptimizationWeights)


class RouteOptimizationResponse(BaseModel):
    request_id: str
    algorithm: str
    objective: str
    path: list[str]
    steps: list[RouteStep]
    total_distance_km: float
    total_travel_time_min: float
    total_cost: float
    optimization_status: str


class DemoNetworkResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
