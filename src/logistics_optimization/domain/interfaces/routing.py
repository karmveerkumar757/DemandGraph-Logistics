from abc import ABC, abstractmethod

from logistics_optimization.schemas.routing import RouteOptimizationRequest, RouteOptimizationResponse


class RoutingEngine(ABC):
    @abstractmethod
    def optimize(self, request: RouteOptimizationRequest) -> RouteOptimizationResponse:
        """Compute the best route for the current optimization objective."""

