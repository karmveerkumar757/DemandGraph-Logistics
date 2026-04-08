from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import get_logger
from logistics_optimization.evaluation.metrics import average_travel_time, route_distance
from logistics_optimization.ml.forecasting.predictor import DemandForecastPipeline
from logistics_optimization.schemas.routing import RouteOptimizationResponse


class EvaluationService:
    def __init__(self) -> None:
        settings = get_settings()
        logger = get_logger("logistics.evaluation", level=settings.log_level, log_dir=settings.log_dir)
        self.pipeline = DemandForecastPipeline(logger=logger)

    def forecasting_metrics(self) -> dict:
        return self.pipeline.latest_metrics()

    def route_metrics(self, response: RouteOptimizationResponse | None) -> dict:
        if response is None:
            return {
                "total_distance_km": 0.0,
                "average_travel_time_min": 0.0,
            }
        step_payload = [step.model_dump() for step in response.steps]
        return {
            "total_distance_km": route_distance(step_payload),
            "average_travel_time_min": average_travel_time(step_payload),
        }

