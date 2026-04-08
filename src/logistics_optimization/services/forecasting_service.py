from sqlalchemy.orm import Session

from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import get_logger
from logistics_optimization.ml.forecasting.predictor import DemandForecastPipeline
from logistics_optimization.repositories.demand_repository import DemandRepository
from logistics_optimization.schemas.forecast import (
    ForecastMetricsResponse,
    ForecastPrediction,
    ForecastRequest,
    ForecastObservation,
    HeatmapPoint,
    TrainForecastRequest,
    TrainForecastResponse,
)


class ForecastingService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.settings = get_settings()
        self.logger = get_logger(
            "logistics.forecasting",
            level=self.settings.log_level,
            log_dir=self.settings.log_dir,
        )
        self.repository = DemandRepository(session)
        self.pipeline = DemandForecastPipeline(logger=self.logger)

    def train(self, payload: TrainForecastRequest) -> TrainForecastResponse:
        dataset_path = payload.dataset_path or str(self.settings.sample_data_path)
        summary = self.pipeline.train(
            dataset_path=dataset_path,
            sequence_length=payload.sequence_length,
            epochs=payload.epochs,
        )
        return TrainForecastResponse(**summary)

    def predict(self, payload: ForecastRequest) -> ForecastPrediction:
        prediction = self.pipeline.predict(payload.observations)
        self.logger.info(
            "Generated demand forecast",
            model_name=prediction.model_name,
            inference_latency_ms=prediction.inference_latency_ms,
        )
        return prediction

    def get_recent_observations(self, zone_id: str, limit: int = 12) -> list[ForecastObservation]:
        return self.repository.recent_by_zone(zone_id=zone_id, limit=limit)

    def get_heatmap(self) -> list[HeatmapPoint]:
        return self.repository.heatmap_points()

    def get_metrics(self) -> ForecastMetricsResponse:
        return ForecastMetricsResponse(**self.pipeline.latest_metrics())

