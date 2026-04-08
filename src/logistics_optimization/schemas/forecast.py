from datetime import datetime

from pydantic import BaseModel, Field


class ForecastObservation(BaseModel):
    zone_id: str
    timestamp: datetime
    demand: float
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    avg_trip_distance: float = 0.0
    avg_travel_time: float = 0.0


class ForecastPrediction(BaseModel):
    zone_id: str
    predicted_demand: float
    model_name: str
    inference_latency_ms: float


class ForecastRequest(BaseModel):
    observations: list[ForecastObservation] = Field(min_length=2)


class TrainForecastRequest(BaseModel):
    dataset_path: str | None = None
    sequence_length: int = 4
    epochs: int = 8


class TrainForecastResponse(BaseModel):
    model_name: str
    sequence_length: int
    epochs: int
    training_loss: float
    validation_mae: float
    rmse: float
    model_path: str


class HeatmapPoint(BaseModel):
    zone_id: str
    timestamp: datetime
    demand: float
    hour_of_day: int


class ForecastMetricsResponse(BaseModel):
    model_name: str
    mae: float
    rmse: float
    latest_training_loss: float

