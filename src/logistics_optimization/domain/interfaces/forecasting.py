from abc import ABC, abstractmethod
from typing import Sequence

from logistics_optimization.schemas.forecast import ForecastObservation, ForecastPrediction


class ForecastingEngine(ABC):
    @abstractmethod
    def train(self, dataset_path: str, sequence_length: int, epochs: int) -> dict:
        """Train the forecasting engine and return summary metrics."""

    @abstractmethod
    def predict(self, observations: Sequence[ForecastObservation]) -> ForecastPrediction:
        """Return the next demand prediction from historical observations."""

