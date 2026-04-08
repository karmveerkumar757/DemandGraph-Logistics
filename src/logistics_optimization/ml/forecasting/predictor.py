import json
import time
from pathlib import Path
from typing import Sequence

from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import LogisticsLogger
from logistics_optimization.domain.interfaces.forecasting import ForecastingEngine
from logistics_optimization.ml.forecasting.datasets import observations_to_model_input
from logistics_optimization.ml.forecasting.trainer import TransformerTrainer
from logistics_optimization.ml.forecasting.transformer import DemandForecastTransformer
from logistics_optimization.ml.preprocessing.nyc_taxi import NycTaxiPreprocessor
from logistics_optimization.schemas.forecast import ForecastObservation, ForecastPrediction


class DemandForecastPipeline(ForecastingEngine):
    def __init__(self, logger: LogisticsLogger) -> None:
        self.settings = get_settings()
        self.logger = logger
        self.preprocessor = NycTaxiPreprocessor()
        self.trainer = TransformerTrainer(
            logger=logger,
            model_dir=self.settings.model_dir,
            model_name=self.settings.forecast_model_name,
        )

    def train(self, dataset_path: str, sequence_length: int, epochs: int) -> dict:
        frame = self.preprocessor.load(dataset_path)
        summary = self.trainer.train(frame=frame, sequence_length=sequence_length, epochs=epochs)
        return summary.__dict__

    def predict(self, observations: Sequence[ForecastObservation]) -> ForecastPrediction:
        if not observations:
            raise ValueError("At least one observation is required for forecasting.")

        started = time.perf_counter()
        zone_id = observations[-1].zone_id
        checkpoint = self.trainer.checkpoint_path

        if checkpoint.exists():
            try:
                prediction = self._predict_with_model(observations=observations, checkpoint_path=checkpoint)
                model_name = self.settings.forecast_model_name
            except RuntimeError as exc:
                self.logger.warning(
                    "Transformer checkpoint unavailable at runtime; using baseline forecast instead",
                    exception_trace=str(exc),
                )
                prediction = self._predict_with_baseline(observations=observations)
                model_name = f"{self.settings.forecast_model_name}_baseline"
        else:
            prediction = self._predict_with_baseline(observations=observations)
            model_name = f"{self.settings.forecast_model_name}_baseline"

        latency_ms = (time.perf_counter() - started) * 1000
        return ForecastPrediction(
            zone_id=zone_id,
            predicted_demand=round(prediction, 3),
            model_name=model_name,
            inference_latency_ms=round(latency_ms, 3),
        )

    def latest_metrics(self) -> dict:
        if not self.trainer.metadata_path.exists():
            return {
                "model_name": self.settings.forecast_model_name,
                "mae": 0.0,
                "rmse": 0.0,
                "latest_training_loss": 0.0,
            }
        payload = json.loads(self.trainer.metadata_path.read_text(encoding="utf-8"))
        return {
            "model_name": payload["model_name"],
            "mae": payload["validation_mae"],
            "rmse": payload["rmse"],
            "latest_training_loss": payload["training_loss"],
        }

    def _predict_with_model(self, observations: Sequence[ForecastObservation], checkpoint_path: Path) -> float:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyTorch is required to load trained transformer checkpoints.") from exc
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        sequence_length = int(checkpoint["sequence_length"])
        model = DemandForecastTransformer(input_dim=int(checkpoint["input_dim"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model_input = observations_to_model_input(observations, sequence_length=sequence_length)
        with torch.no_grad():
            output = model(model_input)
        return float(output.item())

    def _predict_with_baseline(self, observations: Sequence[ForecastObservation]) -> float:
        recent = [item.demand for item in observations[-3:]]
        slope = 0.0
        if len(recent) >= 2:
            slope = recent[-1] - recent[-2]
        return max(0.0, (sum(recent) / len(recent)) + 0.3 * slope)
