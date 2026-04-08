from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, random_split
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    Adam = None
    DataLoader = None
    random_split = None

from logistics_optimization.core.logger import LogisticsLogger
from logistics_optimization.evaluation.metrics import mean_absolute_error, root_mean_squared_error
from logistics_optimization.ml.forecasting.datasets import ForecastSequenceDataset, build_windowed_dataset
from logistics_optimization.ml.forecasting.transformer import DemandForecastTransformer


@dataclass(slots=True)
class TrainingSummary:
    model_name: str
    sequence_length: int
    epochs: int
    training_loss: float
    validation_mae: float
    rmse: float
    model_path: str


class TransformerTrainer:
    def __init__(self, logger: LogisticsLogger, model_dir: Path, model_name: str = "demand_transformer") -> None:
        self.logger = logger
        self.model_dir = model_dir
        self.model_name = model_name

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / f"{self.model_name}.pt"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / f"{self.model_name}_metrics.json"

    def train(self, frame, sequence_length: int, epochs: int) -> TrainingSummary:
        if torch is None or nn is None or Adam is None or DataLoader is None or random_split is None:
            raise RuntimeError("PyTorch is required to train the transformer forecaster. Install the 'torch' package.")
        payload = build_windowed_dataset(frame, sequence_length=sequence_length)
        dataset = ForecastSequenceDataset(payload)

        if len(dataset) == 1:
            train_dataset = dataset
            validation_dataset = dataset
        else:
            train_size = max(1, int(len(dataset) * 0.8))
            validation_size = len(dataset) - train_size
            if validation_size == 0:
                validation_size = 1
                train_size = len(dataset) - 1
            train_dataset, validation_dataset = random_split(
                dataset,
                lengths=[train_size, validation_size],
                generator=torch.Generator().manual_seed(42),
            )

        train_loader = DataLoader(train_dataset, batch_size=min(16, len(train_dataset)), shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=min(16, len(validation_dataset)))

        model = DemandForecastTransformer(input_dim=payload.input_dim)
        optimizer = Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        final_loss = 0.0
        final_mae = 0.0
        final_rmse = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for sequences, targets in train_loader:
                optimizer.zero_grad()
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            final_loss = running_loss / max(1, len(train_loader))
            final_mae, final_rmse = self._evaluate(model, validation_loader)
            self.logger.training_event(
                "Transformer training epoch completed",
                model_name=self.model_name,
                epoch=epoch,
                training_loss=final_loss,
                validation_mae=final_mae,
                rmse=final_rmse,
            )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": payload.input_dim,
                "sequence_length": sequence_length,
                "model_name": self.model_name,
            },
            self.checkpoint_path,
        )

        self.metadata_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "sequence_length": sequence_length,
                    "epochs": epochs,
                    "training_loss": round(final_loss, 4),
                    "validation_mae": round(final_mae, 4),
                    "rmse": round(final_rmse, 4),
                    "model_path": str(self.checkpoint_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return TrainingSummary(
            model_name=self.model_name,
            sequence_length=sequence_length,
            epochs=epochs,
            training_loss=round(final_loss, 4),
            validation_mae=round(final_mae, 4),
            rmse=round(final_rmse, 4),
            model_path=str(self.checkpoint_path),
        )

    def _evaluate(self, model: DemandForecastTransformer, loader: DataLoader) -> tuple[float, float]:
        model.eval()
        actual: list[float] = []
        predicted: list[float] = []
        with torch.no_grad():
            for sequences, targets in loader:
                outputs = model(sequences)
                actual.extend(targets.tolist())
                predicted.extend(outputs.tolist())
        return mean_absolute_error(actual, predicted), root_mean_squared_error(actual, predicted)
