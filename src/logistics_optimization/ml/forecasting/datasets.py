from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None

    class Dataset:  # type: ignore[override]
        pass

from logistics_optimization.schemas.forecast import ForecastObservation


@dataclass(slots=True)
class WindowedDataset:
    sequences: np.ndarray
    targets: np.ndarray
    zone_ids: list[str]
    input_dim: int


def feature_vector_from_observation(observation: ForecastObservation) -> list[float]:
    hour_angle = (2 * pi * observation.hour_of_day) / 24
    day_angle = (2 * pi * observation.day_of_week) / 7
    return [
        float(observation.demand),
        sin(hour_angle),
        cos(hour_angle),
        sin(day_angle),
        cos(day_angle),
        float(observation.is_weekend),
        float(observation.avg_trip_distance),
        float(observation.avg_travel_time),
    ]


def build_windowed_dataset(frame: pd.DataFrame, sequence_length: int) -> WindowedDataset:
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    zone_ids: list[str] = []

    for zone_id, zone_frame in frame.sort_values(["zone_id", "timestamp"]).groupby("zone_id"):
        observations = [
            ForecastObservation(
                zone_id=str(row.zone_id),
                timestamp=row.timestamp.to_pydatetime(),
                demand=float(row.demand),
                hour_of_day=int(row.hour_of_day),
                day_of_week=int(row.day_of_week),
                is_weekend=int(row.is_weekend),
                avg_trip_distance=float(row.avg_trip_distance),
                avg_travel_time=float(row.avg_travel_time),
            )
            for row in zone_frame.itertuples(index=False)
        ]
        zone_features = np.asarray([feature_vector_from_observation(item) for item in observations], dtype=np.float32)
        zone_targets = np.asarray([item.demand for item in observations], dtype=np.float32)

        for index in range(sequence_length, len(zone_features)):
            sequences.append(zone_features[index - sequence_length : index])
            targets.append(zone_targets[index])
            zone_ids.append(str(zone_id))

    if not sequences:
        raise ValueError("Not enough historical data to create transformer windows.")

    sequence_tensor = np.stack(sequences)
    target_tensor = np.asarray(targets, dtype=np.float32)
    return WindowedDataset(
        sequences=sequence_tensor,
        targets=target_tensor,
        zone_ids=zone_ids,
        input_dim=sequence_tensor.shape[-1],
    )


class ForecastSequenceDataset(Dataset):
    def __init__(self, payload: WindowedDataset) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to create training datasets for the transformer model.")
        self.sequences = torch.tensor(payload.sequences, dtype=torch.float32)
        self.targets = torch.tensor(payload.targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.targets[index]


def observations_to_model_input(
    observations: Sequence[ForecastObservation],
    *,
    sequence_length: int,
) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is required to prepare model inputs for the transformer forecaster.")
    ordered = sorted(observations, key=lambda item: item.timestamp)
    if len(ordered) < sequence_length:
        pad_count = sequence_length - len(ordered)
        ordered = [ordered[0]] * pad_count + list(ordered)
    trimmed = ordered[-sequence_length:]
    payload = [feature_vector_from_observation(item) for item in trimmed]
    return torch.tensor([payload], dtype=torch.float32)
