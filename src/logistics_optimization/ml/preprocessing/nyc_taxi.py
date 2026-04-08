from pathlib import Path

import pandas as pd

from logistics_optimization.schemas.forecast import ForecastObservation


class NycTaxiPreprocessor:
    """Prepare NYC Taxi style demand data for the forecasting pipeline."""

    datetime_candidates = ["timestamp", "pickup_datetime", "tpep_pickup_datetime"]
    zone_candidates = ["zone_id", "PULocationID"]

    def load(self, dataset_path: str | Path) -> pd.DataFrame:
        frame = pd.read_csv(dataset_path)
        return self._normalize(frame)

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        timestamp_column = next((column for column in self.datetime_candidates if column in frame.columns), None)
        zone_column = next((column for column in self.zone_candidates if column in frame.columns), None)

        if timestamp_column is None or zone_column is None:
            raise ValueError("Dataset must include a timestamp column and a zone identifier column.")

        normalized = frame.copy()
        normalized["timestamp"] = pd.to_datetime(normalized[timestamp_column])
        normalized["zone_id"] = normalized[zone_column].astype(str)

        if "demand" not in normalized.columns:
            normalized["demand"] = 1.0

        grouped = (
            normalized.groupby(["zone_id", "timestamp"], as_index=False)
            .agg(
                demand=("demand", "sum"),
                avg_trip_distance=("avg_trip_distance", "mean") if "avg_trip_distance" in normalized else ("demand", "size"),
                avg_travel_time=("avg_travel_time", "mean") if "avg_travel_time" in normalized else ("demand", "size"),
            )
            .sort_values(["zone_id", "timestamp"])
        )

        if "avg_trip_distance" not in normalized:
            grouped["avg_trip_distance"] = grouped["avg_trip_distance"].astype(float) * 0.8
        if "avg_travel_time" not in normalized:
            grouped["avg_travel_time"] = grouped["avg_travel_time"].astype(float) * 5.0

        grouped["hour_of_day"] = grouped["timestamp"].dt.hour
        grouped["day_of_week"] = grouped["timestamp"].dt.dayofweek
        grouped["is_weekend"] = grouped["day_of_week"].isin([5, 6]).astype(int)
        return grouped

    def to_observations(self, frame: pd.DataFrame) -> list[ForecastObservation]:
        return [
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
            for row in frame.itertuples(index=False)
        ]

