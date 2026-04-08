from pathlib import Path

from logistics_optimization.ml.preprocessing.nyc_taxi import NycTaxiPreprocessor


def test_preprocessor_loads_aggregated_sample() -> None:
    sample_path = Path("data/sample/nyc_taxi_sample.csv")
    frame = NycTaxiPreprocessor().load(sample_path)
    assert {"zone_id", "timestamp", "demand", "hour_of_day", "day_of_week", "is_weekend"} <= set(frame.columns)
    assert len(frame) >= 12

