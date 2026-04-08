from logistics_optimization.evaluation.metrics import (
    average_travel_time,
    mean_absolute_error,
    root_mean_squared_error,
    route_distance,
)


def test_forecasting_metrics() -> None:
    actual = [10.0, 15.0, 13.0]
    predicted = [11.0, 14.0, 12.0]
    assert round(mean_absolute_error(actual, predicted), 4) == 1.0
    assert round(root_mean_squared_error(actual, predicted), 4) == 1.0


def test_route_metrics() -> None:
    steps = [
        {"distance_km": 2.0, "travel_time_min": 10.0},
        {"distance_km": 3.0, "travel_time_min": 15.0},
    ]
    assert route_distance(steps) == 5.0
    assert average_travel_time(steps) == 12.5

