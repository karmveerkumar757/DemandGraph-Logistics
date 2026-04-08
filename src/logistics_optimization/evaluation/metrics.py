from math import sqrt
from typing import Iterable


def mean_absolute_error(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_values = list(actual)
    predicted_values = list(predicted)
    if not actual_values:
        return 0.0
    errors = [abs(a - p) for a, p in zip(actual_values, predicted_values, strict=False)]
    return sum(errors) / len(errors)


def root_mean_squared_error(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_values = list(actual)
    predicted_values = list(predicted)
    if not actual_values:
        return 0.0
    squared_errors = [(a - p) ** 2 for a, p in zip(actual_values, predicted_values, strict=False)]
    return sqrt(sum(squared_errors) / len(squared_errors))


def route_distance(steps: Iterable[dict]) -> float:
    return sum(step["distance_km"] for step in steps)


def average_travel_time(steps: Iterable[dict]) -> float:
    step_list = list(steps)
    if not step_list:
        return 0.0
    return sum(step["travel_time_min"] for step in step_list) / len(step_list)

