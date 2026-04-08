import os
from typing import Any

import requests


class LogisticsApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("LOGISTICS_API_BASE_URL") or "http://localhost:8000").rstrip("/")

    def _get(self, path: str, **params: Any) -> Any:
        response = requests.get(f"{self.base_url}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def heatmap(self) -> list[dict[str, Any]]:
        return self._get("/api/v1/forecast/heatmap")

    def recent_zone_observations(self, zone_id: str, limit: int = 8) -> list[dict[str, Any]]:
        return self._get(f"/api/v1/forecast/zones/{zone_id}/recent", limit=limit)

    def predict_demand(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        return self._post("/api/v1/forecast/predict", {"observations": observations})

    def train_forecaster(self, sequence_length: int, epochs: int) -> dict[str, Any]:
        return self._post(
            "/api/v1/forecast/train",
            {
                "sequence_length": sequence_length,
                "epochs": epochs,
            },
        )

    def forecasting_metrics(self) -> dict[str, Any]:
        return self._get("/api/v1/forecast/metrics")

    def demo_network(self) -> dict[str, Any]:
        return self._get("/api/v1/routes/demo-network")

    def optimize_route(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/routes/optimize", payload)

    def latest_route(self) -> dict[str, Any] | None:
        return self._get("/api/v1/routes/latest")

