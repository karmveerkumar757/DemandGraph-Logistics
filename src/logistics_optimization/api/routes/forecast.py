from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from logistics_optimization.db.database import get_db_session
from logistics_optimization.schemas.forecast import (
    ForecastMetricsResponse,
    ForecastPrediction,
    ForecastRequest,
    ForecastObservation,
    HeatmapPoint,
    TrainForecastRequest,
    TrainForecastResponse,
)
from logistics_optimization.services.forecasting_service import ForecastingService


router = APIRouter(prefix="/forecast", tags=["Demand Forecasting"])


@router.post("/train", response_model=TrainForecastResponse)
def train_forecasting_model(
    payload: TrainForecastRequest,
    session: Session = Depends(get_db_session),
) -> TrainForecastResponse:
    try:
        return ForecastingService(session).train(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict", response_model=ForecastPrediction)
def predict_demand(
    payload: ForecastRequest,
    session: Session = Depends(get_db_session),
) -> ForecastPrediction:
    try:
        return ForecastingService(session).predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/heatmap", response_model=list[HeatmapPoint])
def demand_heatmap(session: Session = Depends(get_db_session)) -> list[HeatmapPoint]:
    return ForecastingService(session).get_heatmap()


@router.get("/zones/{zone_id}/recent", response_model=list[ForecastObservation])
def recent_zone_observations(
    zone_id: str,
    limit: int = Query(default=8, ge=2, le=48),
    session: Session = Depends(get_db_session),
) -> list[ForecastObservation]:
    return ForecastingService(session).get_recent_observations(zone_id=zone_id, limit=limit)


@router.get("/metrics", response_model=ForecastMetricsResponse)
def forecasting_metrics(session: Session = Depends(get_db_session)) -> ForecastMetricsResponse:
    return ForecastingService(session).get_metrics()
