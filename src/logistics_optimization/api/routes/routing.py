from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from logistics_optimization.db.database import get_db_session
from logistics_optimization.schemas.routing import DemoNetworkResponse, RouteOptimizationRequest, RouteOptimizationResponse
from logistics_optimization.services.routing_service import RoutingService


router = APIRouter(prefix="/routes", tags=["Route Optimization"])


@router.get("/demo-network", response_model=DemoNetworkResponse)
def demo_network(session: Session = Depends(get_db_session)) -> DemoNetworkResponse:
    return RoutingService(session).demo_network()


@router.post("/optimize", response_model=RouteOptimizationResponse)
def optimize_route(
    payload: RouteOptimizationRequest,
    session: Session = Depends(get_db_session),
) -> RouteOptimizationResponse:
    try:
        return RoutingService(session).optimize(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/latest", response_model=RouteOptimizationResponse | None)
def latest_route(session: Session = Depends(get_db_session)) -> RouteOptimizationResponse | None:
    return RoutingService(session).latest()
