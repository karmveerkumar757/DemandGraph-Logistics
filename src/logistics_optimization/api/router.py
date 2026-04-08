from fastapi import APIRouter

from logistics_optimization.api.routes import forecast, health, routing


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(forecast.router, prefix="/api/v1")
api_router.include_router(routing.router, prefix="/api/v1")

