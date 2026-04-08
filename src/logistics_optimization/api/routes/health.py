from fastapi import APIRouter

from logistics_optimization.core.config import get_settings


router = APIRouter(tags=["Health"])


@router.get("/health")
def healthcheck() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "application": settings.app_name,
        "environment": settings.environment,
    }

