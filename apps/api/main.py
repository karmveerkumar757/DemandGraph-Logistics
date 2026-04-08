from contextlib import asynccontextmanager
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI

from logistics_optimization.api.router import api_router
from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import get_logger
from logistics_optimization.db.database import SessionLocal, init_db
from logistics_optimization.services.bootstrap import BootstrapService


settings = get_settings()
logger = get_logger("logistics.api", level=settings.log_level, log_dir=settings.log_dir)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as session:
        BootstrapService(session=session, logger=logger).seed_sample_demand_data()
    logger.info("API startup completed", model_name=settings.forecast_model_name)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(api_router)
