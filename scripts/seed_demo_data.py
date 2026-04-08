from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import get_logger
from logistics_optimization.db.database import SessionLocal, init_db
from logistics_optimization.services.bootstrap import BootstrapService


if __name__ == "__main__":
    settings = get_settings()
    logger = get_logger("logistics.seed", level=settings.log_level, log_dir=settings.log_dir)
    init_db()
    with SessionLocal() as session:
        inserted = BootstrapService(session=session, logger=logger).seed_sample_demand_data()
    print(f"Inserted {inserted} demand records.")
