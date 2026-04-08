from pathlib import Path
import sys
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import get_logger
from logistics_optimization.db.database import SessionLocal, init_db
from logistics_optimization.schemas.routing import RouteOptimizationRequest
from logistics_optimization.services.evaluation_service import EvaluationService
from logistics_optimization.services.routing_service import RoutingService


if __name__ == "__main__":
    settings = get_settings()
    logger = get_logger("logistics.evaluate", level=settings.log_level, log_dir=settings.log_dir)
    init_db()

    with SessionLocal() as session:
        routing_service = RoutingService(session)
        network = routing_service.demo_network()
        result = routing_service.optimize(
            payload=RouteOptimizationRequest(
                request_id=str(uuid4()),
                nodes=network.nodes,
                edges=network.edges,
                start_node="A",
                end_node="F",
                algorithm="dijkstra",
                weights={"distance_weight": 0.5, "time_weight": 0.5},
            )
        )

    evaluation = EvaluationService()
    print(evaluation.forecasting_metrics())
    print(evaluation.route_metrics(result))
