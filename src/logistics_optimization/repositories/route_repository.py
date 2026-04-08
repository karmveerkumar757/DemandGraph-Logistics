import json

from sqlalchemy import Select, desc, select
from sqlalchemy.orm import Session

from logistics_optimization.db.models import RouteOptimizationRecord
from logistics_optimization.schemas.routing import RouteOptimizationResponse


class RouteRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save(self, response: RouteOptimizationResponse) -> None:
        record = RouteOptimizationRecord(
            request_id=response.request_id,
            start_node=response.path[0],
            end_node=response.path[-1],
            algorithm=response.algorithm,
            objective=response.objective,
            total_distance=response.total_distance_km,
            total_travel_time=response.total_travel_time_min,
            status=response.optimization_status,
            route_path_json=json.dumps(response.model_dump()),
        )
        self.session.add(record)
        self.session.commit()

    def latest(self) -> RouteOptimizationResponse | None:
        stmt: Select[tuple[RouteOptimizationRecord]] = (
            select(RouteOptimizationRecord).order_by(desc(RouteOptimizationRecord.created_at)).limit(1)
        )
        row = self.session.scalar(stmt)
        if row is None:
            return None
        payload = json.loads(row.route_path_json)
        return RouteOptimizationResponse(**payload)

