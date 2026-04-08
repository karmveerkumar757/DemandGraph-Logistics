from collections.abc import Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from logistics_optimization.db.models import DemandObservationRecord
from logistics_optimization.schemas.forecast import ForecastObservation, HeatmapPoint


class DemandRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def bulk_insert(self, observations: Sequence[ForecastObservation]) -> int:
        records = [
            DemandObservationRecord(
                zone_id=item.zone_id,
                timestamp=item.timestamp,
                demand=item.demand,
                hour_of_day=item.hour_of_day,
                day_of_week=item.day_of_week,
                is_weekend=item.is_weekend,
                avg_trip_distance=item.avg_trip_distance,
                avg_travel_time=item.avg_travel_time,
            )
            for item in observations
        ]
        self.session.add_all(records)
        self.session.commit()
        return len(records)

    def count(self) -> int:
        return self.session.scalar(select(func.count(DemandObservationRecord.id))) or 0

    def recent_by_zone(self, zone_id: str, limit: int = 12) -> list[ForecastObservation]:
        stmt: Select[tuple[DemandObservationRecord]] = (
            select(DemandObservationRecord)
            .where(DemandObservationRecord.zone_id == zone_id)
            .order_by(DemandObservationRecord.timestamp.desc())
            .limit(limit)
        )
        rows = list(self.session.scalars(stmt))
        return [
            ForecastObservation(
                zone_id=row.zone_id,
                timestamp=row.timestamp,
                demand=row.demand,
                hour_of_day=row.hour_of_day,
                day_of_week=row.day_of_week,
                is_weekend=row.is_weekend,
                avg_trip_distance=row.avg_trip_distance,
                avg_travel_time=row.avg_travel_time,
            )
            for row in reversed(rows)
        ]

    def heatmap_points(self) -> list[HeatmapPoint]:
        stmt = select(DemandObservationRecord).order_by(DemandObservationRecord.timestamp.asc())
        rows = list(self.session.scalars(stmt))
        return [
            HeatmapPoint(
                zone_id=row.zone_id,
                timestamp=row.timestamp,
                demand=row.demand,
                hour_of_day=row.hour_of_day,
            )
            for row in rows
        ]

