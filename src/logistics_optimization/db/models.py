from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from logistics_optimization.db.database import Base


class DemandObservationRecord(Base):
    __tablename__ = "demand_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    zone_id: Mapped[str] = mapped_column(String(32), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    demand: Mapped[float] = mapped_column(Float)
    hour_of_day: Mapped[int] = mapped_column(Integer)
    day_of_week: Mapped[int] = mapped_column(Integer)
    is_weekend: Mapped[int] = mapped_column(Integer)
    avg_trip_distance: Mapped[float] = mapped_column(Float, default=0.0)
    avg_travel_time: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)


class RouteOptimizationRecord(Base):
    __tablename__ = "route_optimizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    start_node: Mapped[str] = mapped_column(String(32))
    end_node: Mapped[str] = mapped_column(String(32))
    algorithm: Mapped[str] = mapped_column(String(32))
    objective: Mapped[str] = mapped_column(String(64))
    total_distance: Mapped[float] = mapped_column(Float)
    total_travel_time: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32))
    route_path_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

