from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from logistics_optimization.core.config import get_settings


settings = get_settings()
engine = create_engine(settings.resolved_database_url(), echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


class Base(DeclarativeBase):
    pass


def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    from logistics_optimization.db.models import DemandObservationRecord, RouteOptimizationRecord

    Base.metadata.create_all(bind=engine)

