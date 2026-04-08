from sqlalchemy.orm import Session

from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import LogisticsLogger
from logistics_optimization.ml.preprocessing.nyc_taxi import NycTaxiPreprocessor
from logistics_optimization.repositories.demand_repository import DemandRepository


class BootstrapService:
    def __init__(self, session: Session, logger: LogisticsLogger) -> None:
        self.session = session
        self.logger = logger
        self.settings = get_settings()

    def seed_sample_demand_data(self) -> int:
        repository = DemandRepository(self.session)
        if repository.count() > 0:
            return 0

        preprocessor = NycTaxiPreprocessor()
        frame = preprocessor.load(self.settings.sample_data_path)
        observations = preprocessor.to_observations(frame)
        inserted = repository.bulk_insert(observations)
        self.logger.info("Seeded sample NYC Taxi demand data", model_name="bootstrap")
        return inserted

