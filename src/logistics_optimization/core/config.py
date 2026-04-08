from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "AI-Based Logistics Optimization System"
    environment: str = Field(default="development", alias="LOGISTICS_ENVIRONMENT")
    api_host: str = Field(default="0.0.0.0", alias="LOGISTICS_API_HOST")
    api_port: int = Field(default=8000, alias="LOGISTICS_API_PORT")
    dashboard_port: int = Field(default=8501, alias="LOGISTICS_DASHBOARD_PORT")
    database_url: str = Field(default="sqlite:///data/logistics.db", alias="LOGISTICS_DATABASE_URL")
    log_level: str = Field(default="INFO", alias="LOGISTICS_LOG_LEVEL")
    model_dir: Path = Field(default=ROOT_DIR / "models", alias="LOGISTICS_MODEL_DIR")
    log_dir: Path = Field(default=ROOT_DIR / "logs", alias="LOGISTICS_LOG_DIR")
    sample_data_path: Path = Field(
        default=ROOT_DIR / "data" / "sample" / "nyc_taxi_sample.csv",
        alias="LOGISTICS_SAMPLE_DATA_PATH",
    )
    forecast_model_name: str = "demand_transformer"
    default_sequence_length: int = 4
    default_forecast_horizon: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        protected_namespaces=(),
    )

    def resolved_database_url(self) -> str:
        if self.database_url.startswith("sqlite:///"):
            raw_path = self.database_url.removeprefix("sqlite:///")
            return f"sqlite:///{(ROOT_DIR / raw_path).resolve()}"
        return self.database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings
