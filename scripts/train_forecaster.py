from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from logistics_optimization.core.config import get_settings
from logistics_optimization.core.logger import get_logger
from logistics_optimization.ml.forecasting.predictor import DemandForecastPipeline


if __name__ == "__main__":
    settings = get_settings()
    logger = get_logger("logistics.train", level=settings.log_level, log_dir=settings.log_dir)
    pipeline = DemandForecastPipeline(logger=logger)
    try:
        result = pipeline.train(
            dataset_path=str(settings.sample_data_path),
            sequence_length=settings.default_sequence_length,
            epochs=8,
        )
        print(result)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)
