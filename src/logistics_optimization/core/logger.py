import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


PROJECT_METADATA_FIELDS = (
    "request_id",
    "model_name",
    "epoch",
    "training_loss",
    "validation_mae",
    "rmse",
    "inference_latency_ms",
    "graph_nodes",
    "route_distance",
    "avg_travel_time",
    "optimization_status",
    "exception_trace",
)


class LogisticsFormatter(logging.Formatter):
    """Formatter tailored to ML and routing observability for the project."""

    def format(self, record: logging.LogRecord) -> str:
        record.asctime = self.formatTime(record, self.datefmt)
        metadata = []
        for field in PROJECT_METADATA_FIELDS:
            value = getattr(record, field, None)
            if value not in (None, "", []):
                metadata.append(f"{field}={value}")

        message = record.getMessage()
        base = f"{record.asctime} | {record.levelname:<7} | {record.name} | {message}"
        return f"{base} | {'; '.join(metadata)}" if metadata else base


class LogisticsLogger:
    """Project-specific logger wrapper with console and rotating file output."""

    def __init__(self, name: str, level: str, log_dir: Path) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.upper())
        self._logger.propagate = False

        if not self._logger.handlers:
            formatter = LogisticsFormatter(datefmt="%Y-%m-%d %H:%M:%S")

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            file_handler = RotatingFileHandler(
                log_dir / "logistics_system.log",
                maxBytes=1_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)

            self._logger.addHandler(console_handler)
            self._logger.addHandler(file_handler)

    def _log(self, level: int, message: str, **metadata: Any) -> None:
        filtered_metadata = {key: value for key, value in metadata.items() if key in PROJECT_METADATA_FIELDS}
        self._logger.log(level, message, extra=filtered_metadata)

    def debug(self, message: str, **metadata: Any) -> None:
        self._log(logging.DEBUG, message, **metadata)

    def info(self, message: str, **metadata: Any) -> None:
        self._log(logging.INFO, message, **metadata)

    def warning(self, message: str, **metadata: Any) -> None:
        self._log(logging.WARNING, message, **metadata)

    def error(self, message: str, **metadata: Any) -> None:
        self._log(logging.ERROR, message, **metadata)

    def exception(self, message: str, **metadata: Any) -> None:
        self._log(logging.ERROR, message, **metadata)

    def training_event(
        self,
        message: str,
        *,
        model_name: str,
        epoch: int,
        training_loss: float,
        validation_mae: float | None = None,
        rmse: float | None = None,
    ) -> None:
        self.info(
            message,
            model_name=model_name,
            epoch=epoch,
            training_loss=round(training_loss, 4),
            validation_mae=round(validation_mae, 4) if validation_mae is not None else None,
            rmse=round(rmse, 4) if rmse is not None else None,
        )

    def optimization_event(
        self,
        message: str,
        *,
        graph_nodes: int,
        route_distance: float,
        avg_travel_time: float,
        optimization_status: str,
        request_id: str | None = None,
    ) -> None:
        self.info(
            message,
            request_id=request_id,
            graph_nodes=graph_nodes,
            route_distance=round(route_distance, 3),
            avg_travel_time=round(avg_travel_time, 3),
            optimization_status=optimization_status,
        )


def get_logger(name: str, *, level: str, log_dir: Path) -> LogisticsLogger:
    return LogisticsLogger(name=name, level=level, log_dir=log_dir)

