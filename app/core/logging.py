"""日志配置：统一格式，并保证每条日志都带 trace_id 字段。"""

from __future__ import annotations

import logging
import logging.config
from typing import Any


class TraceIdFilter(logging.Filter):
    """为缺少 trace_id 的日志补齐占位符，避免格式化失败。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        return True


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | trace=%(trace_id)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """配置根日志器（幂等，可重复调用）。"""
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"trace_id": {"()": TraceIdFilter}},
        "formatters": {
            "standard": {"format": LOG_FORMAT, "datefmt": "%Y-%m-%d %H:%M:%S"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["trace_id"],
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"handlers": ["console"], "level": level.upper()},
        "loggers": {
            "uvicorn": {"handlers": ["console"], "level": level.upper(), "propagate": False},
            "uvicorn.error": {"handlers": ["console"], "level": level.upper(), "propagate": False},
            "uvicorn.access": {"handlers": ["console"], "level": level.upper(), "propagate": False},
        },
    }
    logging.config.dictConfig(config)

