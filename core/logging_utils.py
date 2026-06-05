from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def debug_enabled() -> bool:
    return os.getenv("DEBUG", "0") == "1"


def truncate_for_log(value: str, max_len: int = 800) -> str:
    if len(value) <= max_len:
        return value
    return value[:max_len] + "...<truncated>"


def _sanitize_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for k, v in data.items():
        try:
            json.dumps(v)
            sanitized[k] = v
        except TypeError:
            sanitized[k] = str(v)
    return sanitized


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    payload = {"event": event, **_sanitize_for_json(fields)}
    logger.log(level, json.dumps(payload, ensure_ascii=False))
