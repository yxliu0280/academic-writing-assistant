from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "viewport_probe_frontend"
_viewport_probe = components.declare_component(
    "viewport_probe_v1",
    path=str(_FRONTEND_DIR),
)


def viewport_probe(key: str | None = None) -> Dict[str, int]:
    """Return viewport metrics for adaptive workspace sizing."""

    default_payload: Dict[str, Any] = {"height": 0, "available_height": 0, "event_id": 0}
    result = _viewport_probe(default=default_payload, key=key)
    if not isinstance(result, dict):
        return {"height": 0, "available_height": 0}
    try:
        return {
            "height": max(0, int(result.get("height", 0))),
            "available_height": max(0, int(result.get("available_height", 0))),
        }
    except (TypeError, ValueError):
        return {"height": 0, "available_height": 0}
