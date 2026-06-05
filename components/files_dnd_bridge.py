from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "files_dnd_bridge_frontend"
_files_dnd_bridge = components.declare_component(
    "files_dnd_bridge_v1",
    path=str(_FRONTEND_DIR),
)


def files_dnd_bridge(
    panel_mode: str = "",
    doc_id: str = "",
    nodes: List[Dict[str, str]] | None = None,
    key: str | None = None,
) -> Dict[str, Any]:
    default_payload: Dict[str, Any] = {
        "action": "idle",
        "payload": {},
        "event_id": 0,
    }
    result = _files_dnd_bridge(
        panel_mode=panel_mode,
        doc_id=doc_id,
        nodes=nodes or [],
        default=default_payload,
        key=key,
    )
    if result is None:
        return default_payload
    if isinstance(result, dict):
        payload = result.get("payload", {})
        return {
            "action": str(result.get("action", "idle")),
            "payload": payload if isinstance(payload, dict) else {},
            "event_id": int(result.get("event_id", 0) or 0),
        }
    return default_payload
