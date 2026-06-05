from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "pdf_viewer_frontend"
_pdf_viewer = components.declare_component(
    "pdf_viewer_v2",
    path=str(_FRONTEND_DIR),
)


def pdf_viewer(
    pdf_base64: str = "",
    preview_png_base64: str = "",
    preview_png_dpi: int = 300,
    has_compiled: bool = False,
    compile_ok: bool = False,
    error_title: str = "",
    error_items: List[Dict[str, Any]] | None = None,
    error_log: str = "",
    height: int = 560,
    key: str | None = None,
) -> Dict[str, Any]:
    """Render PDF preview with pdf.js and SyncTeX click events."""

    default_payload: Dict[str, Any] = {
        "action": "idle",
        "page": 0,
        "x": 0.0,
        "y": 0.0,
        "event_id": 0,
    }
    result = _pdf_viewer(
        pdf_base64=pdf_base64,
        preview_png_base64=preview_png_base64,
        preview_png_dpi=preview_png_dpi,
        has_compiled=has_compiled,
        compile_ok=compile_ok,
        error_title=error_title,
        error_items=error_items or [],
        error_log=error_log,
        height=height,
        default=default_payload,
        key=key,
    )
    if result is None:
        return default_payload
    if isinstance(result, dict):
        try:
            page = int(result.get("page", 0))
        except (TypeError, ValueError):
            page = 0
        try:
            x = float(result.get("x", 0.0))
        except (TypeError, ValueError):
            x = 0.0
        try:
            y = float(result.get("y", 0.0))
        except (TypeError, ValueError):
            y = 0.0
        try:
            event_id = int(result.get("event_id", 0))
        except (TypeError, ValueError):
            event_id = 0
        return {
            "action": str(result.get("action", "idle")),
            "page": page,
            "x": x,
            "y": y,
            "event_id": event_id,
        }
    return default_payload
