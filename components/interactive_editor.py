from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "interactive_editor_frontend"
_interactive_editor = components.declare_component(
    "interactive_editor_v2",
    path=str(_FRONTEND_DIR),
)


def interactive_editor(
    text: str = "",
    file_path: str = "main.tex",
    height: int = 560,
    highlight_start: int = -1,
    highlight_end: int = -1,
    selection_status: str = "none",
    confirmed_preview: str = "",
    focus_start: int = -1,
    focus_end: int = -1,
    focus_event_id: int = 0,
    backend_sync_event_id: int = 0,
    key: str | None = None,
) -> Dict[str, Any]:
    """Render an interactive editor component with native selection capture.

    Returns a payload:
    {
      "text": str,
      "selection": str,
      "selection_start": int,
      "selection_end": int,
      "trigger": str,
      "confirm_action": str
    }
    """

    default_payload: Dict[str, Any] = {
        "text": text,
        "file_path": file_path,
        "selection": "",
        "selection_start": -1,
        "selection_end": -1,
        "trigger": "init",
        "confirm_action": "",
        "focus_applied": False,
    }
    result = _interactive_editor(
        text=text,
        file_path=file_path,
        height=height,
        highlight_start=highlight_start,
        highlight_end=highlight_end,
        selection_status=selection_status,
        confirmed_preview=confirmed_preview,
        focus_start=focus_start,
        focus_end=focus_end,
        focus_event_id=focus_event_id,
        backend_sync_event_id=backend_sync_event_id,
        default=default_payload,
        key=key,
    )
    if result is None:
        return default_payload
    if isinstance(result, dict):
        return {
            "text": str(result.get("text", text)),
            "file_path": str(result.get("file_path", file_path)),
            "selection": str(result.get("selection", "")),
            "selection_start": int(result.get("selection_start", -1)),
            "selection_end": int(result.get("selection_end", -1)),
            "trigger": str(result.get("trigger", "update")),
            "confirm_action": str(result.get("confirm_action", "")),
            "focus_applied": bool(result.get("focus_applied", False)),
        }
    return default_payload
