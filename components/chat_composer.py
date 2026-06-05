from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit.components.v1 as components

_FRONTEND_DIR = Path(__file__).resolve().parent / "chat_composer_frontend"
_chat_composer = components.declare_component(
    "chat_composer_v5",
    path=str(_FRONTEND_DIR),
)


def chat_composer(
    draft: str = "",
    disabled: bool = False,
    tools_open: bool = False,
    selected_checks: list[str] | None = None,
    check_enabled: dict[str, bool] | None = None,
    placeholder: str = "Discuss findings, ask for revisions, or request grounded papers.",
    key: str | None = None,
) -> Dict[str, Any]:
    """Render a GPT-like composer with inline + and send controls."""

    default_payload: Dict[str, Any] = {
        "action": "idle",
        "draft": draft,
        "submit_text": "",
        "event_id": 0,
        "selected_checks": selected_checks or ["table", "figure", "citation", "terminology"],
        "tools_open": tools_open,
        "check_enabled": dict(check_enabled or {}),
    }
    result = _chat_composer(
        draft=draft,
        disabled=disabled,
        tools_open=tools_open,
        selected_checks=selected_checks or ["table", "figure", "citation", "terminology"],
        check_enabled=dict(check_enabled or {}),
        placeholder=placeholder,
        default=default_payload,
        key=key,
    )
    if result is None:
        return default_payload
    if isinstance(result, dict):
        return {
            "action": str(result.get("action", "idle")),
            "draft": str(result.get("draft", draft)),
            "submit_text": str(result.get("submit_text", "")),
            "event_id": int(result.get("event_id", 0)),
            "selected_checks": list(result.get("selected_checks", selected_checks or [])),
            "tools_open": bool(result.get("tools_open", tools_open)),
        }
    return default_payload
