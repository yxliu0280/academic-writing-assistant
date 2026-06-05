from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from core.schemas import AppState


class FileCentricStateManager:
    """Inspired by InfiAgent: file-centric state externalization and snapshot history.

    Design goal:
    - keep long-horizon state on disk instead of expanding LLM prompt context
    - preserve deterministic undo via Python history snapshots (not model-inferred undo)
    """

    def __init__(self, base_dir: str = ".state", max_history_items: int = 200) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_history_items = max_history_items

    def persist_state(self, doc_id: str, state: AppState, request_id: str = "") -> Path:
        payload = {
            "doc_id": doc_id,
            "request_id": request_id,
            "updated_at": time.time(),
            "state": self._serialize_state(state),
        }
        path = self._state_path(doc_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_state(self, doc_id: str) -> Dict[str, Any] | None:
        path = self._state_path(doc_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def append_history_snapshot(self, doc_id: str, text: str, request_id: str = "") -> None:
        path = self._history_path(doc_id)
        history = self.load_history_snapshots(doc_id)
        history.append(
            {
                "request_id": request_id,
                "timestamp": time.time(),
                "text": text,
            }
        )
        if len(history) > self.max_history_items:
            history = history[-self.max_history_items :]
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_history_snapshots(self, doc_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
        path = self._history_path(doc_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if limit is None or limit <= 0:
            return list(data)
        return list(data)[-limit:]

    def pop_history_snapshot(self, doc_id: str) -> Dict[str, Any] | None:
        path = self._history_path(doc_id)
        history = self.load_history_snapshots(doc_id)
        if not history:
            return None
        item = history.pop()
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        return item

    def _state_path(self, doc_id: str) -> Path:
        return self.base_dir / f"{doc_id}.state.json"

    def _history_path(self, doc_id: str) -> Path:
        return self.base_dir / f"{doc_id}.history.json"

    def _serialize_state(self, state: AppState) -> Dict[str, Any]:
        # Keep state reviewable and stable; strip binary payloads from persisted snapshot.
        data = asdict(state)

        figures = data.get("uploaded_figures", [])
        for fig in figures:
            content = fig.get("content", b"")
            size = len(content) if isinstance(content, (bytes, bytearray)) else 0
            fig["content"] = f"<bytes:{size}>"

        return data
