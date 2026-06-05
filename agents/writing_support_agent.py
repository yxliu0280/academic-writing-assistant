from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from agents.base import BaseAgent
from core.logging_utils import get_logger, log_event
from core.schemas import AppState, Issue
from tools.patch_utils import ReplaceOperation
from tools.providers import TextLLMProvider, build_text_provider
from tools.text_utils import split_sentences_with_offsets

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]

    def Field(*args: object, **kwargs: object) -> object:  # type: ignore[misc]
        return kwargs.get("default", None)


class WritingRewriteResponse(BaseModel):
    rewritten_text: str = Field(default="")
    rationale: str = Field(default="")


class WritingSupportAgent(BaseAgent):
    name = "writing_support"

    def __init__(self, provider_name: str = "none") -> None:
        self.provider: TextLLMProvider = build_text_provider(provider_name)
        self.logger = get_logger(__name__)

    def run(self, state: AppState) -> List[Issue]:
        # Out of main evaluation scope; returns no consistency issues.
        return []

    def diagnose(self, state: AppState) -> List[Dict[str, str]]:
        diagnostics: List[Dict[str, str]] = []
        for sent in split_sentences_with_offsets(state.current_text):
            text = str(sent["text"])
            word_count = len(text.split())
            if word_count > 35:
                diagnostics.append(
                    {
                        "type": "long_sentence",
                        "snippet": text,
                        "message": "Sentence is long; consider splitting for readability.",
                    }
                )
            if self.provider.enabled() and word_count > 18:
                suggestion = self.provider.suggest_rewrite(text, {})
                if suggestion:
                    diagnostics.append(
                        {
                            "type": "rewrite_hint",
                            "snippet": text,
                            "message": suggestion,
                        }
                    )
        return diagnostics

    def build_rewrite_operation(
        self,
        state: AppState,
        instruction: str,
        request_id: str = "",
    ) -> ReplaceOperation | None:
        start, end = self._target_span(state)
        if end <= start:
            return None

        original = state.current_text[start:end]
        rewritten = self._rewrite_with_llm(
            state=state,
            text=original,
            instruction=instruction,
            request_id=request_id,
        )
        rewrite_source = "llm"
        if not rewritten.strip():
            rewritten = self._rule_based_academic_rewrite(original)
            rewrite_source = "rule_fallback"

        if not rewritten.strip():
            return None
        if rewritten == original:
            return None

        log_event(
            self.logger,
            logging.INFO,
            "writing_rewrite_generated",
            request_id=request_id,
            source=rewrite_source,
            span_start=start,
            span_end=end,
            original_len=len(original),
            rewritten_len=len(rewritten),
        )
        return ReplaceOperation(
            start=start,
            end=end,
            replacement=rewritten,
            reason="writing_assist_rewrite",
        )

    def _target_span(self, state: AppState) -> Tuple[int, int]:
        selection = state.active_selection
        if (
            selection
            and bool(selection.metadata.get("confirmed", False))
            and selection.end > selection.start
        ):
            return int(selection.start), int(selection.end)
        return 0, len(state.current_text)

    def _rewrite_with_llm(
        self,
        state: AppState,
        text: str,
        instruction: str,
        request_id: str = "",
    ) -> str:
        if not self.provider.enabled() or not hasattr(WritingRewriteResponse, "model_validate"):
            return ""

        memory_payload = self._memory_context_payload(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic writing assistant. Rewrite text to be formal and clear. "
                    "Preserve LaTeX commands and structure. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rewrite the following selected manuscript span based on the instruction.\n"
                    "Instruction:\n"
                    f"{instruction}\n\n"
                    "Memory payload (JSON):\n"
                    f"{json.dumps(memory_payload, ensure_ascii=False)}\n\n"
                    "Output schema:\n"
                    "{\n"
                    '  "rewritten_text": string,\n'
                    '  "rationale": string\n'
                    "}\n\n"
                    "Selected span:\n"
                    f"{text}"
                ),
            },
        ]
        try:
            payload = self.provider.complete_json(
                schema=WritingRewriteResponse,
                messages=messages,
                request_id=request_id,
            )
        except Exception as exc:
            log_event(
                self.logger,
                logging.WARNING,
                "writing_rewrite_llm_failed",
                request_id=request_id,
                error=str(exc),
            )
            return ""
        return str(payload.get("rewritten_text", "")).strip()

    @staticmethod
    def _memory_context_payload(state: AppState) -> Dict[str, Any]:
        metadata = state.grounded_context.metadata if state.grounded_context else {}
        if not isinstance(metadata, dict):
            return {}
        memory_context = metadata.get("memory_context", {})
        if not isinstance(memory_context, dict):
            return {}
        shared_memory = memory_context.get("shared_memory", {})
        role_memory = memory_context.get("role_memory", {})
        other_published = memory_context.get("other_roles_published_conclusions", {})
        recent_history = memory_context.get("recent_history", [])
        return {
            "shared_memory": shared_memory if isinstance(shared_memory, dict) else {},
            "role_memory": role_memory if isinstance(role_memory, dict) else {},
            "other_roles_published_conclusions": (
                other_published if isinstance(other_published, dict) else {}
            ),
            "recent_history": recent_history[-8:] if isinstance(recent_history, list) else [],
        }

    @staticmethod
    def _rule_based_academic_rewrite(text: str) -> str:
        # Keep LaTeX command lines untouched; only polish narrative lines.
        lines = text.splitlines()
        polished: List[str] = []
        replacements = [
            (r"\bi\b", "I"),
            (r"\bit do not\b", "it does not"),
            (r"\bwe was\b", "we were"),
            (r"\bchat message\b", "informal message"),
            (r"\bmaybe it works maybe not\b", "its effectiveness remains uncertain"),
            (r"\banyway\b", "in addition"),
            (r"\bvery bad\b", "poorly written"),
        ]

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                polished.append(raw_line)
                continue
            if stripped.startswith("\\"):
                polished.append(raw_line)
                continue

            line = raw_line
            for pattern, replacement in replacements:
                line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
            line = re.sub(r"\s+", " ", line).strip()
            if line and line[-1] not in {".", "!", "?"}:
                line += "."
            if line:
                line = line[0].upper() + line[1:]
            polished.append(line)

        return "\n".join(polished)
