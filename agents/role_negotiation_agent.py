from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Sequence, Tuple

from core.logging_utils import debug_enabled, get_logger, log_event, truncate_for_log
from core.schemas import AppState, ChatTurn, PatchProposal
from roles.role_prompt_bank import RolePromptBank
from tools.patch_utils import ReplaceOperation, apply_replace_operations, unified_diff
from tools.providers import LLMProviderError, TextLLMProvider, build_text_provider

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return kwargs.get("default", None)


class NegotiationReply(BaseModel):
    action: Literal["accept", "revise", "reject"] = "revise"
    assistant_message: str
    preserve_ratio: float = Field(default=0.7, ge=0.0, le=1.0)


class RoleNegotiationAgent:
    """Inspired by ResearchAgent and DeepReview: iterative role negotiation policy.

    This module is UI-agnostic. It manages negotiation state and patch refinement
    while keeping author approval in the loop.
    """

    name = "role_negotiation"

    def __init__(self, provider_name: str = "none") -> None:
        self.provider: TextLLMProvider = build_text_provider(provider_name)
        self.logger = get_logger(__name__)

    def maybe_activate_style_protection(
        self,
        state: AppState,
        patch_diff: str,
        patch_operations: Sequence[Any],
        edit_ratio: float,
        sentence_changes: float,
        request_id: str = "",
    ) -> bool:
        if state.role != "Editor" or not patch_operations:
            return False

        exceeded = (
            edit_ratio > state.config.style_protection_edit_ratio
            or sentence_changes > float(state.config.style_protection_sentence_changes)
        )
        if not exceeded:
            return False

        ops_payload = self._serialize_patch_ops(patch_operations)
        state.pending_patch = PatchProposal(
            status="pending",
            stage="negotiation",
            patch_diff=patch_diff,
            operations=ops_payload,
            reason="style_protection_threshold_exceeded",
            edit_ratio=edit_ratio,
            source="consistency_analysis",
            summary="Patch candidate requires style-protection negotiation before apply.",
        )

        message = RolePromptBank.style_protection_message(edit_ratio, sentence_changes)
        state.negotiation_state.active = True
        state.negotiation_state.mode = "style_protection"
        state.negotiation_state.round_index = 0
        state.negotiation_state.max_rounds = max(1, state.config.negotiation_max_rounds)
        state.negotiation_state.latest_user_feedback = ""
        state.negotiation_state.latest_agent_message = message
        state.chat_history.append(
            ChatTurn(
                role="assistant",
                content=message,
                request_id=request_id,
                metadata={
                    "agent": self.name,
                    "mode": "style_protection",
                    "edit_ratio": round(edit_ratio, 6),
                    "sentence_changes": sentence_changes,
                },
            )
        )
        log_event(
            self.logger,
            logging.INFO,
            "style_protection_activated",
            request_id=request_id,
            edit_ratio=edit_ratio,
            sentence_changes=sentence_changes,
            threshold_edit_ratio=state.config.style_protection_edit_ratio,
            threshold_sentence_changes=state.config.style_protection_sentence_changes,
            operation_count=len(ops_payload),
        )
        return True

    def handle_user_feedback(
        self,
        state: AppState,
        feedback: str,
        current_text: str,
        request_id: str = "",
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not state.negotiation_state.active:
            return "Negotiation is not active.", []
        clean_feedback = feedback.strip()
        if not clean_feedback:
            return "Feedback is empty.", []

        state.chat_history.append(
            ChatTurn(
                role="user",
                content=clean_feedback,
                request_id=request_id,
                metadata={"agent": self.name},
            )
        )
        state.negotiation_state.latest_user_feedback = clean_feedback

        action = self._infer_feedback_action(clean_feedback)
        if action == "accept":
            msg = self._close_negotiation(
                state,
                accepted=True,
                request_id=request_id,
                message="Patch accepted. You can apply it now.",
            )
            return msg, list(state.pending_patch.operations)

        if action == "reject":
            msg = self._close_negotiation(
                state,
                accepted=False,
                request_id=request_id,
                message="Patch rejected. I will stay in advisory mode and avoid automatic rewrites.",
            )
            return msg, []

        state.negotiation_state.round_index += 1
        if state.negotiation_state.round_index >= state.negotiation_state.max_rounds:
            msg = self._close_negotiation(
                state,
                accepted=False,
                request_id=request_id,
                message="Negotiation round limit reached. Please run Analyze again for a fresh patch.",
            )
            return msg, []

        refined_ops = self._refine_pending_patch_operations(
            state.pending_patch.operations,
            clean_feedback,
        )
        state.pending_patch.operations = refined_ops
        state.pending_patch.patch_diff = self._build_patch_diff(current_text, refined_ops)
        state.pending_patch.status = "pending"
        state.pending_patch.stage = "negotiation"

        reply = self._build_refinement_reply(state, clean_feedback, request_id=request_id)
        state.negotiation_state.latest_agent_message = reply
        state.chat_history.append(
            ChatTurn(
                role="assistant",
                content=reply,
                request_id=request_id,
                metadata={
                    "agent": self.name,
                    "mode": state.negotiation_state.mode,
                    "round_index": state.negotiation_state.round_index,
                },
            )
        )
        return reply, refined_ops

    @staticmethod
    def snapshot_chat_window(state: AppState, max_turns: int = 8) -> List[ChatTurn]:
        if max_turns <= 0:
            return []
        return list(state.chat_history)[-max_turns:]

    def _build_refinement_reply(
        self,
        state: AppState,
        feedback: str,
        request_id: str = "",
    ) -> str:
        fallback = (
            "I regenerated the patch with stronger style preservation. "
            "If needed, ask for a more formal tone, shorter sentences, or fewer edits."
        )
        if not self.provider.enabled():
            return fallback

        memory_payload = self._memory_context_payload(state)
        messages = [
            {
                "role": "system",
                "content": RolePromptBank.system_prompt(state.role, mode="style_protection"),
            },
            {
                "role": "user",
                "content": (
                    "You are refining an editor patch through negotiation.\n"
                    "Use the memory payload for continuity.\n"
                    "Other roles' private notes are unavailable by design.\n"
                    f"Memory payload (JSON): {json.dumps(memory_payload, ensure_ascii=False)}\n"
                    f"User feedback: {feedback}\n"
                    f"Round: {state.negotiation_state.round_index}/{state.negotiation_state.max_rounds}\n"
                    "Return JSON with action and assistant_message. "
                    "Do not fabricate evidence or citations."
                ),
            },
        ]
        if debug_enabled():
            log_event(
                self.logger,
                logging.DEBUG,
                "negotiation_llm_messages",
                request_id=request_id,
                messages=[
                    {
                        "role": m.get("role", ""),
                        "content": truncate_for_log(str(m.get("content", "")), max_len=900),
                    }
                    for m in messages
                ],
            )

        try:
            payload = self.provider.complete_json(
                schema=NegotiationReply,
                messages=messages,
                request_id=request_id,
            )
            reply = str(payload.get("assistant_message", "")).strip()
            if reply:
                log_event(
                    self.logger,
                    logging.INFO,
                    "negotiation_llm_reply",
                    request_id=request_id,
                    action=payload.get("action", "revise"),
                )
                return reply
        except LLMProviderError as exc:
            log_event(
                self.logger,
                logging.WARNING,
                "negotiation_llm_fallback",
                request_id=request_id,
                error_type=exc.error_type,
                message=str(exc),
            )
        except Exception as exc:  # pragma: no cover
            log_event(
                self.logger,
                logging.WARNING,
                "negotiation_llm_fallback",
                request_id=request_id,
                error_type="unexpected_error",
                message=str(exc),
            )
        return fallback

    @staticmethod
    def _memory_context_payload(state: AppState) -> Dict[str, Any]:
        metadata = state.grounded_context.metadata if state.grounded_context else {}
        memory_context = metadata.get("memory_context", {}) if isinstance(metadata, dict) else {}
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
    def _infer_feedback_action(feedback: str) -> Literal["accept", "revise", "reject"]:
        text = feedback.lower()
        accept_keywords = [
            "accept",
            "apply",
            "looks good",
            "approved",
            "yes",
        ]
        reject_keywords = [
            "reject",
            "cancel",
            "too much",
            "rollback",
            "do not apply",
        ]
        if any(key in text for key in reject_keywords):
            return "reject"
        if any(key in text for key in accept_keywords):
            return "accept"
        return "revise"

    def _close_negotiation(
        self,
        state: AppState,
        accepted: bool,
        request_id: str,
        message: str,
    ) -> str:
        state.negotiation_state.active = False
        state.negotiation_state.latest_agent_message = message
        state.pending_patch.status = "accepted" if accepted else "rejected"
        state.pending_patch.stage = "awaiting_apply_consent" if accepted else "rejected"
        if not accepted:
            state.pending_patch.patch_diff = ""
            state.pending_patch.operations = []
        state.chat_history.append(
            ChatTurn(
                role="assistant",
                content=message,
                request_id=request_id,
                metadata={"agent": self.name, "closed": True, "accepted": accepted},
            )
        )
        log_event(
            self.logger,
            logging.INFO,
            "negotiation_closed",
            request_id=request_id,
            accepted=accepted,
        )
        return message

    def _refine_pending_patch_operations(
        self,
        operations: List[Dict[str, Any]],
        feedback: str,
    ) -> List[Dict[str, Any]]:
        if not operations:
            return []

        text = feedback.lower()
        refined = list(operations)

        # Preserve more original content if user asks to keep style.
        preserve_markers = [
            "preserve",
            "keep original",
            "lighter edit",
            "minimal",
            "less change",
            "keep my style",
        ]
        if any(marker in text for marker in preserve_markers) and len(refined) > 1:
            refined = refined[: max(1, len(refined) // 2)]

        # Formal tone request is represented in operation reasons for traceability.
        if "formal" in text or "academic" in text:
            for op in refined:
                reason = str(op.get("reason", "editor_patch"))
                if "formalized" not in reason:
                    op["reason"] = f"{reason}_formalized"
        return refined

    @staticmethod
    def _build_patch_diff(current_text: str, operations: List[Dict[str, Any]]) -> str:
        replace_ops: List[ReplaceOperation] = []
        for op in operations:
            replace_ops.append(
                ReplaceOperation(
                    start=int(op.get("start", 0)),
                    end=int(op.get("end", 0)),
                    replacement=str(op.get("replacement", "")),
                    reason=str(op.get("reason", "editor_patch")),
                )
            )
        modified = apply_replace_operations(current_text, replace_ops)
        return unified_diff(current_text, modified)

    @staticmethod
    def _serialize_patch_ops(patch_operations: Sequence[Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for op in patch_operations:
            if hasattr(op, "to_dict"):
                raw = op.to_dict()
                if isinstance(raw, dict):
                    result.append(raw)
                continue
            if isinstance(op, dict):
                result.append(dict(op))
        return result
