from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

from agents.role_negotiation_agent import RoleNegotiationAgent
from agents.writing_support_agent import WritingSupportAgent
from core.logging_utils import get_logger, log_event
from core.schemas import AppState, ChatTurn
from tools.patch_utils import (
    ReplaceOperation,
    apply_replace_operations,
    compute_edit_ratio,
    count_modified_sentences,
    patch_guardrail_ok,
    unified_diff,
)


class PatchManager:
    """Owns patch proposal/apply/negotiation lifecycle outside checks orchestration."""

    def __init__(self, provider_name: str = "none") -> None:
        self.writing_support_agent = WritingSupportAgent(provider_name)
        self.role_negotiation_agent = RoleNegotiationAgent(provider_name)
        self.logger = get_logger(__name__)
        self._last_patch_operations: List[ReplaceOperation] = []
        self._last_patch_diff: str = ""

    def clear_compat_patch(self) -> None:
        self._last_patch_operations = []
        self._last_patch_diff = ""

    def sync_compat_patch(
        self,
        patch_diff: str,
        patch_operations: Sequence[Dict[str, Any]] | Sequence[ReplaceOperation],
    ) -> None:
        self._last_patch_diff = str(patch_diff or "")
        ops: List[ReplaceOperation] = []
        for item in patch_operations:
            if isinstance(item, ReplaceOperation):
                ops.append(item)
            else:
                ops.append(
                    ReplaceOperation(
                        start=int(item.get("start", 0)),
                        end=int(item.get("end", 0)),
                        replacement=str(item.get("replacement", "")),
                        reason=str(item.get("reason", "editor_patch")),
                    )
                )
        self._last_patch_operations = ops

    def apply_patch(self, state: AppState) -> Tuple[AppState, str]:
        if state.role != "Editor":
            return state, "Patch application is only available in Editor role."

        if state.pending_patch.status == "rejected":
            return state, "Patch was rejected in negotiation. Re-run Analyze for a new patch."

        ops = self._resolve_patch_operations(state)
        if not ops:
            return state, "No patch available from the latest analysis."

        state.history.append(state.current_text)
        new_text = apply_replace_operations(state.current_text, ops)
        state.current_text = new_text
        if state.negotiation_state.active:
            state.negotiation_state.active = False
            state.negotiation_state.latest_agent_message = (
                "Patch was applied by user action. Negotiation closed."
            )
        if state.pending_patch.status == "pending":
            state.pending_patch.status = "accepted"
        state.pending_patch.stage = "applied"
        return state, "Patch applied successfully."

    def negotiate_patch(self, state: AppState, feedback: str) -> Tuple[AppState, str]:
        reply, refined_ops = self.role_negotiation_agent.handle_user_feedback(
            state=state,
            feedback=feedback,
            current_text=state.current_text,
            request_id=state.request_id,
        )
        if refined_ops:
            self._last_patch_operations = [
                ReplaceOperation(
                    start=int(op.get("start", 0)),
                    end=int(op.get("end", 0)),
                    replacement=str(op.get("replacement", "")),
                    reason=str(op.get("reason", "editor_patch")),
                )
                for op in refined_ops
            ]
            self._last_patch_diff = str(state.pending_patch.patch_diff or "")
        elif state.pending_patch.status == "rejected":
            self.clear_compat_patch()
        return state, reply

    def propose_writing_patch(self, state: AppState, instruction: str) -> Tuple[AppState, str]:
        if state.role != "Editor":
            return state, "Writing patch generation is only available in Editor mode."

        op = self.writing_support_agent.build_rewrite_operation(
            state=state,
            instruction=instruction,
            request_id=state.request_id,
        )
        if not op:
            return (
                state,
                "I could not generate a rewrite patch from this request. "
                "Please confirm a span and try again.",
            )

        patched = apply_replace_operations(state.current_text, [op])
        patch_diff = unified_diff(state.current_text, patched)
        edit_ratio = compute_edit_ratio(state.current_text, patched)
        sentence_changes = float(count_modified_sentences(state.current_text, patched))
        ok, guardrail = patch_guardrail_ok(
            state.current_text,
            patched,
            max_edit_ratio=state.config.max_patch_edit_ratio,
            max_sentence_changes=state.config.max_patch_sentence_changes,
        )
        if not ok:
            log_event(
                self.logger,
                logging.WARNING,
                "writing_patch_guardrail_exceeded",
                request_id=state.request_id,
                guardrail=guardrail,
            )

        self._last_patch_operations = [op]
        self._last_patch_diff = patch_diff
        state.pending_patch.status = "pending"
        state.pending_patch.stage = "awaiting_apply_consent"
        state.pending_patch.patch_diff = patch_diff
        state.pending_patch.operations = [op.to_dict()]
        state.pending_patch.reason = "writing_assistance"
        state.pending_patch.edit_ratio = edit_ratio
        state.pending_patch.source = "writing_assistance"
        state.pending_patch.summary = "Editor-generated writing patch candidate."

        style_protection_active = self.role_negotiation_agent.maybe_activate_style_protection(
            state=state,
            patch_diff=patch_diff,
            patch_operations=[op],
            edit_ratio=edit_ratio,
            sentence_changes=sentence_changes,
            request_id=state.request_id,
        )
        if style_protection_active:
            return (
                state,
                "I prepared a major rewrite patch (>30%). "
                "Please review and tell me whether to keep more of your original style.",
            )
        return (
            state,
            "I prepared an academic rewrite patch for the selected span. "
            "Review the diff and click Apply if acceptable.",
        )

    def recent_chat_turns(self, state: AppState, max_turns: int = 8) -> List[ChatTurn]:
        return self.role_negotiation_agent.snapshot_chat_window(state, max_turns=max_turns)

    def latest_patch_diff(self) -> str:
        return str(self._last_patch_diff)

    def latest_patch_operations(self) -> List[ReplaceOperation]:
        return list(self._last_patch_operations)

    def _resolve_patch_operations(self, state: AppState) -> List[ReplaceOperation]:
        if state.pending_patch.operations:
            return [
                ReplaceOperation(
                    start=int(op.get("start", 0)),
                    end=int(op.get("end", 0)),
                    replacement=str(op.get("replacement", "")),
                    reason=str(op.get("reason", "editor_patch")),
                )
                for op in state.pending_patch.operations
            ]
        return list(self._last_patch_operations)
