from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List

from roles.role_prompt_bank import RolePromptBank
from core.schemas import CheckerResultPayload, RolePayload, RoleToolSummary, SystemExceptionPayload, ToolFact
from tools.providers import TextLLMProvider, build_text_provider

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]

    def Field(*args: object, **kwargs: object) -> object:  # type: ignore[misc]
        return kwargs.get("default", None)


class RoleChatResponse(BaseModel):
    assistant_message: str = Field(default="")


class RoleChatAgent:
    _ADVISOR_EXAMPLE_LABEL = "Example phrasing:"
    _ADVISOR_MAX_EXAMPLE_WORDS = 30
    _ADVISOR_MAX_EXAMPLE_CHARS = 180
    _EXAMPLE_LABEL_REGEX = r"(?:example phrasing:|example:|for example:|示例表达[:：]?|示例[:：])"
    _PATCH_DIFF_PATTERNS = (
        r"```diff",
        r"(?m)^---\s",
        r"(?m)^\+\+\+\s",
        r"(?m)^@@\s",
        r"\bpatch diff\b",
    )
    _APPLIED_EDIT_PATTERNS = (
        r"\b(already\s+applied|has\s+been\s+applied|was\s+applied)\b",
        r"\b(i|we)\s+(have\s+)?(applied|updated|rewrote|revised|changed|edited)\b",
        r"(已经|已)(应用|修改|替换|更新)",
    )
    _DIRECT_REWRITE_PATTERNS = (
        r"\brewritten version\b",
        r"\brevised version\b",
        r"\breplacement sentence\b",
        r"\breplacement paragraph\b",
        r"\breplace with\b",
        r"\byou could say\b",
        r"\bwrite (it|this) as\b",
        r"\bword it as\b",
        r"\buse this sentence\b",
        r"\bi would rewrite (it|this) as\b",
        r"\bi would revise (it|this) as\b",
        r"\bhere is a possible rewrite\b",
        r"\bhere is (a|the) revised\b",
        r"\bhere is (a|the) rewritten\b",
        r"\bfinal version\b",
        r"\bcopy this\b",
        r"改写如下",
        r"替换为",
        r"修改如下",
    )

    def __init__(self, provider_name: str = "none") -> None:
        self.provider: TextLLMProvider = build_text_provider(provider_name)

    def respond(self, payload: RolePayload, request_id: str = "") -> str:
        if payload.mode == "system_exception" and payload.system_exception is not None:
            return self._system_exception_response(payload)
        if self.provider.enabled():
            llm_response = self._respond_with_llm(payload, request_id=request_id)
            if llm_response:
                return self._enforce_contract(payload, llm_response)
        return self._fallback_response(payload)

    def _respond_with_llm(self, payload: RolePayload, request_id: str = "") -> str:
        system_prompt = self._system_prompt(payload)
        user_payload = {
            "scope": asdict(payload.scope),
            "user_request": payload.user_request,
            "role_input_summary": self._serialize_role_input_summary(payload),
            "checker_results": self._serialize_checker_results(payload.checker_results),
            "tool_facts": [asdict(item) for item in payload.tool_facts],
            "tool_summary": self._serialize_tool_summary(payload.tool_summary),
            "system_exception": self._serialize_exception(payload.system_exception),
            "response_policy": asdict(payload.response_policy) if payload.response_policy is not None else None,
            "memory": payload.memory,
            "edit_stage": payload.edit_stage,
            "metadata": payload.metadata,
        }
        messages = [
            {"role": "system", "content": system_prompt + "\n\nReturn JSON only."},
            {
                "role": "user",
                "content": (
                    "Role payload (JSON):\n"
                    f"{json.dumps(user_payload, ensure_ascii=False)}\n\n"
                    "Output schema:\n"
                    "{\n"
                    '  "assistant_message": string\n'
                    "}"
                ),
            },
        ]
        try:
            result = self.provider.complete_json(
                schema=RoleChatResponse,
                messages=messages,
                request_id=request_id,
            )
        except Exception:
            return ""
        return str(result.get("assistant_message", "")).strip()

    def _system_prompt(self, payload: RolePayload) -> str:
        if payload.mode == "check_response":
            return RolePromptBank.checker_response_prompt(payload.role)
        if payload.mode == "system_exception":
            return RolePromptBank.system_exception_prompt(payload.role)
        return RolePromptBank.grounded_chat_prompt(payload.role)

    @staticmethod
    def _serialize_checker_results(payload: CheckerResultPayload | None) -> Dict[str, Any]:
        if payload is None:
            return {}
        data = {
            "requested_checks": list(payload.requested_checks),
            "issues": [issue.to_dict() for issue in payload.issues],
        }
        if payload.role_summary is not None:
            data["role_summary"] = asdict(payload.role_summary)
        data["metadata"] = dict(payload.metadata)
        return data

    @staticmethod
    def _serialize_exception(payload: SystemExceptionPayload | None) -> Dict[str, Any] | None:
        return asdict(payload) if payload is not None else None

    @staticmethod
    def _serialize_tool_summary(payload: RoleToolSummary | None) -> Dict[str, Any]:
        return asdict(payload) if payload is not None else {}

    @staticmethod
    def _serialize_role_input_summary(payload: RolePayload) -> Dict[str, Any]:
        return asdict(payload.role_input_summary) if payload.role_input_summary is not None else {}

    @staticmethod
    def _fallback_response(payload: RolePayload) -> str:
        role = payload.role
        if payload.mode == "system_exception" and payload.system_exception is not None:
            return RoleChatAgent._system_exception_response(payload)

        if payload.mode == "check_response" and payload.checker_results is not None:
            return RoleChatAgent._fallback_check_response(payload)

        if payload.mode == "grounded_chat" and payload.role_input_summary is not None and payload.role_input_summary.entries:
            return RoleChatAgent._fallback_from_role_input_summary(payload)

        if payload.mode == "grounded_chat" and payload.tool_summary is not None and payload.tool_summary.entries:
            return RoleChatAgent._fallback_tool_grounded_response(payload)

        if payload.mode == "grounded_chat" and payload.tool_facts:
            fact_summary = "; ".join(
                str(item.summary or "").strip() for item in payload.tool_facts[:3] if str(item.summary or "").strip()
            ).strip()
            if not fact_summary:
                fact_summary = "tool-derived evidence is available for the confirmed selection."
            if role == "Reviewer":
                return (
                    "Reviewer assessment based on available evidence: "
                    f"{fact_summary} I will only report issues, risks, and unsupported claims."
                )
            if role == "Advisor":
                return (
                    "Advisor guidance based on available evidence: "
                    f"{fact_summary} I will keep this as advice rather than an applied edit."
                )
            return (
                "Editor assessment based on available evidence: "
                f"{fact_summary} I can explain what I would change and prepare a patch only with your consent."
            )

        scope_text = payload.scope.text.strip()
        if not scope_text:
            return "No confirmed scope is available for grounded chat."
        if role == "Reviewer":
            return (
                "Reviewer mode is active. I will only point out issues, risks, and unsupported claims "
                "inside the confirmed selection."
            )
        if role == "Advisor":
            return (
                "Advisor mode is active. I will give revision guidance for the confirmed selection "
                "without directly editing it."
            )
        return (
            "Editor mode is active. I can explain what I would change in the confirmed selection, "
            "and prepare a patch only after your consent."
        )

    @staticmethod
    def _fallback_from_role_input_summary(payload: RolePayload) -> str:
        assert payload.role_input_summary is not None
        summary = payload.role_input_summary
        first = summary.entries[0]
        if summary.source_kind == "tool":
            evidence_label = RoleChatAgent._tool_evidence_label(summary.source_context)
            if payload.role == "Reviewer":
                return (
                    f"Reviewer assessment based on {evidence_label}: {first.observation} "
                    f"Risk: {first.risk} Impact: {first.impact} "
                    "I will only report evidence limits, risks, and unsupported interpretation."
                ).strip()
            if payload.role == "Advisor":
                guidance = first.guidance or summary.next_step
                return (
                    f"Advisor guidance based on {evidence_label}: {first.observation} "
                    f"Rationale: {first.impact or first.risk} Next step: {guidance} "
                    "I will keep this as advice rather than a direct edit."
                ).strip()
            target = first.edit_target or summary.next_step
            return (
                f"Editor analysis based on {evidence_label}: {first.observation} "
                f"Patch suitability: {'suitable for a later patch' if first.patch_suitable else 'analysis-only for now'}. "
                f"Next step: {target} "
                "If you want direct editing later, I can prepare a patch for the confirmed selection after your consent."
            ).strip()
        if summary.source_kind == "checker":
            checks = ", ".join(payload.checker_results.requested_checks) if payload.checker_results else "requested checks"
            count = summary.entry_count
            if payload.role == "Reviewer":
                if count == 0:
                    return f"No issue was detected for {checks} within the confirmed selection."
                return f"I found {count} issue(s) in {checks}. Observation: {first.observation} Risk: {first.risk} Impact: {first.impact}"
            if payload.role == "Advisor":
                if count == 0:
                    return f"No issue was detected for {checks} within the confirmed selection."
                rationale = first.impact or first.risk
                return f"I found {count} issue(s) in {checks}. Guidance: {first.guidance or summary.next_step} Rationale: {rationale} Next step: {summary.next_step}"
            if count == 0:
                return f"No issue was detected for {checks} within the confirmed selection."
            patchability = "A safe patch target is available." if first.patch_suitable else "No safe patch target is available yet."
            return f"I found {count} issue(s) in {checks}. Edit analysis: {first.observation} Patch suitability: {patchability} Next step: {summary.next_step}"
        if summary.source_kind == "system_exception":
            if payload.role == "Reviewer":
                return (
                    f"I cannot review or verify this request because {first.observation} "
                    f"Impact: {first.impact} {first.guidance}"
                ).strip()
            if payload.role == "Advisor":
                return (
                    f"I cannot advise on this yet because {first.observation} "
                    f"Impact: {first.impact} {first.guidance}"
                ).strip()
            return (
                f"I cannot safely edit this scope because {first.observation} "
                f"Impact: {first.impact} {first.guidance}"
            ).strip()
        return ""

    @staticmethod
    def _fallback_tool_grounded_response(payload: RolePayload) -> str:
        assert payload.tool_summary is not None
        summary = payload.tool_summary
        first = summary.entries[0]
        evidence_label = RoleChatAgent._tool_evidence_label(summary.source_context)
        if payload.role == "Reviewer":
            return (
                f"Reviewer assessment based on {evidence_label}: {first.fact} "
                f"{first.evidence_status} {first.interpretation_risk} "
                "I will only report evidence limits, risks, and unsupported interpretation."
            ).strip()
        if payload.role == "Advisor":
            guidance = first.guidance or summary.next_step
            return (
                f"Advisor guidance based on {evidence_label}: {first.fact} "
                f"{guidance} "
                "I will keep this as advice rather than a direct edit."
            ).strip()
        target = first.edit_target or summary.next_step
        return (
            f"Editor analysis based on {evidence_label}: {first.fact} "
            f"{target} "
            "If you want direct editing later, I can prepare a patch for the confirmed selection after your consent."
        ).strip()

    @staticmethod
    def _tool_evidence_label(source_context: str) -> str:
        if source_context == "image_related":
            return "visual evidence"
        if source_context == "table_related":
            return "table evidence"
        return "tool-derived evidence"

    @staticmethod
    def _fallback_check_response(payload: RolePayload) -> str:
        if payload.role_input_summary is not None and payload.role_input_summary.entries:
            summary_text = RoleChatAgent._fallback_from_role_input_summary(payload)
            if summary_text:
                return summary_text
        assert payload.checker_results is not None
        role = payload.role
        summary = payload.checker_results.role_summary
        issue_count = len(payload.checker_results.issues)
        checks = ", ".join(payload.checker_results.requested_checks) or "requested checks"
        if role == "Reviewer":
            if issue_count == 0:
                return f"No issue was detected for {checks} within the confirmed selection."
            risk = summary.entries[0].risk if summary and summary.entries else "Reviewer mode reports problems, locations, and risks only."
            return f"I found {issue_count} issue(s) in {checks}. {risk}"
        if role == "Advisor":
            if issue_count == 0:
                return f"No issue was detected for {checks} within the confirmed selection."
            guidance = summary.entries[0].guidance if summary and summary.entries else ""
            return (
                f"I found {issue_count} issue(s) in {checks}. "
                + (guidance if guidance else "Advisor mode will turn them into revision guidance rather than direct edits.")
            )
        if issue_count == 0:
            return f"No issue was detected for {checks} within the confirmed selection."
        next_step = summary.next_step if summary else ""
        return (
            f"I found {issue_count} issue(s) in {checks}. "
            + (next_step if next_step else "Editor mode can explain what should change and prepare a patch only after your consent.")
        )

    @staticmethod
    def _system_exception_response(payload: RolePayload) -> str:
        if payload.role_input_summary is not None and payload.role_input_summary.entries:
            summary_text = RoleChatAgent._fallback_from_role_input_summary(payload)
            if summary_text:
                return summary_text
        exc = payload.system_exception
        if exc is None:
            return "The requested action cannot continue because a required dependency is missing."
        message = str(exc.message or "").strip()
        target = str(exc.target or exc.target_type or "resource").strip()
        if payload.role == "Reviewer":
            return (
                f"I cannot review or verify this request because {message} "
                f"Please restore the required {target} before asking for a grounded judgment."
            ).strip()
        if payload.role == "Advisor":
            return (
                f"I cannot advise on this yet because {message} "
                f"Please provide or correct the required {target} first, then I can continue."
            ).strip()
        return (
            f"I cannot safely edit this scope because {message} "
            "Please restore the missing dependency or reconfirm the target selection before editing."
        ).strip()

    def _enforce_contract(self, payload: RolePayload, message: str) -> str:
        normalized = str(message or "").strip()
        if not normalized:
            return self._fallback_response(payload)
        if self._contains_patch_diff(normalized) or self._claims_applied_edit(normalized):
            return self._fallback_response(payload)
        if payload.mode == "grounded_chat":
            normalized = self._enforce_grounded_chat_contract(payload, normalized)
            if not normalized:
                return self._fallback_response(payload)
        elif payload.role == "Reviewer":
            if self._looks_like_direct_rewrite(normalized):
                return self._fallback_response(payload)
        elif payload.role == "Advisor":
            if self._looks_like_direct_rewrite(normalized) and not self._is_labeled_example(normalized):
                return self._fallback_response(payload)
        return self._ensure_policy_closing(payload, normalized)

    def _enforce_grounded_chat_contract(self, payload: RolePayload, message: str) -> str:
        policy = payload.response_policy
        if payload.role == "Reviewer":
            if self._looks_like_direct_rewrite(message) or self._is_labeled_example(message):
                return ""
            return message
        if payload.role == "Advisor":
            if self._looks_like_direct_rewrite(message) and not self._is_labeled_example(message):
                return ""
            return self._sanitize_advisor_grounded_chat(message, payload)
        if payload.role == "Editor":
            if self._looks_like_direct_rewrite(message) or self._is_labeled_example(message):
                return ""
            return self._ensure_editor_analysis_boundary(message, policy.required_closing if policy else "")
        return message

    def _sanitize_advisor_grounded_chat(self, message: str, payload: RolePayload) -> str:
        policy = payload.response_policy
        count = self._count_example_labels(message)
        if count == 0:
            return message
        max_count = int(policy.example_max_count) if policy else 1
        max_chars = int(policy.example_max_chars) if policy else self._ADVISOR_MAX_EXAMPLE_CHARS
        max_words = int(policy.example_max_words) if policy else self._ADVISOR_MAX_EXAMPLE_WORDS
        if count > 1:
            return ""
        example_text = self._extract_example_text(message)
        if not example_text:
            return ""
        if count > max_count:
            return ""
        if len(example_text) > max_chars:
            return ""
        if len(example_text.split()) > max_words:
            return ""
        if "\n" in example_text.strip():
            return ""
        return message

    @staticmethod
    def _ensure_editor_analysis_boundary(message: str, required_closing: str = "") -> str:
        lowered = message.lower()
        if "patch" in lowered or "consent" in lowered or "prepare" in lowered:
            return message
        closing = required_closing.strip() or "If you want, I can prepare a patch for the confirmed selection after your consent."
        return (
            message.rstrip()
            + " "
            + closing
        ).strip()

    def _ensure_policy_closing(self, payload: RolePayload, message: str) -> str:
        policy = payload.response_policy
        if policy is None:
            return message
        closing = str(policy.required_closing or "").strip()
        if not closing:
            return message
        normalized = message.strip()
        if closing.lower() in normalized.lower():
            return normalized
        if payload.role == "Editor" and policy.editor_phase in {"analysis_only", "invite_patch", "awaiting_apply"}:
            return (normalized + "\n\n" + closing).strip()
        return normalized

    @classmethod
    def _contains_patch_diff(cls, text: str) -> bool:
        lowered = text.lower()
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in cls._PATCH_DIFF_PATTERNS)

    @classmethod
    def _claims_applied_edit(cls, text: str) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in cls._APPLIED_EDIT_PATTERNS)

    @classmethod
    def _looks_like_direct_rewrite(cls, text: str) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in cls._DIRECT_REWRITE_PATTERNS)

    @classmethod
    def _is_labeled_example(cls, text: str) -> bool:
        return bool(re.search(cls._EXAMPLE_LABEL_REGEX, text, flags=re.IGNORECASE))

    @classmethod
    def _count_example_labels(cls, text: str) -> int:
        return len(re.findall(cls._EXAMPLE_LABEL_REGEX, text, flags=re.IGNORECASE))

    @classmethod
    def _extract_example_text(cls, text: str) -> str:
        match = re.search(
            cls._EXAMPLE_LABEL_REGEX + r"\s*(.+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ""
        return str(match.group(1) or "").strip()
