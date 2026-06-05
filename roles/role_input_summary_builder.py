from __future__ import annotations

from core.schemas import (
    CheckerResultPayload,
    RoleCheckSummary,
    RoleInputEntry,
    RoleInputSummary,
    RolePayloadMode,
    RoleToolSummary,
    RoleType,
    SystemExceptionPayload,
)


class RoleInputSummaryBuilder:
    """Unify role-facing checker, tool, and exception summaries into one light abstraction."""

    def build(
        self,
        *,
        role: RoleType,
        mode: RolePayloadMode,
        checker_results: CheckerResultPayload | None = None,
        tool_summary: RoleToolSummary | None = None,
        system_exception: SystemExceptionPayload | None = None,
        source_context: str = "",
    ) -> RoleInputSummary | None:
        if checker_results is not None and checker_results.role_summary is not None:
            return self._from_checker_summary(role, mode, checker_results.role_summary, source_context)
        if tool_summary is not None:
            return self._from_tool_summary(role, mode, tool_summary, source_context)
        if system_exception is not None:
            return self._from_system_exception(role, mode, system_exception, source_context)
        return None

    @staticmethod
    def _summary_style(role: RoleType) -> str:
        if role == "Reviewer":
            return "review"
        if role == "Advisor":
            return "advice"
        return "edit_analysis"

    def _from_checker_summary(
        self,
        role: RoleType,
        mode: RolePayloadMode,
        summary: RoleCheckSummary,
        source_context: str,
    ) -> RoleInputSummary:
        entries = [
            RoleInputEntry(
                source_kind="checker",
                category=str(item.issue_type),
                source="checker_result",
                observation=item.problem,
                location=item.location,
                risk=item.risk,
                impact=item.impact,
                guidance=item.guidance,
                patch_suitable=item.patchable,
            )
            for item in summary.entries
        ]
        return RoleInputSummary(
            role=role,
            mode=mode,
            source_kind="checker",
            summary_style=self._summary_style(role),
            source_context=source_context or "checks",
            entry_count=len(entries),
            entries=entries,
            next_step=summary.next_step,
        )

    def _from_tool_summary(
        self,
        role: RoleType,
        mode: RolePayloadMode,
        summary: RoleToolSummary,
        source_context: str,
    ) -> RoleInputSummary:
        entries = [
            RoleInputEntry(
                source_kind="tool",
                category=str(item.kind or "tool_fact"),
                source=str(item.source or ""),
                observation=item.fact,
                confidence=item.confidence,
                risk=item.interpretation_risk,
                impact=item.evidence_status,
                guidance=item.guidance,
                edit_target=item.edit_target,
                patch_suitable=item.patch_suitable,
            )
            for item in summary.entries
        ]
        return RoleInputSummary(
            role=role,
            mode=mode,
            source_kind="tool",
            summary_style=self._summary_style(role),
            source_context=source_context or summary.source_context or "tool_facts",
            entry_count=len(entries),
            entries=entries,
            next_step=summary.next_step,
        )

    def _from_system_exception(
        self,
        role: RoleType,
        mode: RolePayloadMode,
        exc: SystemExceptionPayload,
        source_context: str,
    ) -> RoleInputSummary:
        target = str(exc.target or exc.target_type or "resource").strip()
        if role == "Reviewer":
            guidance = f"Restore the required {target} before asking for a grounded judgment."
        elif role == "Advisor":
            guidance = f"Provide or correct the required {target} first, then advice can continue."
        else:
            guidance = f"Restore the missing {target} or reconfirm the target selection before editing."
        entry = RoleInputEntry(
            source_kind="system_exception",
            category=str(exc.code or "system_exception"),
            source=target,
            observation=str(exc.message or "").strip() or "A required dependency is missing.",
            risk="The current grounded operation cannot continue safely.",
            impact="The requested review, advice, or editing step is blocked until the dependency is restored.",
            guidance=guidance,
        )
        return RoleInputSummary(
            role=role,
            mode=mode,
            source_kind="system_exception",
            summary_style=self._summary_style(role),
            source_context=source_context or "system_exception",
            entry_count=1,
            entries=[entry],
            next_step=guidance,
        )
