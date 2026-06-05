from __future__ import annotations

from typing import List

from core.schemas import (
    Issue,
    RenderedIssue,
    RequestedCheckName,
    RoleCheckEntry,
    RoleCheckSummary,
    RoleType,
)


class CheckResponseContractBuilder:
    """Build role-facing check summaries for stable `check_response` behavior."""

    def build(
        self,
        *,
        role: RoleType,
        requested_checks: List[RequestedCheckName],
        issues: List[Issue],
        rendered_issues: List[RenderedIssue],
    ) -> RoleCheckSummary:
        rendered_by_id = {item.issue.id: item for item in rendered_issues}
        entries = [
            self._entry_for_issue(role, issue, rendered_by_id.get(issue.id))
            for issue in issues
        ]
        return RoleCheckSummary(
            role=role,
            summary_style=self._summary_style(role),
            issue_count=len(issues),
            requested_checks=list(requested_checks),
            entries=entries,
            next_step=self._next_step(role, entries),
        )

    @staticmethod
    def _summary_style(role: RoleType) -> str:
        if role == "Reviewer":
            return "review"
        if role == "Advisor":
            return "advice"
        return "edit_analysis"

    def _entry_for_issue(
        self,
        role: RoleType,
        issue: Issue,
        rendered: RenderedIssue | None,
    ) -> RoleCheckEntry:
        guidance = ""
        patchable = False
        if role == "Advisor":
            guidance = str(rendered.suggested_fix or "").strip() if rendered else ""
        elif role == "Editor":
            guidance = str(rendered.suggested_fix or "").strip() if rendered else ""
            patchable = bool(rendered and rendered.patch)
        return RoleCheckEntry(
            issue_id=issue.id,
            issue_type=issue.type,
            severity=issue.severity,
            status=issue.status,
            location=self._location_label(issue),
            snippet=str(issue.location.snippet or ""),
            problem=str(issue.message or ""),
            risk=self._risk_label(issue),
            impact=self._impact_label(issue),
            guidance=guidance,
            patchable=patchable,
        )

    @staticmethod
    def _location_label(issue: Issue) -> str:
        line_range = issue.location.line_range
        if line_range:
            return f"lines {line_range[0]}-{line_range[1]}"
        return f"sentence {issue.location.sentence_index}"

    @staticmethod
    def _risk_label(issue: Issue) -> str:
        if issue.severity == "high":
            return "High risk to factual correctness or reader trust."
        if issue.severity == "medium":
            return "Moderate risk to rigor, clarity, or evidence quality."
        return "Low risk but still worth cleaning up for consistency or clarity."

    @staticmethod
    def _impact_label(issue: Issue) -> str:
        if issue.type in {"text_table_mismatch", "text_figure_mismatch"}:
            return "May make the manuscript numerically inconsistent with its evidence."
        if issue.type in {"citation_missing", "citation_suspected_missing", "citation_unverified_suggestion"}:
            return "May weaken scholarly support or create citation reliability problems."
        if issue.type == "terminology_inconsistent":
            return "May reduce terminology consistency and reader clarity."
        if issue.type in {"figure_check_uncertain", "figure_check_skipped"}:
            return "Leaves the related claim insufficiently verified."
        return "May weaken precision, consistency, or trustworthiness."

    @staticmethod
    def _next_step(role: RoleType, entries: List[RoleCheckEntry]) -> str:
        if not entries:
            if role == "Reviewer":
                return "State that no issue was detected within the checked scope."
            if role == "Advisor":
                return "State that no issue was detected and optionally mention low-priority refinements."
            return "State that no edit is needed within the checked scope."
        if role == "Reviewer":
            return "Report the problems, risks, and impacts only; do not rewrite or patch."
        if role == "Advisor":
            return "Turn the findings into revision guidance; do not apply or patch."
        patchable_count = sum(1 for item in entries if item.patchable)
        if patchable_count:
            return "Explain what should change and invite patch preparation without applying edits."
        return "Explain what should change, but note that no safe patch candidate is available yet."
