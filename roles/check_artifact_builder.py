from __future__ import annotations

from typing import Dict, List

from agents.role_negotiation_agent import RoleNegotiationAgent
from core.schemas import AppState, CheckExecutionArtifacts, CheckPresentationArtifacts, RenderedIssue
from roles.advisor import AdvisorRenderer
from roles.editor import EditorRenderer
from roles.reviewer import ReviewerRenderer
from tools.patch_utils import (
    apply_replace_operations,
    compute_edit_ratio,
    count_modified_sentences,
    patch_guardrail_ok,
    unified_diff,
)


class CheckArtifactBuilder:
    """Build role-facing check artifacts from normalized check execution results.

    This builder is intentionally separate from ConsistencyOrchestrator so that
    check execution and role-specific presentation can evolve independently.
    """

    def __init__(self, text_provider_name: str = "none") -> None:
        self.reviewer_renderer = ReviewerRenderer()
        self.advisor_renderer = AdvisorRenderer()
        self.editor_renderer = EditorRenderer()
        self.role_negotiation_agent = RoleNegotiationAgent(text_provider_name)

    def build(
        self,
        state: AppState,
        execution: CheckExecutionArtifacts,
    ) -> CheckPresentationArtifacts:
        issues = list(execution.issues or [])
        writing_support = list(execution.writing_support or [])

        if state.role == "Reviewer":
            rendered = self.reviewer_renderer.render(issues)
            return CheckPresentationArtifacts(
                rendered_issues=rendered,
                report_md=self._build_report_md(state, rendered, writing_support),
                metadata={"renderer": "ReviewerRenderer"},
            )

        if state.role == "Advisor":
            rendered = self.advisor_renderer.render(issues)
            return CheckPresentationArtifacts(
                rendered_issues=rendered,
                report_md=self._build_report_md(state, rendered, writing_support),
                metadata={"renderer": "AdvisorRenderer"},
            )

        rendered = self.editor_renderer.render(issues, state.current_text)
        patch_ops = self.editor_renderer.collect_operations(rendered)
        if not patch_ops:
            state.negotiation_state.active = False
            return CheckPresentationArtifacts(
                rendered_issues=rendered,
                report_md=self._build_report_md(state, rendered, writing_support),
                metadata={"renderer": "EditorRenderer", "patch_ops_count": 0},
            )

        patched = apply_replace_operations(state.current_text, patch_ops)
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
            # Keep the patch candidate. Guardrails are advisory, not a silent downgrade.
            pass

        style_protection_active = self.role_negotiation_agent.maybe_activate_style_protection(
            state=state,
            patch_diff=patch_diff,
            patch_operations=patch_ops,
            edit_ratio=float(guardrail.get("edit_ratio", 0.0)),
            sentence_changes=float(guardrail.get("sentence_changes", 0.0)),
            request_id=state.request_id,
        )

        return CheckPresentationArtifacts(
            rendered_issues=rendered,
            patch_diff=patch_diff,
            patch_operations=[op.to_dict() for op in patch_ops],
            report_md=self._build_report_md(state, rendered, writing_support),
            metadata={
                "renderer": "EditorRenderer",
                "patch_ops_count": len(patch_ops),
                "style_protection_active": style_protection_active,
                "guardrail": guardrail,
            },
        )

    @staticmethod
    def _build_report_md(
        state: AppState,
        rendered: List[RenderedIssue],
        writing_support: List[Dict[str, str]],
    ) -> str:
        lines = [
            f"# Academic Writing Companion Report ({state.role})",
            "",
            "## Consistency Issues",
        ]
        if not rendered:
            lines.append("No issues detected.")
        for idx, item in enumerate(rendered, start=1):
            issue = item.issue
            lines.extend(
                [
                    f"### {idx}. {issue.type}",
                    f"- Severity: {issue.severity}",
                    f"- Status: {issue.status}",
                    f"- Location: sentence {issue.location.sentence_index}, span {issue.location.char_span}",
                    f"- Snippet: `{issue.location.snippet}`",
                    f"- Message: {issue.message}",
                    f"- Evidence: `{issue.evidence}`",
                ]
            )
            if item.suggested_fix:
                lines.append(f"- Suggested fix: {item.suggested_fix}")
            if item.patch:
                lines.append(f"- Patch preview: `{item.patch}`")
            if issue.provenance:
                lines.append(f"- Provenance: `{issue.provenance}`")
            lines.append("")

        lines.append("## Writing Support (Out of Eval Scope)")
        if writing_support:
            for rec in writing_support[:20]:
                lines.append(f"- {rec['type']}: {rec['message']}")
        else:
            lines.append("- No additional writing support notes.")
        return "\n".join(lines)
