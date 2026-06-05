from __future__ import annotations

from typing import List

from core.schemas import Issue, RenderedIssue


class AdvisorRenderer:
    role = "Advisor"

    def render(self, issues: List[Issue]) -> List[RenderedIssue]:
        rendered: List[RenderedIssue] = []
        for issue in issues:
            rendered.append(
                RenderedIssue(
                    issue=issue,
                    view_mode="Advisor",
                    suggested_fix=self._build_suggestion(issue),
                )
            )
        return rendered

    def _build_suggestion(self, issue: Issue) -> str | None:
        if issue.type == "text_table_mismatch":
            expected = issue.evidence.get("expected")
            return (
                f"Use table-derived value {expected:.3f} and keep metric wording consistent "
                "with the table row."
                if isinstance(expected, (int, float))
                else "Revise the numeric claim to match the uploaded table evidence."
            )
        if issue.type == "text_figure_mismatch":
            return "Update the sentence numeric value to match figure evidence, or annotate uncertainty."
        if issue.type == "citation_missing":
            missing = issue.evidence.get("missing_key", "unknown_key")
            return f"Replace '{missing}' with a key that exists in the uploaded .bib allowlist."
        if issue.type == "citation_suspected_missing":
            return "Add a citation if this factual statement is based on prior work."
        if issue.type == "citation_unverified_suggestion":
            return "Abstain from adding this citation until key and metadata provenance are verified."
        if issue.type == "terminology_inconsistent":
            canonical = issue.evidence.get("canonical")
            if canonical:
                return f"Normalize all variants to '{canonical}'."
            return "Use one preferred term consistently across the manuscript."
        if issue.type == "figure_check_uncertain":
            return "Keep current text but flag this claim for manual verification from the source figure."
        if issue.type == "figure_check_skipped":
            return "Enable a VLM provider to run figure consistency verification."
        return None
