from __future__ import annotations

import re
from typing import List

from core.schemas import Issue, RenderedIssue
from tools.patch_utils import ReplaceOperation


class EditorRenderer:
    role = "Editor"

    def render(self, issues: List[Issue], text: str) -> List[RenderedIssue]:
        rendered: List[RenderedIssue] = []
        for issue in issues:
            operations = self._operations_for_issue(issue, text)
            patch = None
            if operations:
                patch = {"operations": [op.to_dict() for op in operations]}
            rendered.append(
                RenderedIssue(
                    issue=issue,
                    view_mode="Editor",
                    suggested_fix=self._build_suggestion(issue),
                    patch=patch,
                )
            )
        return rendered

    def collect_operations(self, rendered_issues: List[RenderedIssue]) -> List[ReplaceOperation]:
        operations: List[ReplaceOperation] = []
        for item in rendered_issues:
            if not item.patch:
                continue
            for op in item.patch.get("operations", []):
                operations.append(
                    ReplaceOperation(
                        start=int(op["start"]),
                        end=int(op["end"]),
                        replacement=str(op["replacement"]),
                        reason=str(op.get("reason", "editor_patch")),
                    )
                )
        # Deduplicate exact operations
        uniq = {(op.start, op.end, op.replacement): op for op in operations}
        return list(uniq.values())

    def _operations_for_issue(self, issue: Issue, text: str) -> List[ReplaceOperation]:
        if issue.status != "detected":
            return []

        if issue.type in {"text_table_mismatch", "text_figure_mismatch"}:
            expected = issue.evidence.get("expected", issue.evidence.get("fact_value"))
            if not isinstance(expected, (int, float)):
                return []
            start, end = issue.location.char_span
            segment = text[start:end]
            num_match = re.search(r"\d+(?:\.\d+)?", segment)
            if not num_match:
                return []
            abs_start = start + num_match.start()
            abs_end = start + num_match.end()
            replacement = self._format_number(float(expected))
            return [
                ReplaceOperation(
                    start=abs_start,
                    end=abs_end,
                    replacement=replacement,
                    reason=f"fix_{issue.type}",
                )
            ]

        if issue.type == "citation_missing":
            allowlist = issue.evidence.get("allowlist_preview", [])
            if not allowlist:
                return []
            missing_key = str(issue.evidence.get("missing_key", ""))
            start, end = issue.location.char_span
            segment = text[start:end]
            if missing_key and missing_key in segment:
                local = segment.find(missing_key)
                return [
                    ReplaceOperation(
                        start=start + local,
                        end=start + local + len(missing_key),
                        replacement=str(allowlist[0]),
                        reason="replace_missing_cite_key",
                    )
                ]
            return []

        if issue.type == "terminology_inconsistent":
            canonical = issue.evidence.get("canonical")
            occs = issue.evidence.get("occurrences", [])
            if not canonical or not occs:
                return []
            ops: List[ReplaceOperation] = []
            for occ in occs:
                variant = occ.get("variant")
                if variant == canonical:
                    continue
                span = occ.get("char_span", [0, 0])
                s, e = int(span[0]), int(span[1])
                ops.append(
                    ReplaceOperation(
                        start=s,
                        end=e,
                        replacement=str(canonical),
                        reason="normalize_terminology",
                    )
                )
                if len(ops) >= 3:
                    break
            return ops

        return []

    def _build_suggestion(self, issue: Issue) -> str | None:
        if issue.type in {"text_table_mismatch", "text_figure_mismatch"}:
            return "Apply a minimal numeric replacement based on verified evidence."
        if issue.type == "citation_missing":
            return "Replace invalid cite key with a key from the uploaded BibTeX allowlist."
        if issue.type == "terminology_inconsistent":
            return "Normalize inconsistent variants to the preferred canonical term."
        return None

    @staticmethod
    def _format_number(value: float) -> str:
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.3f}".rstrip("0").rstrip(".")
