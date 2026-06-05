from __future__ import annotations

from typing import List

from core.schemas import Issue, RenderedIssue


class ReviewerRenderer:
    role = "Reviewer"

    def render(self, issues: List[Issue]) -> List[RenderedIssue]:
        return [RenderedIssue(issue=issue, view_mode="Reviewer") for issue in issues]
