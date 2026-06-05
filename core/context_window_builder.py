from __future__ import annotations

from dataclasses import asdict
from typing import List

from core.schemas import GroundedContextBundle, SelectionContext


class ContextWindowBuilder:
    """Inspired by InfiAgent: minimal, file-centric context packing for model calls.

    Strategy:
    - prioritize user-selected span
    - include bounded surrounding text
    - attach only necessary grounded contexts (table/figure/citation snippets)
    """

    def __init__(self, max_chars: int = 3200, surround_chars: int = 800) -> None:
        self.max_chars = max_chars
        self.surround_chars = surround_chars

    def build(
        self,
        full_text: str,
        selection: SelectionContext | None,
        table_contexts: List[str] | None = None,
        figure_contexts: List[str] | None = None,
        citation_contexts: List[str] | None = None,
    ) -> GroundedContextBundle:
        table_contexts = table_contexts or []
        figure_contexts = figure_contexts or []
        citation_contexts = citation_contexts or []

        selected_text = ""
        surrounding_text = ""

        if selection is not None and selection.end > selection.start:
            start = max(0, selection.start)
            end = min(len(full_text), selection.end)
            selected_text = full_text[start:end]

            left = max(0, start - self.surround_chars)
            right = min(len(full_text), end + self.surround_chars)
            surrounding_text = full_text[left:right]
        else:
            surrounding_text = full_text[: self.max_chars]

        # Hard bound final text payload to reduce prompt growth.
        selected_text = self._clip(selected_text, self.max_chars)
        remaining = max(0, self.max_chars - len(selected_text))
        surrounding_text = self._clip(surrounding_text, remaining if remaining else self.max_chars)

        return GroundedContextBundle(
            selected_text=selected_text,
            surrounding_text=surrounding_text,
            table_contexts=[self._clip(x, 1200) for x in table_contexts],
            figure_contexts=[self._clip(x, 1200) for x in figure_contexts],
            citation_contexts=[self._clip(x, 1200) for x in citation_contexts],
            metadata={
                "max_chars": self.max_chars,
                "surround_chars": self.surround_chars,
                "selection_present": selection is not None,
            },
        )

    @staticmethod
    def to_compact_dict(bundle: GroundedContextBundle) -> dict:
        return asdict(bundle)

    @staticmethod
    def _clip(text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n...<truncated>"
