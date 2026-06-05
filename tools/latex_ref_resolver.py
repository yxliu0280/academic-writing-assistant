from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class LatexRefResolver:
    """Inspired by Paper2Poster and Paper2Agent: LaTeX AST grounded multimodal alignment.

    This utility resolves references inside a selected text snippet and grounds them
    back to concrete LaTeX assets in the full manuscript source.
    """

    REF_CMD_PATTERN = re.compile(r"\\(?:ref|autoref|cref|Cref)\{([^}]+)\}")
    LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
    INCLUDEGRAPHICS_PATTERN = re.compile(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", re.IGNORECASE
    )
    TABLE_NATURAL_REF_PATTERN = re.compile(
        r"\btable(?:s)?\s*(?:~|no\.?\s*)?(\d+)[A-Za-z]?\b",
        re.IGNORECASE,
    )
    FIGURE_NATURAL_REF_PATTERN = re.compile(
        r"\bfigure(?:s)?\s*(?:~|no\.?\s*)?(\d+)[A-Za-z]?\b|\bfig\.?\s*(\d+)[A-Za-z]?\b",
        re.IGNORECASE,
    )

    def resolve(self, selection_text: str, full_latex_text: str) -> Dict[str, Any]:
        labels = self._extract_ref_labels(selection_text)

        table_labels = [l for l in labels if l.startswith("tab:")]
        figure_labels = [l for l in labels if l.startswith("fig:")]
        table_numbers = self._extract_natural_ref_numbers(selection_text, env_name="table")
        figure_numbers = self._extract_natural_ref_numbers(selection_text, env_name="figure")

        table_contexts = [
            self._resolve_environment_by_label(full_latex_text, label, env_name="table")
            for label in table_labels
        ]
        if not table_contexts:
            table_contexts = [
                self._resolve_environment_by_ordinal(full_latex_text, ordinal=n, env_name="table")
                for n in table_numbers
            ]
        figure_contexts = [
            self._resolve_environment_by_label(full_latex_text, label, env_name="figure")
            for label in figure_labels
        ]
        if not figure_contexts:
            figure_contexts = [
                self._resolve_environment_by_ordinal(full_latex_text, ordinal=n, env_name="figure")
                for n in figure_numbers
            ]

        missing_labels = [
            c["label"]
            for c in table_contexts + figure_contexts
            if not c.get("found", False) and c.get("label")
        ]

        return {
            "selection": selection_text,
            "refs": {
                "all": labels,
                "table": table_labels,
                "figure": figure_labels,
                "table_numbers": table_numbers,
                "figure_numbers": figure_numbers,
            },
            "tables": table_contexts,
            "figures": figure_contexts,
            "missing_labels": missing_labels,
        }

    def _extract_ref_labels(self, text: str) -> List[str]:
        labels = [m.group(1).strip() for m in self.REF_CMD_PATTERN.finditer(text)]
        deduped: List[str] = []
        seen = set()
        for label in labels:
            if label in seen:
                continue
            seen.add(label)
            deduped.append(label)
        return deduped

    def _extract_natural_ref_numbers(self, text: str, *, env_name: str) -> List[int]:
        pattern = self.TABLE_NATURAL_REF_PATTERN if env_name == "table" else self.FIGURE_NATURAL_REF_PATTERN
        numbers: List[int] = []
        seen = set()
        for match in pattern.finditer(text):
            captured = next((group for group in match.groups() if group), "")
            if not captured:
                continue
            try:
                ordinal = int(captured)
            except ValueError:
                continue
            if ordinal <= 0 or ordinal in seen:
                continue
            seen.add(ordinal)
            numbers.append(ordinal)
        return numbers

    def _resolve_environment_by_label(
        self,
        full_latex_text: str,
        label: str,
        env_name: str,
    ) -> Dict[str, Any]:
        for block, start, end in self._iter_env_blocks(full_latex_text, env_name):
            labels = [m.group(1).strip() for m in self.LABEL_PATTERN.finditer(block)]
            if label not in labels:
                continue

            payload: Dict[str, Any] = {
                "label": label,
                "env": env_name,
                "found": True,
                "char_span": [start, end],
                "block": block,
            }

            if env_name == "figure":
                payload["image_paths"] = self._extract_figure_paths(block)

            return payload

        # Fallback: label may exist but not inside expected env.
        loose_match = re.search(rf"\\label\{{{re.escape(label)}\}}", full_latex_text)
        if loose_match:
            span = [loose_match.start(), loose_match.end()]
            return {
                "label": label,
                "env": env_name,
                "found": False,
                "char_span": span,
                "block": "",
                "note": "label_found_outside_expected_environment",
                "image_paths": [] if env_name == "figure" else None,
            }

        return {
            "label": label,
            "env": env_name,
            "found": False,
            "char_span": [-1, -1],
            "block": "",
            "note": "label_not_found",
            "image_paths": [] if env_name == "figure" else None,
        }

    def _resolve_environment_by_ordinal(
        self,
        full_latex_text: str,
        ordinal: int,
        env_name: str,
    ) -> Dict[str, Any]:
        blocks = self._iter_env_blocks(full_latex_text, env_name)
        if ordinal <= 0 or ordinal > len(blocks):
            return {
                "label": "",
                "env": env_name,
                "found": False,
                "char_span": [-1, -1],
                "block": "",
                "note": "ordinal_not_found",
                "ordinal": ordinal,
                "image_paths": [] if env_name == "figure" else None,
            }

        block, start, end = blocks[ordinal - 1]
        labels = [m.group(1).strip() for m in self.LABEL_PATTERN.finditer(block)]
        payload: Dict[str, Any] = {
            "label": labels[0] if labels else "",
            "env": env_name,
            "found": True,
            "char_span": [start, end],
            "block": block,
            "ordinal": ordinal,
        }
        if env_name == "figure":
            payload["image_paths"] = self._extract_figure_paths(block)
        return payload

    def _iter_env_blocks(
        self,
        full_latex_text: str,
        env_name: str,
    ) -> List[Tuple[str, int, int]]:
        pattern = re.compile(
            rf"\\begin\{{{re.escape(env_name)}\}}[\s\S]*?\\end\{{{re.escape(env_name)}\}}",
            re.IGNORECASE,
        )
        blocks: List[Tuple[str, int, int]] = []
        for m in pattern.finditer(full_latex_text):
            blocks.append((m.group(0), m.start(), m.end()))
        return blocks

    def _extract_figure_paths(self, figure_block: str) -> List[str]:
        paths = [m.group(1).strip() for m in self.INCLUDEGRAPHICS_PATTERN.finditer(figure_block)]
        deduped: List[str] = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped
