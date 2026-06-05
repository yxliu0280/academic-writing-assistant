from __future__ import annotations

import re
from typing import List

from core.schemas import ToolFact
from tools.latex_table_normalizer import latex_table_block_to_csv
from tools.latex_ref_resolver import LatexRefResolver
from tools.table_utils import parse_table_csv


class TableQATool:
    """Minimal table-grounded helper for ordinary chat.

    It resolves a LaTeX table block, extracts lightweight structured facts, and
    returns normalized `ToolFact` items for the role-facing contract layer.
    """

    _TABLE_ENV_PATTERN = re.compile(
        r"\\begin\{table\}[\s\S]*?\\end\{table\}",
        re.IGNORECASE,
    )
    def analyze(self, *, selection_text: str, full_latex_text: str, target_label: str = "") -> List[ToolFact]:
        table_block, label = self._resolve_table_block(
            selection_text=selection_text,
            full_latex_text=full_latex_text,
            target_label=target_label,
        )
        if not table_block:
            return []

        csv_content = latex_table_block_to_csv(table_block)
        if not csv_content:
            return [
                ToolFact(
                    kind="table_resource",
                    source=label or "table",
                    summary=f"Table resource {label or 'table'} is available, but no structured rows were extracted.",
                    data={"table_label": label},
                )
            ]

        try:
            metric_rows = parse_table_csv(csv_content)
        except Exception:
            return [
                ToolFact(
                    kind="table_resource",
                    source=label or "table",
                    summary=f"Table resource {label or 'table'} is available, but its schema could not be normalized.",
                    data={"table_label": label},
                )
            ]

        facts: List[ToolFact] = []
        for metric_name, row in list(metric_rows.items())[:4]:
            facts.append(
                ToolFact(
                    kind="table_metric_fact",
                    source=label or "table",
                    summary=(
                        f"Table fact: metric={metric_name}, baseline={row.baseline}, "
                        f"ours={row.ours}, table_label={label or 'unknown'}"
                    ),
                    data={
                        "table_label": label,
                        "metric": metric_name,
                        "baseline": row.baseline,
                        "ours": row.ours,
                    },
                )
            )
        if not facts:
            facts.append(
                ToolFact(
                    kind="table_resource",
                    source=label or "table",
                    summary=f"Table resource {label or 'table'} is available, but no comparable metric rows were found.",
                    data={"table_label": label},
                )
            )
        return facts

    def summarize_available_tables(self, *, full_latex_text: str) -> List[ToolFact]:
        labels = []
        for block in self._TABLE_ENV_PATTERN.findall(full_latex_text):
            label_match = re.search(r"\\label\{([^}]+)\}", block)
            label = str(label_match.group(1)).strip() if label_match else ""
            labels.append(label or "table")
        deduped = []
        seen = set()
        for label in labels[:6]:
            if label in seen:
                continue
            seen.add(label)
            deduped.append(
                ToolFact(
                    kind="table_resource",
                    source=label,
                    summary=f"Available table resource: {label}",
                    data={"table_label": label},
                )
            )
        return deduped

    def _resolve_table_block(self, *, selection_text: str, full_latex_text: str, target_label: str) -> tuple[str, str]:
        target_label = str(target_label or "").strip()
        if target_label and target_label != "__single__":
            resolved = LatexRefResolver().resolve(selection_text=selection_text, full_latex_text=full_latex_text)
            for table in resolved.get("tables", []):
                if str(table.get("label", "")) == target_label and table.get("found"):
                    return str(table.get("block", "")), target_label
            return "", target_label

        if target_label == "__single__":
            matches = list(self._TABLE_ENV_PATTERN.finditer(full_latex_text))
            if len(matches) == 1:
                block = matches[0].group(0)
                label_match = re.search(r"\\label\{([^}]+)\}", block)
                label = str(label_match.group(1)).strip() if label_match else ""
                return block, label
            return "", ""

        resolved = LatexRefResolver().resolve(selection_text=selection_text, full_latex_text=full_latex_text)
        found_tables = [item for item in resolved.get("tables", []) if item.get("found")]
        if len(found_tables) == 1:
            table = found_tables[0]
            return str(table.get("block", "")), str(table.get("label", "")).strip()
        return "", ""
