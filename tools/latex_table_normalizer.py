from __future__ import annotations

import csv
from dataclasses import dataclass, field
import io
import re
from typing import List


TABULAR_PATTERN = re.compile(
    r"\\begin\{tabular\}(?:\[[^\]]*\])?\{[^}]*\}(.*?)\\end\{tabular\}",
    re.IGNORECASE | re.DOTALL,
)


def _strip_latex_comments(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line
        i = 0
        while i < len(line):
            if line[i] != "%":
                i += 1
                continue
            backslash_count = 0
            j = i - 1
            while j >= 0 and line[j] == "\\":
                backslash_count += 1
                j -= 1
            # Only strip true comments. Keep escaped percent like `\%`.
            if backslash_count % 2 == 0:
                line = line[:i]
                break
            i += 1
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _canonicalize_header_cell(cell: str) -> str:
    text = str(cell or "").strip()
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    if normalized in {"metric", "metrics", "measure", "measures"}:
        return "metric"
    if normalized in {"baseline", "base", "control"}:
        return "baseline"
    if normalized in {"ours", "our"}:
        return "ours"
    return text


def _header_has_metric_baseline_ours(row: List[str]) -> bool:
    canonical = [_canonicalize_header_cell(cell) for cell in row]
    needed = {"metric", "baseline", "ours"}
    return needed.issubset(set(canonical))


def _flatten_header_rows(rows: List[List[str]], warnings: List[str]) -> List[List[str]]:
    if not rows:
        return rows

    if _header_has_metric_baseline_ours(rows[0]):
        return [[_canonicalize_header_cell(cell) for cell in rows[0]], *rows[1:]]

    max_probe = min(2, len(rows) - 1)
    for idx in range(max_probe):
        top = rows[idx]
        sub = rows[idx + 1]
        canonical_sub = [_canonicalize_header_cell(cell) for cell in sub]
        if "baseline" not in canonical_sub or "ours" not in canonical_sub:
            continue

        merged_header = list(canonical_sub)
        if not merged_header:
            continue
        if _canonicalize_header_cell(merged_header[0]) != "metric":
            merged_header[0] = "metric"
        if _header_has_metric_baseline_ours(merged_header):
            warnings.append("HEADER_FLATTENED")
            return [merged_header, *rows[idx + 2 :]]
    return rows


def _sanitize_latex_cell(cell: str) -> str:
    text = cell.strip()
    # Keep payload text for common wrappers before generic command stripping.
    wrappers = [
        r"\\multicolumn\{[^}]+\}\{[^}]*\}\{([^}]*)\}",
        r"\\multirow\{[^}]+\}\{[^}]*\}\{([^}]*)\}",
        r"\\textbf\{([^}]*)\}",
        r"\\emph\{([^}]*)\}",
        r"\\underline\{([^}]*)\}",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in wrappers:
            new_text = re.sub(pattern, r"\1", text)
            if new_text != text:
                text = new_text
                changed = True

    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = re.sub(r"\\(?:pm|times|cdot|quad|qquad|,|;)\b", " ", text)
    text = text.replace("\\%", "%")
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    text = text.replace("$", "")
    # Strip superscript/subscript footnote markers attached to values.
    text = re.sub(r"\^\{[^}]*\}", "", text)
    text = re.sub(r"_\{[^}]*\}", "", text)
    text = re.sub(r"\^[A-Za-z]+", "", text)
    text = re.sub(r"_[A-Za-z]+", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = text.replace("\\", "")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"(?<=\d)[\*\u2020\u2021]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class LatexTableNormalizationResult:
    csv_content: str
    row_count: int
    col_count: int
    warnings: List[str] = field(default_factory=list)


def normalize_latex_table_block(table_block: str) -> LatexTableNormalizationResult:
    tabular_match = TABULAR_PATTERN.search(table_block or "")
    if not tabular_match:
        return LatexTableNormalizationResult(
            csv_content="",
            row_count=0,
            col_count=0,
            warnings=["TABULAR_NOT_FOUND"],
        )

    tabular_body = tabular_match.group(1)
    tabular_body = _strip_latex_comments(tabular_body)
    tabular_body = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|cmidrule\{[^}]+\})", "", tabular_body)

    rows: List[List[str]] = []
    warnings: List[str] = []
    if re.search(r"\\(?:multicolumn|multirow)\b", tabular_body):
        warnings.append("ADVANCED_CELL_SPAN_PRESENT")
    for raw_row in re.split(r"\\\\", tabular_body):
        row = raw_row.strip()
        if not row or "&" not in row:
            continue
        cells = [_sanitize_latex_cell(cell) for cell in row.split("&")]
        if not any(cells):
            continue
        rows.append(cells)

    if not rows:
        return LatexTableNormalizationResult(
            csv_content="",
            row_count=0,
            col_count=0,
            warnings=[*warnings, "NO_VALID_ROWS"],
        )

    max_cols = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    normalized_rows = _flatten_header_rows(normalized_rows, warnings)
    if normalized_rows:
        max_cols = max(len(row) for row in normalized_rows)
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in normalized_rows]

    output = io.StringIO()
    writer = csv.writer(output)
    for row in normalized_rows:
        writer.writerow(row)
    csv_content = output.getvalue().strip()
    return LatexTableNormalizationResult(
        csv_content=csv_content,
        row_count=len(normalized_rows),
        col_count=max_cols,
        warnings=warnings,
    )


def latex_table_block_to_csv(table_block: str) -> str:
    return normalize_latex_table_block(table_block).csv_content
