from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from tools.numeric_verification_kernel import ClaimIR, EvidenceDatum
from tools.text_utils import split_sentences_with_offsets


@dataclass
class NumericClaim:
    claim_type: str
    metric: Optional[str]
    value: Optional[float]
    unit: Optional[str]
    sentence_index: int
    char_span: Tuple[int, int]
    snippet: str
    direction: Optional[str] = None
    reference: Optional[str] = None
    extractor: str = "regex"
    from_value: Optional[float] = None
    to_value: Optional[float] = None
    comparator: Optional[str] = None
    subject: Optional[str] = None


@dataclass
class MetricRow:
    metric: str
    baseline: float
    ours: float
    baseline_unit: str = "raw"
    ours_unit: str = "raw"
    baseline_scale: str = "unknown"
    ours_scale: str = "unknown"


METRIC_HINT_PATTERN = re.compile(
    r"\b(accuracy|accur[a-z]*|acc|f1|f-?score|precision|recall|auc|bleu|rouge|wer|error(?:\s+rate)?|loss)\b",
    re.IGNORECASE,
)

PERCENT_METRIC_PATTERN = re.compile(
    r"("
    r"accuracy|acc|f1|f-?score|precision|recall|auc|bleu|rouge|wer|error|"
    r"\bem\b|"
    r"mnli|qnli|qqp|rte|sst|cola|sts|mrpc|race|squad|glue|top[- ]?1|"
    r"\bavg\b|\baverage\b|score"
    r")",
    re.IGNORECASE,
)

RAW_METRIC_PATTERN = re.compile(
    r"(params?|flops?|steps?|speedup|latency|time|memory|throughput|ppl|perplexity|loss)",
    re.IGNORECASE,
)

ERROR_REDUCTION_PATTERN = re.compile(
    r"(?:relative\s+)?error\s+reduction(?:\s+(?:is|was|were|equals?|of|by))?\s+(\d+(?:\.\d+)?)\s*(\\?%|percent)?",
    re.IGNORECASE,
)

ERROR_DECREASE_PATTERN = re.compile(
    r"error(?:\s+rate)?\s+(?:(?:is|was|were)\s+)?(?:decreases?|decreased|reduce|reduced)\s+(?:by|of)\s+(\d+(?:\.\d+)?)\s*(\\?%|percent)?",
    re.IGNORECASE,
)


def normalize_metric_name(metric: str) -> str:
    metric = metric.strip().lower()
    metric = metric.replace("-", " ")
    metric = re.sub(r"[^\w\s.%/]", " ", metric)
    metric = re.sub(r"\s+", " ", metric)
    aliases = {
        "acc": "accuracy",
        "f score": "f1",
        "f1 score": "f1",
        "error rate": "error",
    }
    metric = aliases.get(metric, metric)
    return metric


def canonicalize_unit(unit: Optional[str]) -> str:
    if not unit:
        return "raw"
    u = unit.strip().lower().replace("\\", "")
    if u in {"percentage point", "percentage points", "pp"}:
        return "pp"
    if u in {"%", "percent", "percentage"}:
        return "%"
    return "raw"


def kernel_unit_from_table_unit(unit: Optional[str]) -> str:
    normalized = canonicalize_unit(unit)
    if normalized == "%":
        return "percent"
    return normalized


def detect_table_format(df: pd.DataFrame) -> str:
    cols = {c.strip().lower() for c in df.columns}
    if {"metric", "baseline", "ours"}.issubset(cols):
        return "wide"
    if {"metric", "method", "value"}.issubset(cols):
        return "long"
    if _detect_matrix_numeric_columns(df):
        return "matrix"
    return "unknown"


def _column_display_name(col: object) -> str:
    text = str(col or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("unnamed:"):
        return ""
    return text


def _can_parse_numeric(value: object) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    # Reject alphanumeric identifiers like "14B", "Model-A", "6 langs".
    residual = re.sub(r"[0-9eE+\-.,%() \t]", "", lowered)
    if re.search(r"[a-z]", residual):
        return False
    cleaned = raw.replace(",", "")
    cleaned = re.sub(r"percent", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("%", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", cleaned, flags=re.IGNORECASE)
    return bool(match)


def _detect_matrix_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_columns: List[str] = []
    for col in df.columns:
        values = [str(v).strip() for v in df[col].tolist() if str(v or "").strip()]
        if not values:
            continue
        parseable = sum(1 for cell in values if _can_parse_numeric(cell))
        ratio = parseable / max(len(values), 1)
        # Require enough parseable values to avoid treating textual identifier
        # columns as metrics.
        if parseable >= 2 and ratio >= 0.6:
            numeric_columns.append(str(col))
    return numeric_columns


def _build_matrix_subject(
    row: pd.Series,
    *,
    subject_columns: List[str],
    row_index: int,
) -> str:
    parts: List[str] = []
    for col in subject_columns:
        value = str(row.get(col, "") or "").strip()
        if not value:
            continue
        col_name = _column_display_name(col)
        if col_name and len(subject_columns) > 1:
            parts.append(f"{col_name}={value}")
        else:
            parts.append(value)
    subject = " | ".join(parts).strip()
    if subject:
        return subject
    return f"row_{row_index}"


def _normalize_subject_role(subject: str) -> str:
    normalized = str(subject or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in {"baseline", "base", "control"}:
        return "baseline"
    if normalized in {"ours", "our", "proposed", "method", "model"}:
        return "ours"
    return normalized


def _metric_prefers_percent(metric: str) -> bool:
    metric_text = str(metric or "")
    if RAW_METRIC_PATTERN.search(metric_text):
        return False
    return bool(PERCENT_METRIC_PATTERN.search(metric_text))


def _parse_numeric_cell(value: object, metric_hint: str) -> Tuple[float, str, str]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("empty numeric cell")

    has_percent = bool(re.search(r"(?:%|percent)", raw, flags=re.IGNORECASE))
    cleaned = raw.replace(",", "")
    cleaned = re.sub(r"percent", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("%", "").strip()

    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", cleaned, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unable to parse numeric value from cell '{raw}'")
    parsed = float(match.group(0))

    if has_percent:
        return parsed, "percent", "0_100"

    if _metric_prefers_percent(metric_hint):
        if abs(parsed) <= 1.0:
            return parsed, "percent", "0_1"
        if abs(parsed) <= 100.0:
            return parsed, "percent", "0_100"

    return parsed, "raw", "raw"


def parse_table_csv(content: str) -> Dict[str, MetricRow]:
    # Keep textual labels (e.g., method "None") as-is; they are valid subjects.
    df = pd.read_csv(StringIO(content), keep_default_na=False, na_filter=False)
    fmt = detect_table_format(df)
    rows: Dict[str, MetricRow] = {}

    if fmt == "wide":
        for _, row in df.iterrows():
            metric = normalize_metric_name(str(row["metric"]))
            baseline, baseline_unit, baseline_scale = _parse_numeric_cell(row["baseline"], metric)
            ours, ours_unit, ours_scale = _parse_numeric_cell(row["ours"], metric)
            rows[metric] = MetricRow(
                metric=metric,
                baseline=float(baseline),
                ours=float(ours),
                baseline_unit=baseline_unit,
                ours_unit=ours_unit,
                baseline_scale=baseline_scale,
                ours_scale=ours_scale,
            )
        return rows

    if fmt == "long":
        grouped: Dict[str, Dict[str, Tuple[float, str, str]]] = {}
        for _, row in df.iterrows():
            metric = normalize_metric_name(str(row["metric"]))
            method = str(row["method"]).strip().lower()
            parsed = _parse_numeric_cell(row["value"], metric)
            grouped.setdefault(metric, {})[method] = parsed
        for metric, methods in grouped.items():
            if "baseline" in methods and "ours" in methods:
                baseline_value, baseline_unit, baseline_scale = methods["baseline"]
                ours_value, ours_unit, ours_scale = methods["ours"]
                rows[metric] = MetricRow(
                    metric=metric,
                    baseline=float(baseline_value),
                    ours=float(ours_value),
                    baseline_unit=baseline_unit,
                    ours_unit=ours_unit,
                    baseline_scale=baseline_scale,
                    ours_scale=ours_scale,
                )
        return rows

    if fmt == "matrix":
        numeric_columns = _detect_matrix_numeric_columns(df)
        subject_columns = [str(col) for col in df.columns if str(col) not in set(numeric_columns)]
        if subject_columns:
            df[subject_columns] = df[subject_columns].replace("", pd.NA).ffill().fillna("")
        grouped: Dict[str, Dict[str, Tuple[float, str, str]]] = {}
        for idx, row in df.iterrows():
            subject = _normalize_subject_role(
                _build_matrix_subject(row, subject_columns=subject_columns, row_index=int(idx))
            )
            if subject not in {"baseline", "ours"}:
                continue
            for col in numeric_columns:
                metric_name = normalize_metric_name(_column_display_name(col) or str(col))
                if not metric_name:
                    continue
                raw_cell = row.get(col, "")
                if not str(raw_cell or "").strip():
                    continue
                try:
                    value, unit, scale = _parse_numeric_cell(raw_cell, metric_name)
                except Exception:
                    continue
                grouped.setdefault(metric_name, {})[subject] = (float(value), unit, scale)
        for metric, methods in grouped.items():
            if "baseline" in methods and "ours" in methods:
                baseline_value, baseline_unit, baseline_scale = methods["baseline"]
                ours_value, ours_unit, ours_scale = methods["ours"]
                rows[metric] = MetricRow(
                    metric=metric,
                    baseline=float(baseline_value),
                    ours=float(ours_value),
                    baseline_unit=baseline_unit,
                    ours_unit=ours_unit,
                    baseline_scale=baseline_scale,
                    ours_scale=ours_scale,
                )
        return rows

    raise ValueError(
        "Unsupported CSV format; expect wide(metric,baseline,ours), long(metric,method,value), or matrix-style table."
    )


def parse_table_evidence_csv(
    content: str,
    *,
    source_ref: str,
) -> List[EvidenceDatum]:
    # Keep textual labels (e.g., method "None") as-is; they are valid subjects.
    df = pd.read_csv(StringIO(content), keep_default_na=False, na_filter=False)
    fmt = detect_table_format(df)
    evidence: List[EvidenceDatum] = []

    if fmt == "wide":
        for idx, row in df.iterrows():
            metric = normalize_metric_name(str(row["metric"]))
            try:
                baseline, baseline_unit, baseline_scale = _parse_numeric_cell(row["baseline"], metric)
                ours, ours_unit, ours_scale = _parse_numeric_cell(row["ours"], metric)
            except Exception:
                # Real-world normalized CSV can still contain sparse/non-numeric
                # cells in wide layout. Skip bad rows instead of failing
                # the entire table parse.
                continue
            evidence.append(
                EvidenceDatum(
                    datum_id=f"{source_ref}:{metric}:baseline:{idx}",
                    source_type="table",
                    source_ref=source_ref,
                    metric=metric,
                    subject="baseline",
                    value=float(baseline),
                    unit=baseline_unit,
                    scale=baseline_scale,
                    row_header=metric,
                    col_header="baseline",
                    row_index=int(idx),
                    col_index=0,
                )
            )
            evidence.append(
                EvidenceDatum(
                    datum_id=f"{source_ref}:{metric}:ours:{idx}",
                    source_type="table",
                    source_ref=source_ref,
                    metric=metric,
                    subject="ours",
                    value=float(ours),
                    unit=ours_unit,
                    scale=ours_scale,
                    row_header=metric,
                    col_header="ours",
                    row_index=int(idx),
                    col_index=1,
                )
            )
        return evidence

    if fmt == "long":
        for idx, row in df.iterrows():
            metric = normalize_metric_name(str(row["metric"]))
            subject = str(row["method"]).strip()
            if not metric or not subject:
                continue
            try:
                value, unit, scale = _parse_numeric_cell(row["value"], metric)
            except Exception:
                # Long layout frequently includes textual placeholders or empty
                # cells on some rows; keep remaining numeric rows parseable.
                continue
            evidence.append(
                EvidenceDatum(
                    datum_id=f"{source_ref}:{metric}:{subject}:{idx}",
                    source_type="table",
                    source_ref=source_ref,
                    metric=metric,
                    subject=subject,
                    value=float(value),
                    unit=unit,
                    scale=scale,
                    row_header=subject,
                    col_header="value",
                    row_index=int(idx),
                    col_index=0,
                )
            )
        return evidence

    if fmt == "matrix":
        numeric_columns = _detect_matrix_numeric_columns(df)
        subject_columns = [str(col) for col in df.columns if str(col) not in set(numeric_columns)]
        # Forward-fill subject columns to handle compact table layouts where
        # row headers are omitted on continuation rows.
        if subject_columns:
            df[subject_columns] = df[subject_columns].replace("", pd.NA).ffill().fillna("")
        for idx, row in df.iterrows():
            subject = _build_matrix_subject(row, subject_columns=subject_columns, row_index=int(idx))
            for col in numeric_columns:
                metric_name = normalize_metric_name(_column_display_name(col) or str(col))
                if not metric_name:
                    continue
                raw_cell = row.get(col, "")
                if not str(raw_cell or "").strip():
                    continue
                try:
                    value, unit, scale = _parse_numeric_cell(raw_cell, metric_name)
                except Exception:
                    # Matrix tables often mix textual flags (e.g., "Part", "Ltd")
                    # with numeric cells. Skip those cells instead of failing
                    # the whole table parse.
                    continue
                evidence.append(
                    EvidenceDatum(
                        datum_id=f"{source_ref}:{metric_name}:{subject}:{idx}",
                        source_type="table",
                        source_ref=source_ref,
                        metric=metric_name,
                        subject=subject,
                        value=float(value),
                        unit=unit,
                        scale=scale,
                        row_header=subject,
                        col_header=metric_name,
                        row_index=int(idx),
                        col_index=int(list(df.columns).index(col)),
                        meta={
                            "parser_mode": "matrix",
                            "raw_metric_column": str(col),
                        },
                    )
                )
                # Matrix tables are frequently interpreted in both orientations:
                # - metric=column, subject=row-header (default above)
                # - metric=row-header, subject=column
                # Add a mirrored datum so downstream resolver can match either
                # claim convention without relying on fragile fuzzy inference.
                evidence.append(
                    EvidenceDatum(
                        datum_id=f"{source_ref}:{subject}:{metric_name}:{idx}:mirror",
                        source_type="table",
                        source_ref=source_ref,
                        metric=normalize_metric_name(subject),
                        subject=metric_name,
                        value=float(value),
                        unit=unit,
                        scale=scale,
                        row_header=subject,
                        col_header=metric_name,
                        row_index=int(idx),
                        col_index=int(list(df.columns).index(col)),
                        meta={
                            "parser_mode": "matrix",
                            "parser_variant": "mirrored_metric_subject",
                            "raw_metric_column": str(col),
                        },
                    )
                )
        return evidence

    raise ValueError("Unsupported CSV format; expect wide(metric,baseline,ours) or long(metric,method,value)")


def _find_metric_hint(sentence: str) -> Optional[str]:
    m = METRIC_HINT_PATTERN.search(sentence)
    if m:
        return normalize_metric_name(m.group(1))
    return None


def _infer_claim_type_from_change(unit: str, direction: Optional[str]) -> str:
    if unit == "pp":
        return "pp_change"
    if direction in {"decrease", "reduce"}:
        return "relative_decrease"
    return "relative_improvement"


def extract_numeric_claims(text: str) -> List[NumericClaim]:
    claims: List[NumericClaim] = []

    improve_pattern = re.compile(
        r"(improves?|increases?|decreases?|reduces?|outperform(?:s|ed|ing)?)\s+(?:by|of)\s+(\d+(?:\.\d+)?)\s*(percentage\s+points?|pp|\\?%|percent(?:age)?)?(?:\s+(?:over|vs\.?|than)\s+(baseline|control))?",
        re.IGNORECASE,
    )
    outperform_pattern = re.compile(
        r"(outperform(?:s|ed|ing)?)\s+(?:the\s+)?(baseline|control)\s+(?:by|of)\s+(\d+(?:\.\d+)?)\s*(percentage\s+points?|pp|\\?%|percent(?:age)?)?",
        re.IGNORECASE,
    )
    reaches_pattern = re.compile(
        r"(reaches?|achieves?|is|was)\s+(\d+(?:\.\d+)?)\s*(\\?%|percent)?",
        re.IGNORECASE,
    )
    metric_is_value_pattern = re.compile(
        r"\b(accuracy|accur[a-z]*|acc|f1|f-?score|precision|recall|auc|bleu|rouge|wer|error(?:\s+rate)?|loss)\s+(?:is|was|equals?)\s+(\d+(?:\.\d+)?)\s*(\\?%|percent)",
        re.IGNORECASE,
    )
    from_to_pattern = re.compile(
        r"from\s+(\d+(?:\.\d+)?)\s*(\\?%|percent)?\s+to\s+(\d+(?:\.\d+)?)\s*(\\?%|percent)?",
        re.IGNORECASE,
    )

    for sentence in split_sentences_with_offsets(text):
        stext = str(sentence["text"])
        start = int(sentence["start"])
        metric = _find_metric_hint(stext)
        sentence_index = int(sentence["index"])

        for m in ERROR_REDUCTION_PATTERN.finditer(stext):
            matched = str(m.group(0) or "").strip().lower()
            claim_type = "relative_error_reduction" if matched.startswith("relative") else "error_reduction"
            claims.append(
                NumericClaim(
                    claim_type=claim_type,
                    metric=metric,
                    value=float(m.group(1)),
                    unit=canonicalize_unit(m.group(2) if m.group(2) else "%"),
                    sentence_index=sentence_index,
                    char_span=(start + m.start(), start + m.end()),
                    snippet=stext,
                    direction="decrease",
                    comparator="baseline",
                    subject="ours",
                )
            )

        for m in ERROR_DECREASE_PATTERN.finditer(stext):
            claims.append(
                NumericClaim(
                    claim_type="error_reduction",
                    metric=metric,
                    value=float(m.group(1)),
                    unit=canonicalize_unit(m.group(2) if m.group(2) else "%"),
                    sentence_index=sentence_index,
                    char_span=(start + m.start(), start + m.end()),
                    snippet=stext,
                    direction="decrease",
                    comparator="baseline",
                    subject="ours",
                )
            )

        for m in improve_pattern.finditer(stext):
            verb = m.group(1).lower()
            direction = "decrease" if verb.startswith(("decreas", "reduc")) else "improve"
            unit = canonicalize_unit(m.group(3))
            claims.append(
                NumericClaim(
                    claim_type=_infer_claim_type_from_change(unit, direction),
                    metric=metric,
                    value=float(m.group(2)),
                    unit=unit,
                    sentence_index=sentence_index,
                    char_span=(start + m.start(), start + m.end()),
                    snippet=stext,
                    direction=direction,
                    comparator=str(m.group(4) or "baseline").lower(),
                    subject="ours",
                )
            )

        for m in outperform_pattern.finditer(stext):
            claims.append(
                NumericClaim(
                    claim_type="pp_change",
                    metric=metric,
                    value=float(m.group(3)),
                    unit=canonicalize_unit(m.group(4)),
                    sentence_index=sentence_index,
                    char_span=(start + m.start(), start + m.end()),
                    snippet=stext,
                    direction="improve",
                    comparator=str(m.group(2) or "baseline").lower(),
                    subject="ours",
                )
            )

        for m in from_to_pattern.finditer(stext):
            claims.append(
                NumericClaim(
                    claim_type="from_to",
                    metric=metric,
                    value=float(m.group(3)),
                    unit=canonicalize_unit(m.group(4) if m.group(4) else m.group(2)),
                    from_value=float(m.group(1)),
                    to_value=float(m.group(3)),
                    sentence_index=sentence_index,
                    char_span=(start + m.start(), start + m.end()),
                    snippet=stext,
                    comparator="baseline",
                    subject="ours",
                )
            )

        for m in reaches_pattern.finditer(stext):
            verb = m.group(1).lower()
            if verb.startswith("reach") or verb.startswith("achieve"):
                unit = canonicalize_unit(m.group(3))
                claims.append(
                    NumericClaim(
                        claim_type="direct_value",
                        metric=metric,
                        value=float(m.group(2)),
                        unit=unit,
                        sentence_index=sentence_index,
                        char_span=(start + m.start(), start + m.end()),
                        snippet=stext,
                        subject="ours",
                        comparator="baseline",
                    )
                )

        for m in metric_is_value_pattern.finditer(stext):
            metric_phrase = normalize_metric_name(str(m.group(1)))
            unit = canonicalize_unit(m.group(3))
            claims.append(
                NumericClaim(
                    claim_type="direct_value",
                    metric=metric_phrase,
                    value=float(m.group(2)),
                    unit=unit,
                    sentence_index=sentence_index,
                    char_span=(start + m.start(), start + m.end()),
                    snippet=stext,
                    subject="ours",
                    comparator="baseline",
                )
            )

    return claims


def _infer_declared_scale(claim_value: Optional[float], claim_unit: str) -> str:
    if claim_value is None:
        return "unknown"
    if claim_unit in {"%", "pp"}:
        if abs(claim_value) <= 1.0:
            return "0_1"
        return "0_100"
    if abs(claim_value) <= 1.0:
        return "0_1"
    return "raw"


def claim_to_claim_ir(
    claim: NumericClaim,
    *,
    claim_id: str,
    abs_tolerance: float,
    rel_tolerance: float,
    extractor: str,
) -> ClaimIR:
    claim_unit = canonicalize_unit(claim.unit)
    declared_scale = _infer_declared_scale(claim.value, claim_unit)
    decimals = None
    if claim.value is not None:
        raw_text = str(claim.value)
        if "." in raw_text:
            decimals = len(raw_text.split(".", 1)[1])
    return ClaimIR(
        claim_id=claim_id,
        sentence_index=claim.sentence_index,
        char_span=claim.char_span,
        raw_text=claim.snippet,
        claim_type=claim.claim_type,
        relation="approx",
        metric=str(claim.metric or ""),
        subject=str(claim.subject or "ours"),
        comparator=str(claim.comparator or "baseline"),
        denominator=str(claim.comparator or "baseline"),
        declared_value=claim.value,
        declared_from=claim.from_value,
        declared_to=claim.to_value,
        declared_unit=kernel_unit_from_table_unit(claim_unit),
        declared_scale=declared_scale,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
        rounding_mode="round",
        decimals=decimals,
        extractor=extractor,
        ambiguity_flags=[],
        metadata={"reference": claim.reference or "", "direction": claim.direction or ""},
    )


def extract_claim_irs(
    text: str,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
    extractor: str = "regex_fallback",
) -> List[ClaimIR]:
    claims = extract_numeric_claims(text)
    output: List[ClaimIR] = []
    for idx, claim in enumerate(claims):
        output.append(
            claim_to_claim_ir(
                claim,
                claim_id=f"claim_{claim.sentence_index}_{idx}",
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
                extractor=extractor,
            )
        )
    return output


def metric_rows_to_evidence_datums(
    metric_rows: Dict[str, MetricRow],
    *,
    source_ref: str,
) -> List[EvidenceDatum]:
    evidence: List[EvidenceDatum] = []
    for idx, (metric, row) in enumerate(metric_rows.items()):
        evidence.append(
            EvidenceDatum(
                datum_id=f"{source_ref}:{metric}:baseline:{idx}",
                source_type="table",
                source_ref=source_ref,
                metric=metric,
                subject="baseline",
                value=float(row.baseline),
                unit=row.baseline_unit,
                scale=row.baseline_scale,
                row_header=metric,
                col_header="baseline",
                row_index=idx,
                col_index=0,
            )
        )
        evidence.append(
            EvidenceDatum(
                datum_id=f"{source_ref}:{metric}:ours:{idx}",
                source_type="table",
                source_ref=source_ref,
                metric=metric,
                subject="ours",
                value=float(row.ours),
                unit=row.ours_unit,
                scale=row.ours_scale,
                row_header=metric,
                col_header="ours",
                row_index=idx,
                col_index=1,
            )
        )
    return evidence


def compare_claim_to_metric(
    claim: NumericClaim,
    row: MetricRow,
    abs_tolerance: float,
    rel_tolerance: float,
) -> Dict[str, float | str | bool]:
    baseline = row.baseline
    ours = row.ours
    rel_improvement = ((ours - baseline) / baseline * 100.0) if baseline != 0 else 0.0
    pp_improvement = ours - baseline

    expected = None
    comparator = "unknown"
    if claim.claim_type == "direct_value":
        expected = ours
        comparator = "ours"
    elif claim.claim_type in {"pp_change"}:
        expected = pp_improvement
        comparator = "pp_delta"
    elif claim.claim_type in {"relative_improvement", "relative_decrease", "relative_error_reduction", "error_reduction"}:
        expected = rel_improvement if claim.claim_type == "relative_improvement" else -rel_improvement
        comparator = "relative_delta_percent"
    elif claim.claim_type == "from_to":
        expected = claim.to_value
        comparator = "from_to_to_value"

    claimed = claim.value if claim.value is not None else 0.0
    expected = float(expected or 0.0)

    abs_diff = abs(claimed - expected)
    rel_diff = abs_diff / max(abs(expected), 1e-9)
    ok = abs_diff <= abs_tolerance or rel_diff <= rel_tolerance

    return {
        "ok": ok,
        "claimed": claimed,
        "expected": expected,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "baseline": baseline,
        "ours": ours,
        "relative_improvement_percent": rel_improvement,
        "pp_improvement": pp_improvement,
        "comparator": comparator,
    }
