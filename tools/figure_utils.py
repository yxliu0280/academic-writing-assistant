from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from tools.numeric_verification_kernel import ClaimIR, EvidenceDatum
from tools.text_utils import split_sentences_with_offsets

EvidenceType = Literal[
    "label_read", "callout_read", "legend_only", "axis_estimate", "unknown"
]


@dataclass
class FigureClaim:
    claim_type: str
    figure_id: str
    metric: str
    subject: str
    comparator: str
    denominator: str
    claimed_value: float
    claimed_from: Optional[float]
    claimed_to: Optional[float]
    unit: Optional[str]
    scale: str
    sentence_index: int
    char_span: Tuple[int, int]
    snippet: str


@dataclass
class FigureFact:
    figure_id: str
    value: float
    evidence_type: EvidenceType
    confidence: float
    source: str
    metric: str = ""
    subject: str = "ours"
    unit: str = "raw"
    scale: str = "unknown"
    series_name: str = ""
    x_value: Optional[float] = None
    x_unit: str = ""
    extra: Dict[str, object] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


METRIC_HINT_PATTERN = re.compile(
    r"(accuracy|acc|f1|f-?score|precision|recall|auc|bleu|rouge|wer|error|loss)",
    re.IGNORECASE,
)


def normalize_metric_name(metric: str) -> str:
    metric = metric.strip().lower()
    metric = metric.replace("-", " ")
    metric = re.sub(r"\s+", " ", metric)
    aliases = {"acc": "accuracy", "f score": "f1", "f1 score": "f1"}
    return aliases.get(metric, metric)


def canonicalize_unit(unit: Optional[str]) -> str:
    if not unit:
        return "raw"
    normalized = str(unit).strip().lower()
    if normalized in {"%", "percent", "percentage"}:
        return "percent"
    if normalized in {"pp", "percentage points", "percentage point"}:
        return "pp"
    return "raw"


def infer_scale(value: Optional[float], unit: str) -> str:
    if value is None:
        return "unknown"
    if unit in {"percent", "pp"}:
        return "0_1" if abs(value) <= 1.0 else "0_100"
    if abs(value) <= 1.0:
        return "0_1"
    return "raw"


def _find_metric_hint(sentence: str) -> str:
    match = METRIC_HINT_PATTERN.search(sentence)
    if not match:
        return ""
    return normalize_metric_name(match.group(1))


def extract_figure_claims(text: str) -> List[FigureClaim]:
    claims: List[FigureClaim] = []
    direct_pattern = re.compile(
        r"(?:Figure|Fig\.?)[\s]*(\d+)[^\n.!?]*?(\d+(?:\.\d+)?)\s*(%|percent)?",
        re.IGNORECASE,
    )
    from_to_pattern = re.compile(
        r"(?:Figure|Fig\.?)[\s]*(\d+)[^\n.!?]*?from\s+(\d+(?:\.\d+)?)\s*(%|percent)?\s+to\s+(\d+(?:\.\d+)?)\s*(%|percent)?",
        re.IGNORECASE,
    )

    for sentence in split_sentences_with_offsets(text):
        stext = str(sentence["text"])
        base = int(sentence["start"])
        metric = _find_metric_hint(stext)
        occupied_spans: List[Tuple[int, int]] = []
        for m in from_to_pattern.finditer(stext):
            unit = canonicalize_unit(m.group(5) if m.group(5) else m.group(3))
            to_value = float(m.group(4))
            occupied_spans.append((m.start(), m.end()))
            claims.append(
                FigureClaim(
                    claim_type="from_to",
                    figure_id=m.group(1),
                    metric=metric,
                    subject="ours",
                    comparator="baseline",
                    denominator="baseline",
                    claimed_value=to_value,
                    claimed_from=float(m.group(2)),
                    claimed_to=to_value,
                    unit=unit,
                    scale=infer_scale(to_value, unit),
                    sentence_index=int(sentence["index"]),
                    char_span=(base + m.start(), base + m.end()),
                    snippet=stext,
                )
            )
        for m in direct_pattern.finditer(stext):
            if any(not (m.end() <= s or m.start() >= e) for s, e in occupied_spans):
                continue
            unit = canonicalize_unit(m.group(3))
            value = float(m.group(2))
            claims.append(
                FigureClaim(
                    claim_type="direct_value",
                    figure_id=m.group(1),
                    metric=metric,
                    subject="ours",
                    comparator="baseline",
                    denominator="baseline",
                    claimed_value=value,
                    claimed_from=None,
                    claimed_to=None,
                    unit=unit,
                    scale=infer_scale(value, unit),
                    sentence_index=int(sentence["index"]),
                    char_span=(base + m.start(), base + m.end()),
                    snippet=stext,
                )
            )
    return claims


def figure_claim_to_claim_ir(claim: FigureClaim, *, claim_id: str, abs_tolerance: float, rel_tolerance: float) -> ClaimIR:
    return ClaimIR(
        claim_id=claim_id,
        sentence_index=claim.sentence_index,
        char_span=claim.char_span,
        raw_text=claim.snippet,
        claim_type=claim.claim_type,
        relation="approx",
        metric=claim.metric,
        subject=claim.subject,
        comparator=claim.comparator,
        denominator=claim.denominator,
        declared_value=claim.claimed_value,
        declared_from=claim.claimed_from,
        declared_to=claim.claimed_to,
        declared_unit=claim.unit or "raw",
        declared_scale=claim.scale,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
        rounding_mode="round",
        extractor="figure_claim_regex",
        metadata={"figure_id": claim.figure_id},
    )


def figure_fact_to_evidence_datum(
    fact: FigureFact,
    *,
    source_ref: str,
    ordinal: int,
) -> EvidenceDatum:
    metric = normalize_metric_name(fact.metric) if fact.metric else "figure_metric"
    unit = canonicalize_unit(fact.unit)
    scale = fact.scale if fact.scale in {"0_1", "0_100", "raw", "unknown"} else infer_scale(fact.value, unit)
    subject = str(fact.subject or "ours").strip().lower() or "ours"
    extra = fact.extra or {}
    return EvidenceDatum(
        datum_id=f"{source_ref}:{metric}:{subject}:{ordinal}",
        source_type="figure",
        source_ref=source_ref,
        metric=metric,
        subject=subject,
        value=float(fact.value),
        unit=unit,
        scale=scale,
        dataset=str(extra.get("dataset", "")),
        split=str(extra.get("split", "")),
        setting=str(extra.get("setting", "")),
        row_header=fact.series_name or metric,
        col_header="value",
        row_index=ordinal,
        col_index=0,
        meta={
            "figure_id": fact.figure_id,
            "confidence": float(fact.confidence),
            "evidence_type": fact.evidence_type,
            "source": fact.source,
            "x_value": fact.x_value,
            "x_unit": fact.x_unit,
            **extra,
        },
    )


def normalize_figure_facts_to_evidence_datums(facts: List[FigureFact], *, source_ref: str) -> List[EvidenceDatum]:
    output: List[EvidenceDatum] = []
    for idx, fact in enumerate(facts):
        output.append(figure_fact_to_evidence_datum(fact, source_ref=source_ref, ordinal=idx))
    return output


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _claim_target_for_subject(claim: ClaimIR, subject_norm: str) -> Optional[float]:
    subject = _norm_text(claim.subject or "ours")
    comparator = _norm_text(claim.comparator or claim.denominator or "baseline")
    if _norm_text(claim.claim_type) == "from_to":
        if subject_norm == subject:
            return _safe_float(claim.declared_to if claim.declared_to is not None else claim.declared_value)
        if subject_norm == comparator:
            return _safe_float(claim.declared_from)
    return _safe_float(claim.declared_value)


def _align_claim_target_for_datum(claim: ClaimIR, target: float, datum: EvidenceDatum) -> float:
    target_value = float(target)
    claim_unit = _norm_text(claim.declared_unit)
    claim_scale = _norm_text(claim.declared_scale)
    datum_unit = _norm_text(datum.unit)
    datum_scale = _norm_text(datum.scale)

    claim_ratio_like = claim_scale == "0_1" and claim_unit in {"raw", "percent", "pp"}
    claim_percent_like = claim_scale == "0_100" or claim_unit in {"percent", "pp"}
    datum_ratio_like = datum_scale == "0_1"
    datum_percent_like = datum_scale == "0_100" or datum_unit in {"percent", "pp"}

    if claim_ratio_like and datum_percent_like:
        return target_value * 100.0
    if claim_percent_like and datum_ratio_like:
        return target_value / 100.0
    return target_value


def collapse_figure_evidence_for_claim(
    claim: ClaimIR,
    evidence: List[EvidenceDatum],
) -> Tuple[List[EvidenceDatum], Dict[str, int]]:
    """Collapse multi-point figure evidence into one datum per metric+subject.

    This is a figure-specific target-resolution step to avoid treating
    time/epoch series points as comparator ambiguity.
    """

    groups: Dict[Tuple[str, str], List[EvidenceDatum]] = {}
    for datum in evidence:
        key = (_norm_text(datum.metric), _norm_text(datum.subject))
        groups.setdefault(key, []).append(datum)

    selected: List[EvidenceDatum] = []
    collapsed_groups = 0
    dropped_count = 0
    for (metric_key, subject_key), items in groups.items():
        if len(items) == 1:
            selected.append(items[0])
            continue

        collapsed_groups += 1
        dropped_count += max(0, len(items) - 1)
        with_x = []
        for item in items:
            x_value = _safe_float(item.meta.get("x_value"))
            if x_value is not None:
                with_x.append((x_value, _safe_float(item.meta.get("confidence")) or 0.0, item))
        if with_x:
            # Prefer latest x-position (for line/epoch style charts), then higher confidence.
            with_x.sort(key=lambda x: (x[0], x[1]))
            selected.append(with_x[-1][2])
            continue

        target = _claim_target_for_subject(claim, subject_key)
        if target is not None:
            by_target = sorted(
                items,
                key=lambda d: (
                    abs(float(d.value) - _align_claim_target_for_datum(claim, float(target), d)),
                    -(float(d.meta.get("confidence", 0.0) or 0.0)),
                ),
            )
            selected.append(by_target[0])
            continue

        by_confidence = sorted(items, key=lambda d: float(d.meta.get("confidence", 0.0) or 0.0), reverse=True)
        selected.append(by_confidence[0])

    selected.sort(key=lambda d: d.datum_id)
    return selected, {
        "input_count": len(evidence),
        "output_count": len(selected),
        "collapsed_groups": collapsed_groups,
        "dropped_count": dropped_count,
    }
