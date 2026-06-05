from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


ClaimType = str
Verdict = str
Unit = str
Scale = str


SUPPORTED_VERDICTS = {"supported", "contradicted", "unsupported", "uncertain"}

FATAL_REASON_CODES = {
    "TARGET_NO_MATCH",
    "TARGET_MULTI_MATCH",
    "CLAIM_NO_VALUE",
    "DENOMINATOR_MISSING",
    "DENOMINATOR_AMBIGUOUS",
}


@dataclass
class ClaimIR:
    claim_id: str
    sentence_index: int
    char_span: Tuple[int, int]
    raw_text: str
    claim_type: ClaimType
    relation: str = "eq"
    metric: str = ""
    subject: str = "ours"
    comparator: str = "baseline"
    denominator: str = "baseline"
    declared_value: Optional[float] = None
    declared_from: Optional[float] = None
    declared_to: Optional[float] = None
    declared_unit: Unit = "raw"
    declared_scale: Scale = "unknown"
    abs_tolerance: float = 0.2
    rel_tolerance: float = 0.02
    rounding_mode: str = "round"
    decimals: Optional[int] = None
    extractor: str = "regex"
    confidence: Optional[float] = None
    ambiguity_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceDatum:
    datum_id: str
    source_type: str
    source_ref: str
    metric: str
    subject: str
    value: float
    unit: Unit = "raw"
    scale: Scale = "unknown"
    dataset: str = ""
    split: str = ""
    setting: str = ""
    row_header: str = ""
    col_header: str = ""
    row_index: int = -1
    col_index: int = -1
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceStep:
    step_name: str
    status: str
    message: str
    input_refs: List[str] = field(default_factory=list)
    output_refs: List[str] = field(default_factory=list)
    candidate_count: int = -1
    expression: str = ""
    bindings: Dict[str, Any] = field(default_factory=dict)
    intermediate: Dict[str, Any] = field(default_factory=dict)
    error_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationTrace:
    trace_id: str
    claim_id: str
    kernel_version: str
    steps: List[TraceStep] = field(default_factory=list)
    final_verdict: Verdict = "unsupported"
    expected_value: Optional[float] = None
    observed_value: Optional[float] = None
    delta: Optional[float] = None
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "claim_id": self.claim_id,
            "kernel_version": self.kernel_version,
            "steps": [step.to_dict() for step in self.steps],
            "final_verdict": self.final_verdict,
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "delta": self.delta,
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class VerificationResult:
    verdict: Verdict
    expected_value: Optional[float]
    observed_value: Optional[float]
    abs_diff: Optional[float]
    rel_diff: Optional[float]
    reason_codes: List[str]
    trace: VerificationTrace
    claim: ClaimIR
    resolved_metric: str = ""
    resolved_subject: str = ""
    resolved_comparator: str = ""
    formula: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["trace"] = self.trace.to_dict()
        payload["claim"] = self.claim.to_dict()
        return payload

    def to_eval_record(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim.claim_id,
            "claim_type": self.claim.claim_type,
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "abs_diff": self.abs_diff,
            "rel_diff": self.rel_diff,
            "resolved_metric": self.resolved_metric,
            "resolved_subject": self.resolved_subject,
            "resolved_comparator": self.resolved_comparator,
            "formula": self.formula,
            "trace_id": self.trace.trace_id,
            "kernel_version": self.trace.kernel_version,
        }


class NumericVerificationKernel:
    """Deterministic numeric verification kernel.

    This kernel is intentionally modality-agnostic: table-text and future
    figure-text can share the same verification path as long as evidence is
    normalized into `EvidenceDatum`.
    """

    version = "ncvf-mvp-v1"

    def verify_claim(self, claim: ClaimIR, evidence: List[EvidenceDatum]) -> VerificationResult:
        trace = VerificationTrace(
            trace_id=f"trace-{uuid4().hex[:10]}",
            claim_id=claim.claim_id,
            kernel_version=self.version,
        )
        reason_codes: List[str] = []
        self._hydrate_missing_declared_values(claim, trace, reason_codes)

        metric_evidence, resolved_metric = self._resolve_metric_candidates(claim, evidence, trace, reason_codes)
        if not metric_evidence:
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="unsupported",
                expected=None,
                observed=None,
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes or ["TARGET_NO_MATCH"],
            )

        normalized_claim_type = self._normalize_text(claim.claim_type)
        if normalized_claim_type == "unsupported":
            reason_codes.append("TARGET_AMBIGUOUS")
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="unsupported",
                expected=None,
                observed=self._safe_float(claim.declared_value),
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes,
                resolved_metric=resolved_metric,
                formula="unsupported_boundary_claim",
            )

        subject_datum, comparator_datum = self._resolve_subject_pair(
            claim,
            metric_evidence,
            trace,
            reason_codes,
        )
        if subject_datum is None:
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="unsupported",
                expected=None,
                observed=None,
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes or ["TARGET_NO_MATCH"],
            )
        if (
            claim.claim_type == "from_to"
            and comparator_datum is None
            and self._is_figure_evidence(evidence)
        ):
            axis_swap_pair = self._infer_from_to_pair_for_figure_axis_swap(
                claim=claim,
                evidence=evidence,
                fallback_subject=subject_datum,
            )
            if axis_swap_pair is not None:
                axis_subject, axis_comparator = axis_swap_pair
                subject_datum = axis_subject
                comparator_datum = axis_comparator
                reason_codes = [
                    code for code in reason_codes if code not in {"DENOMINATOR_MISSING", "DENOMINATOR_AMBIGUOUS"}
                ]
                reason_codes.append("FROM_TO_INFERRED_FROM_AXIS_SWAP")
                trace.steps.append(
                    TraceStep(
                        step_name="resolve_from_to_axis_swap",
                        status="ok",
                        message="resolved from/to pair via figure axis-swap fallback",
                        output_refs=[subject_datum.datum_id, comparator_datum.datum_id],
                    )
                )

        observed_value, observed_reason = self._normalize_observed_value(claim)
        if observed_reason:
            reason_codes.append(observed_reason)

        if normalized_claim_type == "uncertain":
            reason_codes.append("DENOMINATOR_AMBIGUOUS")
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="uncertain",
                expected=None,
                observed=observed_value,
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes,
                resolved_metric=resolved_metric,
                resolved_subject=subject_datum.subject if subject_datum else "",
                resolved_comparator=comparator_datum.subject if comparator_datum else "",
                formula="uncertain_boundary_claim",
            )

        if claim.claim_type == "from_to":
            return self._verify_from_to(
                claim=claim,
                trace=trace,
                reason_codes=reason_codes,
                subject_datum=subject_datum,
                comparator_datum=comparator_datum,
                resolved_metric=resolved_metric,
            )

        if observed_value is None:
            reason_codes.append("CLAIM_NO_VALUE")
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="unsupported",
                expected=None,
                observed=None,
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes,
                resolved_metric=resolved_metric,
                resolved_subject=subject_datum.subject,
                resolved_comparator=comparator_datum.subject if comparator_datum else "",
            )

        expected_value, formula, compute_reason = self._compute_expected_value(
            claim=claim,
            subject_datum=subject_datum,
            comparator_datum=comparator_datum,
        )
        if compute_reason:
            reason_codes.append(compute_reason)
        trace.steps.append(
            TraceStep(
                step_name="compute",
                status="ok" if expected_value is not None else "fail",
                message="deterministic formula execution",
                input_refs=[subject_datum.datum_id] + ([comparator_datum.datum_id] if comparator_datum else []),
                expression=formula,
                bindings={
                    "claim_type": claim.claim_type,
                    "subject_value": subject_datum.value,
                    "subject_unit": subject_datum.unit,
                    "subject_scale": subject_datum.scale,
                    "comparator_value": comparator_datum.value if comparator_datum else None,
                    "comparator_unit": comparator_datum.unit if comparator_datum else None,
                    "comparator_scale": comparator_datum.scale if comparator_datum else None,
                },
                intermediate={"expected_value": expected_value},
                error_code=compute_reason or "",
            )
        )

        if expected_value is None:
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="unsupported",
                expected=None,
                observed=observed_value,
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes,
                resolved_metric=resolved_metric,
                resolved_subject=subject_datum.subject,
                resolved_comparator=comparator_datum.subject if comparator_datum else "",
                formula=formula,
            )

        rounded_expected = self._apply_rounding(expected_value, claim.rounding_mode, claim.decimals)
        if rounded_expected != expected_value:
            reason_codes.append("ROUNDING_APPLIED")
            trace.steps.append(
                TraceStep(
                    step_name="rounding",
                    status="ok",
                    message="rounded expected value using claim rounding policy",
                    expression=f"{claim.rounding_mode}(expected, decimals={claim.decimals})",
                    intermediate={"expected_raw": expected_value, "expected_rounded": rounded_expected},
                )
            )

        abs_diff = abs(observed_value - rounded_expected)
        rel_diff = abs_diff / max(abs(rounded_expected), 1e-9)
        abs_tol, rel_tol, tol_reason = self._effective_compare_tolerance(
            claim=claim,
            reason_codes=reason_codes,
            subject_datum=subject_datum,
            comparator_datum=comparator_datum,
        )
        if tol_reason:
            reason_codes.append(tol_reason)
            trace.steps.append(
                TraceStep(
                    step_name="adjust_tolerance",
                    status="ok",
                    message="applied derived-claim tolerance adjustment for noisy figure extraction",
                    bindings={
                        "claim_abs_tol": claim.abs_tolerance,
                        "claim_rel_tol": claim.rel_tolerance,
                        "effective_abs_tol": abs_tol,
                        "effective_rel_tol": rel_tol,
                    },
                    error_code=tol_reason,
                )
            )
        within_tolerance = abs_diff <= abs_tol or rel_diff <= rel_tol
        trace.steps.append(
            TraceStep(
                step_name="compare",
                status="ok" if within_tolerance else "fail",
                message="compare observed claim against deterministic expected value",
                expression="abs_diff <= abs_tol OR rel_diff <= rel_tol",
                bindings={
                    "observed": observed_value,
                    "expected": rounded_expected,
                    "expected_raw": expected_value,
                    "abs_tol": abs_tol,
                    "rel_tol": rel_tol,
                    "abs_tol_default": claim.abs_tolerance,
                    "rel_tol_default": claim.rel_tolerance,
                },
                intermediate={"abs_diff": abs_diff, "rel_diff": rel_diff},
                error_code="" if within_tolerance else "OUT_OF_TOL",
            )
        )

        if not within_tolerance:
            reason_codes.append("OUT_OF_TOL")

        verdict = "supported" if within_tolerance else "contradicted"
        if verdict == "supported" and (claim.ambiguity_flags or self._has_assumption_code(reason_codes)):
            verdict = "uncertain"
        if verdict == "supported" and self._needs_comparator(claim.claim_type) and comparator_datum is None:
            verdict = "unsupported"
            reason_codes.append("DENOMINATOR_MISSING")

        return self._finalize(
            claim=claim,
            trace=trace,
            verdict=verdict,
            expected=rounded_expected,
            observed=observed_value,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            reason_codes=reason_codes,
            resolved_metric=resolved_metric,
            resolved_subject=subject_datum.subject,
            resolved_comparator=comparator_datum.subject if comparator_datum else "",
            formula=formula,
        )

    def _verify_from_to(
        self,
        *,
        claim: ClaimIR,
        trace: VerificationTrace,
        reason_codes: List[str],
        subject_datum: EvidenceDatum,
        comparator_datum: EvidenceDatum | None,
        resolved_metric: str,
    ) -> VerificationResult:
        if comparator_datum is None:
            reason_codes.append("DENOMINATOR_MISSING")
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="unsupported",
                expected=None,
                observed=None,
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes,
                resolved_metric=resolved_metric,
                resolved_subject=subject_datum.subject,
                formula="from_to(subject, comparator)",
            )

        observed_from = claim.declared_from
        observed_to = claim.declared_to if claim.declared_to is not None else claim.declared_value
        if observed_from is None or observed_to is None:
            reason_codes.append("CLAIM_NO_VALUE")
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="unsupported",
                expected=None,
                observed=None,
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes,
                resolved_metric=resolved_metric,
                resolved_subject=subject_datum.subject,
                resolved_comparator=comparator_datum.subject,
                formula="from_to(subject, comparator)",
            )

        expected_from = self._datum_value_in_claim_unit(comparator_datum, claim, reason_codes)
        expected_to = self._datum_value_in_claim_unit(subject_datum, claim, reason_codes)
        if expected_from is None or expected_to is None:
            reason_codes.append("UNIT_SCALE_MISMATCH")
            return self._finalize(
                claim=claim,
                trace=trace,
                verdict="unsupported",
                expected=None,
                observed=None,
                abs_diff=None,
                rel_diff=None,
                reason_codes=reason_codes,
                resolved_metric=resolved_metric,
                resolved_subject=subject_datum.subject,
                resolved_comparator=comparator_datum.subject,
                formula="from_to(subject, comparator)",
            )

        rounded_expected_from = self._apply_rounding(expected_from, claim.rounding_mode, claim.decimals)
        rounded_expected_to = self._apply_rounding(expected_to, claim.rounding_mode, claim.decimals)
        if rounded_expected_from != expected_from or rounded_expected_to != expected_to:
            reason_codes.append("ROUNDING_APPLIED")
            trace.steps.append(
                TraceStep(
                    step_name="rounding_from_to",
                    status="ok",
                    message="rounded expected from/to values using claim rounding policy",
                    expression=f"{claim.rounding_mode}(expected_from_to, decimals={claim.decimals})",
                    intermediate={
                        "expected_from_raw": expected_from,
                        "expected_from_rounded": rounded_expected_from,
                        "expected_to_raw": expected_to,
                        "expected_to_rounded": rounded_expected_to,
                    },
                )
            )

        from_abs_diff = abs(observed_from - rounded_expected_from)
        to_abs_diff = abs(observed_to - rounded_expected_to)
        from_rel_diff = from_abs_diff / max(abs(rounded_expected_from), 1e-9)
        to_rel_diff = to_abs_diff / max(abs(rounded_expected_to), 1e-9)
        abs_tol = max(claim.abs_tolerance, 0.0)
        rel_tol = max(claim.rel_tolerance, 0.0)
        # For percentage/pp "from->to" claims, relative tolerance can be too permissive
        # (e.g., a +1.2 absolute mismatch still passes with 2% rel tolerance at ~90%).
        # We therefore enforce absolute tolerance in percent-like claims and keep abs|rel
        # for raw-valued claims.
        declared_unit = str(claim.declared_unit or "").strip().lower()
        declared_scale = str(claim.declared_scale or "").strip().lower()
        use_abs_only = declared_unit in {"percent", "pp"} or declared_scale in {"0_1", "0_100"}
        if use_abs_only:
            from_ok = from_abs_diff <= abs_tol
            to_ok = to_abs_diff <= abs_tol
            compare_expression = "from_abs_diff <= abs_tol AND to_abs_diff <= abs_tol"
        else:
            from_ok = from_abs_diff <= abs_tol or from_rel_diff <= rel_tol
            to_ok = to_abs_diff <= abs_tol or to_rel_diff <= rel_tol
            compare_expression = "(from_abs_diff <= abs_tol OR from_rel_diff <= rel_tol) AND (to_abs_diff <= abs_tol OR to_rel_diff <= rel_tol)"
        trace.steps.append(
            TraceStep(
                step_name="compare_from_to",
                status="ok" if (from_ok and to_ok) else "fail",
                message="compare from/to values with deterministic subject and comparator values",
                expression=compare_expression,
                bindings={
                    "observed_from": observed_from,
                    "expected_from": rounded_expected_from,
                    "expected_from_raw": expected_from,
                    "observed_to": observed_to,
                    "expected_to": rounded_expected_to,
                    "expected_to_raw": expected_to,
                    "abs_tol": abs_tol,
                    "rel_tol": rel_tol,
                    "tolerance_mode": "abs_only" if use_abs_only else "abs_or_rel",
                },
                intermediate={
                    "from_abs_diff": from_abs_diff,
                    "from_rel_diff": from_rel_diff,
                    "to_abs_diff": to_abs_diff,
                    "to_rel_diff": to_rel_diff,
                    "tolerance_mode": "abs_only" if use_abs_only else "abs_or_rel",
                },
                error_code="" if (from_ok and to_ok) else "OUT_OF_TOL",
            )
        )

        verdict = "supported" if (from_ok and to_ok) else "contradicted"
        if verdict == "supported" and (claim.ambiguity_flags or self._has_assumption_code(reason_codes)):
            verdict = "uncertain"
        if verdict == "contradicted":
            reason_codes.append("OUT_OF_TOL")

        combined_abs_diff = max(from_abs_diff, to_abs_diff)
        combined_rel_diff = max(from_rel_diff, to_rel_diff)
        return self._finalize(
            claim=claim,
            trace=trace,
            verdict=verdict,
            expected=rounded_expected_to,
            observed=observed_to,
            abs_diff=combined_abs_diff,
            rel_diff=combined_rel_diff,
            reason_codes=reason_codes,
            resolved_metric=resolved_metric,
            resolved_subject=subject_datum.subject,
            resolved_comparator=comparator_datum.subject,
            formula="from_to(subject, comparator)",
        )

    def _resolve_metric_candidates(
        self,
        claim: ClaimIR,
        evidence: List[EvidenceDatum],
        trace: VerificationTrace,
        reason_codes: List[str],
    ) -> Tuple[List[EvidenceDatum], str]:
        if not evidence:
            reason_codes.append("TARGET_NO_MATCH")
            trace.steps.append(
                TraceStep(
                    step_name="resolve_metric",
                    status="fail",
                    message="no evidence available",
                    candidate_count=0,
                    error_code="TARGET_NO_MATCH",
                )
            )
            return [], ""

        claim_metric = self._normalize_text(claim.metric)
        by_metric: Dict[str, List[EvidenceDatum]] = {}
        for datum in evidence:
            key = self._normalize_text(datum.metric)
            by_metric.setdefault(key, []).append(datum)

        if claim_metric:
            exact = by_metric.get(claim_metric, [])
            if exact:
                trace.steps.append(
                    TraceStep(
                        step_name="resolve_metric",
                        status="ok",
                        message=f"resolved metric '{claim_metric}' via exact match",
                        candidate_count=len(exact),
                        output_refs=[item.datum_id for item in exact[:16]],
                    )
                )
                return exact, claim_metric

            fuzzy_by_metric: Dict[str, List[EvidenceDatum]] = {}
            for key, rows in by_metric.items():
                if self._text_fuzzy_match(claim_metric, key):
                    fuzzy_by_metric[key] = rows
            if fuzzy_by_metric:
                if len(fuzzy_by_metric) > 1:
                    if not self._is_figure_evidence(evidence):
                        inferred_text = self._infer_metric_by_token_overlap(claim_metric, fuzzy_by_metric)
                        if inferred_text is not None:
                            inferred_metric, inferred_rows, inferred_score = inferred_text
                            reason_codes.append("TARGET_DISAMBIGUATED_CONTEXT")
                            trace.steps.append(
                                TraceStep(
                                    step_name="resolve_metric",
                                    status="ok",
                                    message=(
                                        f"metric '{claim_metric}' fuzzy-matched multiple candidates; "
                                        f"resolved via token-overlap tie-break -> '{inferred_metric}'"
                                    ),
                                    candidate_count=len(inferred_rows),
                                    output_refs=[item.datum_id for item in inferred_rows[:16]],
                                    bindings={
                                        "requested_metric": claim_metric,
                                        "inferred_metric": inferred_metric,
                                        "token_overlap_score": inferred_score,
                                    },
                                    error_code="TARGET_DISAMBIGUATED_CONTEXT",
                                )
                            )
                            return inferred_rows, inferred_metric
                        inferred_fuzzy = self._infer_metric_by_declared_value(claim, fuzzy_by_metric)
                        if inferred_fuzzy is not None:
                            inferred_metric, inferred_rows, inferred_score = inferred_fuzzy
                            reason_codes.append("TARGET_DISAMBIGUATED_DECLARED_VALUE")
                            trace.steps.append(
                                TraceStep(
                                    step_name="resolve_metric",
                                    status="ok",
                                    message=(
                                        f"metric '{claim_metric}' fuzzy-matched multiple candidates; "
                                        f"resolved via declared-value tie-break -> '{inferred_metric}'"
                                    ),
                                    candidate_count=len(inferred_rows),
                                    output_refs=[item.datum_id for item in inferred_rows[:16]],
                                    bindings={
                                        "requested_metric": claim_metric,
                                        "inferred_metric": inferred_metric,
                                        "inference_score": inferred_score,
                                    },
                                    error_code="TARGET_DISAMBIGUATED_DECLARED_VALUE",
                                )
                            )
                            return inferred_rows, inferred_metric
                    reason_codes.extend(["TARGET_FUZZY_MATCH", "TARGET_MULTI_MATCH"])
                    trace.steps.append(
                        TraceStep(
                            step_name="resolve_metric",
                            status="fail",
                            message=f"metric '{claim_metric}' fuzzy-matched multiple candidates",
                            candidate_count=sum(len(rows) for rows in fuzzy_by_metric.values()),
                            error_code="TARGET_MULTI_MATCH",
                        )
                    )
                    return [], claim_metric

                fuzzy_key, fuzzy = next(iter(fuzzy_by_metric.items()))
                fuzzy_reason = "TARGET_FUZZY_MATCH"
                if not self._is_figure_evidence(evidence):
                    anchor_score = self._metric_anchor_score(claim, metric_key=fuzzy_key, rows=fuzzy)
                    if anchor_score >= 1.0:
                        fuzzy_reason = "TARGET_DISAMBIGUATED_CONTEXT"
                reason_codes.append(fuzzy_reason)
                trace.steps.append(
                    TraceStep(
                        step_name="resolve_metric",
                        status="ok",
                        message=f"resolved metric '{claim_metric}' via fuzzy match",
                        candidate_count=len(fuzzy),
                        output_refs=[item.datum_id for item in fuzzy[:16]],
                        error_code=fuzzy_reason,
                    )
                )
                return fuzzy, claim_metric

            if self._is_figure_evidence(evidence):
                inferred = self._infer_metric_for_figure_claim(claim, by_metric)
                if inferred is not None:
                    inferred_metric, inferred_rows, inferred_score = inferred
                    reason_codes.append("TARGET_METRIC_INFERRED")
                    trace.steps.append(
                        TraceStep(
                            step_name="resolve_metric",
                            status="ok",
                            message=(
                                f"metric '{claim_metric}' inferred from figure subject/target matching "
                                f"-> '{inferred_metric}'"
                            ),
                            candidate_count=len(inferred_rows),
                            output_refs=[item.datum_id for item in inferred_rows[:16]],
                            bindings={
                                "requested_metric": claim_metric,
                                "inferred_metric": inferred_metric,
                                "inference_score": inferred_score,
                            },
                            error_code="TARGET_METRIC_INFERRED",
                        )
                    )
                    return inferred_rows, inferred_metric
            else:
                inferred_table = self._infer_metric_by_declared_value(claim, by_metric)
                if inferred_table is not None:
                    inferred_metric, inferred_rows, inferred_score = inferred_table
                    reason_codes.append("TARGET_METRIC_INFERRED")
                    trace.steps.append(
                        TraceStep(
                            step_name="resolve_metric",
                            status="ok",
                            message=(
                                f"metric '{claim_metric}' inferred from declared-value consistency "
                                f"-> '{inferred_metric}'"
                            ),
                            candidate_count=len(inferred_rows),
                            output_refs=[item.datum_id for item in inferred_rows[:16]],
                            bindings={
                                "requested_metric": claim_metric,
                                "inferred_metric": inferred_metric,
                                "inference_score": inferred_score,
                            },
                            error_code="TARGET_METRIC_INFERRED",
                        )
                    )
                    return inferred_rows, inferred_metric

            reason_codes.append("TARGET_NO_MATCH")
            trace.steps.append(
                TraceStep(
                    step_name="resolve_metric",
                    status="fail",
                    message=f"metric '{claim_metric}' not found in evidence",
                    candidate_count=0,
                    error_code="TARGET_NO_MATCH",
                )
            )
            return [], claim_metric

        if len(by_metric) == 1:
            only_metric = next(iter(by_metric.keys()))
            rows = by_metric[only_metric]
            trace.steps.append(
                TraceStep(
                    step_name="resolve_metric",
                    status="ok",
                    message=f"single metric '{only_metric}' selected",
                    candidate_count=len(rows),
                    output_refs=[item.datum_id for item in rows[:16]],
                )
            )
            return rows, only_metric

        reason_codes.append("TARGET_MULTI_MATCH")
        trace.steps.append(
            TraceStep(
                step_name="resolve_metric",
                status="fail",
                message="claim metric missing and multiple metric candidates exist",
                candidate_count=sum(len(v) for v in by_metric.values()),
                error_code="TARGET_MULTI_MATCH",
            )
        )
        return [], ""

    def _infer_metric_by_declared_value(
        self,
        claim: ClaimIR,
        by_metric: Dict[str, List[EvidenceDatum]],
    ) -> Optional[Tuple[str, List[EvidenceDatum], float]]:
        if not by_metric:
            return None

        claim_type = self._normalize_text(claim.claim_type)
        observed_value, _ = self._normalize_observed_value(claim)
        observed_from = self._safe_float(claim.declared_from)
        observed_to = self._safe_float(claim.declared_to if claim.declared_to is not None else claim.declared_value)

        scored: List[Tuple[float, float, str, List[EvidenceDatum]]] = []
        for metric_key, rows in by_metric.items():
            if not rows:
                continue
            best = float("inf")
            if claim_type == "from_to":
                if observed_from is None or observed_to is None:
                    continue
                for subject in rows:
                    expected_to = self._datum_value_in_claim_unit(subject, claim, reason_codes=None)
                    if expected_to is None:
                        continue
                    for comparator in rows:
                        if comparator.datum_id == subject.datum_id:
                            continue
                        expected_from = self._datum_value_in_claim_unit(comparator, claim, reason_codes=None)
                        if expected_from is None:
                            continue
                        score = abs(observed_to - expected_to) + abs(observed_from - expected_from)
                        if score < best:
                            best = score
            elif self._needs_comparator(claim_type):
                if observed_value is None:
                    continue
                for subject in rows:
                    for comparator in rows:
                        if comparator.datum_id == subject.datum_id:
                            continue
                        expected, _, error_code = self._compute_expected_value(
                            claim=claim,
                            subject_datum=subject,
                            comparator_datum=comparator,
                        )
                        if expected is None or error_code:
                            continue
                        score = abs(observed_value - expected)
                        if score < best:
                            best = score
            else:
                if observed_value is None:
                    continue
                for subject in rows:
                    expected = self._datum_value_in_claim_unit(subject, claim, reason_codes=None)
                    if expected is None:
                        continue
                    score = abs(observed_value - expected)
                    if score < best:
                        best = score

            if best != float("inf"):
                anchor_score = self._metric_anchor_score(claim, metric_key=metric_key, rows=rows)
                scored.append((best, anchor_score, metric_key, rows))

        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], -item[1]))
        best_score, best_anchor_score, best_metric, best_rows = scored[0]
        if len(scored) > 1:
            second_score, second_anchor_score, _, _ = scored[1]
            if abs(best_score - second_score) <= 1e-9 and abs(best_anchor_score - second_anchor_score) <= 1e-9:
                return None

        max_error = self._metric_inference_max_error(
            claim,
            observed_value=observed_value,
            observed_from=observed_from,
            observed_to=observed_to,
        )
        if best_score > max_error:
            return None
        return best_metric, best_rows, best_score

    def _metric_inference_max_error(
        self,
        claim: ClaimIR,
        *,
        observed_value: Optional[float],
        observed_from: Optional[float],
        observed_to: Optional[float],
    ) -> float:
        claim_type = self._normalize_text(claim.claim_type)
        if claim_type == "ratio":
            return max(0.2, abs(observed_value or 0.0) * 0.15)
        if claim_type == "from_to":
            baseline = abs(observed_from or 0.0) + abs(observed_to or 0.0)
            return max(2.0, baseline * 0.1)
        if claim_type in {"pp_change", "relative_change", "relative_improvement", "relative_error_reduction", "error_reduction"}:
            return max(2.0, abs(observed_value or 0.0) * 0.15)
        if claim_type == "scale_normalize":
            return 2.0
        return max(2.0, abs(observed_value or 0.0) * 0.2)

    def _infer_metric_by_token_overlap(
        self,
        claim_metric: str,
        by_metric: Dict[str, List[EvidenceDatum]],
    ) -> Optional[Tuple[str, List[EvidenceDatum], float]]:
        claim_tokens = [tok for tok in re.findall(r"[a-z0-9]+", self._normalize_text(claim_metric)) if tok]
        if not claim_tokens:
            return None
        claim_set = set(claim_tokens)
        claim_num = {tok for tok in claim_set if tok.isdigit()}

        scored: List[Tuple[float, str, List[EvidenceDatum]]] = []
        for metric_key, rows in by_metric.items():
            metric_tokens = set(re.findall(r"[a-z0-9]+", self._normalize_text(metric_key)))
            if not metric_tokens:
                continue
            overlap = len(claim_set & metric_tokens)
            if overlap <= 0:
                continue
            metric_num = {tok for tok in metric_tokens if tok.isdigit()}
            num_overlap = len(claim_num & metric_num)
            # Prefer numeric anchor agreement (e.g., "2-14" vs "2-8").
            score = float(overlap) + float(num_overlap) * 2.0
            scored.append((score, metric_key, rows))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_metric, best_rows = scored[0]
        if len(scored) > 1 and abs(best_score - scored[1][0]) <= 1e-9:
            return None
        return best_metric, best_rows, best_score

    def _resolve_subject_pair(
        self,
        claim: ClaimIR,
        metric_evidence: List[EvidenceDatum],
        trace: VerificationTrace,
        reason_codes: List[str],
    ) -> Tuple[EvidenceDatum | None, EvidenceDatum | None]:
        subject_buckets: Dict[str, List[EvidenceDatum]] = {}
        for item in metric_evidence:
            key = self._normalize_text(item.subject)
            subject_buckets.setdefault(key, []).append(item)
        requested_subject = self._normalize_text(claim.subject) or "ours"
        requested_comparator = self._normalize_text(claim.comparator or claim.denominator or "baseline")
        claim_type_norm = self._normalize_text(claim.claim_type)
        needs_comparator = self._needs_comparator(claim.claim_type)
        figure_anchor_binding = self._is_figure_evidence(metric_evidence) and self._supports_from_to_anchor_binding(
            claim.claim_type
        )
        anchor_subject_target = self._safe_float(
            claim.declared_to
            if claim.declared_to is not None and claim_type_norm in {
                "from_to",
                "ratio",
                "pp_change",
                "relative_change",
                "relative_improvement",
                "relative_error_reduction",
                "error_reduction",
            }
            else (
                claim.declared_value
                if claim_type_norm in {"direct_value", "from_to"}
                else None
            )
        )
        anchor_comparator_target = self._safe_float(claim.declared_from)

        subject_candidates = list(subject_buckets.get(requested_subject, []))
        if not subject_candidates and figure_anchor_binding and anchor_subject_target is not None:
            inferred_subject_anchor = self._infer_datum_by_value(
                candidates=metric_evidence,
                target=anchor_subject_target,
                label_hint=requested_subject,
                secondary_hint=self._normalize_text(claim.metric),
                target_unit=claim.declared_unit,
                target_scale=claim.declared_scale,
                max_abs_gap=self._from_to_anchor_max_gap(anchor_subject_target, claim),
            )
            if inferred_subject_anchor is not None:
                subject_candidates = [inferred_subject_anchor]
                reason_codes.append("SUBJECT_INFERRED_BY_FROM_TO_VALUE")
        if not subject_candidates and requested_subject:
            fuzzy_subject_keys = [
                key for key in subject_buckets.keys() if self._text_fuzzy_match(requested_subject, key)
            ]
            if len(fuzzy_subject_keys) == 1:
                subject_candidates = list(subject_buckets.get(fuzzy_subject_keys[0], []))
                reason_codes.append("TARGET_FUZZY_MATCH")
            elif len(fuzzy_subject_keys) > 1:
                reason_codes.extend(["TARGET_FUZZY_MATCH", "TARGET_MULTI_MATCH"])
                trace.steps.append(
                    TraceStep(
                        step_name="resolve_subject",
                        status="fail",
                        message=f"subject '{requested_subject}' fuzzy-matched multiple candidates",
                        candidate_count=sum(len(subject_buckets[key]) for key in fuzzy_subject_keys),
                        error_code="TARGET_MULTI_MATCH",
                    )
                )
                return None, None
        if not subject_candidates and requested_subject == "ours":
            non_baseline_keys = [k for k in subject_buckets.keys() if k not in {"baseline", "control"}]
            if len(non_baseline_keys) == 1:
                assumed = subject_buckets.get(non_baseline_keys[0], [])
                if len(assumed) == 1:
                    subject_candidates = [assumed[0]]
                    reason_codes.append("SUBJECT_ASSUMED")
        if not subject_candidates and not self._is_figure_evidence(metric_evidence):
            normalized_claim_type = self._normalize_text(claim.claim_type)
            if normalized_claim_type == "from_to":
                subject_target = self._safe_float(
                    claim.declared_to if claim.declared_to is not None else claim.declared_value
                )
            else:
                subject_target = self._safe_float(claim.declared_value)
            inferred_subject_table = self._infer_datum_by_value(
                candidates=metric_evidence,
                target=subject_target,
                label_hint=requested_subject,
                secondary_hint=self._normalize_text(claim.metric),
                target_unit=claim.declared_unit,
                target_scale=claim.declared_scale,
            )
            if inferred_subject_table is not None:
                subject_candidates = [inferred_subject_table]
                reason_codes.append("SUBJECT_INFERRED_BY_VALUE")
        if not subject_candidates and self._is_figure_evidence(metric_evidence):
            normalized_claim_type = self._normalize_text(claim.claim_type)
            if normalized_claim_type in {
                "from_to",
                "ratio",
                "pp_change",
                "relative_change",
                "relative_improvement",
                "relative_error_reduction",
                "error_reduction",
            } and claim.declared_to is not None:
                subject_target = self._safe_float(
                    claim.declared_to if claim.declared_to is not None else claim.declared_value
                )
            else:
                subject_target = self._safe_float(claim.declared_value)
            inferred_subject = self._infer_datum_by_value(
                candidates=metric_evidence,
                target=subject_target,
                label_hint=requested_subject,
                secondary_hint=self._normalize_text(claim.metric),
                target_unit=claim.declared_unit,
                target_scale=claim.declared_scale,
            )
            if inferred_subject is not None:
                subject_candidates = [inferred_subject]
                reason_codes.append("SUBJECT_INFERRED_BY_VALUE")
        if not subject_candidates and self._is_figure_evidence(metric_evidence):
            axis_hint = self._normalize_text(claim.metric)
            axis_rows = self._resolve_subject_rows(subject_buckets, axis_hint)
            if axis_rows:
                if self._normalize_text(claim.claim_type) in {
                    "from_to",
                    "ratio",
                    "pp_change",
                    "relative_change",
                    "relative_improvement",
                    "relative_error_reduction",
                    "error_reduction",
                } and claim.declared_to is not None:
                    subject_target = self._safe_float(
                        claim.declared_to if claim.declared_to is not None else claim.declared_value
                    )
                else:
                    subject_target = self._safe_float(claim.declared_value)
                inferred_axis_subject = self._infer_datum_by_value(
                    candidates=axis_rows,
                    target=subject_target,
                    label_hint=axis_hint,
                    secondary_hint=requested_subject,
                    target_unit=claim.declared_unit,
                    target_scale=claim.declared_scale,
                )
                if inferred_axis_subject is None and len(axis_rows) == 1:
                    inferred_axis_subject = axis_rows[0]
                if inferred_axis_subject is not None:
                    subject_candidates = [inferred_axis_subject]
                    reason_codes.append("SUBJECT_INFERRED_BY_METRIC_AXIS")
        if not subject_candidates and self._is_figure_evidence(metric_evidence):
            inferred_extrema_subject = self._infer_axis_extrema_subject(claim=claim, candidates=metric_evidence)
            if inferred_extrema_subject is not None:
                subject_candidates = [inferred_extrema_subject]
                reason_codes.append("SUBJECT_INFERRED_BY_AXIS_EXTREMA")

        if not subject_candidates:
            reason_codes.append("TARGET_NO_MATCH")
            trace.steps.append(
                TraceStep(
                    step_name="resolve_subject",
                    status="fail",
                    message=f"subject '{requested_subject}' not found",
                    candidate_count=len(subject_buckets),
                    error_code="TARGET_NO_MATCH",
                )
            )
            return None, None

        # Lightweight disambiguation for table evidence:
        # when multiple subject rows exist for the same label, prefer the
        # candidate that best aligns with comparator-side context (dataset/split/
        # setting/column). This avoids failing fast on TARGET_MULTI_MATCH for
        # otherwise resolvable anchored claims.
        if (
            len(subject_candidates) > 1
            and not self._is_figure_evidence(metric_evidence)
            and self._supports_context_disambiguation(claim.claim_type)
        ):
            comparator_hint_pool = self._resolve_subject_rows(subject_buckets, requested_comparator)
            resolved_subject = self._disambiguate_subject_by_context(
                subject_candidates=subject_candidates,
                comparator_candidates=comparator_hint_pool,
            )
            if resolved_subject is not None:
                subject_candidates = [resolved_subject]
                reason_codes.append("TARGET_DISAMBIGUATED_CONTEXT")
                trace.steps.append(
                    TraceStep(
                        step_name="resolve_subject_disambiguation",
                        status="ok",
                        message="resolved subject via context-based tie-break",
                        candidate_count=len(subject_buckets),
                        output_refs=[resolved_subject.datum_id],
                        error_code="TARGET_DISAMBIGUATED_CONTEXT",
                    )
                )

        if (
            len(subject_candidates) > 1
            and self._supports_context_disambiguation(claim.claim_type)
            and (
                not self._is_figure_evidence(metric_evidence)
                or self._supports_figure_declared_disambiguation(claim.claim_type)
            )
        ):
            resolved_subject_declared = self._disambiguate_subject_by_declared_value(
                claim=claim,
                subject_candidates=subject_candidates,
                requested_comparator=requested_comparator,
                subject_buckets=subject_buckets,
                metric_evidence=metric_evidence,
            )
            if resolved_subject_declared is not None:
                subject_candidates = [resolved_subject_declared]
                reason_codes.append("TARGET_DISAMBIGUATED_DECLARED_VALUE")
                trace.steps.append(
                    TraceStep(
                        step_name="resolve_subject_disambiguation",
                        status="ok",
                        message="resolved subject via declared-value formula tie-break",
                        candidate_count=len(subject_buckets),
                        output_refs=[resolved_subject_declared.datum_id],
                        error_code="TARGET_DISAMBIGUATED_DECLARED_VALUE",
                    )
                )

        if (
            len(subject_candidates) > 1
            and self._normalize_text(claim.claim_type) == "uncertain"
            and not self._is_figure_evidence(metric_evidence)
        ):
            fallback_subject = self._pick_best_candidate(subject_candidates)
            if fallback_subject is not None:
                subject_candidates = [fallback_subject]
                reason_codes.append("TARGET_DISAMBIGUATED_BY_CONFIDENCE")
                trace.steps.append(
                    TraceStep(
                        step_name="resolve_subject_disambiguation",
                        status="ok",
                        message="resolved uncertain-claim subject by confidence/order tie-break",
                        candidate_count=len(subject_buckets),
                        output_refs=[fallback_subject.datum_id],
                        error_code="TARGET_DISAMBIGUATED_BY_CONFIDENCE",
                    )
                )

        if len(subject_candidates) > 1:
            reason_codes.append("TARGET_MULTI_MATCH")
            trace.steps.append(
                TraceStep(
                    step_name="resolve_subject",
                    status="fail",
                    message=f"subject '{requested_subject}' has multiple candidates",
                    candidate_count=len(subject_candidates),
                    output_refs=[item.datum_id for item in subject_candidates[:16]],
                    error_code="TARGET_MULTI_MATCH",
                )
            )
            return None, None

        subject_datum = subject_candidates[0]

        comparator_datum: EvidenceDatum | None = None
        comparator_error = ""
        comparator_candidates = list(subject_buckets.get(requested_comparator, []))
        if needs_comparator:
            if (
                not comparator_candidates
                and not comparator_error
                and figure_anchor_binding
                and anchor_comparator_target is not None
            ):
                inferred_comparator_anchor = self._infer_datum_by_value(
                    candidates=metric_evidence,
                    target=anchor_comparator_target,
                    label_hint=requested_comparator,
                    secondary_hint=self._normalize_text(claim.metric),
                    target_unit=claim.declared_unit,
                    target_scale=claim.declared_scale,
                    exclude_ids={subject_datum.datum_id},
                    max_abs_gap=self._from_to_anchor_max_gap(anchor_comparator_target, claim),
                )
                if inferred_comparator_anchor is not None:
                    comparator_candidates = [inferred_comparator_anchor]
                    reason_codes.append("DENOMINATOR_INFERRED_BY_FROM_TO_VALUE")
            if not comparator_candidates and requested_comparator:
                fuzzy_comparator_keys = [
                    key for key in subject_buckets.keys() if self._text_fuzzy_match(requested_comparator, key)
                ]
                if len(fuzzy_comparator_keys) == 1:
                    comparator_candidates = list(subject_buckets.get(fuzzy_comparator_keys[0], []))
                    reason_codes.append("TARGET_FUZZY_MATCH")
                elif len(fuzzy_comparator_keys) > 1:
                    reason_codes.extend(["TARGET_FUZZY_MATCH", "DENOMINATOR_AMBIGUOUS"])
                    comparator_error = "DENOMINATOR_AMBIGUOUS"
            if (
                not comparator_candidates
                and not comparator_error
                and self._is_figure_evidence(metric_evidence)
                and self._supports_from_to_anchor_binding(claim.claim_type)
                and self._safe_float(claim.declared_from) is not None
            ):
                inferred_comparator = self._infer_datum_by_value(
                    candidates=metric_evidence,
                    target=self._safe_float(claim.declared_from),
                    label_hint=requested_comparator,
                    secondary_hint=self._normalize_text(claim.metric),
                    target_unit=claim.declared_unit,
                    target_scale=claim.declared_scale,
                    exclude_ids={subject_datum.datum_id},
                )
                if inferred_comparator is not None:
                    comparator_candidates = [inferred_comparator]
                    reason_codes.append("DENOMINATOR_INFERRED_BY_VALUE")
            if (
                not comparator_candidates
                and not comparator_error
                and self._is_figure_evidence(metric_evidence)
                and self._supports_from_to_anchor_binding(claim.claim_type)
                and self._safe_float(claim.declared_from) is not None
            ):
                axis_hint = self._normalize_text(claim.metric)
                axis_rows = self._resolve_subject_rows(subject_buckets, axis_hint)
                if axis_rows:
                    inferred_comparator_axis = self._infer_datum_by_value(
                        candidates=axis_rows,
                        target=self._safe_float(claim.declared_from),
                        label_hint=axis_hint,
                        secondary_hint=requested_comparator,
                        target_unit=claim.declared_unit,
                        target_scale=claim.declared_scale,
                        exclude_ids={subject_datum.datum_id},
                    )
                    if inferred_comparator_axis is not None:
                        comparator_candidates = [inferred_comparator_axis]
                        reason_codes.append("DENOMINATOR_INFERRED_BY_METRIC_AXIS")
            if (
                not comparator_candidates
                and not comparator_error
                and self._is_figure_evidence(metric_evidence)
            ):
                inferred_formula_comparator = self._infer_comparator_for_figure_by_declared_formula(
                    claim=claim,
                    subject_datum=subject_datum,
                    candidates=metric_evidence,
                )
                if inferred_formula_comparator is not None:
                    comparator_candidates = [inferred_formula_comparator]
                    reason_codes.append("DENOMINATOR_INFERRED_BY_FORMULA")
            if (
                not comparator_candidates
                and not comparator_error
                and not self._is_figure_evidence(metric_evidence)
            ):
                inferred_formula_comparator_table = self._infer_comparator_for_figure_by_declared_formula(
                    claim=claim,
                    subject_datum=subject_datum,
                    candidates=metric_evidence,
                )
                if inferred_formula_comparator_table is not None:
                    comparator_candidates = [inferred_formula_comparator_table]
                    reason_codes.append("DENOMINATOR_INFERRED_BY_FORMULA")
            if not comparator_candidates and not comparator_error:
                reason_codes.append("DENOMINATOR_MISSING")
                comparator_error = "DENOMINATOR_MISSING"
            elif len(comparator_candidates) > 1:
                if (
                    not self._is_figure_evidence(metric_evidence)
                    and self._supports_context_disambiguation(claim.claim_type)
                ):
                    resolved_comparator = self._disambiguate_comparator_by_context(
                        subject_datum=subject_datum,
                        comparator_candidates=comparator_candidates,
                    )
                    if resolved_comparator is not None:
                        comparator_candidates = [resolved_comparator]
                        reason_codes.append("DENOMINATOR_DISAMBIGUATED_CONTEXT")
                if (
                    len(comparator_candidates) > 1
                    and self._supports_context_disambiguation(claim.claim_type)
                    and (
                        not self._is_figure_evidence(metric_evidence)
                        or self._supports_figure_declared_disambiguation(claim.claim_type)
                    )
                ):
                    resolved_comparator_declared = self._disambiguate_comparator_by_declared_value(
                        claim=claim,
                        subject_datum=subject_datum,
                        comparator_candidates=comparator_candidates,
                    )
                    if resolved_comparator_declared is not None:
                        comparator_candidates = [resolved_comparator_declared]
                        reason_codes.append("DENOMINATOR_DISAMBIGUATED_DECLARED_VALUE")
                if len(comparator_candidates) > 1:
                    reason_codes.append("DENOMINATOR_AMBIGUOUS")
                    comparator_error = "DENOMINATOR_AMBIGUOUS"
                elif comparator_candidates:
                    comparator_datum = comparator_candidates[0]
            elif comparator_candidates:
                comparator_datum = comparator_candidates[0]
        elif comparator_candidates:
            comparator_datum = comparator_candidates[0]

        trace.steps.append(
            TraceStep(
                step_name="resolve_subject_and_comparator",
                status="ok" if (subject_datum and (comparator_datum or not needs_comparator)) else "fail",
                message="resolved subject/comparator candidates",
                candidate_count=len(subject_buckets),
                output_refs=[subject_datum.datum_id] + ([comparator_datum.datum_id] if comparator_datum else []),
                error_code=comparator_error,
            )
        )
        if (
            self._is_figure_evidence(metric_evidence)
            and needs_comparator
            and self._supports_from_to_anchor_binding(claim.claim_type)
        ):
            pair_by_anchor = self._infer_subject_comparator_pair_by_from_to(
                claim=claim,
                candidates=metric_evidence,
            )
            if pair_by_anchor is not None:
                inferred_subject, inferred_comparator, pair_score = pair_by_anchor
                subject_datum = inferred_subject
                comparator_datum = inferred_comparator
                reason_codes.append("PAIR_INFERRED_BY_FROM_TO")
                trace.steps.append(
                    TraceStep(
                        step_name="resolve_subject_pair_anchor",
                        status="ok",
                        message="resolved subject/comparator via from-to joint anchor optimization",
                        output_refs=[subject_datum.datum_id, comparator_datum.datum_id],
                        bindings={"pair_score": pair_score},
                        error_code="PAIR_INFERRED_BY_FROM_TO",
                    )
                )
        return subject_datum, comparator_datum

    @staticmethod
    def _supports_context_disambiguation(claim_type: str) -> bool:
        normalized = " ".join(str(claim_type or "").strip().lower().split())
        return normalized in {
            "ratio",
            "pp_change",
            "relative_change",
            "from_to",
            "relative_improvement",
            "relative_error_reduction",
            "relative_decrease",
            "error_reduction",
            "uncertain",
        }

    @staticmethod
    def _supports_figure_declared_disambiguation(claim_type: str) -> bool:
        normalized = " ".join(str(claim_type or "").strip().lower().split())
        return normalized in {
            "ratio",
            "pp_change",
            "relative_change",
            "from_to",
            "direct_value",
            "relative_improvement",
            "relative_error_reduction",
            "relative_decrease",
            "error_reduction",
        }

    @staticmethod
    def _supports_from_to_anchor_binding(claim_type: str) -> bool:
        normalized = " ".join(str(claim_type or "").strip().lower().split())
        return normalized in {
            "from_to",
            "ratio",
            "pp_change",
            "relative_change",
            "relative_improvement",
            "relative_decrease",
            "error_reduction",
            "relative_error_reduction",
            "direct_value",
        }

    def _from_to_anchor_max_gap(self, target: float, claim: ClaimIR) -> float:
        claim_unit = self._normalize_unit(claim.declared_unit)
        claim_scale = self._normalize_scale(claim.declared_scale)
        claim_type = self._normalize_text(claim.claim_type)
        base = max(1e-9, abs(float(target)))
        if claim_type == "ratio":
            return max(0.08, base * 0.12)
        if claim_scale == "0_1":
            return max(0.06, base * 0.15)
        if claim_scale == "0_100" or claim_unit in {"percent", "pp"}:
            return max(4.0, base * 0.15)
        return max(2.0, base * 0.2)

    def _disambiguate_subject_by_context(
        self,
        *,
        subject_candidates: List[EvidenceDatum],
        comparator_candidates: List[EvidenceDatum],
    ) -> Optional[EvidenceDatum]:
        if len(subject_candidates) <= 1:
            return subject_candidates[0] if subject_candidates else None
        if not comparator_candidates:
            return None

        scored: List[Tuple[float, float, EvidenceDatum]] = []
        for subject in subject_candidates:
            best_pair_score = max(
                (
                    self._context_pair_score(subject, comp)
                    for comp in comparator_candidates
                    if comp.datum_id != subject.datum_id
                ),
                default=-1.0,
            )
            confidence = self._safe_float(subject.meta.get("confidence")) or 0.0
            scored.append((best_pair_score, confidence, subject))

        scored.sort(key=lambda item: (item[0], item[1], -item[2].row_index, -item[2].col_index), reverse=True)
        if not scored:
            return None
        top_score, top_conf, top_subject = scored[0]
        if top_score <= 0:
            return None
        if len(scored) > 1:
            second_score, second_conf, _ = scored[1]
            if abs(top_score - second_score) < 1e-9 and abs(top_conf - second_conf) < 1e-9:
                best_ctx = -1.0
                best_item: Optional[EvidenceDatum] = None
                for _, _, candidate in scored:
                    ctx = max(
                        (
                            self._context_pair_score(candidate, comp)
                            for comp in comparator_pool
                            if comp.datum_id != candidate.datum_id
                        ),
                        default=-1.0,
                    )
                    if ctx > best_ctx:
                        best_ctx = ctx
                        best_item = candidate
                if best_item is None or best_ctx <= 0:
                    return None
                return best_item
        return top_subject

    def _disambiguate_comparator_by_context(
        self,
        *,
        subject_datum: EvidenceDatum,
        comparator_candidates: List[EvidenceDatum],
    ) -> Optional[EvidenceDatum]:
        if len(comparator_candidates) <= 1:
            return comparator_candidates[0] if comparator_candidates else None

        scored: List[Tuple[float, float, EvidenceDatum]] = []
        for comparator in comparator_candidates:
            score = self._context_pair_score(subject_datum, comparator)
            confidence = self._safe_float(comparator.meta.get("confidence")) or 0.0
            scored.append((score, confidence, comparator))

        scored.sort(key=lambda item: (item[0], item[1], -item[2].row_index, -item[2].col_index), reverse=True)
        if not scored:
            return None
        top_score, top_conf, top_item = scored[0]
        if top_score <= 0:
            return None
        if len(scored) > 1:
            second_score, second_conf, _ = scored[1]
            if abs(top_score - second_score) < 1e-9 and abs(top_conf - second_conf) < 1e-9:
                return None
        return top_item

    def _pick_best_candidate(self, candidates: List[EvidenceDatum]) -> Optional[EvidenceDatum]:
        if not candidates:
            return None

        def _rank(item: EvidenceDatum) -> Tuple[float, int, int]:
            confidence = self._safe_float(item.meta.get("confidence")) or 0.0
            row = item.row_index if item.row_index >= 0 else 10**9
            col = item.col_index if item.col_index >= 0 else 10**9
            return (confidence, -row, -col)

        ranked = sorted(candidates, key=_rank, reverse=True)
        return ranked[0]

    def _disambiguate_subject_by_declared_value(
        self,
        *,
        claim: ClaimIR,
        subject_candidates: List[EvidenceDatum],
        requested_comparator: str,
        subject_buckets: Dict[str, List[EvidenceDatum]],
        metric_evidence: List[EvidenceDatum],
    ) -> Optional[EvidenceDatum]:
        if len(subject_candidates) <= 1:
            return subject_candidates[0] if subject_candidates else None

        comparator_pool = list(subject_buckets.get(requested_comparator, []))
        if not comparator_pool:
            comparator_pool = self._resolve_subject_rows(subject_buckets, requested_comparator)
        if not comparator_pool and self._needs_comparator(claim.claim_type):
            comparator_pool = [item for item in metric_evidence]

        scored: List[Tuple[float, float, EvidenceDatum]] = []
        for subject in subject_candidates:
            score = self._score_candidate_pair_by_declared_value(
                claim=claim,
                subject_datum=subject,
                comparator_candidates=comparator_pool,
            )
            confidence = self._safe_float(subject.meta.get("confidence")) or 0.0
            scored.append((score, confidence, subject))

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=False)
        if not scored:
            return None
        top_score, top_conf, top_subject = scored[0]
        if top_score == float("inf"):
            return None
        if len(scored) > 1:
            second_score, second_conf, _ = scored[1]
            if abs(top_score - second_score) < 1e-9 and abs(top_conf - second_conf) < 1e-9:
                return None
        return top_subject

    def _disambiguate_comparator_by_declared_value(
        self,
        *,
        claim: ClaimIR,
        subject_datum: EvidenceDatum,
        comparator_candidates: List[EvidenceDatum],
    ) -> Optional[EvidenceDatum]:
        if len(comparator_candidates) <= 1:
            return comparator_candidates[0] if comparator_candidates else None

        scored: List[Tuple[float, float, EvidenceDatum]] = []
        for comparator in comparator_candidates:
            score = self._score_candidate_pair_by_declared_value(
                claim=claim,
                subject_datum=subject_datum,
                comparator_candidates=[comparator],
            )
            confidence = self._safe_float(comparator.meta.get("confidence")) or 0.0
            scored.append((score, confidence, comparator))

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=False)
        if not scored:
            return None
        top_score, top_conf, top_item = scored[0]
        if top_score == float("inf"):
            return None
        if len(scored) > 1:
            second_score, second_conf, _ = scored[1]
            if abs(top_score - second_score) < 1e-9 and abs(top_conf - second_conf) < 1e-9:
                return None
        return top_item

    def _score_candidate_pair_by_declared_value(
        self,
        *,
        claim: ClaimIR,
        subject_datum: EvidenceDatum,
        comparator_candidates: List[EvidenceDatum],
    ) -> float:
        claim_type = self._normalize_text(claim.claim_type)
        if claim_type == "from_to":
            observed_from = self._safe_float(claim.declared_from)
            observed_to = self._safe_float(claim.declared_to if claim.declared_to is not None else claim.declared_value)
            if observed_from is None or observed_to is None:
                return float("inf")
            expected_to = self._datum_value_in_claim_unit(subject_datum, claim, reason_codes=None)
            if expected_to is None:
                return float("inf")
            best = float("inf")
            for comparator in comparator_candidates:
                if comparator.datum_id == subject_datum.datum_id:
                    continue
                expected_from = self._datum_value_in_claim_unit(comparator, claim, reason_codes=None)
                if expected_from is None:
                    continue
                score = abs(observed_to - expected_to) + abs(observed_from - expected_from)
                if score < best:
                    best = score
            return best

        observed_value, _ = self._normalize_observed_value(claim)
        if observed_value is None:
            return float("inf")
        if self._needs_comparator(claim_type):
            best = float("inf")
            for comparator in comparator_candidates:
                if comparator.datum_id == subject_datum.datum_id:
                    continue
                expected, _, error_code = self._compute_expected_value(
                    claim=claim,
                    subject_datum=subject_datum,
                    comparator_datum=comparator,
                )
                if error_code or expected is None:
                    continue
                score = abs(observed_value - expected)
                if score < best:
                    best = score
            return best

        expected_direct = self._datum_value_in_claim_unit(subject_datum, claim, reason_codes=None)
        if expected_direct is None:
            return float("inf")
        return abs(observed_value - expected_direct)

    def _context_pair_score(self, left: EvidenceDatum, right: EvidenceDatum) -> float:
        score = 0.0
        if left.datum_id == right.datum_id:
            return -1.0

        def _norm(value: Any) -> str:
            return self._normalize_text(str(value or ""))

        if _norm(left.dataset) and _norm(left.dataset) == _norm(right.dataset):
            score += 2.0
        if _norm(left.split) and _norm(left.split) == _norm(right.split):
            score += 1.5
        if _norm(left.setting) and _norm(left.setting) == _norm(right.setting):
            score += 1.5

        if left.col_index >= 0 and right.col_index >= 0 and left.col_index == right.col_index:
            score += 1.5
        elif _norm(left.col_header) and _norm(left.col_header) == _norm(right.col_header):
            score += 1.0

        if left.row_index >= 0 and right.row_index >= 0:
            row_gap = abs(left.row_index - right.row_index)
            if row_gap <= 1:
                score += 0.5
        return score

    def _is_figure_evidence(self, evidence: List[EvidenceDatum]) -> bool:
        return any(str(item.source_type or "").strip().lower() == "figure" for item in evidence)

    def _infer_metric_for_figure_claim(
        self,
        claim: ClaimIR,
        by_metric: Dict[str, List[EvidenceDatum]],
    ) -> Optional[Tuple[str, List[EvidenceDatum], float]]:
        requested_subject = self._normalize_text(claim.subject) or "ours"
        requested_comparator = self._normalize_text(claim.comparator or claim.denominator or "baseline")
        needs_comparator = self._needs_comparator(claim.claim_type)

        if claim.claim_type == "from_to":
            target_subject = self._safe_float(claim.declared_to if claim.declared_to is not None else claim.declared_value)
            target_comparator = self._safe_float(claim.declared_from)
        else:
            target_subject = self._safe_float(claim.declared_value)
            target_comparator = None

        if target_subject is None:
            return None

        best_metric = ""
        best_rows: List[EvidenceDatum] = []
        best_score: Optional[float] = None
        for metric_key, rows in by_metric.items():
            subject_buckets: Dict[str, List[EvidenceDatum]] = {}
            for item in rows:
                key = self._normalize_text(item.subject)
                subject_buckets.setdefault(key, []).append(item)

            subject_rows = self._resolve_subject_rows(subject_buckets, requested_subject)
            if not subject_rows:
                # Some figure extractors swap series/category axes:
                # claim.metric may appear in subject slots while series labels
                # (e.g., improved/about-the-same) become metrics.
                subject_rows = self._resolve_subject_rows(subject_buckets, self._normalize_text(claim.metric))
            if not subject_rows:
                continue

            comparator_rows: List[EvidenceDatum] = []
            if needs_comparator:
                comparator_rows = self._resolve_subject_rows(subject_buckets, requested_comparator)
                if not comparator_rows:
                    axis_hint = self._normalize_text(claim.metric)
                    comparator_rows = self._resolve_subject_rows(subject_buckets, axis_hint)
                if not comparator_rows:
                    continue

            _, subject_score = self._pick_best_subject_row(
                subject_rows,
                target_subject,
                target_unit=claim.declared_unit,
                target_scale=claim.declared_scale,
            )
            total_score = subject_score
            if needs_comparator:
                _, comparator_score = self._pick_best_subject_row(
                    comparator_rows,
                    target_comparator,
                    target_unit=claim.declared_unit,
                    target_scale=claim.declared_scale,
                )
                total_score += comparator_score

            if best_score is None or total_score < best_score:
                best_metric = metric_key
                best_rows = rows
                best_score = total_score

        if best_score is None:
            return None
        return best_metric, best_rows, best_score

    def _text_fuzzy_match(self, left: str, right: str) -> bool:
        lhs = self._normalize_text(left)
        rhs = self._normalize_text(right)
        if not lhs or not rhs:
            return False
        if lhs == rhs:
            return True
        if lhs not in rhs and rhs not in lhs:
            return False
        lhs_tokens = set(lhs.split())
        rhs_tokens = set(rhs.split())
        for unit_token in {"percent", "pp"}:
            if (unit_token in lhs_tokens) != (unit_token in rhs_tokens):
                return False
        return True

    def _metric_anchor_score(
        self,
        claim: ClaimIR,
        *,
        metric_key: str,
        rows: List[EvidenceDatum],
    ) -> float:
        score = 0.0
        claim_metric = self._normalize_text(claim.metric)
        metric_norm = self._normalize_text(metric_key)
        if claim_metric and self._text_fuzzy_match(claim_metric, metric_norm):
            score += 1.0

        requested_subject = self._normalize_text(claim.subject)
        requested_comparator = self._normalize_text(claim.comparator or claim.denominator)
        labels: List[str] = []
        for item in rows:
            labels.extend(
                [
                    self._normalize_text(item.subject),
                    self._normalize_text(item.row_header),
                    self._normalize_text(item.col_header),
                    self._normalize_text(item.dataset),
                    self._normalize_text(item.split),
                    self._normalize_text(item.setting),
                ]
            )
        if requested_subject and any(self._text_fuzzy_match(requested_subject, lbl) for lbl in labels if lbl):
            score += 1.0
        if requested_comparator and any(self._text_fuzzy_match(requested_comparator, lbl) for lbl in labels if lbl):
            score += 1.0
        return score

    def _resolve_subject_rows(
        self,
        subject_buckets: Dict[str, List[EvidenceDatum]],
        requested_subject: str,
    ) -> List[EvidenceDatum]:
        exact = list(subject_buckets.get(requested_subject, []))
        if exact:
            return exact
        if not requested_subject:
            return []
        fuzzy_keys = [k for k in subject_buckets.keys() if self._text_fuzzy_match(requested_subject, k)]
        if len(fuzzy_keys) == 1:
            return list(subject_buckets.get(fuzzy_keys[0], []))
        return []

    def _pick_best_subject_row(
        self,
        rows: List[EvidenceDatum],
        target: Optional[float],
        *,
        target_unit: str = "",
        target_scale: str = "",
    ) -> Tuple[Optional[EvidenceDatum], float]:
        if not rows:
            return None, float("inf")

        def _score(item: EvidenceDatum) -> Tuple[float, float]:
            confidence = self._safe_float(item.meta.get("confidence")) or 0.0
            if target is None:
                return (0.0, -confidence)
            aligned_target = self._align_target_value_for_datum(
                target,
                target_unit=target_unit,
                target_scale=target_scale,
                datum=item,
            )
            return (abs(float(item.value) - float(aligned_target)), -confidence)

        best = sorted(rows, key=_score)[0]
        if target is None:
            return best, 0.0
        best_target = self._align_target_value_for_datum(
            target,
            target_unit=target_unit,
            target_scale=target_scale,
            datum=best,
        )
        return best, abs(float(best.value) - float(best_target))

    def _infer_datum_by_value(
        self,
        *,
        candidates: List[EvidenceDatum],
        target: Optional[float],
        label_hint: str,
        secondary_hint: str = "",
        target_unit: str = "",
        target_scale: str = "",
        exclude_ids: Optional[set[str]] = None,
        max_abs_gap: Optional[float] = None,
    ) -> Optional[EvidenceDatum]:
        if target is None:
            return None
        blocked = exclude_ids or set()
        pool = [item for item in candidates if item.datum_id not in blocked]
        if not pool:
            return None

        hint = self._normalize_text(label_hint)
        secondary = self._normalize_text(secondary_hint)

        def _score(item: EvidenceDatum) -> Tuple[float, float, float, float]:
            aligned_target = self._align_target_value_for_datum(
                target,
                target_unit=target_unit,
                target_scale=target_scale,
                datum=item,
            )
            value_gap = abs(float(item.value) - float(aligned_target))
            subject_norm = self._normalize_text(item.subject)
            metric_norm = self._normalize_text(item.metric)

            def _text_match(lhs: str, rhs: str) -> bool:
                return bool(lhs and rhs and self._text_fuzzy_match(lhs, rhs))

            primary_match = _text_match(hint, subject_norm) or _text_match(hint, metric_norm)
            secondary_match = _text_match(secondary, subject_norm) or _text_match(secondary, metric_norm)
            hint_penalty = 0.0 if primary_match else 0.25
            secondary_penalty = 0.0 if secondary_match else (0.05 if secondary else 0.0)
            confidence = self._safe_float(item.meta.get("confidence")) or 0.0
            return (value_gap + hint_penalty + secondary_penalty, value_gap, hint_penalty + secondary_penalty, -confidence)

        ranked = sorted(pool, key=_score)
        if not ranked:
            return None
        if max_abs_gap is not None:
            best_target = self._align_target_value_for_datum(
                target,
                target_unit=target_unit,
                target_scale=target_scale,
                datum=ranked[0],
            )
            best_gap = abs(float(ranked[0].value) - float(best_target))
            if best_gap > max_abs_gap:
                return None
        if len(ranked) >= 2:
            first = _score(ranked[0])[0]
            second = _score(ranked[1])[0]
            if abs(first - second) <= 1e-9:
                return None
        return ranked[0]

    def _infer_comparator_for_figure_by_declared_formula(
        self,
        *,
        claim: ClaimIR,
        subject_datum: EvidenceDatum,
        candidates: List[EvidenceDatum],
    ) -> Optional[EvidenceDatum]:
        target = self._expected_comparator_target_by_formula(claim=claim, subject_datum=subject_datum)
        if target is None:
            return None
        target_value, target_unit, target_scale = target
        return self._infer_datum_by_value(
            candidates=candidates,
            target=target_value,
            label_hint=self._normalize_text(claim.comparator or claim.denominator),
            secondary_hint=self._normalize_text(claim.metric),
            target_unit=target_unit,
            target_scale=target_scale,
            exclude_ids={subject_datum.datum_id},
        )

    def _infer_subject_comparator_pair_by_from_to(
        self,
        *,
        claim: ClaimIR,
        candidates: List[EvidenceDatum],
    ) -> Optional[Tuple[EvidenceDatum, EvidenceDatum, float]]:
        if not candidates:
            return None
        target_to = self._safe_float(claim.declared_to if claim.declared_to is not None else claim.declared_value)
        target_from = self._safe_float(claim.declared_from)
        if target_to is None or target_from is None:
            return None

        best_pair: Optional[Tuple[EvidenceDatum, EvidenceDatum, float]] = None
        for subject in candidates:
            expected_to = self._datum_value_in_claim_unit(subject, claim, reason_codes=None)
            if expected_to is None:
                continue
            for comparator in candidates:
                if comparator.datum_id == subject.datum_id:
                    continue
                expected_from = self._datum_value_in_claim_unit(comparator, claim, reason_codes=None)
                if expected_from is None:
                    continue
                score = abs(float(target_to) - float(expected_to)) + abs(float(target_from) - float(expected_from))
                if self._normalize_text(subject.metric) != self._normalize_text(comparator.metric):
                    score += 0.25
                observed_value = self._safe_float(claim.declared_value)
                if observed_value is not None and self._normalize_text(claim.claim_type) != "from_to":
                    expected_value, _, error_code = self._compute_expected_value(
                        claim=claim,
                        subject_datum=subject,
                        comparator_datum=comparator,
                    )
                    if expected_value is not None and not error_code:
                        score += 0.6 * abs(float(observed_value) - float(expected_value))
                if best_pair is None or score < best_pair[2]:
                    best_pair = (subject, comparator, score)

        if best_pair is None:
            return None
        max_gap = self._from_to_anchor_max_gap(target_to, claim) + self._from_to_anchor_max_gap(target_from, claim)
        if best_pair[2] > max(max_gap * 2.0, 8.0):
            return None
        return best_pair

    def _expected_comparator_target_by_formula(
        self,
        *,
        claim: ClaimIR,
        subject_datum: EvidenceDatum,
    ) -> Optional[Tuple[float, str, str]]:
        observed_value, _ = self._normalize_observed_value(claim)
        if observed_value is None:
            return None
        claim_type = self._normalize_text(claim.claim_type)
        subject_ratio = self._datum_to_ratio(subject_datum)
        subject_percent = self._datum_to_percent(subject_datum)
        metric_hint = self._normalize_text(claim.metric)

        if claim_type == "ratio":
            if subject_ratio is None or abs(observed_value) <= 1e-12:
                return None
            return subject_ratio / observed_value, "raw", "0_1"

        if claim_type == "pp_change":
            if subject_percent is None:
                return None
            return subject_percent - observed_value, "percent", "0_100"

        if claim_type in {"relative_improvement", "relative_decrease", "error_reduction", "relative_error_reduction"}:
            if subject_ratio is None:
                return None
            ratio_factor = 1.0 - (observed_value / 100.0)
            if claim_type == "relative_improvement":
                ratio_factor = 1.0 + (observed_value / 100.0)
            if abs(ratio_factor) <= 1e-12:
                return None

            if claim_type in {"error_reduction", "relative_error_reduction"} and self._metric_prefers_error_rate_reduction(metric_hint):
                subject_error = 1.0 - subject_ratio
                if abs(ratio_factor) <= 1e-12:
                    return None
                comparator_error = subject_error / ratio_factor
                comparator_ratio = 1.0 - comparator_error
                return comparator_ratio, "raw", "0_1"

            comparator_ratio = subject_ratio / ratio_factor
            return comparator_ratio, "raw", "0_1"

        return None

    def _infer_from_to_pair_for_figure_axis_swap(
        self,
        *,
        claim: ClaimIR,
        evidence: List[EvidenceDatum],
        fallback_subject: EvidenceDatum,
    ) -> Optional[Tuple[EvidenceDatum, EvidenceDatum]]:
        axis_hint = self._normalize_text(claim.metric)
        if not axis_hint:
            return None
        subject_target = self._safe_float(claim.declared_to if claim.declared_to is not None else claim.declared_value)
        comparator_target = self._safe_float(claim.declared_from)
        if subject_target is None or comparator_target is None:
            return None

        subject_buckets: Dict[str, List[EvidenceDatum]] = {}
        for item in evidence:
            key = self._normalize_text(item.subject)
            subject_buckets.setdefault(key, []).append(item)
        axis_rows = self._resolve_subject_rows(subject_buckets, axis_hint)
        if not axis_rows:
            return None

        axis_subject = self._infer_datum_by_value(
            candidates=axis_rows,
            target=subject_target,
            label_hint=axis_hint,
            secondary_hint=self._normalize_text(claim.subject),
            target_unit=claim.declared_unit,
            target_scale=claim.declared_scale,
        )
        if axis_subject is None:
            axis_subject = fallback_subject
        if axis_subject is None:
            return None

        axis_comparator = self._infer_datum_by_value(
            candidates=axis_rows,
            target=comparator_target,
            label_hint=axis_hint,
            secondary_hint=self._normalize_text(claim.comparator or claim.denominator),
            target_unit=claim.declared_unit,
            target_scale=claim.declared_scale,
            exclude_ids={axis_subject.datum_id},
        )
        if axis_comparator is None:
            return None
        return axis_subject, axis_comparator

    def _axis_extrema_mode_from_subject(self, subject_text: str) -> str:
        text = self._normalize_text(subject_text)
        if not text:
            return ""
        has_axis = ("axis" in text) or ("y axis" in text) or ("yaxis" in text)
        if not has_axis:
            return ""
        if any(token in text for token in {"maximum", "max", "upper", "top", "highest", "peak"}):
            return "max"
        if any(token in text for token in {"minimum", "min", "lower", "bottom", "lowest"}):
            return "min"
        return ""

    def _infer_axis_extrema_subject(
        self,
        *,
        claim: ClaimIR,
        candidates: List[EvidenceDatum],
    ) -> Optional[EvidenceDatum]:
        extrema_mode = self._axis_extrema_mode_from_subject(claim.subject)
        if not extrema_mode:
            return None
        if not candidates:
            return None

        metric_hint = self._normalize_text(claim.metric)
        filtered = [
            item
            for item in candidates
            if not metric_hint
            or self._text_fuzzy_match(metric_hint, self._normalize_text(item.metric))
            or self._text_fuzzy_match(metric_hint, self._normalize_text(item.col_header))
        ]
        pool = filtered if filtered else list(candidates)
        if not pool:
            return None

        ranked: List[Tuple[float, float, EvidenceDatum]] = []
        for item in pool:
            value_in_claim_unit = self._datum_value_in_claim_unit(item, claim, reason_codes=None)
            if value_in_claim_unit is None:
                continue
            confidence = self._safe_float(item.meta.get("confidence")) or 0.0
            ranked.append((float(value_in_claim_unit), confidence, item))
        if not ranked:
            return None
        if extrema_mode == "max":
            ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        else:
            ranked.sort(key=lambda row: (row[0], -row[1]), reverse=False)
        return ranked[0][2]

    def _align_target_value_for_datum(
        self,
        target: float,
        *,
        target_unit: str,
        target_scale: str,
        datum: EvidenceDatum,
    ) -> float:
        aligned = float(target)
        claim_unit = self._normalize_unit(target_unit)
        claim_scale = self._normalize_scale(target_scale)
        datum_unit = self._normalize_unit(datum.unit)
        datum_scale = self._normalize_scale(datum.scale)

        claim_ratio_like = claim_scale == "0_1" and claim_unit in {"raw", "percent", "pp"}
        claim_percent_like = claim_scale == "0_100" or claim_unit in {"percent", "pp"}
        datum_ratio_like = datum_scale == "0_1"
        datum_percent_like = datum_scale == "0_100" or datum_unit in {"percent", "pp"}

        if claim_ratio_like and datum_percent_like:
            return aligned * 100.0
        if claim_percent_like and datum_ratio_like:
            return aligned / 100.0
        return aligned

    def _hydrate_missing_declared_values(
        self,
        claim: ClaimIR,
        trace: VerificationTrace,
        reason_codes: List[str],
    ) -> None:
        has_from_to = claim.declared_from is not None and claim.declared_to is not None
        has_value = claim.declared_value is not None
        if has_value and has_from_to:
            return

        numeric_values = self._extract_numeric_candidates_from_text(claim.raw_text)
        if not numeric_values:
            return

        claim_type = self._normalize_text(claim.claim_type)
        recovered: Dict[str, float] = {}
        if claim_type == "from_to":
            if claim.declared_from is None and len(numeric_values) >= 2:
                claim.declared_from = numeric_values[0]
                recovered["declared_from"] = claim.declared_from
            if claim.declared_to is None:
                if len(numeric_values) >= 2:
                    claim.declared_to = numeric_values[1]
                else:
                    claim.declared_to = numeric_values[0]
                recovered["declared_to"] = claim.declared_to
            if claim.declared_value is None and claim.declared_to is not None:
                claim.declared_value = claim.declared_to
                recovered["declared_value"] = claim.declared_value
        elif claim.declared_value is None:
            claim.declared_value = numeric_values[0]
            recovered["declared_value"] = claim.declared_value

        if recovered:
            reason_codes.append("CLAIM_VALUE_RECOVERED_TEXT")
            trace.steps.append(
                TraceStep(
                    step_name="hydrate_claim_values",
                    status="ok",
                    message="recovered missing declared numeric value(s) from claim text",
                    intermediate={
                        "recovered_fields": recovered,
                        "numeric_candidates": numeric_values,
                    },
                    error_code="CLAIM_VALUE_RECOVERED_TEXT",
                )
            )

    @staticmethod
    def _extract_numeric_candidates_from_text(text: str) -> List[float]:
        raw = str(text or "")
        if not raw:
            return []
        # Support signed integers/decimals and ignore separators like commas.
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", raw)
        values: List[float] = []
        for token in matches:
            normalized = token.replace(",", "")
            try:
                values.append(float(normalized))
            except ValueError:
                continue
        return values

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _compute_expected_value(
        self,
        *,
        claim: ClaimIR,
        subject_datum: EvidenceDatum,
        comparator_datum: EvidenceDatum | None,
    ) -> Tuple[Optional[float], str, str]:
        claim_type = self._normalize_text(claim.claim_type)

        if claim_type == "direct_value":
            expected = self._datum_value_in_claim_unit(subject_datum, claim, reason_codes=None)
            return expected, "subject_value", "" if expected is not None else "UNIT_SCALE_MISMATCH"

        if claim_type in {"relative_improvement", "relative_decrease", "error_reduction", "relative_error_reduction"}:
            if comparator_datum is None:
                return None, "((subject-comparator)/comparator)*100", "DENOMINATOR_MISSING"
            subject_ratio = self._datum_to_ratio(subject_datum)
            comparator_ratio = self._datum_to_ratio(comparator_datum)
            if subject_ratio is None or comparator_ratio is None or abs(comparator_ratio) <= 1e-12:
                return None, "((subject-comparator)/comparator)*100", "F_DIV_ZERO"
            if claim_type in {"relative_improvement"}:
                expected = ((subject_ratio - comparator_ratio) / comparator_ratio) * 100.0
            elif claim_type in {"relative_decrease"}:
                expected = ((comparator_ratio - subject_ratio) / comparator_ratio) * 100.0
            else:
                metric_hint = self._normalize_text(claim.metric)
                if self._metric_prefers_error_rate_reduction(metric_hint):
                    comparator_error = 1.0 - comparator_ratio
                    subject_error = 1.0 - subject_ratio
                    if abs(comparator_error) <= 1e-12:
                        return None, "((err_comparator-err_subject)/err_comparator)*100", "F_DIV_ZERO"
                    expected = ((comparator_error - subject_error) / comparator_error) * 100.0
                    return expected, "((err_comparator-err_subject)/err_comparator)*100", ""
                expected = ((comparator_ratio - subject_ratio) / comparator_ratio) * 100.0
            return expected, "((subject-comparator)/comparator)*100", ""

        if claim_type == "ratio":
            if comparator_datum is None:
                return None, "subject/comparator", "DENOMINATOR_MISSING"
            subject_ratio = self._datum_to_ratio(subject_datum)
            comparator_ratio = self._datum_to_ratio(comparator_datum)
            if subject_ratio is None or comparator_ratio is None or abs(comparator_ratio) <= 1e-12:
                return None, "subject/comparator", "F_DIV_ZERO"
            expected = subject_ratio / comparator_ratio
            return expected, "subject_ratio / comparator_ratio", ""

        if claim_type == "scale_normalize":
            target_scale = self._normalize_scale(claim.declared_scale)
            if target_scale == "0_100":
                expected = self._datum_to_percent(subject_datum)
                if expected is not None:
                    datum_unit = self._normalize_unit(subject_datum.unit)
                    datum_scale = self._normalize_scale(subject_datum.scale)
                    # Raw+unknown table cells are common in curated hard cases.
                    # Resolve 0_1<->0_100 ambiguity via declared-value proximity.
                    if datum_unit == "raw" and datum_scale == "raw":
                        raw = float(subject_datum.value)
                        options = [raw, raw * 100.0]
                        if claim.declared_value is not None:
                            expected = min(options, key=lambda v: abs(float(v) - float(claim.declared_value)))
                        else:
                            expected = raw * 100.0 if abs(raw) <= 1.0 else raw
                return expected, "scale_normalize_to_0_100", "" if expected is not None else "UNIT_SCALE_MISMATCH"
            if target_scale == "0_1":
                expected = self._datum_to_ratio(subject_datum)
                if expected is not None:
                    datum_unit = self._normalize_unit(subject_datum.unit)
                    datum_scale = self._normalize_scale(subject_datum.scale)
                    if datum_unit == "raw" and datum_scale == "raw":
                        raw = float(subject_datum.value)
                        options = [raw, raw / 100.0]
                        if claim.declared_value is not None:
                            expected = min(options, key=lambda v: abs(float(v) - float(claim.declared_value)))
                        else:
                            expected = raw / 100.0 if abs(raw) > 1.0 else raw
                return expected, "scale_normalize_to_0_1", "" if expected is not None else "UNIT_SCALE_MISMATCH"
            return None, "scale_normalize", "UNIT_SCALE_MISMATCH"

        if claim_type == "pp_change":
            if comparator_datum is None:
                return None, "subject_pp - comparator_pp", "DENOMINATOR_MISSING"
            subject_percent = self._datum_to_percent(subject_datum)
            comparator_percent = self._datum_to_percent(comparator_datum)
            if subject_percent is None or comparator_percent is None:
                return None, "subject_pp - comparator_pp", "UNIT_SCALE_MISMATCH"
            expected = subject_percent - comparator_percent
            return expected, "subject_percent - comparator_percent", ""

        return None, "", "FORMULA_INVALID"

    def _effective_compare_tolerance(
        self,
        *,
        claim: ClaimIR,
        reason_codes: List[str],
        subject_datum: EvidenceDatum,
        comparator_datum: EvidenceDatum | None,
    ) -> Tuple[float, float, str]:
        abs_tol = max(claim.abs_tolerance, 0.0)
        rel_tol = max(claim.rel_tolerance, 0.0)
        claim_type = self._normalize_text(claim.claim_type)
        if claim_type not in {
            "ratio",
            "pp_change",
            "relative_change",
            "relative_improvement",
            "error_reduction",
            "relative_error_reduction",
            "scale_normalize",
        }:
            return abs_tol, rel_tol, ""
        if self._normalize_text(subject_datum.source_type) != "figure":
            return abs_tol, rel_tol, ""
        has_noisy_binding = any(
            code in {
                "DENOMINATOR_INFERRED_BY_FORMULA",
                "DENOMINATOR_INFERRED_BY_VALUE",
                "DENOMINATOR_DISAMBIGUATED_DECLARED_VALUE",
                "TARGET_DISAMBIGUATED_DECLARED_VALUE",
                "PAIR_INFERRED_BY_FROM_TO",
                "SUBJECT_INFERRED_BY_VALUE",
                "SUBJECT_INFERRED_BY_FROM_TO_VALUE",
                "UNIT_SCALE_ASSUMED",
            }
            for code in reason_codes
        )
        if not has_noisy_binding:
            return abs_tol, rel_tol, ""
        if claim_type == "scale_normalize":
            target_scale = self._normalize_scale(claim.declared_scale)
            if target_scale == "0_1":
                tuned_abs_tol = max(abs_tol, 0.03)
                tuned_rel_tol = max(rel_tol, 0.05)
            elif target_scale == "0_100":
                tuned_abs_tol = max(abs_tol, 0.5)
                tuned_rel_tol = max(rel_tol, 0.05)
            else:
                tuned_abs_tol = max(abs_tol, 0.25)
                tuned_rel_tol = max(rel_tol, 0.05)
            return tuned_abs_tol, tuned_rel_tol, "SCALE_TOLERANCE_RELAXED"
        # Figure extraction on derived claims is noisier than table: allow slightly
        # looser tolerance after deterministic disambiguation succeeds.
        tuned_abs_tol = max(abs_tol, 0.35)
        tuned_rel_tol = max(rel_tol, 0.05)
        return tuned_abs_tol, tuned_rel_tol, "DERIVED_TOLERANCE_RELAXED"

    def _normalize_observed_value(self, claim: ClaimIR) -> Tuple[Optional[float], str]:
        if claim.declared_value is None:
            return None, ""
        value = float(claim.declared_value)
        if self._normalize_text(claim.claim_type) == "scale_normalize":
            # For explicit scale-normalize claims, declared value already represents
            # the target scale in claim text; compare it directly.
            return value, ""
        unit = self._normalize_unit(claim.declared_unit)
        scale = self._normalize_scale(claim.declared_scale)

        if unit in {"percent", "pp"}:
            if scale == "0_1":
                return value * 100.0, "CLAIM_SCALE_NORMALIZED"
            return value, ""
        return value, ""

    def _datum_to_percent(self, datum: EvidenceDatum) -> Optional[float]:
        scale = self._normalize_scale(datum.scale)
        unit = self._normalize_unit(datum.unit)
        value = float(datum.value)

        if unit == "percent":
            if scale == "0_1":
                return value * 100.0
            if scale in {"0_100", "unknown"}:
                if scale == "unknown" and abs(value) <= 1.0:
                    return value * 100.0
                return value
        if unit == "raw":
            if scale == "0_1":
                return value * 100.0
            if scale == "0_100":
                return value
            return value
        return None

    def _datum_to_ratio(self, datum: EvidenceDatum) -> Optional[float]:
        scale = self._normalize_scale(datum.scale)
        unit = self._normalize_unit(datum.unit)
        value = float(datum.value)
        if unit == "percent":
            if scale == "0_100":
                return value / 100.0
            if scale in {"0_1", "unknown"}:
                if scale == "unknown" and abs(value) > 1.0:
                    return value / 100.0
                return value
        if unit == "raw":
            if scale == "0_100":
                return value / 100.0
            return value
        return None

    def _datum_value_in_claim_unit(
        self,
        datum: EvidenceDatum,
        claim: ClaimIR,
        reason_codes: Optional[List[str]],
    ) -> Optional[float]:
        unit = self._normalize_unit(claim.declared_unit)
        scale = self._normalize_scale(claim.declared_scale)

        if unit in {"percent", "pp"}:
            return self._datum_to_percent(datum)

        ratio = self._datum_to_ratio(datum)
        percent = self._datum_to_percent(datum)
        if ratio is None and percent is None:
            return None

        if scale == "0_1" and ratio is not None:
            return ratio
        if scale == "0_100" and percent is not None:
            return percent
        if scale == "raw":
            if claim.declared_value is not None and abs(claim.declared_value) <= 1.0 and ratio is not None:
                if reason_codes is not None:
                    reason_codes.append("UNIT_SCALE_ASSUMED")
                return ratio
            if percent is not None:
                if reason_codes is not None:
                    reason_codes.append("UNIT_SCALE_ASSUMED")
                return percent
        if ratio is not None:
            if reason_codes is not None:
                reason_codes.append("UNIT_SCALE_ASSUMED")
            return ratio
        return percent

    @staticmethod
    def _needs_comparator(claim_type: str) -> bool:
        normalized = " ".join(str(claim_type or "").strip().lower().split())
        return normalized in {
            "relative_improvement",
            "relative_decrease",
            "pp_change",
            "error_reduction",
            "relative_error_reduction",
            "ratio",
            "from_to",
        }

    @staticmethod
    def _apply_rounding(value: float, mode: str, decimals: Optional[int]) -> float:
        if decimals is None or decimals < 0:
            return float(value)
        normalized_mode = str(mode or "round").strip().lower()
        quant = Decimal("1").scaleb(-decimals)
        decimal_value = Decimal(str(value))
        if normalized_mode == "floor":
            return float(decimal_value.quantize(quant, rounding=ROUND_FLOOR))
        if normalized_mode == "ceil":
            return float(decimal_value.quantize(quant, rounding=ROUND_CEILING))
        return float(decimal_value.quantize(quant, rounding=ROUND_HALF_UP))

    def _finalize(
        self,
        *,
        claim: ClaimIR,
        trace: VerificationTrace,
        verdict: Verdict,
        expected: Optional[float],
        observed: Optional[float],
        abs_diff: Optional[float],
        rel_diff: Optional[float],
        reason_codes: List[str],
        resolved_metric: str = "",
        resolved_subject: str = "",
        resolved_comparator: str = "",
        formula: str = "",
    ) -> VerificationResult:
        dedup_codes = self._dedup_reason_codes(reason_codes)
        if verdict not in SUPPORTED_VERDICTS:
            verdict = "unsupported"

        if verdict == "supported" and any(code in FATAL_REASON_CODES for code in dedup_codes):
            verdict = "unsupported"
        if verdict == "supported" and self._has_assumption_code(dedup_codes):
            allow_supported = self._allow_supported_with_assumption(claim=claim, codes=dedup_codes)
            if not allow_supported:
                verdict = "uncertain"

        trace.final_verdict = verdict
        trace.expected_value = expected
        trace.observed_value = observed
        trace.delta = abs_diff
        trace.reason_codes = list(dedup_codes)
        trace.steps.append(
            TraceStep(
                step_name="judge",
                status="ok",
                message="final deterministic verdict",
                intermediate={"verdict": verdict, "reason_codes": dedup_codes},
            )
        )

        return VerificationResult(
            verdict=verdict,
            expected_value=expected,
            observed_value=observed,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            reason_codes=dedup_codes,
            trace=trace,
            claim=claim,
            resolved_metric=resolved_metric,
            resolved_subject=resolved_subject,
            resolved_comparator=resolved_comparator,
            formula=formula,
        )

    @staticmethod
    def _allow_supported_with_assumption(*, claim: ClaimIR, codes: List[str]) -> bool:
        code_set = {str(code or "") for code in (codes or [])}
        if "UNIT_SCALE_ASSUMED" not in code_set:
            return False
        if code_set.intersection({"TARGET_FUZZY_MATCH", "TARGET_MULTI_MATCH", "TARGET_NO_MATCH"}):
            return False
        claim_type = " ".join(str(claim.claim_type or "").strip().lower().split())
        if claim_type not in {
            "from_to",
            "ratio",
            "pp_change",
            "relative_change",
            "relative_improvement",
            "relative_error_reduction",
            "error_reduction",
            "scale_normalize",
        }:
            return False
        strong_anchor_codes = {
            "PAIR_INFERRED_BY_FROM_TO",
            "TARGET_DISAMBIGUATED_DECLARED_VALUE",
            "DENOMINATOR_DISAMBIGUATED_DECLARED_VALUE",
        }
        return bool(code_set.intersection(strong_anchor_codes))

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = str(text or "").strip().lower()
        normalized = normalized.replace("\\%", " percent ")
        normalized = normalized.replace("%", " percent ")
        normalized = re.sub(r"[-/]", " ", normalized)
        normalized = re.sub(r"[^\w\s\.]", " ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _metric_prefers_error_rate_reduction(metric: str) -> bool:
        # Metrics that are typically "higher is better" bounded scores:
        # for these, error-reduction should be computed on (1 - score).
        score_like = re.search(
            r"(accuracy|acc|f1|f\.?1|f score|em\b|bleu|rouge|auc|score|"
            r"mnli|qnli|qqp|rte|sst|cola|sts|mrpc|race|squad|glue|top ?1)",
            metric,
        )
        # Metrics that are already error/cost style should use direct relative decrease.
        non_score = re.search(r"(error|loss|wer|ppl|perplexity|speedup|flops|params|steps?|time|latency)", metric)
        return bool(score_like) and not bool(non_score)

    @staticmethod
    def _normalize_unit(unit: str) -> Unit:
        normalized = str(unit or "").strip().lower()
        if normalized in {"%", "percent", "percentage"}:
            return "percent"
        if normalized in {"pp", "percentage_points", "percentage points"}:
            return "pp"
        return "raw"

    @staticmethod
    def _normalize_scale(scale: str) -> Scale:
        normalized = str(scale or "").strip().lower()
        if normalized in {"0_1", "01", "fraction"}:
            return "0_1"
        if normalized in {"0_100", "0100", "percent"}:
            return "0_100"
        if normalized in {"raw"}:
            return "raw"
        return "unknown"

    @staticmethod
    def _dedup_reason_codes(codes: List[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for code in codes:
            if not code:
                continue
            if code in seen:
                continue
            seen.add(code)
            deduped.append(code)
        return deduped

    @staticmethod
    def _has_assumption_code(codes: List[str]) -> bool:
        return any(code in {"UNIT_SCALE_ASSUMED", "TARGET_FUZZY_MATCH", "SUBJECT_ASSUMED"} for code in codes)
