from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from agents.base import BaseAgent
from core.logging_utils import get_logger, log_event
from core.schemas import AppState, Issue, Location
from tools.numeric_verification_kernel import ClaimIR, NumericVerificationKernel, VerificationResult
from tools.providers import LLMProviderError, TextLLMProvider, build_text_provider
from tools.table_claim_schema import LLMTableClaimsResponse
from tools.table_utils import (
    canonicalize_unit,
    extract_claim_irs,
    kernel_unit_from_table_unit,
    metric_rows_to_evidence_datums,
    normalize_metric_name,
    parse_table_csv,
)
from tools.text_utils import char_span_to_line_range, split_sentences_with_offsets


class TableConsistencyAgent(BaseAgent):
    name = "table_consistency"

    def __init__(self, provider_name: str = "none") -> None:
        self.provider: TextLLMProvider = build_text_provider(provider_name)
        self.logger = get_logger(__name__)
        self.kernel = NumericVerificationKernel()
        self.last_extractor: str = "regex_fallback"

    def run(self, state: AppState) -> List[Issue]:
        if not state.uploaded_tables:
            return []

        claims, extractor = self._extract_claims(state)
        self.last_extractor = extractor
        log_event(
            self.logger,
            logging.INFO,
            "table_claim_extraction_done",
            request_id=state.request_id,
            extractor=extractor,
            claim_count=len(claims),
        )
        if not claims:
            return []

        evidence = self._build_table_evidence(state)
        if not evidence:
            return []

        issues: List[Issue] = []
        traces: List[Dict[str, object]] = []
        records: List[Dict[str, object]] = []
        for idx, claim in enumerate(claims):
            result = self.kernel.verify_claim(claim, evidence)
            traces.append(result.trace.to_dict())
            records.append(result.to_eval_record())
            if result.verdict == "supported":
                continue
            issues.append(self._issue_from_result(state, idx, result))

        metadata = state.grounded_context.metadata if isinstance(state.grounded_context.metadata, dict) else {}
        previous = list(metadata.get("table_verification_traces", []))
        previous_records = list(metadata.get("table_verification_records", []))
        metadata["table_verification_traces"] = (previous + traces)[-160:]
        metadata["table_verification_records"] = (previous_records + records)[-320:]
        metadata["table_kernel_version"] = self.kernel.version
        metadata["table_claim_count"] = len(claims)
        metadata["table_evidence_count"] = len(evidence)
        state.grounded_context.metadata = metadata
        return issues

    def _build_table_evidence(self, state: AppState):
        evidence = []
        for table in state.uploaded_tables:
            source_ref = str(table.meta.get("name", "table")).strip() or "table"
            try:
                metric_rows = parse_table_csv(table.content)
                evidence.extend(metric_rows_to_evidence_datums(metric_rows, source_ref=source_ref))
            except Exception as exc:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "table_csv_parse_failed",
                    request_id=state.request_id,
                    table_name=source_ref,
                    error=str(exc),
                )
        return evidence

    def _issue_from_result(self, state: AppState, idx: int, result: VerificationResult) -> Issue:
        claim = result.claim
        start, end = claim.char_span

        status = "detected" if result.verdict == "contradicted" else "uncertain"
        severity = "high" if result.verdict == "contradicted" else "medium"
        if result.verdict == "unsupported":
            message = "Claim could not be deterministically verified from available table evidence."
        elif result.verdict == "uncertain":
            message = "Claim verification remains uncertain due to ambiguity, assumptions, or weakly constrained evidence."
        else:
            observed_text = (
                f"{result.observed_value:.4f}" if isinstance(result.observed_value, (int, float)) else "unknown"
            )
            expected_text = (
                f"{result.expected_value:.4f}" if isinstance(result.expected_value, (int, float)) else "unknown"
            )
            message = (
                f"Claimed value {observed_text} contradicts deterministic expected value "
                f"{expected_text} for the resolved table target."
            )

        snippet = claim.raw_text.strip()
        if not snippet and start >= 0 and end > start:
            snippet = state.current_text[start:end].strip()

        return Issue(
            id=f"table_claim_{idx}_{start}",
            type="text_table_mismatch",
            severity=severity,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            location=Location(
                sentence_index=claim.sentence_index,
                char_span=claim.char_span,
                snippet=snippet,
                line_range=char_span_to_line_range(state.current_text, start, end),
            ),
            evidence={
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "extractor": claim.extractor,
                "declared_unit": claim.declared_unit,
                "declared_scale": claim.declared_scale,
                "declared_value": claim.declared_value,
                "declared_from": claim.declared_from,
                "declared_to": claim.declared_to,
                "resolved_metric": result.resolved_metric,
                "resolved_subject": result.resolved_subject,
                "resolved_comparator": result.resolved_comparator,
                "formula": result.formula,
                "expected": result.expected_value,
                "observed": result.observed_value,
                "abs_diff": result.abs_diff,
                "rel_diff": result.rel_diff,
                "verdict": result.verdict,
                "reason_codes": result.reason_codes,
                "trace": result.trace.to_dict(),
                "thresholds": {
                    "abs_tolerance": claim.abs_tolerance,
                    "rel_tolerance": claim.rel_tolerance,
                    "rounding_mode": claim.rounding_mode,
                    "decimals": claim.decimals,
                },
            },
            message=message,
            provenance={
                "source": "numeric_verification_kernel",
                "kernel_version": self.kernel.version,
                "claim_extractor": claim.extractor,
                "trace_id": result.trace.trace_id,
            },
        )

    def _extract_claims(self, state: AppState) -> Tuple[List[ClaimIR], str]:
        regex_claims = extract_claim_irs(
            state.current_text,
            abs_tolerance=state.config.table_abs_tolerance,
            rel_tolerance=state.config.table_rel_tolerance,
            extractor="regex_fallback",
        )
        if regex_claims:
            return regex_claims, "regex_fallback"

        if self.provider.enabled():
            try:
                llm_claims = self._extract_claims_with_llm(state)
                if llm_claims:
                    return llm_claims, "llm"
                log_event(
                    self.logger,
                    logging.WARNING,
                    "table_claim_extraction_empty_llm",
                    request_id=state.request_id,
                )
            except LLMProviderError as exc:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "table_claim_extraction_llm_failed",
                    request_id=state.request_id,
                    error_type=exc.error_type,
                    error=str(exc),
                )

        return ([], "regex_fallback")

    def _extract_claims_with_llm(self, state: AppState) -> List[ClaimIR]:
        text = state.current_text
        sentence_payload = []
        for sent in split_sentences_with_offsets(text):
            sentence_payload.append(
                {
                    "sentence_index": sent["index"],
                    "char_span": [sent["start"], sent["end"]],
                    "text": sent["text"],
                }
            )

        system_prompt = (
            "You extract table-related numeric claims from manuscript text. "
            "Output JSON only and do not perform final numeric verification."
        )
        user_prompt = (
            "Extract table-related numeric claims.\n"
            "Schema:\n"
            "{\n"
            '  "claims": [\n'
            "    {\n"
            '      "sentence_index": int,\n'
            '      "char_span": [start, end],\n'
            '      "claim_type": "direct_value"|"from_to"|"relative_improvement"|"relative_decrease"|"pp_change"|"error_reduction"|"relative_error_reduction"|null,\n'
            '      "metric": string|null,\n'
            '      "subject": string|null,\n'
            '      "comparator": string|null,\n'
            '      "value": number|null,\n'
            '      "from_value": number|null,\n'
            '      "to_value": number|null,\n'
            '      "unit": "%"|"pp"|"raw",\n'
            '      "scale": "0_1"|"0_100"|"raw"|"unknown"|null,\n'
            '      "direction": "increase"|"decrease"|"improve"|null,\n'
            '      "reference": string|null\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Keep char_span within original text offsets.\n"
            "- Do not infer final correctness.\n"
            "- If uncertain, leave optional fields null.\n"
            f"\nText:\n{text}\n"
            f"\nSentences(JSON):\n{json.dumps(sentence_payload, ensure_ascii=False)}\n"
        )

        response = self.provider.complete_json(
            schema=LLMTableClaimsResponse,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            request_id=state.request_id,
        )

        claims: List[ClaimIR] = []
        text_len = len(text)
        sentence_lookup = {int(s["index"]): str(s["text"]) for s in split_sentences_with_offsets(text)}

        for idx, item in enumerate(response.get("claims", [])):
            sentence_index = int(item["sentence_index"])
            start, end = self._normalize_span(item["char_span"], text_len)
            snippet = text[start:end].strip() or sentence_lookup.get(sentence_index, "")

            claim_type = str(item.get("claim_type") or "").strip().lower()
            unit = canonicalize_unit(item.get("unit"))
            direction = str(item.get("direction") or "").strip().lower()
            if not claim_type:
                claim_type = self._infer_claim_type(unit=unit, direction=direction, snippet=snippet)

            declared_value = self._to_optional_float(item.get("value"))
            declared_from = self._to_optional_float(item.get("from_value"))
            declared_to = self._to_optional_float(item.get("to_value"))
            if claim_type == "from_to" and (declared_from is None or declared_to is None):
                parsed_from, parsed_to = self._parse_from_to_from_snippet(snippet)
                declared_from = declared_from if declared_from is not None else parsed_from
                declared_to = declared_to if declared_to is not None else parsed_to
            if declared_value is None and declared_to is not None:
                declared_value = declared_to

            raw_scale = str(item.get("scale") or "").strip().lower()
            declared_scale = raw_scale if raw_scale in {"0_1", "0_100", "raw", "unknown"} else ""
            if not declared_scale:
                declared_scale = self._infer_declared_scale(declared_value, unit)

            decimals = None
            if declared_value is not None:
                value_text = str(declared_value)
                if "." in value_text:
                    decimals = len(value_text.split(".", 1)[1].rstrip("0"))

            metric = normalize_metric_name(str(item.get("metric") or "")) if item.get("metric") else ""
            subject = str(item.get("subject") or "ours").strip().lower() or "ours"
            comparator = str(item.get("comparator") or "baseline").strip().lower() or "baseline"

            claims.append(
                ClaimIR(
                    claim_id=f"claim_{sentence_index}_{idx}",
                    sentence_index=sentence_index,
                    char_span=(start, end),
                    raw_text=snippet,
                    claim_type=claim_type,
                    relation="approx",
                    metric=metric,
                    subject=subject,
                    comparator=comparator,
                    denominator=comparator,
                    declared_value=declared_value,
                    declared_from=declared_from,
                    declared_to=declared_to,
                    declared_unit=kernel_unit_from_table_unit(unit),
                    declared_scale=declared_scale,
                    abs_tolerance=state.config.table_abs_tolerance,
                    rel_tolerance=state.config.table_rel_tolerance,
                    rounding_mode="round",
                    decimals=decimals if decimals and decimals > 0 else None,
                    extractor="llm",
                    confidence=None,
                    ambiguity_flags=[],
                    metadata={"reference": item.get("reference"), "direction": direction},
                )
            )

        return claims

    @staticmethod
    def _to_optional_float(value: object) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_span(span: List[int], text_len: int) -> Tuple[int, int]:
        start = max(0, min(int(span[0]), text_len))
        end = max(start, min(int(span[1]), text_len))
        return start, end

    @staticmethod
    def _parse_from_to_from_snippet(snippet: str) -> Tuple[Optional[float], Optional[float]]:
        match = re.search(
            r"from\s+(\d+(?:\.\d+)?)\s*(?:%|percent)?\s+to\s+(\d+(?:\.\d+)?)\s*(?:%|percent)?",
            snippet,
            flags=re.IGNORECASE,
        )
        if not match:
            return None, None
        return float(match.group(1)), float(match.group(2))

    @staticmethod
    def _infer_declared_scale(value: Optional[float], unit: str) -> str:
        if value is None:
            return "unknown"
        if unit in {"%", "pp"}:
            if abs(value) <= 1.0:
                return "0_1"
            return "0_100"
        if abs(value) <= 1.0:
            return "0_1"
        return "raw"

    @staticmethod
    def _infer_claim_type(unit: str, direction: str, snippet: str) -> str:
        snippet_l = snippet.lower()
        if "error reduction" in snippet_l:
            return "relative_error_reduction"
        if "from" in snippet_l and "to" in snippet_l:
            return "from_to"
        if unit == "pp":
            return "pp_change"
        if direction in {"decrease"} or "decrease" in snippet_l or "reduce" in snippet_l:
            return "relative_decrease"
        if "reach" in snippet_l or "achieve" in snippet_l:
            return "direct_value"
        return "relative_improvement"
