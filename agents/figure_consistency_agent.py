from __future__ import annotations

import logging
from typing import Dict, List

from agents.base import BaseAgent
from core.logging_utils import get_logger, log_event
from core.schemas import AppState, Issue, Location
from tools.figure_utils import (
    FigureClaim,
    FigureFact,
    extract_figure_claims,
    figure_claim_to_claim_ir,
    normalize_figure_facts_to_evidence_datums,
)
from tools.numeric_verification_kernel import NumericVerificationKernel, VerificationResult
from tools.providers import VLMProvider, build_vlm_provider


class FigureConsistencyAgent(BaseAgent):
    name = "figure_consistency"

    def __init__(self, provider_name: str = "none") -> None:
        self.provider: VLMProvider = build_vlm_provider(provider_name)
        self.logger = get_logger(__name__)
        self.kernel = NumericVerificationKernel()

    def run(self, state: AppState) -> List[Issue]:
        text = state.current_text
        claims = extract_figure_claims(text)

        if not claims and not state.uploaded_figures:
            return []

        if not self.provider.enabled():
            return [
                Issue(
                    id="figure_check_skipped_provider_disabled",
                    type="figure_check_skipped",
                    severity="low",
                    status="skipped",
                    location=Location(
                        sentence_index=claims[0].sentence_index if claims else -1,
                        char_span=claims[0].char_span if claims else (0, 0),
                        snippet=claims[0].snippet if claims else "",
                    ),
                    evidence={
                        "reason": "vlm_provider_disabled",
                        "evidence_type": "unknown",
                        "confidence": 0.0,
                        "provider": state.config.vlm_provider_name,
                    },
                    message=(
                        "Figure consistency check skipped because no VLM provider is configured. "
                        "Configure multimodal_provider, multimodal_model, and api_key in "
                        "model_config.toml, or use environment variables to enable it."
                    ),
                    provenance={"source": "system_config"},
                )
            ]

        facts = self._collect_facts(state)
        evidence_by_figure = self._build_figure_evidence(facts)
        issues: List[Issue] = []
        traces: List[Dict[str, object]] = []
        records: List[Dict[str, object]] = []

        for idx, claim in enumerate(claims):
            evidence = list(evidence_by_figure.get(str(claim.figure_id), []))
            if not evidence:
                issues.append(self._make_uncertain_issue(claim, idx, "No verifiable numeric fact extracted from the referenced figure."))
                continue

            claim_ir = figure_claim_to_claim_ir(
                claim,
                claim_id=f"figure_claim_{claim.figure_id}_{idx}",
                abs_tolerance=state.config.figure_abs_tolerance,
                rel_tolerance=max(state.config.table_rel_tolerance, 0.01),
            )
            result = self.kernel.verify_claim(claim_ir, evidence)
            traces.append(result.trace.to_dict())
            record = result.to_eval_record()
            record["figure_id"] = claim.figure_id
            records.append(record)

            low_conf, weak_evidence = self._figure_evidence_quality_flags(evidence, state.config.figure_confidence_threshold)
            force_uncertain = low_conf or weak_evidence

            if result.verdict == "supported" and not force_uncertain:
                continue
            if result.verdict == "contradicted" and not force_uncertain:
                issues.append(self._make_mismatch_issue(claim, idx, result))
                continue

            uncertain_message = "Figure evidence confidence is insufficient; abstaining from hard mismatch detection."
            if result.verdict == "unsupported":
                uncertain_message = "Figure claim cannot be deterministically verified from extracted evidence."
            issues.append(
                self._make_uncertain_issue(
                    claim,
                    idx,
                    uncertain_message,
                    result=result,
                    force_uncertain=force_uncertain,
                )
            )

        metadata = state.grounded_context.metadata if isinstance(state.grounded_context.metadata, dict) else {}
        previous_traces = list(metadata.get("figure_verification_traces", []))
        previous_records = list(metadata.get("figure_verification_records", []))
        metadata["figure_verification_traces"] = (previous_traces + traces)[-160:]
        metadata["figure_verification_records"] = (previous_records + records)[-320:]
        metadata["figure_kernel_version"] = self.kernel.version
        metadata["figure_claim_count"] = len(claims)
        metadata["figure_evidence_count"] = sum(len(v) for v in evidence_by_figure.values())
        state.grounded_context.metadata = metadata
        return issues

    def _build_figure_evidence(self, facts: List[FigureFact]) -> Dict[str, List]:
        grouped: Dict[str, List[FigureFact]] = {}
        for fact in facts:
            grouped.setdefault(str(fact.figure_id), []).append(fact)
        output: Dict[str, List] = {}
        for figure_id, figure_facts in grouped.items():
            source_ref = str(figure_facts[0].extra.get("source_ref", f"figure_{figure_id}"))
            output[figure_id] = normalize_figure_facts_to_evidence_datums(figure_facts, source_ref=source_ref)
        return output

    def _collect_facts(self, state: AppState) -> List[FigureFact]:
        facts: List[FigureFact] = []
        for fig in state.uploaded_figures:
            meta = dict(fig.meta)
            meta["request_id"] = state.request_id
            source_ref = str(fig.meta.get("name", "figure"))
            try:
                extracted = self.provider.extract_numeric_facts(fig.content, meta)
                for fact in extracted:
                    fact.extra["source_ref"] = source_ref
                    if not fact.figure_id and meta.get("figure_id"):
                        fact.figure_id = str(meta["figure_id"])
                facts.extend(extracted)
            except Exception as exc:
                log_event(
                    self.logger,
                    logging.WARNING,
                    "figure_fact_extraction_failed",
                    request_id=state.request_id,
                    figure_name=fig.meta.get("name", ""),
                    error=str(exc),
                )
        return facts

    @staticmethod
    def _figure_evidence_quality_flags(evidence: List, threshold: float) -> tuple[bool, bool]:
        confidences = [float(d.meta.get("confidence", 0.0)) for d in evidence]
        best_conf = max(confidences) if confidences else 0.0
        evidence_types = {str(d.meta.get("evidence_type", "unknown")) for d in evidence}
        weak_types = {"axis_estimate", "legend_only", "unknown"}
        low_conf = best_conf < threshold
        weak_evidence = bool(evidence_types) and evidence_types.issubset(weak_types)
        return low_conf, weak_evidence

    @staticmethod
    def _make_mismatch_issue(claim: FigureClaim, idx: int, result: VerificationResult) -> Issue:
        return Issue(
            id=f"figure_mismatch_{idx}",
            type="text_figure_mismatch",
            severity="high",
            status="detected",
            location=Location(
                sentence_index=claim.sentence_index,
                char_span=claim.char_span,
                snippet=claim.snippet,
            ),
            evidence={
                "figure_id": claim.figure_id,
                "claim_type": claim.claim_type,
                "claimed": claim.claimed_value,
                "expected": result.expected_value,
                "abs_diff": result.abs_diff,
                "rel_diff": result.rel_diff,
                "verdict": result.verdict,
                "reason_codes": result.reason_codes,
                "formula": result.formula,
                "trace": result.trace.to_dict(),
            },
            message=(
                f"Claimed value {claim.claimed_value:.3f} conflicts with deterministic figure evidence "
                f"{(result.expected_value if result.expected_value is not None else 0.0):.3f}."
            ),
            provenance={
                "source": "numeric_verification_kernel",
                "kernel_version": result.trace.kernel_version,
                "trace_id": result.trace.trace_id,
            },
        )

    @staticmethod
    def _make_uncertain_issue(
        claim: FigureClaim,
        idx: int,
        message: str,
        *,
        result: VerificationResult | None = None,
        force_uncertain: bool = False,
    ) -> Issue:
        evidence = {
            "figure_id": claim.figure_id,
            "claim_type": claim.claim_type,
            "claimed": claim.claimed_value,
            "force_uncertain": force_uncertain,
        }
        provenance = {"source": "vlm"}
        if result is not None:
            evidence.update(
                {
                    "expected": result.expected_value,
                    "abs_diff": result.abs_diff,
                    "rel_diff": result.rel_diff,
                    "verdict": result.verdict,
                    "reason_codes": result.reason_codes,
                    "formula": result.formula,
                    "trace": result.trace.to_dict(),
                }
            )
            provenance = {
                "source": "numeric_verification_kernel",
                "kernel_version": result.trace.kernel_version,
                "trace_id": result.trace.trace_id,
            }
        return Issue(
            id=f"figure_uncertain_{idx}",
            type="figure_check_uncertain",
            severity="medium",
            status="uncertain",
            location=Location(
                sentence_index=claim.sentence_index,
                char_span=claim.char_span,
                snippet=claim.snippet,
            ),
            evidence=evidence,
            message=message,
            provenance=provenance,
        )
