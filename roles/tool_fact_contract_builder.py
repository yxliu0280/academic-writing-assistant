from __future__ import annotations

from typing import List

from core.schemas import RoleToolEntry, RoleToolSummary, RoleType, ToolFact


class ToolFactContractBuilder:
    """Build role-facing tool summaries for stable tool-grounded chat behavior."""

    def build(
        self,
        *,
        role: RoleType,
        tool_facts: List[ToolFact],
        source_context: str = "",
    ) -> RoleToolSummary:
        entries = [self._entry_for_fact(role, fact) for fact in tool_facts]
        return RoleToolSummary(
            role=role,
            summary_style=self._summary_style(role),
            source_context=source_context or "tool_facts",
            fact_count=len(entries),
            entries=entries,
            next_step=self._next_step(role, entries),
        )

    @staticmethod
    def _summary_style(role: RoleType) -> str:
        if role == "Reviewer":
            return "review_evidence"
        if role == "Advisor":
            return "advice_guidance"
        return "edit_analysis"

    def _entry_for_fact(self, role: RoleType, fact: ToolFact) -> RoleToolEntry:
        data = dict(fact.data or {})
        confidence_value = self._confidence_value(data.get("confidence"))
        confidence = self._confidence_label(confidence_value)
        factual_statement = self._factual_statement(fact)
        evidence_status = self._evidence_status(fact, confidence_value)
        interpretation_risk = self._interpretation_risk(fact, confidence_value)
        guidance = ""
        edit_target = ""
        patch_suitable = False

        if role == "Advisor":
            guidance = self._guidance(fact, confidence_value)
        elif role == "Editor":
            edit_target = self._edit_target(fact, confidence_value)
            patch_suitable = self._patch_suitable(fact, confidence_value)

        return RoleToolEntry(
            kind=str(fact.kind or ""),
            source=str(fact.source or ""),
            fact=factual_statement,
            confidence=confidence,
            evidence_status=evidence_status,
            interpretation_risk=interpretation_risk,
            guidance=guidance,
            edit_target=edit_target,
            patch_suitable=patch_suitable,
        )

    @staticmethod
    def _confidence_value(value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return -1.0
        return max(0.0, min(parsed, 1.0))

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence < 0:
            return "Confidence unknown."
        if confidence >= 0.8:
            return "High-confidence tool-derived evidence."
        if confidence >= 0.6:
            return "Moderate-confidence tool-derived evidence."
        return "Low-confidence tool-derived evidence."

    @staticmethod
    def _factual_statement(fact: ToolFact) -> str:
        data = dict(fact.data or {})
        if fact.kind == "image_numeric_fact":
            figure_id = str(data.get("figure_id") or "the figure").strip()
            value = data.get("value")
            evidence_type = str(data.get("evidence_type") or "visual observation").strip()
            if value not in (None, ""):
                return f"{figure_id} indicates {evidence_type} with extracted value {value}."
            return f"{figure_id} provides a {evidence_type} observation, but no stable numeric value was extracted."
        if fact.kind == "table_metric_fact":
            metric = str(data.get("metric") or "the table metric").strip()
            baseline = data.get("baseline")
            ours = data.get("ours")
            table_label = str(data.get("table_label") or "the table").strip()
            return f"{table_label} reports {metric} with baseline={baseline} and ours={ours}."
        if fact.kind == "table_resource":
            resource = str(data.get("table_label") or fact.source or "table resource").strip()
            return f"{resource} is available, but only coarse table evidence is currently extracted."
        if fact.kind == "image_resource":
            resource = str(data.get("path") or fact.source or "image resource").strip()
            return f"{resource} is available, but no specific visual fact has been extracted yet."
        return str(fact.summary or "").strip() or "Tool-derived evidence is available."

    @staticmethod
    def _evidence_status(fact: ToolFact, confidence: float) -> str:
        if fact.kind == "image_resource":
            return "The image resource exists, but the available evidence is still coarse."
        if fact.kind == "table_resource":
            return "The table resource exists, but the available evidence is still coarse."
        if fact.kind == "table_metric_fact":
            return "The available table evidence is structured enough for grounded quantitative wording."
        if confidence < 0:
            return "Tool-derived evidence is present, but its reliability is not quantified."
        if confidence >= 0.8:
            return "The available evidence is strong enough for careful grounded claims."
        if confidence >= 0.6:
            return "The evidence is usable, but interpretation should stay conservative."
        return "The evidence is weak and should not support a strong textual claim."

    @staticmethod
    def _interpretation_risk(fact: ToolFact, confidence: float) -> str:
        if fact.kind == "image_resource":
            return "Any detailed visual interpretation would currently be weakly supported."
        if fact.kind == "table_resource":
            return "Any detailed numeric interpretation would currently be weakly supported."
        if fact.kind == "table_metric_fact":
            return "Interpretation risk is moderate if the prose adds unsupported comparisons beyond the extracted table values."
        if confidence < 0:
            return "Interpretation risk is uncertain because the available evidence has no confidence estimate."
        if confidence >= 0.8:
            return "Interpretation risk is low if the text stays close to the extracted fact."
        if confidence >= 0.6:
            return "Interpretation risk is moderate; avoid overstating trends or unsupported numeric precision."
        return "Interpretation risk is high; the current evidence should be treated as tentative."

    @staticmethod
    def _guidance(fact: ToolFact, confidence: float) -> str:
        if fact.kind == "image_resource":
            return "State only that the referenced image exists, or ask a more specific visual question before revising the wording."
        if fact.kind == "table_resource":
            return "State only that the referenced table exists, or narrow the question before revising the wording."
        if fact.kind == "table_metric_fact":
            return "Describe the table values directly and avoid adding unsupported comparisons or stronger conclusions than the table shows."
        if confidence >= 0.8:
            return "Describe the extracted fact directly, and keep the wording tightly aligned with the evidence."
        if confidence >= 0.6:
            return "Add a caveat and avoid precise or causal wording unless the evidence clearly supports it."
        return "Use cautious wording, or avoid making a strong visual claim until the figure can be interpreted more reliably."

    @staticmethod
    def _edit_target(fact: ToolFact, confidence: float) -> str:
        if fact.kind == "image_resource":
            return "Stay in analysis mode until a more specific visual fact is available."
        if fact.kind == "table_resource":
            return "Stay in analysis mode until a more specific table fact is available."
        if fact.kind == "table_metric_fact":
            return "An edit can align the text more tightly with the extracted table values."
        if confidence >= 0.8:
            return "An edit can align the text more tightly with the extracted evidence."
        if confidence >= 0.6:
            return "A cautious wording adjustment may be possible, but the edit should preserve uncertainty."
        return "Do not patch the prose yet; first narrow the claim or obtain clearer visual support."

    @staticmethod
    def _patch_suitable(fact: ToolFact, confidence: float) -> bool:
        if fact.kind == "table_metric_fact":
            return True
        if fact.kind != "image_numeric_fact":
            return False
        if confidence < 0:
            return False
        return confidence >= 0.7

    @staticmethod
    def _next_step(role: RoleType, entries: List[RoleToolEntry]) -> str:
        if not entries:
            if role == "Reviewer":
                return "State that no stable tool-derived visual fact is available yet."
            if role == "Advisor":
                return "State that guidance must stay cautious because no stable tool-derived visual fact is available yet."
            return "Stay in analysis mode because no stable tool-derived edit target is available yet."
        if role == "Reviewer":
            return "Report the evidence status and interpretation risks only; do not rewrite."
        if role == "Advisor":
            return "Turn the available evidence into wording guidance and caveats without behaving like an editor."
        patchable = sum(1 for item in entries if item.patch_suitable)
        if patchable:
            return "Explain the edit target and, if asked, invite a patch later without applying anything now."
        return "Explain the edit target conservatively and stay in analysis mode unless stronger visual support becomes available."
