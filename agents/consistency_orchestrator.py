from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from agents.citation_agent import CitationAgent
from agents.consistency_orchestrator_compat import ConsistencyOrchestratorCompatAdapter
from agents.figure_consistency_agent import FigureConsistencyAgent
from agents.table_consistency_agent import TableConsistencyAgent
from agents.terminology_agent import TerminologyAgent
from agents.writing_support_agent import WritingSupportAgent
from core.logging_utils import get_logger, log_event
from core.schemas import AppState, ChatTurn, CheckExecutionArtifacts, Issue, RenderedIssue
from core.state_machine import PipelinePhase, next_phase
from tools.patch_utils import ReplaceOperation


class ConsistencyOrchestrator:
    """Checks execution submodule invoked by Router.

    This class is not the system-level router. Router decides when checks should run;
    ConsistencyOrchestrator only orchestrates the check pipeline and produces check-side artifacts
    such as issues, report text, and optional editor patch candidates.
    """

    def __init__(self, state: AppState) -> None:
        self.table_agent = TableConsistencyAgent(state.config.text_provider_name)
        self.figure_agent = FigureConsistencyAgent(state.config.vlm_provider_name)
        self.citation_agent = CitationAgent()
        self.terminology_agent = TerminologyAgent()
        self.writing_support_agent = WritingSupportAgent(state.config.text_provider_name)
        self._compat = ConsistencyOrchestratorCompatAdapter(state.config.text_provider_name)
        self.logger = get_logger(__name__)

        self._last_execution_artifacts = CheckExecutionArtifacts()

    def analyze(
        self,
        state: AppState,
        enabled_checks: Optional[Set[str]] = None,
    ) -> Tuple[AppState, List[RenderedIssue], List[Dict[str, str]]]:
        """Backward-compatible wrapper around `run_checks()`.

        New mainline code should call `run_checks()` and consume
        `CheckExecutionArtifacts` directly. This wrapper only exists so old
        direct callers of `analyze()` do not break immediately.
        """
        state, execution_artifacts = self.run_checks(state, enabled_checks=enabled_checks)
        rendered, writing_support = self._compat.build_legacy_analyze_output(state, execution_artifacts)
        return state, rendered, writing_support

    def run_checks(
        self,
        state: AppState,
        enabled_checks: Optional[Set[str]] = None,
    ) -> Tuple[AppState, CheckExecutionArtifacts]:
        """Primary checks entrypoint used by Router/app mainline.

        This method only runs the checks pipeline and returns normalized execution
        artifacts. It intentionally does not produce role-facing rendered output.
        """
        return self._run_checks_impl(state, enabled_checks=enabled_checks)

    def _run_checks_impl(
        self,
        state: AppState,
        enabled_checks: Optional[Set[str]] = None,
    ) -> Tuple[AppState, CheckExecutionArtifacts]:
        state.request_id = uuid4().hex[:12]
        enabled = enabled_checks or {"table", "figure", "citation", "terminology"}
        # Compatibility caches are not part of the new mainline semantics.
        self._compat.clear_patch_cache()
        log_event(
            self.logger,
            logging.INFO,
            "analyze_start",
            request_id=state.request_id,
            role=state.role,
            text_length=len(state.current_text),
            table_count=len(state.uploaded_tables),
            figure_count=len(state.uploaded_figures),
            has_bib=bool(state.uploaded_bib.content),
            text_provider=state.config.text_provider_name,
            vlm_provider=state.config.vlm_provider_name,
            enabled_checks=sorted(list(enabled)),
        )
        phase = PipelinePhase.INGEST
        table_issues: List[Issue] = []
        figure_issues: List[Issue] = []
        citation_issues: List[Issue] = []
        terminology_issues: List[Issue] = []

        writing_support = []

        while phase != PipelinePhase.DONE:
            if phase == PipelinePhase.TABLE_CHECK:
                if "table" in enabled:
                    table_issues = self.table_agent.run(state)
                else:
                    table_issues = []
                log_event(
                    self.logger,
                    logging.DEBUG,
                    "phase_complete",
                    request_id=state.request_id,
                    phase=str(phase),
                    issue_count=len(table_issues),
                    extractor=getattr(self.table_agent, "last_extractor", "unknown"),
                )
            elif phase == PipelinePhase.FIGURE_CHECK:
                if "figure" in enabled:
                    figure_issues = self.figure_agent.run(state)
                else:
                    figure_issues = []
                log_event(
                    self.logger,
                    logging.DEBUG,
                    "phase_complete",
                    request_id=state.request_id,
                    phase=str(phase),
                    issue_count=len(figure_issues),
                )
            elif phase == PipelinePhase.CITATION_CHECK:
                if "citation" in enabled:
                    citation_issues = self.citation_agent.run(state)
                else:
                    citation_issues = []
                log_event(
                    self.logger,
                    logging.DEBUG,
                    "phase_complete",
                    request_id=state.request_id,
                    phase=str(phase),
                    issue_count=len(citation_issues),
                )
            elif phase == PipelinePhase.TERMINOLOGY_CHECK:
                if "terminology" in enabled:
                    terminology_issues = self.terminology_agent.run(state)
                else:
                    terminology_issues = []
                log_event(
                    self.logger,
                    logging.DEBUG,
                    "phase_complete",
                    request_id=state.request_id,
                    phase=str(phase),
                    issue_count=len(terminology_issues),
                )
            elif phase == PipelinePhase.WRITING_SUPPORT:
                writing_support = self.writing_support_agent.diagnose(state)
            elif phase == PipelinePhase.MERGE:
                merged = self._merge_and_sort(
                    table_issues + figure_issues + citation_issues + terminology_issues
                )
                scoped = self._scope_issues_to_confirmed_selection(state, merged)
                if len(scoped) != len(merged):
                    log_event(
                        self.logger,
                        logging.INFO,
                        "issues_scoped_to_selection",
                        request_id=state.request_id,
                        before_count=len(merged),
                        after_count=len(scoped),
                        selection_start=state.active_selection.start if state.active_selection else -1,
                        selection_end=state.active_selection.end if state.active_selection else -1,
                    )
                state.issues = scoped
            elif phase == PipelinePhase.RENDER:
                self._last_execution_artifacts = CheckExecutionArtifacts(
                    issues=list(state.issues),
                    writing_support=list(writing_support),
                    metadata={"request_id": state.request_id, "role": state.role},
                )
                log_event(
                    self.logger,
                    logging.INFO,
                    "checks_execution_complete",
                    request_id=state.request_id,
                    role=state.role,
                    issue_count=len(self._last_execution_artifacts.issues),
                    writing_support_count=len(self._last_execution_artifacts.writing_support),
                    negotiation_active=state.negotiation_state.active,
                )
            phase = next_phase(phase)

        log_event(
            self.logger,
            logging.INFO,
            "checks_done",
            request_id=state.request_id,
            total_issues=len(state.issues),
            role=state.role,
        )
        return state, self.latest_execution_artifacts()

    def apply_patch(self, state: AppState) -> Tuple[AppState, str]:
        """Compatibility delegate.

        Patch lifecycle now belongs to PatchManager and should not be added back
        into the checks mainline.
        """
        return self._compat.apply_patch(state)

    def undo(self, state: AppState) -> Tuple[AppState, str]:
        if not state.history:
            return state, "Undo stack is empty."
        state.current_text = state.history.pop()
        return state, "Undo successful."

    def negotiate_patch(self, state: AppState, feedback: str) -> Tuple[AppState, str]:
        """Compatibility delegate for legacy negotiation callers."""
        return self._compat.negotiate_patch(state, feedback)

    def propose_writing_patch(self, state: AppState, instruction: str) -> Tuple[AppState, str]:
        """Compatibility delegate for legacy editor-writing callers."""
        return self._compat.propose_writing_patch(state, instruction)

    def recent_chat_turns(self, state: AppState, max_turns: int = 8) -> List[ChatTurn]:
        return self._compat.recent_chat_turns(state, max_turns=max_turns)

    def latest_execution_artifacts(self) -> CheckExecutionArtifacts:
        """Compatibility accessor for the latest normalized execution artifacts.

        New mainline code should prefer the explicit `run_checks()` return value.
        """
        return CheckExecutionArtifacts(
            issues=list(self._last_execution_artifacts.issues),
            writing_support=list(self._last_execution_artifacts.writing_support),
            metadata=dict(self._last_execution_artifacts.metadata),
        )

    def latest_patch_diff(self) -> str:
        """Compatibility accessor used by older patch-call sites."""
        return self._compat.latest_patch_diff()

    def latest_patch_operations(self) -> List[ReplaceOperation]:
        """Compatibility accessor used by older patch-call sites."""
        return self._compat.latest_patch_operations()

    @staticmethod
    def _merge_and_sort(issues: List[Issue]) -> List[Issue]:
        dedup: Dict[tuple, Issue] = {}
        for issue in issues:
            key = (
                issue.type,
                issue.location.char_span,
                issue.message,
                issue.status,
            )
            dedup[key] = issue

        order = {"high": 0, "medium": 1, "low": 2}
        merged = sorted(
            dedup.values(),
            key=lambda i: (
                order.get(i.severity, 9),
                i.location.char_span[0],
                i.id,
            ),
        )
        return merged

    @staticmethod
    def _scope_issues_to_confirmed_selection(state: AppState, issues: List[Issue]) -> List[Issue]:
        selection = state.active_selection
        if not selection:
            return issues
        if not bool(selection.metadata.get("confirmed", False)):
            return issues
        if selection.end <= selection.start:
            return issues

        start = int(selection.start)
        end = int(selection.end)
        scoped: List[Issue] = []
        for issue in issues:
            issue_start, issue_end = issue.location.char_span
            s = int(issue_start)
            e = int(issue_end)
            if e > s and not (e <= start or s >= end):
                scoped.append(issue)
        return scoped


# Backward-compatible alias for older tests, scripts, and docs.
ControllerAgent = ConsistencyOrchestrator
