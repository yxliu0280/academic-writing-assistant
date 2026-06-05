from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from core.logging_utils import get_logger, log_event
from core.patch_manager import PatchManager
from core.schemas import AppState, CheckExecutionArtifacts, RenderedIssue
from roles.check_artifact_builder import CheckArtifactBuilder
from tools.patch_utils import ReplaceOperation


class ConsistencyOrchestratorCompatAdapter:
    """Compatibility-only adapter around legacy controller-style surfaces.

    New mainline code should not depend on this adapter.
    """

    def __init__(self, provider_name: str = "none") -> None:
        self._patch_manager = PatchManager(provider_name)
        self._last_rendered: List[RenderedIssue] = []
        self.logger = get_logger(__name__)

    def build_legacy_analyze_output(
        self,
        state: AppState,
        execution_artifacts: CheckExecutionArtifacts,
    ) -> Tuple[List[RenderedIssue], List[Dict[str, str]]]:
        compatibility = CheckArtifactBuilder(state.config.text_provider_name).build(
            state,
            execution_artifacts,
        )
        self._last_rendered = list(compatibility.rendered_issues)
        self._patch_manager.sync_compat_patch(
            str(compatibility.patch_diff or ""),
            compatibility.patch_operations,
        )
        log_event(
            self.logger,
            logging.INFO,
            "analyze_compat_complete",
            request_id=state.request_id,
            role=state.role,
            rendered_issue_count=len(compatibility.rendered_issues),
            patch_ops_count=len(self._patch_manager.latest_patch_operations()),
            negotiation_active=state.negotiation_state.active,
        )
        return list(self._last_rendered), list(execution_artifacts.writing_support)

    def clear_patch_cache(self) -> None:
        self._patch_manager.clear_compat_patch()
        self._last_rendered = []

    def apply_patch(self, state: AppState) -> Tuple[AppState, str]:
        return self._patch_manager.apply_patch(state)

    def negotiate_patch(self, state: AppState, feedback: str) -> Tuple[AppState, str]:
        return self._patch_manager.negotiate_patch(state, feedback)

    def propose_writing_patch(self, state: AppState, instruction: str) -> Tuple[AppState, str]:
        return self._patch_manager.propose_writing_patch(state, instruction)

    def recent_chat_turns(self, state: AppState, max_turns: int = 8):
        return self._patch_manager.recent_chat_turns(state, max_turns=max_turns)

    def latest_patch_diff(self) -> str:
        return self._patch_manager.latest_patch_diff()

    def latest_patch_operations(self) -> List[ReplaceOperation]:
        return self._patch_manager.latest_patch_operations()
