from __future__ import annotations

from enum import Enum
from typing import Dict


class PipelinePhase(str, Enum):
    INGEST = "ingest"
    TABLE_CHECK = "table_check"
    FIGURE_CHECK = "figure_check"
    CITATION_CHECK = "citation_check"
    TERMINOLOGY_CHECK = "terminology_check"
    WRITING_SUPPORT = "writing_support"
    MERGE = "merge"
    RENDER = "render"
    DONE = "done"


TRANSITIONS: Dict[PipelinePhase, PipelinePhase] = {
    PipelinePhase.INGEST: PipelinePhase.TABLE_CHECK,
    PipelinePhase.TABLE_CHECK: PipelinePhase.FIGURE_CHECK,
    PipelinePhase.FIGURE_CHECK: PipelinePhase.CITATION_CHECK,
    PipelinePhase.CITATION_CHECK: PipelinePhase.TERMINOLOGY_CHECK,
    PipelinePhase.TERMINOLOGY_CHECK: PipelinePhase.WRITING_SUPPORT,
    PipelinePhase.WRITING_SUPPORT: PipelinePhase.MERGE,
    PipelinePhase.MERGE: PipelinePhase.RENDER,
    PipelinePhase.RENDER: PipelinePhase.DONE,
}


def next_phase(phase: PipelinePhase) -> PipelinePhase:
    return TRANSITIONS[phase]
