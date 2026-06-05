from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

IssueType = Literal[
    "text_table_mismatch",
    "text_figure_mismatch",
    "citation_missing",
    "citation_suspected_missing",
    "citation_unverified_suggestion",
    "terminology_inconsistent",
    "figure_check_skipped",
    "figure_check_uncertain",
]
Severity = Literal["low", "medium", "high"]
IssueStatus = Literal["detected", "skipped", "uncertain"]
RoleType = Literal["Reviewer", "Advisor", "Editor"]
PatchStatus = Literal["none", "pending", "accepted", "rejected"]
PatchLifecycleStage = Literal[
    "idle",
    "awaiting_preview_consent",
    "awaiting_apply_consent",
    "negotiation",
    "applied",
    "rejected",
]
RequestedCheckName = Literal["table", "figure", "citation", "terminology"]
RouterActionType = Literal[
    "send_message",
    "run_checks",
    "confirm_patch_prepare",
    "confirm_patch_apply",
    "undo",
]
RouterDecisionKind = Literal[
    "CHAT",
    "RUN_CHECKS_UI",
    "RUN_CHECKS_NL",
    "APPLY_PATCH",
    "UNDO",
    "BLOCKED_PRECONDITION",
    "SYSTEM_EXCEPTION",
    "PATCH_NEGOTIATION",
    "PATCH_PREVIEW_CONSENT",
    "PATCH_APPLY_CONSENT",
    "ADVISOR_SUGGESTION_CONSENT",
]
ScopeType = Literal["selected_span", "current_file", "full_manuscript"]
BlockedReason = Literal[
    "no_role",
    "no_document",
    "no_confirmed_selection",
    "busy",
    "invalid_scope",
    "module_inapplicable",
    "missing_bib",
]
SystemExceptionCode = Literal[
    "target_missing",
    "figure_missing",
    "image_missing",
    "table_missing",
    "unsupported_request",
    "tool_unavailable",
    "invalid_scope",
]
RolePayloadMode = Literal[
    "grounded_chat",
    "check_response",
    "system_exception",
]
EditorResponsePhase = Literal["n/a", "analysis_only", "invite_patch", "awaiting_apply"]
RoleInputSourceKind = Literal["checker", "tool", "system_exception"]


@dataclass
class Location:
    sentence_index: int
    char_span: Tuple[int, int]
    snippet: str
    line_range: Optional[Tuple[int, int]] = None


@dataclass
class Issue:
    id: str
    type: IssueType
    severity: Severity
    status: IssueStatus
    location: Location
    evidence: Dict[str, Any]
    message: str
    suggested_fix: Optional[str] = None
    patch: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UploadedBib:
    content: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadedTable:
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UploadedFigure:
    content: bytes
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionContext:
    """Inspired by InfiAgent: selection-scoped state for context minimization."""

    start: int = 0
    end: int = 0
    snippet: str = ""
    sentence_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatTurn:
    role: str
    content: str
    request_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundedContextBundle:
    """Inspired by InfiAgent: file-centric context externalization bundle."""

    selected_text: str = ""
    surrounding_text: str = ""
    table_contexts: List[str] = field(default_factory=list)
    figure_contexts: List[str] = field(default_factory=list)
    citation_contexts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatchProposal:
    status: PatchStatus = "none"
    stage: PatchLifecycleStage = "idle"
    patch_diff: str = ""
    operations: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    edit_ratio: float = 0.0
    source: str = ""
    summary: str = ""


@dataclass
class NegotiationState:
    active: bool = False
    mode: str = ""
    round_index: int = 0
    max_rounds: int = 3
    latest_user_feedback: str = ""
    latest_agent_message: str = ""


@dataclass
class AppConfig:
    table_abs_tolerance: float = 0.2
    table_rel_tolerance: float = 0.02
    figure_abs_tolerance: float = 1.0
    figure_confidence_threshold: float = 0.75
    max_patch_edit_ratio: float = 0.2
    max_patch_sentence_changes: int = 3
    style_protection_edit_ratio: float = 0.30
    style_protection_sentence_changes: int = 999
    negotiation_max_rounds: int = 3
    routing_policy: str = "deterministic"
    text_provider_name: str = "none"
    vlm_provider_name: str = "none"
    providers_enabled: Dict[str, bool] = field(
        default_factory=lambda: {"text_llm": False, "vlm": False}
    )


@dataclass
class AppState:
    current_text: str
    role: RoleType = "Reviewer"
    uploaded_bib: UploadedBib = field(default_factory=UploadedBib)
    uploaded_tables: List[UploadedTable] = field(default_factory=list)
    uploaded_figures: List[UploadedFigure] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    report_md: str = ""
    patch_preview: str = ""
    history: List[str] = field(default_factory=list)
    config: AppConfig = field(default_factory=AppConfig)
    request_id: str = ""

    # Phase-1 state extension for future iterative UX pipeline
    active_selection: Optional[SelectionContext] = None
    chat_history: List[ChatTurn] = field(default_factory=list)
    grounded_context: GroundedContextBundle = field(default_factory=GroundedContextBundle)
    pending_patch: PatchProposal = field(default_factory=PatchProposal)
    negotiation_state: NegotiationState = field(default_factory=NegotiationState)
    state_doc_id: str = "default"


@dataclass
class RenderedIssue:
    issue: Issue
    view_mode: RoleType
    suggested_fix: Optional[str] = None
    patch: Optional[Dict[str, Any]] = None


@dataclass
class ScopePayload:
    type: ScopeType
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockedResponse:
    reason: BlockedReason
    message: str
    role: str = ""
    required_scope: str = "confirmed_selection"


@dataclass
class SystemExceptionPayload:
    code: SystemExceptionCode
    message: str
    target: str = ""
    target_type: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    blocking: bool = True


@dataclass
class ToolFact:
    kind: str
    source: str
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleCheckEntry:
    issue_id: str
    issue_type: IssueType
    severity: Severity
    status: IssueStatus
    location: str
    snippet: str
    problem: str
    risk: str = ""
    impact: str = ""
    guidance: str = ""
    patchable: bool = False


@dataclass
class RoleCheckSummary:
    role: RoleType
    summary_style: Literal["review", "advice", "edit_analysis"]
    issue_count: int = 0
    requested_checks: List[RequestedCheckName] = field(default_factory=list)
    entries: List[RoleCheckEntry] = field(default_factory=list)
    next_step: str = ""


@dataclass
class RoleToolEntry:
    kind: str
    source: str
    fact: str
    confidence: str = ""
    evidence_status: str = ""
    interpretation_risk: str = ""
    guidance: str = ""
    edit_target: str = ""
    patch_suitable: bool = False


@dataclass
class RoleToolSummary:
    role: RoleType
    summary_style: Literal["review_evidence", "advice_guidance", "edit_analysis"]
    source_context: str = ""
    fact_count: int = 0
    entries: List[RoleToolEntry] = field(default_factory=list)
    next_step: str = ""


@dataclass
class RoleInputEntry:
    source_kind: RoleInputSourceKind
    category: str
    source: str = ""
    observation: str = ""
    location: str = ""
    confidence: str = ""
    risk: str = ""
    impact: str = ""
    guidance: str = ""
    edit_target: str = ""
    patch_suitable: bool = False


@dataclass
class RoleInputSummary:
    role: RoleType
    mode: RolePayloadMode
    source_kind: RoleInputSourceKind
    summary_style: Literal["review", "advice", "edit_analysis"]
    source_context: str = ""
    entry_count: int = 0
    entries: List[RoleInputEntry] = field(default_factory=list)
    next_step: str = ""


@dataclass
class RoleResponsePolicy:
    role: RoleType
    mode: RolePayloadMode
    focus_points: List[str] = field(default_factory=list)
    framing_guidance: str = ""
    forbid_rewrite: bool = False
    forbid_patch_diff: bool = True
    forbid_applied_claims: bool = True
    allow_example: bool = False
    example_max_count: int = 0
    example_max_words: int = 0
    example_max_chars: int = 0
    editor_phase: EditorResponsePhase = "n/a"
    required_closing: str = ""


@dataclass
class CheckerResultPayload:
    requested_checks: List[RequestedCheckName] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    role_summary: Optional[RoleCheckSummary] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckExecutionArtifacts:
    issues: List[Issue] = field(default_factory=list)
    writing_support: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckPresentationArtifacts:
    rendered_issues: List[RenderedIssue] = field(default_factory=list)
    patch_diff: str = ""
    patch_operations: List[Dict[str, Any]] = field(default_factory=list)
    report_md: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolePayload:
    mode: RolePayloadMode
    role: str
    scope: ScopePayload
    user_request: str = ""
    checker_results: Optional[CheckerResultPayload] = None
    tool_facts: List[ToolFact] = field(default_factory=list)
    tool_summary: Optional[RoleToolSummary] = None
    system_exception: Optional[SystemExceptionPayload] = None
    role_input_summary: Optional[RoleInputSummary] = None
    response_policy: Optional[RoleResponsePolicy] = None
    memory: Dict[str, Any] = field(default_factory=dict)
    edit_stage: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterRequest:
    action: RouterActionType
    role: str
    user_text: str = ""
    document_text: str = ""
    scope: Optional[ScopePayload] = None
    ui_selected_checks: List[RequestedCheckName] = field(default_factory=list)
    ui_checks_explicit: bool = False
    workspace_resources: Dict[str, Any] = field(default_factory=dict)
    tool_availability: Dict[str, bool] = field(default_factory=dict)
    busy: bool = False
    busy_kind: str = ""
    memory: Dict[str, Any] = field(default_factory=dict)
    pending_patch_negotiation: bool = False
    pending_patch_preview_consent: bool = False
    pending_patch_apply_consent: bool = False
    pending_advisor_suggestion_consent: bool = False


@dataclass
class RouterDecision:
    kind: RouterDecisionKind
    requested_checks: List[RequestedCheckName] = field(default_factory=list)
    blocked_response: Optional[BlockedResponse] = None
    system_exception: Optional[SystemExceptionPayload] = None
    role_payload: Optional[RolePayload] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
