from __future__ import annotations

import copy
import csv
import base64
import html
import io
import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as st_components

from agents.consistency_orchestrator import ConsistencyOrchestrator
from agents.role_chat_agent import RoleChatAgent
from agents.router import RouterV1
from agents.terminology_agent import TerminologyAgent
from components import chat_composer, interactive_editor, pdf_viewer, viewport_probe
from components.files_dnd_bridge import files_dnd_bridge as render_files_dnd_bridge
from core import ContextWindowBuilder, FileCentricStateManager
from core.config import load_config_from_env
from core.patch_manager import PatchManager
from core.schemas import (
    AppState,
    BlockedResponse,
    CheckExecutionArtifacts,
    CheckerResultPayload,
    ChatTurn,
    RolePayload,
    RouterDecision,
    RouterRequest,
    SelectionContext,
    ScopePayload,
    SystemExceptionPayload,
    ToolFact,
    UploadedBib,
    UploadedFigure,
    UploadedTable,
)
from roles.check_artifact_builder import CheckArtifactBuilder
from roles.check_response_contract_builder import CheckResponseContractBuilder
from roles.role_input_summary_builder import RoleInputSummaryBuilder
from roles.response_policy_builder import RoleResponsePolicyBuilder
from roles.tool_fact_contract_builder import ToolFactContractBuilder
from tools.image_qa_tool import ImageQATool
from tools.latex_table_normalizer import latex_table_block_to_csv
from tools.table_qa_tool import TableQATool
from tools.bib_utils import extract_cite_keys, parse_bibtex_keys
from tools.latex_utils import detect_latex_compiler
from tools.latex_ref_resolver import LatexRefResolver
from tools.providers import build_text_provider, build_vlm_provider

ROLE_AVATAR_FALLBACK = {
    "Reviewer": "🔎",
    "Advisor": "🧭",
    "Editor": "🛠️",
}

ROLE_CHAT_AVATAR_FILES = {
    "Reviewer": "assets/avatars/reviewer_robot.svg",
    "Advisor": "assets/avatars/advisor_robot.svg",
    "Editor": "assets/avatars/editor_robot.svg",
}

WORKSPACE_UI_KEYS = [
    "selection_text",
    "selection_start",
    "selection_end",
    "pending_selection_text",
    "pending_selection_start",
    "pending_selection_end",
    "suppress_next_selection_payload",
    "last_component_action_fingerprint",
    "confirmed_selection_text",
    "confirmed_selection_start",
    "confirmed_selection_end",
    "active_selection",
    "issues_json",
    "last_grounding",
    "editor_waiting_patch_consent",
    "editor_waiting_apply_consent",
    "advisor_waiting_suggestion_consent",
    "selected_checks",
    "selected_checks_explicit",
    "project_root",
    "composer_text",
    "composer_triggered",
    "composer_tools_open",
    "pdf_compiled_text_hash",
    "pdf_compiled_base64",
    "pdf_compiled_bytes",
    "pdf_compile_ok",
    "pdf_compile_log",
    "pdf_last_export_path",
    "pdf_compiled_pdf_path",
    "pdf_compiled_synctex_path",
    "pdf_compiled_tex_path",
    "pdf_compile_errors",
    "pdf_preview_png_hash",
    "pdf_preview_png_bytes",
    "compile_in_progress",
]

ROLE_ALIASES = {
    "reviewer": "Reviewer",
    "advisor": "Advisor",
    "editor": "Editor",
}

LOTTIE_ROLE_FILES = {
    "Reviewer": Path("assets/lottie/reviewer.json"),
    "Advisor": Path("assets/lottie/advisor.json"),
    "Editor": Path("assets/lottie/editor.json"),
}

COMPOSER_CLOSED_HEIGHT = 72
COMPOSER_OPEN_HEIGHT = 262
PANEL_MIN_HEIGHT = 460
PANEL_MAX_HEIGHT = 1280
PANEL_VIEWPORT_RESERVED = 180
PANEL_COLUMNS_OVERHEAD = 0
PANEL_HEIGHT_BOOST = 170
PANEL_FALLBACK_HEIGHT = 640
# Lift editor bottom edge a bit while keeping right input bottom aligned.
EDITOR_BOTTOM_RAISE_PX = 10
# Compensate Streamlit stack spacing so the right composer never gets clipped at the drawer bottom.
CHAT_PANEL_BOTTOM_BUFFER = 28
# Push composer downward inside the fixed drawer while keeping history area aligned.
CHAT_COMPOSER_VERTICAL_SHIFT_PX = 50
# Global top-level vertical gap (header row vs editor/chat row).
TOP_LEVEL_ROW_GAP_REM = 0.02
# Pull the second row upward to visually tighten header-to-panel spacing.
HEADER_TO_PANEL_PULLUP_REM = 1.02
# Fine tune section title spacing above each panel.
PANEL_TITLE_MARGIN_TOP_REM = -0.16
PANEL_TITLE_MARGIN_BOTTOM_REM = 0.12
MAIN_WORKSPACE_LIFT_REM = 3.2
EDITOR_PDF_RATIO = [1, 1]
PDF_EXPORT_NAME_PREFIX = "compiled_preview"
CHAT_DRAWER_WIDTH_PX = 420
APP_ROOT_DIR = Path(__file__).resolve().parent
PAGE_ICON_PATH = APP_ROOT_DIR / "assets" / "icons" / "robot.png"
STATE_ROOT_DIR = APP_ROOT_DIR / ".state"
PROJECTS_ROOT_DIR = STATE_ROOT_DIR / "projects"
PDF_BUILDS_ROOT_DIR = STATE_ROOT_DIR / "pdf_builds"
DEMO_WORKSPACE_SOURCE_DIR = APP_ROOT_DIR / "test_assets" / "full_system_walkthrough"
DEMO_WORKSPACE_MAIN_TEX_PATH = DEMO_WORKSPACE_SOURCE_DIR / "sample_paper.tex"
DEMO_WORKSPACE_BIB_PATH = DEMO_WORKSPACE_SOURCE_DIR / "references.bib"
DEMO_WORKSPACE_PLOT_PATH = DEMO_WORKSPACE_SOURCE_DIR / "dummy_plot.png"
SYSTEM_STATE_FILENAMES = {"conversation.json", "memory.json", "metadata.json", "agent_chats.json"}
TEXT_FILE_SUFFIXES = {
    ".tex",
    ".bib",
}
RESOURCE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}
RESOURCE_PDF_SUFFIXES = {".pdf"}
RESOURCE_TEXT_PREVIEW_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".log",
    ".xml",
    ".html",
    ".htm",
    ".ini",
    ".cfg",
    ".toml",
    ".bib",
}
RESOURCE_TEXT_PREVIEW_MAX_BYTES = 320_000
DEFAULT_MAIN_TEX_TEMPLATE = """\\documentclass[12pt, a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[english]{babel}

\\title{Untitled Draft}
\\author{}
\\date{\\today}

\\begin{document}

\\maketitle

\\section{Introduction}
Write your manuscript here.

\\end{document}
"""
# Keep editor and preview at the same base height.
PDF_HEIGHT_COMPENSATION_PX = 0
# Increase PNG preview clarity so rendered PDF text does not look blurry.
PDF_PREVIEW_DPI = 300
CHECK_NAMES: Tuple[str, str, str, str] = ("table", "figure", "citation", "terminology")
CHECK_NAME_SET = set(CHECK_NAMES)
CHECK_ENABLE_DEFAULTS: Dict[str, bool] = {name: True for name in CHECK_NAMES}
TABLE_REF_PATTERN = re.compile(r"\\(?:ref|autoref|cref|Cref)\{[^}]*tab:[^}]*\}")
FIGURE_REF_PATTERN = re.compile(r"\\(?:ref|autoref|cref|Cref)\{[^}]*fig:[^}]*\}")
TABLE_NATURAL_REF_PATTERN = re.compile(
    r"\b(?:Table|Tables|Tab\.)\s*(?:~?\s*)?(?:[A-Za-z]?\d+[A-Za-z]?|[IVXLCM]+)\b"
)
FIGURE_NATURAL_REF_PATTERN = re.compile(
    r"\b(?:Figure|Figures|Fig\.)\s*(?:~?\s*)?(?:[A-Za-z]?\d+[A-Za-z]?|[IVXLCM]+)\b"
)
CITATION_CMD_PATTERN = re.compile(r"\\cite\w*\{[^}]+\}", flags=re.IGNORECASE)
REF_OR_CITE_MARKER_PATTERN = re.compile(
    r"\\(?:ref|autoref|cref|Cref)\{[^}]+\}|\\cite\w*\{[^}]+\}",
    flags=re.IGNORECASE,
)
TERMINOLOGY_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")
TERMINOLOGY_TITLE_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
LATEX_INCLUDE_PATTERN = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}", flags=re.IGNORECASE)
CHECK_REASON_MESSAGES: Dict[str, str] = {
    "no_confirmed_selection": "Please confirm the text span you want to analyze first.",
    "selection_has_no_table_ref": "Table consistency requires at least one table reference like Table 1 or \\ref{tab:...} in the confirmed selection.",
    "selection_has_no_figure_ref": "Figure consistency requires at least one figure reference like Figure 1 or \\ref{fig:...} in the confirmed selection.",
    "workspace_missing_figure_file": "Figure consistency requires at least one image file in the workspace file tree.",
    "selection_has_no_citation_cmd": "Citation consistency requires at least one citation command like \\cite{...} in the confirmed selection.",
    "selection_has_no_terminology_signal": "Terminology consistency requires at least one terminology signal (e.g., acronym or named term) in the confirmed selection.",
    "workspace_missing_bib_file": "Citation consistency requires a .bib file in the workspace file tree.",
}
TERMINOLOGY_AVAILABILITY_AGENT = TerminologyAgent()


def _render_fallback_companion(role: str, key: str, compact: bool = False) -> None:
    normalized = _normalize_role_name(role) or "Reviewer"
    theme_map = {
        "Reviewer": {"accent": "#2563eb", "surface": "#dbeafe", "tool": "lens"},
        "Advisor": {"accent": "#0891b2", "surface": "#cffafe", "tool": "bubble"},
        "Editor": {"accent": "#16a34a", "surface": "#dcfce7", "tool": "cursor"},
    }
    theme = theme_map[normalized]
    element_id = re.sub(r"[^a-zA-Z0-9_]", "_", f"companion_{normalized}_{key}")
    shell_scale = 0.46 if compact else 1.0
    frame_height = 56 if compact else 126
    st_components.html(
        f"""
<div id="{element_id}" class="companion-wrap">
  <div class="companion-shell">
    <div class="robot">
      <div class="halo"></div>
      <div class="head">
        <div class="eye"></div>
        <div class="eye"></div>
      </div>
      <div class="body"></div>
      <div class="tool tool-{theme["tool"]}"></div>
    </div>
    <div class="label">{normalized}</div>
  </div>
</div>
<style>
#{element_id}.companion-wrap {{
  height: {frame_height}px;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}}
#{element_id} .companion-shell {{
  width: 112px;
  margin: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  transform: scale({shell_scale});
  transform-origin: top center;
}}
#{element_id} .robot {{
  position: relative;
  width: 88px;
  height: 92px;
  animation: float_{element_id} 2.4s ease-in-out infinite;
}}
#{element_id} .halo {{
  position: absolute;
  left: 17px;
  bottom: 5px;
  width: 52px;
  height: 11px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.35);
}}
#{element_id} .head {{
  position: absolute;
  left: 21px;
  top: 10px;
  width: 46px;
  height: 30px;
  border-radius: 12px;
  border: 2px solid {theme["accent"]};
  background: #ffffff;
  display: flex;
  align-items: center;
  justify-content: space-evenly;
}}
#{element_id} .eye {{
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #0f172a;
  animation: blink_{element_id} 3.2s infinite;
}}
#{element_id} .body {{
  position: absolute;
  left: 15px;
  top: 38px;
  width: 58px;
  height: 39px;
  border-radius: 14px;
  border: 2px solid {theme["accent"]};
  background: {theme["surface"]};
}}
#{element_id} .tool {{
  position: absolute;
}}
#{element_id} .tool-lens {{
  right: 4px;
  top: 45px;
  width: 14px;
  height: 14px;
  border: 2px solid {theme["accent"]};
  border-radius: 999px;
  box-shadow: 8px 8px 0 -6px {theme["accent"]};
  animation: scan_{element_id} 2.1s ease-in-out infinite;
}}
#{element_id} .tool-bubble {{
  right: 1px;
  top: 9px;
  width: 16px;
  height: 12px;
  border-radius: 10px;
  background: {theme["accent"]};
  opacity: 0.3;
  animation: pulse_{element_id} 2.1s ease-in-out infinite;
}}
#{element_id} .tool-cursor {{
  right: 8px;
  top: 56px;
  width: 3px;
  height: 16px;
  border-radius: 2px;
  background: {theme["accent"]};
  animation: typing_{element_id} 0.8s step-end infinite;
}}
#{element_id} .label {{
  font-size: 11px;
  font-weight: 700;
  color: #0f172a;
}}
@keyframes float_{element_id} {{
  0%, 100% {{ transform: translateY(0); }}
  50% {{ transform: translateY(-5px); }}
}}
@keyframes blink_{element_id} {{
  0%, 46%, 100% {{ transform: scaleY(1); }}
  48% {{ transform: scaleY(0.2); }}
}}
@keyframes scan_{element_id} {{
  0%, 100% {{ transform: translateX(0); }}
  50% {{ transform: translateX(-7px); }}
}}
@keyframes pulse_{element_id} {{
  0%, 100% {{ transform: scale(0.85); opacity: 0.3; }}
  50% {{ transform: scale(1.08); opacity: 0.52; }}
}}
@keyframes typing_{element_id} {{
  0%, 50% {{ opacity: 1; }}
  51%, 100% {{ opacity: 0.25; }}
}}
</style>
""",
        height=frame_height,
        scrolling=False,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_doc_id(doc_id: str) -> str:
    normalized = str(doc_id or "").strip()
    if not normalized:
        raise ValueError("doc_id is empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", normalized):
        raise ValueError(f"Invalid doc_id: {normalized}")
    return normalized


def _safe_remove_tree(path: Path, base_dir: Path) -> None:
    try:
        base_resolved = base_dir.resolve()
        target_resolved = path.resolve()
    except OSError as exc:
        raise ValueError(f"Unable to resolve deletion path: {path}") from exc

    if target_resolved == base_resolved or base_resolved not in target_resolved.parents:
        raise ValueError(f"Unsafe delete target: {target_resolved}")
    if target_resolved.exists():
        shutil.rmtree(target_resolved)


def _default_shared_memory() -> Dict[str, Any]:
    return {
        "task_summary": "",
        "current_focus": "",
        "confirmed_issues": [],
        "confirmed_plans": [],
        "active_file": "main.tex",
        "last_compile_status": "",
    }


def _default_role_memory() -> Dict[str, Any]:
    return {
        "working_summary": "",
        "open_items": [],
        "private_notes": [],
        "published_conclusions": [],
    }


def _default_memory_payload(current_role: str = "") -> Dict[str, Any]:
    return {
        "shared_memory": _default_shared_memory(),
        "role_memories": {
            "Reviewer": _default_role_memory(),
            "Advisor": _default_role_memory(),
            "Editor": _default_role_memory(),
        },
        "meta": {
            "current_role": _normalize_role_name(current_role),
            "last_updated": _utc_now_iso(),
        },
    }


def _normalize_memory_payload(payload: Dict[str, Any] | None, current_role: str = "") -> Dict[str, Any]:
    merged = _default_memory_payload(current_role=current_role)
    if not isinstance(payload, dict):
        return merged

    shared = payload.get("shared_memory")
    if isinstance(shared, dict):
        for key in list(merged["shared_memory"].keys()):
            if key in shared:
                merged["shared_memory"][key] = shared[key]

    role_memories = payload.get("role_memories")
    if isinstance(role_memories, dict):
        for role_name in ["Reviewer", "Advisor", "Editor"]:
            role_payload = role_memories.get(role_name)
            if not isinstance(role_payload, dict):
                continue
            for key in list(merged["role_memories"][role_name].keys()):
                if key in role_payload:
                    merged["role_memories"][role_name][key] = role_payload[key]

    meta = payload.get("meta")
    if isinstance(meta, dict):
        merged["meta"]["current_role"] = _normalize_role_name(
            str(meta.get("current_role", current_role))
        )
        merged["meta"]["last_updated"] = str(meta.get("last_updated", _utc_now_iso()))
    return merged


def _default_metadata(doc_id: str, title: str = "Conversation 1", current_role: str = "") -> Dict[str, Any]:
    normalized_role = _normalize_role_name(current_role)
    return {
        "doc_id": _validate_doc_id(doc_id),
        "title": str(title or "Conversation 1").strip() or "Conversation 1",
        "created_at": _utc_now_iso(),
        "current_role": normalized_role,
        "project_root": str(Path.cwd()),
        "active_file_path": "main.tex",
        "updated_at": _utc_now_iso(),
    }


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(fallback)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(fallback)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_project_dir(doc_id: str) -> Path:
    normalized_id = _validate_doc_id(doc_id)
    return PROJECTS_ROOT_DIR / normalized_id


def _normalize_project_relative_path(path_text: str | None) -> str:
    raw = str(path_text or "").strip().replace("\\", "/")
    normalized_raw = raw.lstrip("/")
    if not normalized_raw:
        return "main.tex"
    parts = Path(normalized_raw).parts
    cleaned_parts: List[str] = []
    for part in parts:
        part_text = str(part).strip()
        if not part_text or part_text in {".", ".."}:
            raise ValueError(f"Invalid project-relative path: {raw}")
        cleaned_parts.append(part_text)
    if not cleaned_parts:
        return "main.tex"
    return "/".join(cleaned_parts)


def _resolve_project_path(doc_id: str, relative_path: str | None) -> Path:
    project_dir = ensure_project_dir(doc_id)
    normalized_relative = _normalize_project_relative_path(relative_path)
    target = project_dir / normalized_relative
    try:
        project_resolved = project_dir.resolve()
        target_resolved = target.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"Unable to resolve project path: {relative_path}") from exc
    if target_resolved != project_resolved and project_resolved not in target_resolved.parents:
        raise ValueError(f"Unsafe project path: {relative_path}")
    return target


def _is_internal_state_relative_path(relative_path: str | None) -> bool:
    try:
        normalized = _normalize_project_relative_path(relative_path)
    except ValueError:
        return False
    rel = Path(normalized)
    return len(rel.parts) == 1 and rel.name in SYSTEM_STATE_FILENAMES


def _is_text_editable_project_file(relative_path: str | None) -> bool:
    try:
        normalized = _normalize_project_relative_path(relative_path)
    except ValueError:
        return False
    suffix = Path(normalized).suffix.lower()
    return suffix in TEXT_FILE_SUFFIXES


def ensure_project_dir(doc_id: str) -> Path:
    normalized_id = _validate_doc_id(doc_id)
    PROJECTS_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    project_dir = get_project_dir(normalized_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    main_tex_path = project_dir / "main.tex"
    conversation_path = project_dir / "conversation.json"
    memory_path = project_dir / "memory.json"
    metadata_path = project_dir / "metadata.json"

    if not main_tex_path.exists():
        main_tex_path.write_text(DEFAULT_MAIN_TEX_TEMPLATE, encoding="utf-8")
    if not conversation_path.exists():
        _write_json(conversation_path, [])
    if not memory_path.exists():
        _write_json(memory_path, _default_memory_payload())
    if not metadata_path.exists():
        _write_json(metadata_path, _default_metadata(normalized_id, title="Conversation 1"))
    return project_dir


def _seed_demo_workspace(doc_id: str) -> bool:
    if not (
        DEMO_WORKSPACE_MAIN_TEX_PATH.is_file()
        and DEMO_WORKSPACE_BIB_PATH.is_file()
        and DEMO_WORKSPACE_PLOT_PATH.is_file()
    ):
        return False
    project_dir = ensure_project_dir(doc_id)
    try:
        (project_dir / "main.tex").write_text(DEMO_WORKSPACE_MAIN_TEX_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        (project_dir / "references.bib").write_text(DEMO_WORKSPACE_BIB_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        shutil.copy2(DEMO_WORKSPACE_PLOT_PATH, project_dir / "dummy_plot.png")
    except OSError:
        return False
    return True


def load_main_tex(doc_id: str) -> str:
    main_tex_path = _resolve_project_path(doc_id, "main.tex")
    try:
        return main_tex_path.read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_MAIN_TEX_TEMPLATE


def save_main_tex(doc_id: str, text: str | None = None) -> Path:
    main_tex_path = _resolve_project_path(doc_id, "main.tex")
    if text is None:
        state = st.session_state.get("app_state")
        if state is not None:
            text = str(getattr(state, "current_text", ""))
        else:
            text = ""
    main_tex_path.write_text(str(text), encoding="utf-8")
    return main_tex_path


def load_project_text_file(doc_id: str, relative_path: str | None) -> str:
    file_path = _resolve_project_path(doc_id, relative_path)
    if _is_internal_state_relative_path(relative_path):
        raise ValueError("Internal state files cannot be opened in editor.")
    if file_path.is_dir():
        raise ValueError("Cannot open directory in editor.")
    if not file_path.exists():
        return ""
    try:
        return file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def save_project_text_file(doc_id: str, relative_path: str | None, text: str | None = None) -> Path:
    file_path = _resolve_project_path(doc_id, relative_path)
    if _is_internal_state_relative_path(relative_path):
        raise ValueError("Internal state files cannot be edited.")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        state = st.session_state.get("app_state")
        text = str(getattr(state, "current_text", "")) if state is not None else ""
    file_path.write_text(str(text), encoding="utf-8")
    return file_path


def load_conversation(doc_id: str) -> List[Dict[str, Any]]:
    project_dir = ensure_project_dir(doc_id)
    conversation_path = project_dir / "conversation.json"
    payload = _read_json(conversation_path, [])
    return payload if isinstance(payload, list) else []


def save_conversation(doc_id: str, conversation: List[Dict[str, Any]] | None = None) -> Path:
    project_dir = ensure_project_dir(doc_id)
    conversation_path = project_dir / "conversation.json"
    if conversation is None:
        conversation = []
    _write_json(conversation_path, list(conversation))
    return conversation_path


def load_memory(doc_id: str) -> Dict[str, Any]:
    project_dir = ensure_project_dir(doc_id)
    memory_path = project_dir / "memory.json"
    fallback_payload = _default_memory_payload()
    # When read fails, keep last_updated empty so version arbitration does not
    # treat fallback as "newer" than an existing in-memory cache.
    fallback_payload["meta"]["last_updated"] = ""
    payload = _read_json(memory_path, fallback_payload)
    return _normalize_memory_payload(payload)


def save_memory(doc_id: str, memory: Dict[str, Any] | None = None) -> Path:
    project_dir = ensure_project_dir(doc_id)
    memory_path = project_dir / "memory.json"
    normalized = _normalize_memory_payload(memory or {})
    normalized["meta"]["last_updated"] = _utc_now_iso()
    _write_json(memory_path, normalized)
    return memory_path


def load_metadata(doc_id: str) -> Dict[str, Any]:
    normalized_id = _validate_doc_id(doc_id)
    project_dir = ensure_project_dir(normalized_id)
    metadata_path = project_dir / "metadata.json"
    payload = _read_json(metadata_path, _default_metadata(normalized_id))
    merged = _default_metadata(normalized_id)
    if isinstance(payload, dict):
        merged["title"] = str(payload.get("title", merged["title"])).strip() or merged["title"]
        merged["created_at"] = str(payload.get("created_at", merged["created_at"]))
        merged["current_role"] = _normalize_role_name(str(payload.get("current_role", "")))
        merged["project_root"] = str(payload.get("project_root", merged["project_root"])).strip() or str(Path.cwd())
        try:
            merged["active_file_path"] = _normalize_project_relative_path(
                payload.get("active_file_path", merged.get("active_file_path", "main.tex"))
            )
        except ValueError:
            merged["active_file_path"] = "main.tex"
        if _is_internal_state_relative_path(merged["active_file_path"]) or not _is_text_editable_project_file(
            merged["active_file_path"]
        ):
            merged["active_file_path"] = "main.tex"
        merged["updated_at"] = str(payload.get("updated_at", merged["updated_at"]))
    return merged


def save_metadata(
    doc_id: str,
    metadata: Dict[str, Any] | None = None,
    touch_updated_at: bool = True,
) -> Path:
    normalized_id = _validate_doc_id(doc_id)
    project_dir = ensure_project_dir(normalized_id)
    metadata_path = project_dir / "metadata.json"
    current = load_metadata(normalized_id)
    incoming = metadata or {}
    active_file_path = _normalize_project_relative_path(
        incoming.get("active_file_path", current.get("active_file_path", "main.tex"))
    )
    if _is_internal_state_relative_path(active_file_path) or not _is_text_editable_project_file(active_file_path):
        active_file_path = "main.tex"
    preserved_updated_at = (
        str(incoming.get("updated_at", current.get("updated_at", ""))).strip()
        or str(current.get("updated_at", "")).strip()
        or _utc_now_iso()
    )
    merged = {
        "doc_id": normalized_id,
        "title": str(incoming.get("title", current.get("title", "Conversation 1"))).strip() or "Conversation 1",
        "created_at": str(incoming.get("created_at", current.get("created_at", _utc_now_iso()))),
        "current_role": _normalize_role_name(str(incoming.get("current_role", current.get("current_role", "")))),
        "project_root": str(incoming.get("project_root", current.get("project_root", str(Path.cwd())))).strip()
        or str(Path.cwd()),
        "active_file_path": active_file_path,
        "updated_at": _utc_now_iso() if touch_updated_at else preserved_updated_at,
    }
    _write_json(metadata_path, merged)
    return metadata_path


def delete_project(doc_id: str) -> None:
    normalized_id = _validate_doc_id(doc_id)
    project_dir = get_project_dir(normalized_id)
    pdf_build_dir = PDF_BUILDS_ROOT_DIR / normalized_id
    if project_dir.exists():
        _safe_remove_tree(project_dir, PROJECTS_ROOT_DIR)
    if pdf_build_dir.exists():
        _safe_remove_tree(pdf_build_dir, PDF_BUILDS_ROOT_DIR)


def _chat_turns_to_payload(turns: List[ChatTurn], fallback_role: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    normalized_fallback_role = _normalize_role_name(fallback_role)
    for turn in turns:
        metadata = dict(turn.metadata or {})
        timestamp = str(metadata.get("timestamp", "")).strip() or _utc_now_iso()
        role_name = _normalize_role_name(
            str(
                metadata.get("agent_role")
                or metadata.get("companion_role")
                or normalized_fallback_role
            )
        )
        item: Dict[str, Any] = {
            "role": str(turn.role or ""),
            "content": str(turn.content or ""),
            "timestamp": timestamp,
        }
        if role_name and item["role"] == "assistant":
            item["agent_role"] = role_name
        if metadata:
            item["metadata"] = metadata
        records.append(item)
    return records


def _payload_to_chat_turns(payload: List[Dict[str, Any]]) -> List[ChatTurn]:
    turns: List[ChatTurn] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", ""))
        if role not in {"user", "assistant"}:
            continue
        metadata: Dict[str, Any] = {}
        raw_metadata = item.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        timestamp = str(item.get("timestamp", "")).strip()
        if timestamp:
            metadata["timestamp"] = timestamp
        agent_role = _normalize_role_name(str(item.get("agent_role", "")))
        if agent_role:
            metadata.setdefault("companion_role", agent_role)
            metadata.setdefault("agent_role", agent_role)
        turns.append(
            ChatTurn(
                role=role,
                content=content,
                request_id=str(item.get("request_id", "")),
                metadata=metadata,
            )
        )
    return turns


def _new_agent_chat_id() -> str:
    return f"agentchat-{uuid4().hex[:10]}"


def _next_agent_chat_title(agent_chats_payload: Dict[str, Any]) -> str:
    chats = agent_chats_payload.get("chats", {}) if isinstance(agent_chats_payload, dict) else {}
    normalized_titles = {
        str(item.get("name", "")).strip().casefold()
        for item in list(chats.values()) if isinstance(item, dict) and str(item.get("name", "")).strip()
    }
    candidate = 1
    while True:
        name = f"Chat {candidate}"
        if name.casefold() not in normalized_titles:
            return name
        candidate += 1


def _normalize_agent_chat_conversation_payload(payload: Any) -> List[Dict[str, Any]]:
    records = payload if isinstance(payload, list) else []
    turns = _payload_to_chat_turns(records)
    return _chat_turns_to_payload(turns, fallback_role="")


def _normalize_agent_chats_payload(
    payload: Dict[str, Any] | None,
    fallback_conversation: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    fallback_records = _normalize_agent_chat_conversation_payload(fallback_conversation or [])
    raw = payload if isinstance(payload, dict) else {}
    raw_chats = raw.get("chats")
    raw_order = raw.get("order")
    raw_active = str(raw.get("active_chat_id", "")).strip()

    chats_map = raw_chats if isinstance(raw_chats, dict) else {}
    normalized_chats: Dict[str, Dict[str, Any]] = {}
    for chat_id_raw, chat_payload in chats_map.items():
        chat_id = str(chat_id_raw or "").strip()
        if not chat_id:
            continue
        data = chat_payload if isinstance(chat_payload, dict) else {}
        created_at = str(data.get("created_at", "")).strip() or _utc_now_iso()
        updated_at = str(data.get("updated_at", "")).strip() or created_at
        normalized_chats[chat_id] = {
            "name": str(data.get("name", "")).strip() or f"Chat {len(normalized_chats) + 1}",
            "created_at": created_at,
            "updated_at": updated_at,
            "conversation": _normalize_agent_chat_conversation_payload(data.get("conversation", [])),
        }

    order_list = [str(item or "").strip() for item in (raw_order if isinstance(raw_order, list) else [])]
    order: List[str] = []
    seen = set()
    for chat_id in order_list:
        if chat_id and chat_id in normalized_chats and chat_id not in seen:
            order.append(chat_id)
            seen.add(chat_id)
    for chat_id in list(normalized_chats.keys()):
        if chat_id not in seen:
            order.append(chat_id)
            seen.add(chat_id)

    if not order:
        now = _utc_now_iso()
        default_id = _new_agent_chat_id()
        normalized_chats[default_id] = {
            "name": "Chat 1",
            "created_at": now,
            "updated_at": now,
            "conversation": fallback_records,
        }
        order = [default_id]

    active_chat_id = raw_active if raw_active in normalized_chats else order[0]
    if active_chat_id not in order:
        order = [active_chat_id] + [item for item in order if item != active_chat_id]

    return {
        "active_chat_id": active_chat_id,
        "order": order,
        "chats": normalized_chats,
    }


def load_agent_chats(doc_id: str) -> Dict[str, Any]:
    project_dir = ensure_project_dir(doc_id)
    path = project_dir / "agent_chats.json"
    raw_payload = _read_json(path, {})
    fallback_conversation = load_conversation(doc_id)
    return _normalize_agent_chats_payload(
        raw_payload if isinstance(raw_payload, dict) else {},
        fallback_conversation=fallback_conversation,
    )


def save_agent_chats(doc_id: str, payload: Dict[str, Any] | None = None) -> Path:
    project_dir = ensure_project_dir(doc_id)
    path = project_dir / "agent_chats.json"
    normalized = _normalize_agent_chats_payload(payload if isinstance(payload, dict) else {}, fallback_conversation=[])
    _write_json(path, normalized)
    return path


def _save_component_debug_payload(doc_id: str, component_name: str, payload: Dict[str, Any]) -> None:
    project_dir = ensure_project_dir(doc_id)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(component_name or "").strip()).strip("._-") or "component"
    path = project_dir / f".{safe_name}_debug.json"
    _write_json(
        path,
        {
            "saved_at": _utc_now_iso(),
            "payload": payload if isinstance(payload, dict) else {},
        },
    )


def _save_processing_debug_payload(doc_id: str, stage: str, payload: Dict[str, Any] | None = None) -> None:
    project_dir = ensure_project_dir(doc_id)
    path = project_dir / ".processing_debug.json"
    _write_json(
        path,
        {
            "saved_at": _utc_now_iso(),
            "stage": str(stage or "").strip(),
            "payload": payload if isinstance(payload, dict) else {},
        },
    )


def _workspace_memory_defaults(current_role: str = "") -> Dict[str, Any]:
    return _default_memory_payload(current_role=current_role)


def _workspace_doc_ids_from_disk() -> List[str]:
    if not PROJECTS_ROOT_DIR.exists():
        return []
    doc_ids: List[str] = []
    for child in sorted(PROJECTS_ROOT_DIR.iterdir()):
        if not child.is_dir():
            continue
        try:
            doc_ids.append(_validate_doc_id(child.name))
        except ValueError:
            continue
    return doc_ids


def _all_workspace_titles() -> List[str]:
    titles: List[str] = []
    store = st.session_state.get("workspace_store", {})
    order = st.session_state.get("workspace_order", [])
    if isinstance(store, dict) and isinstance(order, list):
        for workspace_id in order:
            payload = store.get(str(workspace_id))
            if not isinstance(payload, dict):
                continue
            name = str(payload.get("name", "")).strip()
            if name:
                titles.append(name)
    for doc_id in _workspace_doc_ids_from_disk():
        metadata = load_metadata(doc_id)
        title = str(metadata.get("title", "")).strip()
        if title:
            titles.append(title)
    return titles


def _next_conversation_title() -> str:
    titles = _all_workspace_titles()
    used_numbers: set[int] = set()
    normalized_titles = {title.casefold() for title in titles if title}

    for title in titles:
        match = re.fullmatch(r"\s*(?i:conversation)\s+(\d+)\s*", title)
        if not match:
            continue
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        if number >= 1:
            used_numbers.add(number)

    candidate = 1
    while True:
        if candidate not in used_numbers:
            proposed = f"Conversation {candidate}"
            if proposed.casefold() not in normalized_titles:
                return proposed
        candidate += 1


def init_state() -> None:
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState(current_text="")
        st.session_state.app_state.config = load_config_from_env(
            st.session_state.app_state.config
        )
    if "consistency_orchestrator" not in st.session_state:
        st.session_state.consistency_orchestrator = ConsistencyOrchestrator(st.session_state.app_state)
    st.session_state.controller = st.session_state.consistency_orchestrator
    if "resolver" not in st.session_state:
        st.session_state.resolver = LatexRefResolver()
    if "context_builder" not in st.session_state:
        st.session_state.context_builder = ContextWindowBuilder(max_chars=4000, surround_chars=900)
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = FileCentricStateManager(base_dir=str(STATE_ROOT_DIR))
    if "router_v1" not in st.session_state:
        st.session_state.router_v1 = RouterV1()
    _sync_patch_preview_compat(st.session_state.app_state)
    _sync_patch_gate_flags(st.session_state.app_state)
    if "selection_text" not in st.session_state:
        st.session_state.selection_text = ""
    if "selection_start" not in st.session_state:
        st.session_state.selection_start = -1
    if "selection_end" not in st.session_state:
        st.session_state.selection_end = -1
    if "pending_selection_text" not in st.session_state:
        st.session_state.pending_selection_text = ""
    if "pending_selection_start" not in st.session_state:
        st.session_state.pending_selection_start = -1
    if "pending_selection_end" not in st.session_state:
        st.session_state.pending_selection_end = -1
    if "suppress_next_selection_payload" not in st.session_state:
        st.session_state.suppress_next_selection_payload = False
    if "last_component_action_fingerprint" not in st.session_state:
        st.session_state.last_component_action_fingerprint = ""
    if "confirmed_selection_text" not in st.session_state:
        st.session_state.confirmed_selection_text = ""
    if "confirmed_selection_start" not in st.session_state:
        st.session_state.confirmed_selection_start = -1
    if "confirmed_selection_end" not in st.session_state:
        st.session_state.confirmed_selection_end = -1
    if "active_selection" not in st.session_state:
        st.session_state.active_selection = ""
    if "selected_role" not in st.session_state:
        st.session_state.selected_role = ""
    if "current_role" not in st.session_state:
        if st.session_state.selected_role in LOTTIE_ROLE_FILES:
            st.session_state.current_role = st.session_state.selected_role
        else:
            st.session_state.current_role = ""
    if "issues_json" not in st.session_state:
        st.session_state.issues_json = []
    if "project_root" not in st.session_state:
        st.session_state.project_root = str(Path.cwd())
    if "selected_checks" not in st.session_state:
        st.session_state.selected_checks = list(CHECK_NAMES)
    if "check_enabled_map" not in st.session_state:
        st.session_state.check_enabled_map = dict(CHECK_ENABLE_DEFAULTS)
    if "selected_checks_explicit" not in st.session_state:
        st.session_state.selected_checks_explicit = False
    if "composer_text" not in st.session_state:
        st.session_state.composer_text = ""
    if "composer_triggered" not in st.session_state:
        st.session_state.composer_triggered = False
    if "composer_tools_open" not in st.session_state:
        st.session_state.composer_tools_open = False
    if "composer_last_event_id" not in st.session_state:
        st.session_state.composer_last_event_id = 0
    if "agent_chat_history_open" not in st.session_state:
        st.session_state.agent_chat_history_open = False
    if "agent_chat_delete_confirm" not in st.session_state:
        st.session_state.agent_chat_delete_confirm = {}
    if "processing_busy" not in st.session_state:
        st.session_state.processing_busy = False
    if "processing_kind" not in st.session_state:
        st.session_state.processing_kind = ""
    if "processing_pending_action" not in st.session_state:
        st.session_state.processing_pending_action = {}
    if "role_hint_dismissed" not in st.session_state:
        st.session_state.role_hint_dismissed = False
    if "last_grounding" not in st.session_state:
        st.session_state.last_grounding = {}
    if "editor_waiting_patch_consent" not in st.session_state:
        st.session_state.editor_waiting_patch_consent = False
    if "editor_waiting_apply_consent" not in st.session_state:
        st.session_state.editor_waiting_apply_consent = False
    if "advisor_waiting_suggestion_consent" not in st.session_state:
        st.session_state.advisor_waiting_suggestion_consent = False
    if "workspace_order" not in st.session_state:
        st.session_state.workspace_order = []
    if "workspace_store" not in st.session_state:
        st.session_state.workspace_store = {}
    if "active_workspace_id" not in st.session_state:
        st.session_state.active_workspace_id = ""
    if "workspace_delete_confirm" not in st.session_state:
        st.session_state.workspace_delete_confirm = {}
    if "workspace_bootstrapped" not in st.session_state:
        st.session_state.workspace_bootstrapped = False
    if "workspace_text_hashes" not in st.session_state:
        st.session_state.workspace_text_hashes = {}
    if "workspace_activity_marks" not in st.session_state:
        st.session_state.workspace_activity_marks = {}
    if "active_file_path" not in st.session_state:
        st.session_state.active_file_path = "main.tex"
    if "files_selected_path" not in st.session_state:
        st.session_state.files_selected_path = "main.tex"
    if "files_expanded_dirs" not in st.session_state:
        st.session_state.files_expanded_dirs = []
    if "files_delete_confirm" not in st.session_state:
        st.session_state.files_delete_confirm = {}
    if "files_dnd_bridge_action" not in st.session_state:
        st.session_state.files_dnd_bridge_action = ""
    if "files_dnd_bridge_payload" not in st.session_state:
        st.session_state.files_dnd_bridge_payload = ""
    if "files_dnd_bridge_nonce" not in st.session_state:
        st.session_state.files_dnd_bridge_nonce = ""
    if "files_dnd_bridge_last_nonce" not in st.session_state:
        st.session_state.files_dnd_bridge_last_nonce = ""
    if "pdf_pending_dblclick_event" not in st.session_state:
        st.session_state.pdf_pending_dblclick_event = None
    if "resource_preview_visible" not in st.session_state:
        st.session_state.resource_preview_visible = False
    if "resource_preview_path" not in st.session_state:
        st.session_state.resource_preview_path = ""
    if "resource_preview_doc_id" not in st.session_state:
        st.session_state.resource_preview_doc_id = ""
    if "sidebar_panel_mode" not in st.session_state:
        st.session_state.sidebar_panel_mode = "chats"
    if "viewport_height" not in st.session_state:
        st.session_state.viewport_height = 0
    if "viewport_available_height" not in st.session_state:
        st.session_state.viewport_available_height = 0
    if "pdf_compiled_text_hash" not in st.session_state:
        st.session_state.pdf_compiled_text_hash = ""
    if "pdf_compiled_base64" not in st.session_state:
        st.session_state.pdf_compiled_base64 = ""
    if "pdf_compiled_bytes" not in st.session_state:
        st.session_state.pdf_compiled_bytes = b""
    if "pdf_compile_ok" not in st.session_state:
        st.session_state.pdf_compile_ok = False
    if "pdf_compile_log" not in st.session_state:
        st.session_state.pdf_compile_log = ""
    if "pdf_last_export_path" not in st.session_state:
        st.session_state.pdf_last_export_path = ""
    if "pdf_compiled_pdf_path" not in st.session_state:
        st.session_state.pdf_compiled_pdf_path = ""
    if "pdf_compiled_synctex_path" not in st.session_state:
        st.session_state.pdf_compiled_synctex_path = ""
    if "pdf_compiled_tex_path" not in st.session_state:
        st.session_state.pdf_compiled_tex_path = ""
    if "pdf_compile_errors" not in st.session_state:
        st.session_state.pdf_compile_errors = []
    if "pdf_compiled_extracted_text_hash" not in st.session_state:
        st.session_state.pdf_compiled_extracted_text_hash = ""
    if "pdf_compiled_extracted_text" not in st.session_state:
        st.session_state.pdf_compiled_extracted_text = ""
    if "pdf_preview_png_hash" not in st.session_state:
        st.session_state.pdf_preview_png_hash = ""
    if "pdf_preview_png_bytes" not in st.session_state:
        st.session_state.pdf_preview_png_bytes = b""
    if "pdf_preview_png_pages_hash" not in st.session_state:
        st.session_state.pdf_preview_png_pages_hash = ""
    if "pdf_preview_png_pages_bytes" not in st.session_state:
        st.session_state.pdf_preview_png_pages_bytes = []
    if "compile_in_progress" not in st.session_state:
        st.session_state.compile_in_progress = False
    if "pdf_viewer_last_event_id" not in st.session_state:
        st.session_state.pdf_viewer_last_event_id = 0
    if "export_hint_nonce" not in st.session_state:
        st.session_state.export_hint_nonce = 0
    if "export_hint_message" not in st.session_state:
        st.session_state.export_hint_message = ""
    if "export_hint_duration_ms" not in st.session_state:
        st.session_state.export_hint_duration_ms = 3000
    if "export_hint_last_rendered_nonce" not in st.session_state:
        st.session_state.export_hint_last_rendered_nonce = 0
    if "browser_pdf_export_nonce" not in st.session_state:
        st.session_state.browser_pdf_export_nonce = 0
    if "browser_pdf_export_name" not in st.session_state:
        st.session_state.browser_pdf_export_name = ""
    if "browser_pdf_export_b64" not in st.session_state:
        st.session_state.browser_pdf_export_b64 = ""
    if "editor_focus_start" not in st.session_state:
        st.session_state.editor_focus_start = -1
    if "editor_focus_end" not in st.session_state:
        st.session_state.editor_focus_end = -1
    if "editor_focus_event_id" not in st.session_state:
        st.session_state.editor_focus_event_id = 0
    if "editor_backend_sync_event_id" not in st.session_state:
        st.session_state.editor_backend_sync_event_id = 0
    if not st.session_state.workspace_bootstrapped:
        _bootstrap_workspaces_from_disk()
        st.session_state.workspace_bootstrapped = True


def _set_consistency_orchestrator(state: AppState) -> ConsistencyOrchestrator:
    orchestrator = ConsistencyOrchestrator(state)
    st.session_state.consistency_orchestrator = orchestrator
    st.session_state.controller = orchestrator
    return orchestrator


def _consistency_orchestrator() -> ConsistencyOrchestrator:
    orchestrator = st.session_state.get("consistency_orchestrator")
    if isinstance(orchestrator, ConsistencyOrchestrator):
        st.session_state.controller = orchestrator
        return orchestrator
    legacy = st.session_state.get("controller")
    if isinstance(legacy, ConsistencyOrchestrator):
        st.session_state.consistency_orchestrator = legacy
        return legacy
    return _set_consistency_orchestrator(st.session_state.app_state)


def _normalize_role_name(raw_role: str) -> str:
    role = str(raw_role).strip()
    if role in LOTTIE_ROLE_FILES:
        return role
    return ROLE_ALIASES.get(role.lower(), "")


def _sync_role_state_from_session(write_widget_key: bool = False) -> None:
    current_value = st.session_state.get("current_role", "")
    normalized = _normalize_role_name(current_value)
    if write_widget_key and current_value != normalized:
        st.session_state.current_role = normalized
    st.session_state.selected_role = normalized
    app_state = st.session_state.get("app_state")
    if app_state is not None and normalized:
        app_state.role = normalized


def _on_role_change() -> None:
    _sync_role_state_from_session()
    active_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if not active_id:
        return
    store = st.session_state.get("workspace_store", {})
    payload = store.get(active_id)
    if not isinstance(payload, dict):
        return
    normalized_role = _normalize_role_name(str(st.session_state.get("current_role", "")))
    ui_payload = dict(payload.get("ui", {}))
    ui_payload["current_role"] = normalized_role
    ui_payload["selected_role"] = normalized_role
    payload["ui"] = ui_payload

    memory_payload = _normalize_memory_payload(payload.get("memory", {}), current_role=normalized_role)
    memory_payload["meta"]["current_role"] = normalized_role
    memory_payload["meta"]["last_updated"] = _utc_now_iso()
    payload["memory"] = memory_payload

    metadata = load_metadata(active_id)
    metadata["current_role"] = normalized_role
    save_metadata(active_id, metadata, touch_updated_at=False)
    payload["metadata"] = metadata
    _persist_workspace_to_disk(active_id, touch_activity=False)


def _dismiss_role_hint() -> None:
    st.session_state.role_hint_dismissed = True


def _append_chat(state: AppState, role: str, content: str, **metadata: Any) -> None:
    turn_metadata = dict(metadata or {})
    turn_metadata.setdefault("timestamp", _utc_now_iso())
    if role == "assistant":
        companion_role = _normalize_role_name(
            str(turn_metadata.get("companion_role") or turn_metadata.get("agent_role") or state.role)
        )
        if companion_role:
            turn_metadata["companion_role"] = companion_role
            turn_metadata["agent_role"] = companion_role
    state.chat_history.append(
        ChatTurn(
            role=role,
            content=content,
            request_id=state.request_id,
            metadata=turn_metadata,
        )
    )


def _confirmed_scope_payload(state: AppState) -> ScopePayload | None:
    selection = state.active_selection
    if not isinstance(selection, SelectionContext):
        return None
    if not bool(selection.metadata.get("confirmed", False)):
        return None
    if selection.end <= selection.start:
        return None
    scope_text = str(selection.snippet or "").strip()
    if not scope_text:
        return None
    return ScopePayload(
        type="selected_span",
        text=scope_text,
        metadata={
            "start": int(selection.start),
            "end": int(selection.end),
            "active_file": _active_file_for_state(state),
            "confirmed": True,
        },
    )


def _strip_latex_comments(text: str) -> str:
    lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = str(raw_line)
        # Keep escaped \% and remove true comments.
        line = re.sub(r"(?<!\\)%.*$", "", line)
        lines.append(line)
    return "\n".join(lines)


def _resolve_tex_include_path(
    *,
    project_root: Path,
    current_file: Path,
    include_target: str,
) -> Path | None:
    raw = str(include_target or "").strip().strip('"').strip("'")
    if not raw:
        return None
    if raw.startswith("/"):
        return None
    path_like = Path(raw)
    candidates: List[Path] = []
    if path_like.suffix.lower() == ".tex":
        candidates.extend([current_file.parent / path_like, project_root / path_like])
    else:
        candidates.extend(
            [
                current_file.parent / path_like,
                current_file.parent / f"{raw}.tex",
                project_root / path_like,
                project_root / f"{raw}.tex",
            ]
        )
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(project_root.resolve())
        except Exception:
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        try:
            rel = resolved.relative_to(project_root).as_posix()
        except Exception:
            continue
        if _is_internal_state_relative_path(rel):
            continue
        if resolved.suffix.lower() != ".tex":
            continue
        return resolved
    return None


def _expand_latex_main_document(
    *,
    doc_id: str,
    entry_rel: str = "main.tex",
    max_files: int = 120,
    max_depth: int = 24,
) -> Dict[str, Any]:
    project_root = ensure_project_dir(doc_id)
    try:
        entry_path = _resolve_project_path(doc_id, entry_rel)
    except ValueError:
        entry_path = _resolve_project_path(doc_id, "main.tex")

    visited: List[str] = []
    visited_set = set()
    unresolved: List[Dict[str, str]] = []

    def _expand(path: Path, depth: int) -> str:
        if depth > max_depth:
            return ""
        if len(visited) >= max_files:
            return ""
        try:
            rel = path.relative_to(project_root).as_posix()
        except Exception:
            rel = path.name
        if rel in visited_set:
            return ""
        visited_set.add(rel)
        visited.append(rel)
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            unresolved.append({"from": rel, "target": "<read_error>"})
            return ""
        clean_text = _strip_latex_comments(raw_text)

        def _replace_include(match: re.Match[str]) -> str:
            target = str(match.group(1) or "").strip()
            include_path = _resolve_tex_include_path(
                project_root=project_root,
                current_file=path,
                include_target=target,
            )
            if include_path is None:
                unresolved.append({"from": rel, "target": target})
                return ""
            return "\n" + _expand(include_path, depth + 1) + "\n"

        return LATEX_INCLUDE_PATTERN.sub(_replace_include, clean_text)

    expanded = _expand(entry_path, 0)
    return {
        "text": str(expanded or ""),
        "visited_files": visited,
        "unresolved_includes": unresolved,
    }


def _extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 180) -> str:
    if not pdf_bytes:
        return ""
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return ""

    pages: List[str] = []
    for idx, page in enumerate(reader.pages):
        if idx >= max_pages:
            break
        try:
            text = str(page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(text)
    if not pages:
        return ""
    merged = "\n\n".join(pages)
    merged = re.sub(r"[ \t]+\n", "\n", merged)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()


def _extract_text_from_docx_bytes(docx_bytes: bytes, max_chars: int = RESOURCE_TEXT_PREVIEW_MAX_BYTES) -> str:
    if not docx_bytes:
        return ""
    try:
        archive = zipfile.ZipFile(io.BytesIO(docx_bytes))
    except Exception:
        return ""

    names = set(archive.namelist())
    xml_targets: List[str] = []
    if "word/document.xml" in names:
        xml_targets.append("word/document.xml")
    xml_targets.extend(
        sorted(
            name
            for name in names
            if re.fullmatch(r"word/(?:header|footer)\d+\.xml", str(name or "").strip())
        )
    )
    if not xml_targets:
        return ""

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    lines: List[str] = []
    char_count = 0

    for xml_name in xml_targets:
        try:
            xml_bytes = archive.read(xml_name)
            root = ET.fromstring(xml_bytes)
        except Exception:
            continue

        paragraphs = root.findall(".//w:p", ns)
        for para in paragraphs:
            chunks: List[str] = []
            for node in para.iter():
                tag_text = str(getattr(node, "tag", ""))
                if tag_text.endswith("}t"):
                    text = str(getattr(node, "text", "") or "")
                    if text:
                        chunks.append(text)
                elif tag_text.endswith("}tab"):
                    chunks.append("\t")
                elif tag_text.endswith("}br") or tag_text.endswith("}cr"):
                    chunks.append("\n")

            para_text = "".join(chunks).strip()
            if not para_text:
                continue
            lines.append(para_text)
            char_count += len(para_text) + 2
            if char_count >= max_chars:
                break
        if char_count >= max_chars:
            break

    merged = "\n\n".join(lines).strip()
    if not merged:
        return ""
    if len(merged) > max_chars:
        merged = f"{merged[:max_chars].rstrip()}\n\n...<preview truncated>"
    return merged


def _extract_text_via_textutil(file_path: Path, max_chars: int = RESOURCE_TEXT_PREVIEW_MAX_BYTES) -> str:
    if not file_path.exists() or not file_path.is_file():
        return ""
    textutil_bin = shutil.which("textutil")
    if not textutil_bin:
        return ""
    cmd = [textutil_bin, "-convert", "txt", "-stdout", str(file_path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    text = str(proc.stdout or "").strip()
    if not text:
        return ""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = f"{text[:max_chars].rstrip()}\n\n...<preview truncated>"
    return text


def _compiled_pdf_text_snapshot() -> Dict[str, Any]:
    source_hash = str(st.session_state.get("pdf_compiled_text_hash", "") or "").strip()
    if not source_hash:
        return {"text": "", "source": "pdf_unavailable", "chars": 0}
    if not bool(st.session_state.get("pdf_compile_ok", False)):
        return {"text": "", "source": "pdf_not_compiled", "chars": 0}
    cached_hash = str(st.session_state.get("pdf_compiled_extracted_text_hash", "") or "").strip()
    cached_text = str(st.session_state.get("pdf_compiled_extracted_text", "") or "")
    if cached_hash == source_hash and cached_text.strip():
        return {"text": cached_text, "source": "compiled_pdf_cache", "chars": len(cached_text)}

    pdf_bytes = bytes(st.session_state.get("pdf_compiled_bytes", b"") or b"")
    if not pdf_bytes:
        return {"text": "", "source": "pdf_bytes_missing", "chars": 0}
    extracted = _extract_text_from_pdf_bytes(pdf_bytes)
    st.session_state.pdf_compiled_extracted_text_hash = source_hash
    st.session_state.pdf_compiled_extracted_text = extracted
    return {"text": extracted, "source": "compiled_pdf_extract", "chars": len(extracted)}


def _selection_covers_full_active_file(state: AppState, scope: ScopePayload | None) -> bool:
    if scope is None:
        return False
    if scope.type != "selected_span":
        return False
    selection = state.active_selection
    if not isinstance(selection, SelectionContext):
        return False
    if not bool(selection.metadata.get("confirmed", False)):
        return False
    if selection.end <= selection.start:
        return False
    current_text = str(state.current_text or "")
    if not current_text:
        return False
    active_file = _active_file_for_state(state)
    if active_file != "main.tex":
        return False
    if int(selection.start) <= 0 and int(selection.end) >= len(current_text):
        return True
    scope_text = str(scope.text or "").strip()
    if scope_text and scope_text == current_text.strip():
        return True
    return False


def _analysis_document_text_for_scope(state: AppState, scope: ScopePayload | None) -> Dict[str, Any]:
    active_file = _active_file_for_state(state)
    payload: Dict[str, Any] = {
        "document_text": str(state.current_text or ""),
        "check_document_text": str(state.current_text or ""),
        "source": "active_file",
        "check_source": "active_file",
        "expanded_main": False,
        "expanded_active_file": False,
        "active_file": active_file,
        "visited_files": [active_file],
        "unresolved_includes": [],
    }
    if active_file != "main.tex" and active_file.lower().endswith(".tex"):
        expanded_local = _expand_latex_main_document(doc_id=state.state_doc_id, entry_rel=active_file)
        expanded_local_text = str(expanded_local.get("text", "") or "")
        visited_local = list(expanded_local.get("visited_files", []))
        if expanded_local_text.strip() and len(visited_local) > 1:
            payload.update(
                {
                    "check_document_text": expanded_local_text,
                    "check_source": "expanded_active_file_latex",
                    "expanded_active_file": True,
                    "visited_files": visited_local,
                    "unresolved_includes": list(expanded_local.get("unresolved_includes", [])),
                }
            )
    if not _selection_covers_full_active_file(state, scope):
        return payload

    expanded = _expand_latex_main_document(doc_id=state.state_doc_id, entry_rel="main.tex")
    expanded_text = str(expanded.get("text", "") or "")
    if not expanded_text.strip():
        return payload
    pdf_snapshot = _compiled_pdf_text_snapshot()
    pdf_text = str(pdf_snapshot.get("text", "") or "")
    if pdf_text.strip() and len(pdf_text) >= 120:
        payload.update(
            {
                "document_text": pdf_text,
                "check_document_text": expanded_text,
                "source": "compiled_pdf_text",
                "check_source": "expanded_main_latex",
                "expanded_main": True,
                "visited_files": list(expanded.get("visited_files", [])),
                "unresolved_includes": list(expanded.get("unresolved_includes", [])),
                "pdf_text_chars": int(pdf_snapshot.get("chars", len(pdf_text))),
                "pdf_text_source": str(pdf_snapshot.get("source", "compiled_pdf_extract")),
            }
        )
        return payload
    payload.update(
        {
            "document_text": expanded_text,
            "check_document_text": expanded_text,
            "source": "expanded_main",
            "check_source": "expanded_main_latex",
            "expanded_main": True,
            "visited_files": list(expanded.get("visited_files", [])),
            "unresolved_includes": list(expanded.get("unresolved_includes", [])),
        }
    )
    return payload


def _selection_has_terminology_signal(text: str) -> bool:
    probe = _strip_latex_comments(str(text or ""))
    try:
        if bool(TERMINOLOGY_AVAILABILITY_AGENT._extract_variants(probe)):  # noqa: SLF001
            return True
    except Exception:
        pass
    if TERMINOLOGY_ACRONYM_PATTERN.search(probe):
        return True
    if TERMINOLOGY_TITLE_PATTERN.search(probe):
        return True
    return False


def _selection_has_table_signal(text: str) -> bool:
    probe = _strip_latex_comments(str(text or ""))
    if not probe.strip():
        return False
    return bool(TABLE_REF_PATTERN.search(probe) or TABLE_NATURAL_REF_PATTERN.search(probe))


def _selection_has_figure_signal(text: str) -> bool:
    probe = _strip_latex_comments(str(text or ""))
    if not probe.strip():
        return False
    return bool(FIGURE_REF_PATTERN.search(probe) or FIGURE_NATURAL_REF_PATTERN.search(probe))


def _normalize_check_enabled_map(raw: Dict[str, Any] | None) -> Dict[str, bool]:
    normalized = dict(CHECK_ENABLE_DEFAULTS)
    if isinstance(raw, dict):
        for key in CHECK_NAMES:
            if key in raw:
                normalized[key] = bool(raw.get(key))
    return normalized


def _compute_check_availability(
    *,
    selection_text: str,
    workspace_resources: Dict[str, Any],
    document_text: str | None = None,
) -> Dict[str, Any]:
    clean = _strip_latex_comments(str(selection_text or ""))
    if not clean.strip():
        return {
            "enabled": {name: False for name in CHECK_NAMES},
            "reasons": {name: "no_confirmed_selection" for name in CHECK_NAMES},
            "has_bib": bool(workspace_resources.get("has_bib", False)),
        }

    has_table = _selection_has_table_signal(clean)
    has_figure = _selection_has_figure_signal(clean)
    has_citation = bool(CITATION_CMD_PATTERN.search(clean))
    has_terminology = _selection_has_terminology_signal(clean)
    has_bib = bool(workspace_resources.get("has_bib", False))
    has_images = bool(list(workspace_resources.get("images", []))) if isinstance(workspace_resources, dict) else False

    enabled = {
        "table": has_table,
        "figure": has_figure and has_images,
        "citation": has_citation and has_bib,
        "terminology": has_terminology,
    }
    reasons = {
        "table": "" if enabled["table"] else "selection_has_no_table_ref",
        "figure": (
            ""
            if enabled["figure"]
            else (
                "selection_has_no_figure_ref"
                if not has_figure
                else "workspace_missing_figure_file"
            )
        ),
        "citation": (
            ""
            if enabled["citation"]
            else ("selection_has_no_citation_cmd" if not has_citation else "workspace_missing_bib_file")
        ),
        "terminology": "" if enabled["terminology"] else "selection_has_no_terminology_signal",
    }
    return {
        "enabled": enabled,
        "reasons": reasons,
        "has_bib": has_bib,
    }


def _format_skipped_checks_note(blocked_checks: List[Dict[str, str]]) -> str:
    if not blocked_checks:
        return ""
    snippets: List[str] = []
    for item in blocked_checks:
        check_name = str(item.get("check", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not check_name:
            continue
        reason_message = CHECK_REASON_MESSAGES.get(reason, "this check is not applicable to the current confirmed selection")
        snippets.append(f"{check_name}: {reason_message}")
    if not snippets:
        return ""
    return "Skipped checks due to scope preconditions:\n- " + "\n- ".join(snippets)


def _tool_availability_snapshot(state: AppState) -> Dict[str, bool]:
    text_provider = build_text_provider(state.config.text_provider_name)
    multimodal_provider = build_vlm_provider(state.config.vlm_provider_name)
    return {
        "text_provider": bool(text_provider.enabled()),
        "multimodal_provider": bool(multimodal_provider.enabled()),
        "image_qa": bool(multimodal_provider.enabled()),
    }


def _workspace_resource_snapshot(state: AppState) -> Dict[str, Any]:
    workspace_id = str(state.state_doc_id or "").strip()
    if not workspace_id:
        return {"images": [], "text_files": [], "has_bib": False}

    project_dir = ensure_project_dir(workspace_id)
    images: List[Dict[str, str]] = []
    text_files: List[Dict[str, str]] = []
    has_bib = False
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir).as_posix()
        if _is_internal_state_relative_path(rel):
            continue
        suffix = path.suffix.lower()
        entry = {
            "path": rel,
            "name": path.name,
            "stem": path.stem,
            "suffix": suffix,
        }
        if suffix in RESOURCE_IMAGE_SUFFIXES:
            images.append(entry)
        elif suffix == ".bib":
            has_bib = True
            text_files.append(entry)
        elif suffix in TEXT_FILE_SUFFIXES:
            text_files.append(entry)
    return {
        "images": images,
        "text_files": text_files,
        "has_bib": has_bib or bool(state.uploaded_bib.content.strip()),
    }


def _build_router_request(state: AppState, pending_action: Dict[str, Any]) -> RouterRequest:
    action = str(pending_action.get("action", "")).strip().lower()
    mapped_action = {
        "send": "send_message",
        "send_message": "send_message",
        "run_checks": "run_checks",
        "confirm_patch_prepare": "confirm_patch_prepare",
        "confirm_patch_apply": "confirm_patch_apply",
        "undo": "undo",
    }.get(action, "send_message")
    current_role = _normalize_role_name(str(st.session_state.get("current_role", "")))
    if current_role:
        state.role = current_role
    pending_patch_negotiation = bool(state.negotiation_state.active)
    pending_patch_preview_consent = (str(state.pending_patch.stage or "") == "awaiting_preview_consent")
    pending_patch_apply_consent = (str(state.pending_patch.stage or "") == "awaiting_apply_consent")
    pending_advisor_suggestion_consent = bool(st.session_state.get("advisor_waiting_suggestion_consent", False))
    ui_selected_checks = [
        x
        for x in list(pending_action.get("selected_checks", st.session_state.get("selected_checks", [])))
        if x in CHECK_NAME_SET
    ]

    # Consent/negotiation turns do not need full document expansion or workspace scans.
    # Keeping these turns lightweight avoids long "loading" delays after a simple "yes/no".
    if mapped_action == "send_message" and (
        pending_patch_negotiation
        or pending_patch_preview_consent
        or pending_patch_apply_consent
        or pending_advisor_suggestion_consent
    ):
        return RouterRequest(
            action=mapped_action,  # type: ignore[arg-type]
            role=current_role,
            user_text=str(pending_action.get("submit_text", "")).strip(),
            document_text=str(state.current_text or ""),
            scope=_confirmed_scope_payload(state),
            ui_selected_checks=ui_selected_checks,
            ui_checks_explicit=bool(pending_action.get("ui_checks_explicit", False)),
            workspace_resources={},
            tool_availability={},
            busy=False,
            busy_kind="",
            memory=_memory_context_for_prompt(state, current_role or state.role),
            pending_patch_negotiation=pending_patch_negotiation,
            pending_patch_preview_consent=pending_patch_preview_consent,
            pending_patch_apply_consent=pending_patch_apply_consent,
            pending_advisor_suggestion_consent=pending_advisor_suggestion_consent,
        )
    scope = _confirmed_scope_payload(state)
    analysis_doc = _analysis_document_text_for_scope(state, scope)
    if scope is not None and bool(analysis_doc.get("expanded_main", False)):
        scope = ScopePayload(
            type=scope.type,
            text=str(analysis_doc.get("document_text", scope.text) or scope.text),
            metadata={
                **dict(scope.metadata or {}),
                "analysis_source": str(analysis_doc.get("source", "expanded_main")),
                "expanded_main": True,
            },
        )
    workspace_resources = _workspace_resource_snapshot(state)
    workspace_resources["analysis_document"] = {
        "source": str(analysis_doc.get("source", "active_file")),
        "check_source": str(analysis_doc.get("check_source", analysis_doc.get("source", "active_file"))),
        "expanded_main": bool(analysis_doc.get("expanded_main", False)),
        "visited_files_count": len(list(analysis_doc.get("visited_files", []))),
    }
    return RouterRequest(
        action=mapped_action,  # type: ignore[arg-type]
        role=current_role,
        user_text=str(pending_action.get("submit_text", "")).strip(),
        document_text=str(
            analysis_doc.get(
                "check_document_text",
                analysis_doc.get("document_text", state.current_text),
            )
            or ""
        ),
        scope=scope,
        ui_selected_checks=ui_selected_checks,
        ui_checks_explicit=bool(pending_action.get("ui_checks_explicit", False)),
        workspace_resources=workspace_resources,
        tool_availability=_tool_availability_snapshot(state),
        busy=False,
        busy_kind="",
        memory=_memory_context_for_prompt(state, current_role or state.role),
        pending_patch_negotiation=pending_patch_negotiation,
        pending_patch_preview_consent=pending_patch_preview_consent,
        pending_patch_apply_consent=pending_patch_apply_consent,
        pending_advisor_suggestion_consent=pending_advisor_suggestion_consent,
    )


def _sync_patch_gate_flags(state: AppState) -> None:
    patch_stage = str(state.pending_patch.stage or "idle")
    st.session_state.editor_waiting_patch_consent = patch_stage == "awaiting_preview_consent"
    st.session_state.editor_waiting_apply_consent = patch_stage == "awaiting_apply_consent"


def _sync_patch_preview_compat(state: AppState) -> None:
    # Legacy mirror only. Router-driven patch flows should read
    # `state.pending_patch.patch_diff` directly.
    state.patch_preview = str(state.pending_patch.patch_diff or "")


def _set_patch_lifecycle_stage(state: AppState, stage: str) -> None:
    state.pending_patch.stage = str(stage or "idle")
    _sync_patch_gate_flags(state)


def _reset_turn_gates(state: AppState) -> None:
    _set_patch_lifecycle_stage(state, "idle")
    st.session_state.advisor_waiting_suggestion_consent = False


def _clear_pending_patch_payload(state: AppState, *, keep_stage: bool = False) -> None:
    state.pending_patch.status = "none"
    if not keep_stage:
        _set_patch_lifecycle_stage(state, "idle")
    state.pending_patch.patch_diff = ""
    state.pending_patch.operations = []
    state.pending_patch.reason = ""
    state.pending_patch.edit_ratio = 0.0
    state.pending_patch.source = ""
    state.pending_patch.summary = ""
    _sync_patch_preview_compat(state)


def _mark_workspace_activity(workspace_id: str) -> None:
    workspace_key = str(workspace_id or "").strip()
    if not workspace_key:
        return
    marks = dict(st.session_state.get("workspace_activity_marks", {}))
    marks[workspace_key] = True
    st.session_state.workspace_activity_marks = marks
    order = list(st.session_state.get("workspace_order", []))
    if workspace_key in order:
        order = [workspace_key] + [item for item in order if item != workspace_key]
        st.session_state.workspace_order = order


def _consume_workspace_activity_mark(workspace_id: str) -> bool:
    workspace_key = str(workspace_id or "").strip()
    if not workspace_key:
        return False
    marks = dict(st.session_state.get("workspace_activity_marks", {}))
    flagged = bool(marks.pop(workspace_key, False))
    st.session_state.workspace_activity_marks = marks
    return flagged


def _set_processing_busy(kind: str, pending_action: Dict[str, Any]) -> None:
    st.session_state.processing_busy = True
    st.session_state.processing_kind = str(kind or "").strip().lower()
    st.session_state.processing_pending_action = dict(pending_action)
    active_workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if active_workspace_id:
        _save_processing_debug_payload(
            active_workspace_id,
            "busy_set",
            {
                "kind": st.session_state.processing_kind,
                "pending_action": dict(pending_action),
            },
        )


def _clear_processing_busy() -> None:
    st.session_state.processing_busy = False
    st.session_state.processing_kind = ""
    st.session_state.processing_pending_action = {}


def _role_chat_agent_for_state(state: AppState) -> RoleChatAgent:
    return RoleChatAgent(state.config.text_provider_name)


def _patch_manager_for_state(state: AppState) -> PatchManager:
    return PatchManager(state.config.text_provider_name)


def _append_blocked_response(
    state: AppState,
    request: RouterRequest,
    blocked: BlockedResponse,
) -> None:
    if request.action == "send_message" and request.user_text.strip() and request.role:
        _append_chat(
            state,
            role="user",
            content=request.user_text.strip(),
            companion_role=request.role,
            mode="user_message",
        )
    companion_role = request.role or _normalize_role_name(str(st.session_state.get("current_role", ""))) or state.role
    _append_chat(
        state,
        role="assistant",
        content=blocked.message,
        companion_role=companion_role,
        mode=f"blocked_{blocked.reason}",
    )
    _persist_state(state)


def _compose_role_payload(
    state: AppState,
    request: RouterRequest,
    *,
    mode: str,
    user_request: Optional[str] = None,
    checker_results: Optional[CheckerResultPayload] = None,
    tool_facts: Optional[List[ToolFact]] = None,
    system_exception: Optional[SystemExceptionPayload] = None,
    edit_stage: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> RolePayload:
    scope = request.scope or _confirmed_scope_payload(state) or ScopePayload(type="selected_span", text="", metadata={})
    route = str((metadata or {}).get("route", ""))
    normalized_tool_facts = list(tool_facts or [])
    tool_summary = None
    if normalized_tool_facts:
        tool_summary = ToolFactContractBuilder().build(
            role=state.role,
            tool_facts=normalized_tool_facts,
            source_context=route or "tool_facts",
        )
    role_input_summary = RoleInputSummaryBuilder().build(
        role=state.role,
        mode=mode,  # type: ignore[arg-type]
        checker_results=checker_results,
        tool_summary=tool_summary,
        system_exception=system_exception,
        source_context=route or mode,
    )
    policy = RoleResponsePolicyBuilder().build(
        role=state.role,
        mode=mode,
        route=route,
        patch_stage=str(state.pending_patch.stage or "idle"),
        has_patch_candidate=bool(str(state.pending_patch.patch_diff or "").strip() and state.pending_patch.operations),
        checker_results=checker_results,
    )
    return RolePayload(
        mode=mode,  # type: ignore[arg-type]
        role=state.role,
        scope=scope,
        user_request=request.user_text if user_request is None else str(user_request),
        checker_results=checker_results,
        tool_facts=normalized_tool_facts,
        tool_summary=tool_summary,
        system_exception=system_exception,
        role_input_summary=role_input_summary,
        response_policy=policy,
        memory=_memory_context_for_prompt(state, state.role),
        edit_stage=edit_stage,
        metadata=dict(metadata or {}),
    )


def _build_check_role_payload(
    state: AppState,
    request: RouterRequest,
    requested_checks: List[str],
    presentation: CheckPresentationArtifacts,
    metadata: Optional[Dict[str, Any]] = None,
) -> RolePayload:
    summary = CheckResponseContractBuilder().build(
        role=state.role,
        requested_checks=[x for x in requested_checks if x in CHECK_NAME_SET],
        issues=list(state.issues),
        rendered_issues=list(presentation.rendered_issues),
    )
    checker_results = CheckerResultPayload(
        requested_checks=[x for x in requested_checks if x in CHECK_NAME_SET],
        issues=list(state.issues),
        role_summary=summary,
        metadata={
            "issue_count": len(state.issues),
            **dict(metadata or {}),
        },
    )
    return _compose_role_payload(
        state,
        request,
        mode="check_response",
        checker_results=checker_results,
        edit_stage="analysis",
        metadata={"route": "checks"},
    )


def _normalize_role_payload_for_response(state: AppState, payload: RolePayload) -> RolePayload:
    """Ensure the runtime response path sees a fully composed role payload.

    Router may hand off a minimal payload for plain grounded chat or
    system_exception. Before invoking RoleChatAgent, normalize those payloads so
    response policy and unified role-facing input summaries are consistently
    available on the real call path.
    """
    if (
        payload.response_policy is not None
        and payload.memory
        and (
            payload.role_input_summary is not None
            or (
                payload.checker_results is None
                and payload.system_exception is None
                and not payload.tool_facts
            )
        )
    ):
        return payload

    request = RouterRequest(
        action="send_message",
        role=payload.role,
        user_text=payload.user_request,
        document_text=str(state.current_text or ""),
        scope=payload.scope,
        memory=payload.memory or _memory_context_for_prompt(state, payload.role),
    )
    return _compose_role_payload(
        state,
        request,
        mode=payload.mode,
        user_request=payload.user_request,
        checker_results=payload.checker_results,
        tool_facts=payload.tool_facts,
        system_exception=payload.system_exception,
        edit_stage=payload.edit_stage,
        metadata=payload.metadata,
    )


def _run_role_payload_response(state: AppState, payload: RolePayload) -> str:
    payload = _normalize_role_payload_for_response(state, payload)
    agent = _role_chat_agent_for_state(state)
    return agent.respond(payload, request_id=str(state.request_id or ""))


def _ingest_check_artifacts_from_execution(
    state: AppState,
    execution_artifacts: CheckExecutionArtifacts,
) -> tuple[bool, CheckPresentationArtifacts]:
    presentation = CheckArtifactBuilder(state.config.text_provider_name).build(state, execution_artifacts)
    presentation_meta = dict(presentation.metadata or {})

    state.report_md = str(presentation.report_md or "")
    patch_diff = str(presentation.patch_diff or "").strip()
    patch_ops = list(presentation.patch_operations or [])

    if state.role != "Editor":
        _clear_pending_patch_payload(state)
        return False, presentation

    if state.negotiation_state.active:
        if not state.pending_patch.patch_diff and patch_diff:
            state.pending_patch.patch_diff = patch_diff
        if not state.pending_patch.operations and patch_ops:
            state.pending_patch.operations = [op.to_dict() for op in patch_ops]
        if not state.pending_patch.source:
            state.pending_patch.source = "consistency_analysis"
        if not state.pending_patch.summary:
            state.pending_patch.summary = "Editor patch candidate generated from consistency checks."
        _sync_patch_preview_compat(state)
        return bool(state.pending_patch.patch_diff.strip() and state.pending_patch.operations), presentation

    if patch_diff and patch_ops:
        state.pending_patch.status = "pending"
        state.pending_patch.patch_diff = patch_diff
        state.pending_patch.operations = [dict(op) for op in patch_ops]
        state.pending_patch.reason = "consistency_analysis"
        state.pending_patch.source = "consistency_analysis"
        state.pending_patch.summary = "Editor patch candidate generated from consistency checks."
        guardrail = presentation_meta.get("guardrail", {})
        if isinstance(guardrail, dict):
            try:
                state.pending_patch.edit_ratio = float(guardrail.get("edit_ratio", 0.0))
            except (TypeError, ValueError):
                state.pending_patch.edit_ratio = 0.0
        _sync_patch_preview_compat(state)
        return True, presentation

    _clear_pending_patch_payload(state)
    return False, presentation


def _finalize_check_followup_gates(state: AppState, *, has_patch_candidate: bool) -> None:
    issue_count = len(state.issues)
    st.session_state.advisor_waiting_suggestion_consent = False
    _set_patch_lifecycle_stage(state, "idle")
    if issue_count == 0:
        return
    if state.role == "Advisor":
        st.session_state.advisor_waiting_suggestion_consent = True
    elif state.role == "Editor":
        if has_patch_candidate and not state.negotiation_state.active:
            _set_patch_lifecycle_stage(state, "awaiting_preview_consent")


def _enforce_check_followup_message(state: AppState, answer: str) -> str:
    normalized = answer.strip()
    if state.role == "Advisor" and st.session_state.get("advisor_waiting_suggestion_consent", False):
        prompt = "Would you like targeted revision suggestions for these issues?"
        return normalized if prompt.lower() in normalized.lower() else (normalized + "\n\n" + prompt).strip()
    if state.role == "Editor" and str(state.pending_patch.stage or "") == "awaiting_preview_consent":
        prompt = "Do you want me to show and prepare a patch diff now?"
        return normalized if prompt.lower() in normalized.lower() else (normalized + "\n\n" + prompt).strip()
    return normalized


def _run_requested_checks_via_router(
    state: AppState,
    request: RouterRequest,
    requested_checks: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    _reset_turn_gates(state)
    normalized_selection = request.scope.text.strip() if request.scope else ""
    if not normalized_selection:
        return

    analysis_doc = _analysis_document_text_for_scope(state, request.scope)
    analysis_document_text = str(analysis_doc.get("document_text", state.current_text) or "")
    check_document_text = str(analysis_doc.get("check_document_text", analysis_document_text) or "")
    availability_selection_text = (
        check_document_text
        if bool(analysis_doc.get("expanded_main", False))
        else normalized_selection
    )
    availability = _compute_check_availability(
        selection_text=availability_selection_text,
        workspace_resources=request.workspace_resources,
        document_text=check_document_text,
    )
    enabled_map = _normalize_check_enabled_map(availability.get("enabled"))
    effective_checks = [name for name in requested_checks if enabled_map.get(name, False)]
    blocked_checks = [
        {"check": name, "reason": str(availability.get("reasons", {}).get(name, ""))}
        for name in requested_checks
        if not enabled_map.get(name, False)
    ]
    if not effective_checks:
        summary_reasons = [
            CHECK_REASON_MESSAGES.get(item["reason"], "the check is not applicable to the confirmed selection")
            for item in blocked_checks
            if str(item.get("reason", "")).strip()
        ]
        message = (
            "The requested consistency checks cannot run on the current confirmed selection. "
            + (" ".join(summary_reasons[:3]) if summary_reasons else "")
        ).strip()
        system_exception = SystemExceptionPayload(
            code="invalid_scope",
            message=message or "The requested consistency checks cannot run on the current confirmed selection.",
            target_type="selection",
            details={
                "requested_checks": list(requested_checks),
                "blocked_checks": blocked_checks,
                "analysis_source": str(analysis_doc.get("source", "active_file")),
            },
        )
        exception_payload = _compose_role_payload(
            state,
            request,
            mode="system_exception",
            system_exception=system_exception,
            metadata={
                "route": "check_preflight",
                "blocked_checks": blocked_checks,
            },
        )
        _run_system_exception_response(state, request, exception_payload)
        return

    selection_span = None
    if state.active_selection and state.active_selection.end > state.active_selection.start:
        selection_span = (state.active_selection.start, state.active_selection.end)

    project_root = Path(str(st.session_state.get("project_root", Path.cwd())))
    grounding = _hydrate_assets_from_selection(
        state=state,
        selection_text=normalized_selection,
        project_root=project_root,
        selection_span=selection_span,
        full_latex_text=check_document_text,
    )
    grounding["analysis_source"] = str(analysis_doc.get("source", "active_file"))
    grounding["analysis_check_source"] = str(
        analysis_doc.get("check_source", analysis_doc.get("source", "active_file"))
    )
    grounding["analysis_expanded_main"] = bool(analysis_doc.get("expanded_main", False))
    grounding["analysis_expanded_active_file"] = bool(analysis_doc.get("expanded_active_file", False))
    grounding["analysis_visited_files"] = list(analysis_doc.get("visited_files", []))
    grounding["blocked_checks"] = blocked_checks
    grounding["effective_checks"] = list(effective_checks)
    st.session_state.last_grounding = grounding
    state.grounded_context.metadata["memory_context"] = _build_role_memory_context(state, state.role)

    orchestrator = _set_consistency_orchestrator(state)
    run_state = copy.deepcopy(state)
    run_state.current_text = analysis_document_text
    if bool(analysis_doc.get("expanded_main", False)):
        run_state.current_text = check_document_text
        run_state.active_selection = SelectionContext(
            start=0,
            end=max(0, len(check_document_text)),
            snippet=normalized_selection,
            sentence_indices=[],
            metadata={
                "confirmed": True,
                "source": str(analysis_doc.get("check_source", "expanded_main_latex")),
            },
        )
    run_state, execution_artifacts = orchestrator.run_checks(
        run_state,
        enabled_checks=set(effective_checks),
    )
    state.request_id = str(run_state.request_id or state.request_id)
    state.issues = list(run_state.issues)
    st.session_state.app_state = state
    st.session_state.issues_json = _issues_json_payload(state)
    has_patch_candidate, presentation = _ingest_check_artifacts_from_execution(state, execution_artifacts)
    if bool(analysis_doc.get("expanded_main", False)) and state.role == "Editor":
        # Patch operations are scoped to one editable file; expanded-main analysis can span multiple files.
        _clear_pending_patch_payload(state)
        has_patch_candidate = False
    _finalize_check_followup_gates(state, has_patch_candidate=has_patch_candidate)

    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "analysis_source": str(analysis_doc.get("source", "active_file")),
            "analysis_check_source": str(
                analysis_doc.get("check_source", analysis_doc.get("source", "active_file"))
            ),
            "requested_checks": list(requested_checks),
            "effective_checks": list(effective_checks),
            "blocked_checks": blocked_checks,
        }
    )
    payload = _build_check_role_payload(state, request, effective_checks, presentation, metadata=payload_metadata)
    answer = _enforce_check_followup_message(state, _run_role_payload_response(state, payload))
    skipped_note = _format_skipped_checks_note(blocked_checks)
    if skipped_note:
        answer = (answer.strip() + "\n\n" + skipped_note).strip()
    if request.action == "send_message" and request.user_text.strip():
        _append_chat(
            state,
            role="user",
            content=request.user_text.strip(),
            companion_role=state.role,
            mode="user_message",
        )
    _append_chat(
        state,
        role="assistant",
        content=answer,
        companion_role=state.role,
        mode="analysis",
    )
    _update_memory_after_turn(
        state=state,
        role=state.role,
        user_text=request.user_text.strip() or normalized_selection,
        assistant_text=answer,
        mode="analysis",
    )
    _persist_state(state)


def _run_system_exception_response(
    state: AppState,
    request: RouterRequest,
    payload: RolePayload,
) -> None:
    if request.action == "send_message" and request.user_text.strip():
        _append_chat(
            state,
            role="user",
            content=request.user_text.strip(),
            companion_role=state.role,
            mode="user_message",
        )
    answer = _run_role_payload_response(state, payload)
    _append_chat(
        state,
        role="assistant",
        content=answer,
        companion_role=state.role,
        mode="system_exception",
    )
    _update_memory_after_turn(
        state=state,
        role=state.role,
        user_text=request.user_text.strip(),
        assistant_text=answer,
        mode="system_exception",
    )
    _persist_state(state)


def _build_scope_grounding_tool_facts(state: AppState, request: RouterRequest) -> List[ToolFact]:
    scope_text = str(request.scope.text if request.scope is not None else "").strip()
    if not scope_text:
        return []
    full_text = str(request.document_text or state.current_text or "")
    facts: List[ToolFact] = []

    selection_for_refs = scope_text
    if request.scope is not None and bool((request.scope.metadata or {}).get("expanded_main", False)):
        if not REF_OR_CITE_MARKER_PATTERN.search(selection_for_refs):
            selection_for_refs = full_text

    resolved: Dict[str, Any] = {}
    try:
        resolver = st.session_state.get("resolver")
        if resolver is not None:
            resolved = resolver.resolve(selection_text=selection_for_refs, full_latex_text=full_text)
    except Exception:
        resolved = {}

    refs = dict(resolved.get("refs") or {})
    table_labels = [str(x).strip() for x in list(refs.get("table") or []) if str(x).strip()]
    figure_labels = [str(x).strip() for x in list(refs.get("figure") or []) if str(x).strip()]
    missing_labels = [str(x).strip() for x in list(resolved.get("missing_labels") or []) if str(x).strip()]
    if table_labels or figure_labels or missing_labels:
        facts.append(
            ToolFact(
                kind="latex_refs",
                source="latex_ref_resolver",
                summary=(
                    f"Resolved refs in scope: tables={len(table_labels)}, figures={len(figure_labels)}, "
                    f"missing_labels={len(missing_labels)}."
                ),
                data={
                    "table_labels": table_labels[:12],
                    "figure_labels": figure_labels[:12],
                    "missing_labels": missing_labels[:12],
                },
            )
        )

    images = list(request.workspace_resources.get("images", [])) if isinstance(request.workspace_resources, dict) else []
    figure_entries = list(resolved.get("figures") or [])
    if figure_labels:
        requested_paths: List[str] = []
        for item in figure_entries:
            for raw_path in list(item.get("image_paths") or []):
                path = str(raw_path or "").strip()
                if path:
                    requested_paths.append(path)
        normalized_pool = {
            str(entry.get("path", "")).strip().lower()
            for entry in images
            if str(entry.get("path", "")).strip()
        }
        matched = 0
        for raw in requested_paths:
            stem = Path(raw).stem.lower()
            if any(stem and stem in path for path in normalized_pool):
                matched += 1
        facts.append(
            ToolFact(
                kind="figure_assets",
                source="workspace_resources",
                summary=(
                    f"Figure grounding: requested_paths={len(requested_paths)}, matched_workspace_images={matched}, "
                    f"available_images={len(images)}."
                ),
                data={
                    "requested_image_paths": requested_paths[:12],
                    "available_image_paths": [str(item.get("path", "")) for item in images[:12]],
                },
            )
        )

    cite_blocks = extract_cite_keys(selection_for_refs)
    cite_keys: List[str] = []
    seen = set()
    for block in cite_blocks:
        for key in list(block.get("keys") or []):
            normalized = str(key).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cite_keys.append(normalized)
    if cite_keys:
        project_root = Path(str(st.session_state.get("project_root", Path.cwd())))
        bib = _load_bib_from_latex(full_text, project_root)
        allowlist = parse_bibtex_keys(bib.content)
        missing_keys = [key for key in cite_keys if key not in allowlist]
        facts.append(
            ToolFact(
                kind="citation_grounding",
                source=str((bib.meta or {}).get("source", "bib_loader")),
                summary=(
                    f"Citation grounding: cite_keys={len(cite_keys)}, missing_in_bib={len(missing_keys)}, "
                    f"bib_keys={len(allowlist)}."
                ),
                data={
                    "cite_keys": cite_keys[:20],
                    "missing_keys": missing_keys[:20],
                    "bib_source_paths": list((bib.meta or {}).get("paths") or [])[:8],
                },
            )
        )

    return facts


def _run_grounded_chat_response(
    state: AppState,
    request: RouterRequest,
    payload: RolePayload,
) -> None:
    response_payload = payload
    if (
        payload.mode == "grounded_chat"
        and not payload.tool_facts
        and request.scope is not None
        and str(request.scope.text or "").strip()
    ):
        inferred_facts = _build_scope_grounding_tool_facts(state, request)
        if inferred_facts:
            response_payload = _compose_role_payload(
                state,
                request,
                mode="grounded_chat",
                tool_facts=inferred_facts,
                metadata=dict(payload.metadata or {}),
            )
    _append_chat(
        state,
        role="user",
        content=request.user_text.strip(),
        companion_role=state.role,
        mode="user_message",
    )
    answer = _run_role_payload_response(state, response_payload)
    _append_chat(
        state,
        role="assistant",
        content=answer,
        companion_role=state.role,
        mode="grounded_chat",
    )
    _update_memory_after_turn(
        state=state,
        role=state.role,
        user_text=request.user_text.strip(),
        assistant_text=answer,
        mode="grounded_chat",
    )
    _persist_state(state)


def _run_image_related_chat_response(
    state: AppState,
    request: RouterRequest,
    payload: RolePayload,
    decision: RouterDecision,
) -> None:
    target_rel = str(decision.metadata.get("image_target", "")).strip()
    if not target_rel:
        _run_grounded_chat_response(state, request, payload)
        return

    image_tool = ImageQATool(state.config.vlm_provider_name)
    if not image_tool.enabled():
        system_exception = decision.system_exception or payload.system_exception or SystemExceptionPayload(
            code="tool_unavailable",
            message="image analysis is unavailable because the multimodal provider is not configured.",
            target=target_rel,
            target_type="image",
        )
        exception_payload = _compose_role_payload(
            state,
            request,
            mode="system_exception",
            system_exception=system_exception,
            metadata={"route": "image_related", "image_target": target_rel},
        )
        _run_system_exception_response(state, request, exception_payload)
        return

    try:
        image_path = _resolve_project_path(state.state_doc_id, target_rel)
    except Exception:
        exception_payload = _compose_role_payload(
            state,
            request,
            mode="system_exception",
            system_exception=SystemExceptionPayload(
                code="image_missing",
                message="the requested image file is not available in the current workspace path.",
                target=target_rel,
                target_type="image",
            ),
            metadata={"route": "image_related", "image_target": target_rel},
        )
        _run_system_exception_response(state, request, exception_payload)
        return
    tool_facts = image_tool.analyze(image_path, user_request=request.user_text)
    if not tool_facts:
        matched = [
            item
            for item in list(request.workspace_resources.get("images", []))
            if str(item.get("path", "")) == target_rel
        ]
        tool_facts = ImageQATool.summarize_available_images(matched) if matched else []
    grounded_payload = _compose_role_payload(
        state,
        request,
        mode="grounded_chat",
        tool_facts=tool_facts,
        metadata={"route": "image_related", "image_target": target_rel},
    )
    _run_grounded_chat_response(state, request, grounded_payload)


def _run_table_related_chat_response(
    state: AppState,
    request: RouterRequest,
    payload: RolePayload,
    decision: RouterDecision,
) -> None:
    table_target = str(decision.metadata.get("table_target", "")).strip()
    if not table_target:
        _run_grounded_chat_response(state, request, payload)
        return

    table_tool = TableQATool()
    document_text = str(request.document_text or state.current_text or "")
    tool_facts = table_tool.analyze(
        selection_text=str(request.scope.text if request.scope else ""),
        full_latex_text=document_text,
        target_label=table_target,
    )
    if not tool_facts:
        tool_facts = table_tool.summarize_available_tables(full_latex_text=document_text)
    grounded_payload = _compose_role_payload(
        state,
        request,
        mode="grounded_chat",
        tool_facts=tool_facts,
        metadata={"route": "table_related", "table_target": table_target},
    )
    _run_grounded_chat_response(state, request, grounded_payload)


def _handle_editor_patch_preview_consent(state: AppState, user_text: str, metadata: Dict[str, Any]) -> str:
    if bool(metadata.get("affirmative")):
        _set_patch_lifecycle_stage(state, "idle")
        if state.pending_patch.patch_diff.strip():
            _set_patch_lifecycle_stage(state, "awaiting_apply_consent")
            return (
                "I prepared the patch diff below.\n\n"
                + _format_patch_for_chat(state.pending_patch.patch_diff)
                + "\n\nDo you want me to apply this patch to the confirmed selection now?"
            )
        return "No patch is available from the latest analysis."
    if bool(metadata.get("negative")):
        _set_patch_lifecycle_stage(state, "idle")
        return "Understood. I will not prepare an edit patch now."
    return "Please reply with a clear yes or no so I know whether to prepare the patch diff."


def _handle_editor_patch_apply_consent(state: AppState, user_text: str, metadata: Dict[str, Any]) -> str:
    if bool(metadata.get("affirmative")):
        if _confirmed_scope_payload(state) is None:
            _set_patch_lifecycle_stage(state, "idle")
            return (
                "The confirmed selection is no longer valid because the text changed. "
                "Please confirm the target span again before applying a patch."
            )
        message = _apply_patch_with_history(state)
        _sync_patch_gate_flags(state)
        return (
            f"{message} The confirmed selection has been updated. "
            "If you want to revert it, use Undo."
        )
    if bool(metadata.get("negative")):
        _set_patch_lifecycle_stage(state, "idle")
        return "Patch was not applied."
    return "Please reply with a clear yes or no so I know whether to apply the prepared patch."


def _handle_patch_negotiation_feedback(state: AppState, user_text: str) -> str:
    if state.role != "Editor":
        _append_chat(state, role="user", content=user_text.strip(), companion_role=state.role, mode="user_message")
        message = "Patch negotiation is only available in Editor mode."
        _append_chat(state, role="assistant", content=message, companion_role=state.role, mode="editor_patch_negotiation")
        return message
    if not state.negotiation_state.active:
        _append_chat(state, role="user", content=user_text.strip(), companion_role=state.role, mode="user_message")
        message = "Patch negotiation is no longer active."
        _append_chat(state, role="assistant", content=message, companion_role=state.role, mode="editor_patch_negotiation")
        return message
    if state.pending_patch.status != "pending":
        _append_chat(state, role="user", content=user_text.strip(), companion_role=state.role, mode="user_message")
        message = "No pending patch is available for negotiation."
        _append_chat(state, role="assistant", content=message, companion_role=state.role, mode="editor_patch_negotiation")
        return message
    if not state.pending_patch.operations:
        _append_chat(state, role="user", content=user_text.strip(), companion_role=state.role, mode="user_message")
        message = "The pending patch is no longer valid. Please generate it again."
        _append_chat(state, role="assistant", content=message, companion_role=state.role, mode="editor_patch_negotiation")
        return message

    state, reply = _patch_manager_for_state(state).negotiate_patch(state, user_text)
    st.session_state.app_state = state
    _sync_patch_preview_compat(state)
    _sync_patch_gate_flags(state)

    if (
        str(state.pending_patch.stage or "") == "awaiting_apply_consent"
        and str(state.pending_patch.patch_diff or "").strip()
    ):
        return (
            reply
            + "\n\n"
            + _format_patch_for_chat(state.pending_patch.patch_diff)
            + "\n\nDo you want me to apply this patch to the confirmed selection now?"
        )
    return reply


def _handle_advisor_suggestion_consent(state: AppState, user_text: str, metadata: Dict[str, Any]) -> str:
    if bool(metadata.get("affirmative")):
        st.session_state.advisor_waiting_suggestion_consent = False
        return _format_advisor_suggestions(state)
    if bool(metadata.get("negative")):
        st.session_state.advisor_waiting_suggestion_consent = False
        return "Understood. I will keep the report only."
    return "Please reply with a clear yes or no if you want targeted revision suggestions."


def _run_blocked_precondition_response(
    state: AppState,
    request: RouterRequest,
    blocked: BlockedResponse,
) -> None:
    # Keep `no_role` / `busy` as hard precondition messages.
    # For role-bound scope/doc errors, route through role agent so reminders
    # follow the active persona style.
    if (not str(request.role or "").strip()) or blocked.reason in {"no_role", "busy"}:
        _append_blocked_response(state, request, blocked)
        return
    if blocked.reason not in {
        "no_document",
        "no_confirmed_selection",
        "invalid_scope",
        "module_inapplicable",
        "missing_bib",
    }:
        _append_blocked_response(state, request, blocked)
        return

    if blocked.reason == "no_document":
        target_type = "document"
    elif blocked.reason == "missing_bib":
        target_type = "resource"
    else:
        target_type = "selection"
    system_exception = SystemExceptionPayload(
        code="invalid_scope",
        message=str(blocked.message or "").strip() or "The request cannot run because required scope preconditions are missing.",
        target_type=target_type,
        details={
            "blocked_reason": blocked.reason,
            "required_scope": str(blocked.required_scope or ""),
            "action": str(request.action or ""),
        },
    )
    payload = _compose_role_payload(
        state,
        request,
        mode="system_exception",
        system_exception=system_exception,
        metadata={
            "route": "blocked_precondition",
            "blocked_reason": blocked.reason,
        },
    )
    _run_system_exception_response(state, request, payload)


def _execute_router_decision(state: AppState, request: RouterRequest, decision: RouterDecision) -> None:
    if decision.kind == "BLOCKED_PRECONDITION" and decision.blocked_response is not None:
        _run_blocked_precondition_response(state, request, decision.blocked_response)
        return

    if decision.kind == "UNDO":
        message = _undo_last_change(state)
        if message:
            _append_chat(
                state,
                role="assistant",
                content=message,
                companion_role=state.role,
                mode="undo",
            )
            _persist_state(state)
        return

    if decision.kind in {"RUN_CHECKS_UI", "RUN_CHECKS_NL"}:
        _run_requested_checks_via_router(state, request, decision.requested_checks, metadata=decision.metadata)
        return

    if decision.kind == "SYSTEM_EXCEPTION" and decision.role_payload is not None:
        _run_system_exception_response(state, request, decision.role_payload)
        return

    if decision.kind == "PATCH_PREVIEW_CONSENT":
        _append_chat(
            state,
            role="user",
            content=request.user_text.strip(),
            companion_role=state.role,
            mode="user_message",
        )
        answer = _handle_editor_patch_preview_consent(state, request.user_text, decision.metadata)
        _append_chat(
            state,
            role="assistant",
            content=answer,
            companion_role=state.role,
            mode="editor_patch_consent",
        )
        _persist_state(state)
        return

    if decision.kind == "PATCH_NEGOTIATION":
        answer = _handle_patch_negotiation_feedback(state, request.user_text)
        _persist_state(state)
        return

    if decision.kind == "PATCH_APPLY_CONSENT":
        _append_chat(
            state,
            role="user",
            content=request.user_text.strip(),
            companion_role=state.role,
            mode="user_message",
        )
        answer = _handle_editor_patch_apply_consent(state, request.user_text, decision.metadata)
        _append_chat(
            state,
            role="assistant",
            content=answer,
            companion_role=state.role,
            mode="editor_patch_apply",
        )
        _persist_state(state)
        return

    if decision.kind == "ADVISOR_SUGGESTION_CONSENT":
        _append_chat(
            state,
            role="user",
            content=request.user_text.strip(),
            companion_role=state.role,
            mode="user_message",
        )
        answer = _handle_advisor_suggestion_consent(state, request.user_text, decision.metadata)
        _append_chat(
            state,
            role="assistant",
            content=answer,
            companion_role=state.role,
            mode="advisor_suggestions",
        )
        _persist_state(state)
        return

    if decision.kind == "CHAT" and decision.role_payload is not None:
        if decision.metadata.get("route") == "editor_rewrite" and state.role == "Editor":
            _append_chat(
                state,
                role="user",
                content=request.user_text.strip(),
                companion_role=state.role,
                mode="user_message",
            )
            state, answer = _patch_manager_for_state(state).propose_writing_patch(
                state,
                request.user_text.strip(),
            )
            st.session_state.app_state = state
            _sync_patch_preview_compat(state)
            _sync_patch_gate_flags(state)
            if state.pending_patch.patch_diff.strip():
                answer = (
                    answer
                    + "\n\n"
                    + _format_patch_for_chat(state.pending_patch.patch_diff)
                    + "\n\nDo you want me to apply this patch to the confirmed selection now?"
                )
            _append_chat(
                state,
                role="assistant",
                content=answer,
                companion_role=state.role,
                mode="writing_patch",
            )
            _update_memory_after_turn(
                state=state,
                role=state.role,
                user_text=request.user_text.strip(),
                assistant_text=answer,
                mode="writing_patch",
            )
            _persist_state(state)
            return

        if decision.metadata.get("route") == "image_related":
            _run_image_related_chat_response(state, request, decision.role_payload, decision)
            return

        if decision.metadata.get("route") == "table_related":
            _run_table_related_chat_response(state, request, decision.role_payload, decision)
            return

        _run_grounded_chat_response(state, request, decision.role_payload)


def _decide_router_pending_action(
    state: AppState,
    pending_action: Dict[str, Any],
) -> Tuple[RouterRequest, RouterDecision]:
    request = _build_router_request(state, pending_action)
    decision = st.session_state.router_v1.decide(request)
    return request, decision


def _dispatch_router_pending_action(
    state: AppState,
    pending_action: Dict[str, Any],
) -> Tuple[RouterRequest, RouterDecision]:
    request, decision = _decide_router_pending_action(state, pending_action)
    _execute_router_decision(state, request, decision)
    return request, decision


def _drain_processing_pending(state: AppState) -> None:
    if not bool(st.session_state.get("processing_busy", False)):
        return
    pending = st.session_state.get("processing_pending_action", {})
    if not isinstance(pending, dict) or not pending:
        _clear_processing_busy()
        return

    st.session_state.processing_pending_action = {}
    _save_processing_debug_payload(
        state.state_doc_id,
        "drain_before_dispatch",
        {
            "kind": str(st.session_state.get("processing_kind", "")).strip(),
            "pending_action": dict(pending),
        },
    )
    try:
        _dispatch_router_pending_action(state, pending)
        _save_processing_debug_payload(
            state.state_doc_id,
            "drain_after_dispatch",
            {
                "kind": str(st.session_state.get("processing_kind", "")).strip(),
                "pending_action": dict(pending),
            },
        )
    except Exception as exc:
        _save_processing_debug_payload(
            state.state_doc_id,
            "drain_exception",
            {
                "kind": str(st.session_state.get("processing_kind", "")).strip(),
                "pending_action": dict(pending),
                "error": repr(exc),
            },
        )
        raise
    finally:
        _clear_processing_busy()

    st.rerun()


def _reset_project(state: AppState) -> None:
    state.current_text = ""
    state.chat_history = []
    state.issues = []
    state.patch_preview = ""
    state.report_md = ""
    state.history = []
    state.uploaded_tables = []
    state.uploaded_figures = []
    state.uploaded_bib = UploadedBib()
    state.active_selection = None
    state.pending_patch.status = "none"
    state.pending_patch.stage = "idle"
    state.pending_patch.patch_diff = ""
    state.pending_patch.operations = []
    state.pending_patch.reason = ""
    state.pending_patch.edit_ratio = 0.0
    state.pending_patch.source = ""
    state.pending_patch.summary = ""
    state.negotiation_state.active = False
    state.negotiation_state.latest_agent_message = ""

    st.session_state.selection_text = ""
    st.session_state.selection_start = -1
    st.session_state.selection_end = -1
    st.session_state.pending_selection_text = ""
    st.session_state.pending_selection_start = -1
    st.session_state.pending_selection_end = -1
    st.session_state.confirmed_selection_text = ""
    st.session_state.confirmed_selection_start = -1
    st.session_state.confirmed_selection_end = -1
    st.session_state.active_selection = ""
    st.session_state.issues_json = []
    st.session_state.last_grounding = {}
    st.session_state.pdf_preview_png_hash = ""
    st.session_state.pdf_preview_png_bytes = b""
    st.session_state.pdf_preview_png_pages_hash = ""
    st.session_state.pdf_preview_png_pages_bytes = []
    st.session_state.pdf_compiled_extracted_text_hash = ""
    st.session_state.pdf_compiled_extracted_text = ""
    st.session_state.compile_in_progress = False
    st.session_state.resource_preview_visible = False
    st.session_state.resource_preview_path = ""
    st.session_state.resource_preview_doc_id = ""
    _clear_processing_busy()
    _reset_turn_gates(state)
    _set_consistency_orchestrator(state)


def _workspace_ui_defaults() -> Dict[str, Any]:
    return {
        "selection_text": "",
        "selection_start": -1,
        "selection_end": -1,
        "pending_selection_text": "",
        "pending_selection_start": -1,
        "pending_selection_end": -1,
        "suppress_next_selection_payload": False,
        "last_component_action_fingerprint": "",
        "confirmed_selection_text": "",
        "confirmed_selection_start": -1,
        "confirmed_selection_end": -1,
        "active_selection": "",
        "issues_json": [],
        "last_grounding": {},
        "editor_waiting_patch_consent": False,
        "editor_waiting_apply_consent": False,
        "advisor_waiting_suggestion_consent": False,
        "selected_checks": list(CHECK_NAMES),
        "selected_checks_explicit": False,
        "project_root": str(Path.cwd()),
        "active_file_path": "main.tex",
        "files_selected_path": "main.tex",
        "files_expanded_dirs": [],
        "composer_text": "",
        "composer_triggered": False,
        "composer_tools_open": False,
        "composer_last_event_id": 0,
        "agent_chat_history_open": False,
        "agent_chat_delete_confirm": {},
        "current_role": "",
        "selected_role": "",
        "pdf_compiled_text_hash": "",
        "pdf_compiled_base64": "",
        "pdf_compiled_bytes": b"",
        "pdf_compile_ok": False,
        "pdf_compile_log": "",
        "pdf_last_export_path": "",
        "pdf_compiled_pdf_path": "",
        "pdf_compiled_synctex_path": "",
        "pdf_compiled_tex_path": "",
        "pdf_compile_errors": [],
        "pdf_compiled_extracted_text_hash": "",
        "pdf_compiled_extracted_text": "",
        "pdf_preview_png_hash": "",
        "pdf_preview_png_bytes": b"",
        "pdf_preview_png_pages_hash": "",
        "pdf_preview_png_pages_bytes": [],
        "compile_in_progress": False,
    }


def _capture_workspace_ui() -> Dict[str, Any]:
    defaults = _workspace_ui_defaults()
    payload: Dict[str, Any] = {}
    for key, default in defaults.items():
        payload[key] = copy.deepcopy(st.session_state.get(key, default))
    return payload


def _sync_workspace_ui_to_store(workspace_id: str) -> None:
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return
    payload["ui"] = _capture_workspace_ui()
    state = st.session_state.get("app_state")
    if isinstance(state, AppState):
        payload["state"] = copy.deepcopy(state)


def _restore_workspace_ui(payload: Dict[str, Any]) -> None:
    defaults = _workspace_ui_defaults()
    for key, default in defaults.items():
        st.session_state[key] = copy.deepcopy(payload.get(key, default))
    st.session_state.project_root_input = st.session_state.project_root
    try:
        normalized_active_file = _normalize_project_relative_path(
            st.session_state.get("active_file_path", "main.tex")
        )
    except ValueError:
        normalized_active_file = "main.tex"
    if _is_internal_state_relative_path(normalized_active_file) or not _is_text_editable_project_file(
        normalized_active_file
    ):
        normalized_active_file = "main.tex"
    st.session_state.active_file_path = normalized_active_file
    if not st.session_state.get("files_selected_path"):
        st.session_state.files_selected_path = normalized_active_file


def _create_state_from_project(doc_id: str, role: str = "", active_file_path: str = "main.tex") -> AppState:
    normalized_active_file = _normalize_project_relative_path(active_file_path)
    try:
        initial_text = load_project_text_file(doc_id, normalized_active_file)
    except ValueError:
        normalized_active_file = "main.tex"
        initial_text = load_main_tex(doc_id)
    state = AppState(current_text=initial_text)
    state.config = load_config_from_env(state.config)
    state.state_doc_id = doc_id
    normalized_role = _normalize_role_name(role)
    if normalized_role:
        state.role = normalized_role
    conversation_payload = load_conversation(doc_id)
    state.chat_history = _payload_to_chat_turns(conversation_payload)
    return state


def _workspace_payload_from_disk(doc_id: str) -> Dict[str, Any]:
    metadata = load_metadata(doc_id)
    role_from_meta = _normalize_role_name(str(metadata.get("current_role", "")))
    active_file_path = _normalize_project_relative_path(str(metadata.get("active_file_path", "main.tex")))
    if _is_internal_state_relative_path(active_file_path) or not _is_text_editable_project_file(active_file_path):
        active_file_path = "main.tex"
    try:
        active_path_obj = _resolve_project_path(doc_id, active_file_path)
    except ValueError:
        active_path_obj = _resolve_project_path(doc_id, "main.tex")
        active_file_path = "main.tex"
    if active_path_obj.exists() and active_path_obj.is_dir():
        active_file_path = "main.tex"
    memory = load_memory(doc_id)
    state = _create_state_from_project(doc_id, role=role_from_meta, active_file_path=active_file_path)
    agent_chats = load_agent_chats(doc_id)
    active_chat_id = str(agent_chats.get("active_chat_id", "")).strip()
    active_chat_payload = (
        agent_chats.get("chats", {}).get(active_chat_id, {})
        if isinstance(agent_chats.get("chats"), dict)
        else {}
    )
    state.chat_history = _payload_to_chat_turns(active_chat_payload.get("conversation", []))

    ui = _workspace_ui_defaults()
    ui["current_role"] = role_from_meta
    ui["selected_role"] = role_from_meta
    ui["project_root"] = str(metadata.get("project_root", str(Path.cwd())))
    ui["active_file_path"] = active_file_path
    ui["files_selected_path"] = active_file_path

    name = str(metadata.get("title", "")).strip()
    if not name:
        existing_payload = st.session_state.get("workspace_store", {}).get(doc_id, {})
        existing_name = (
            str(existing_payload.get("name", "")).strip()
            if isinstance(existing_payload, dict)
            else ""
        )
        name = existing_name or _next_conversation_title()
        metadata["title"] = name
        save_metadata(doc_id, metadata, touch_updated_at=False)
    return {
        "name": name,
        "state": copy.deepcopy(state),
        "ui": ui,
        "memory": memory,
        "agent_chats": agent_chats,
        "metadata": metadata,
    }


def _refresh_memory_from_state(
    memory: Dict[str, Any],
    state: AppState,
    role: str,
    active_file_path: str = "main.tex",
) -> Dict[str, Any]:
    normalized_role = _normalize_role_name(role)
    merged = _normalize_memory_payload(memory, current_role=normalized_role)
    shared = merged["shared_memory"]
    shared["active_file"] = _normalize_project_relative_path(active_file_path)
    if state.active_selection and str(state.active_selection.snippet).strip():
        shared["current_focus"] = str(state.active_selection.snippet).strip()[:1200]
    elif state.current_text.strip():
        shared["current_focus"] = state.current_text.strip()[:1200]
    if state.issues:
        shared["confirmed_issues"] = [str(issue.message) for issue in state.issues[:12] if str(issue.message).strip()]
    role_memory = merged["role_memories"].get(normalized_role) if normalized_role else None
    if isinstance(role_memory, dict):
        role_memory["open_items"] = _normalized_memory_list(
            [issue.message for issue in state.issues],
            item_limit=16,
            item_max_chars=360,
        )
        last_assistant = ""
        for turn in reversed(state.chat_history):
            if turn.role != "assistant":
                continue
            turn_role = _normalize_role_name(
                str(turn.metadata.get("agent_role") or turn.metadata.get("companion_role") or "")
            )
            if not normalized_role or turn_role == normalized_role:
                last_assistant = str(turn.content).strip()
                break
        if last_assistant:
            role_memory["working_summary"] = last_assistant[:1500]
    merged["meta"]["current_role"] = normalized_role
    merged["meta"]["last_updated"] = _utc_now_iso()
    return merged


def _workspace_memory_payload(workspace_id: str) -> Dict[str, Any]:
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id, {})
    disk_memory = load_memory(workspace_id)
    cached_memory = payload.get("memory") if isinstance(payload, dict) else None
    if isinstance(cached_memory, dict):
        normalized_cached = _normalize_memory_payload(cached_memory)
        cache_epoch = _memory_last_updated_epoch(normalized_cached)
        disk_epoch = _memory_last_updated_epoch(disk_memory)
        # Tie goes to disk so the file source-of-truth wins when timestamps match.
        chosen = normalized_cached if cache_epoch > disk_epoch else disk_memory
    else:
        chosen = disk_memory
    if isinstance(payload, dict):
        payload["memory"] = chosen
    return chosen


def _normalized_memory_list(values: Any, item_limit: int, item_max_chars: int) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if len(text) > item_max_chars:
            text = text[:item_max_chars] + "...<truncated>"
        normalized.append(text)
    return normalized[-item_limit:]


def _active_file_for_state(state: AppState) -> str:
    workspace_id = str(getattr(state, "state_doc_id", "")).strip()
    active_workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if workspace_id and workspace_id == active_workspace_id:
        try:
            normalized = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
            if _is_internal_state_relative_path(normalized) or not _is_text_editable_project_file(normalized):
                return "main.tex"
            return normalized
        except ValueError:
            return "main.tex"
    if workspace_id:
        payload = st.session_state.get("workspace_store", {}).get(workspace_id, {})
        if isinstance(payload, dict):
            ui_payload = payload.get("ui", {})
            if isinstance(ui_payload, dict):
                try:
                    normalized = _normalize_project_relative_path(ui_payload.get("active_file_path", "main.tex"))
                    if _is_internal_state_relative_path(normalized) or not _is_text_editable_project_file(normalized):
                        return "main.tex"
                    return normalized
                except ValueError:
                    return "main.tex"
    try:
        normalized = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
        if _is_internal_state_relative_path(normalized) or not _is_text_editable_project_file(normalized):
            return "main.tex"
        return normalized
    except ValueError:
        return "main.tex"


def _memory_last_updated_epoch(memory_payload: Any) -> float:
    if not isinstance(memory_payload, dict):
        return -1.0
    meta = memory_payload.get("meta", {})
    if not isinstance(meta, dict):
        return -1.0
    raw = str(meta.get("last_updated", "")).strip()
    if not raw:
        return -1.0
    normalized_raw = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(normalized_raw).timestamp()
    except ValueError:
        return -1.0


def _build_role_memory_context(state: AppState, role: str) -> Dict[str, Any]:
    normalized_role = _normalize_role_name(role)
    memory_payload = _workspace_memory_payload(state.state_doc_id)
    shared_memory = memory_payload.get("shared_memory", {})
    role_memories = memory_payload.get("role_memories", {})
    role_memory = role_memories.get(normalized_role, _default_role_memory())
    other_conclusions: Dict[str, List[str]] = {}
    for role_name in ["Reviewer", "Advisor", "Editor"]:
        if role_name == normalized_role:
            continue
        role_payload = role_memories.get(role_name, {})
        published = role_payload.get("published_conclusions", [])
        if isinstance(published, list) and published:
            other_conclusions[role_name] = [str(x) for x in published[-8:]]

    recent_history: List[Dict[str, Any]] = []
    for turn in state.chat_history[-14:]:
        agent_role = _normalize_role_name(
            str(turn.metadata.get("agent_role") or turn.metadata.get("companion_role") or "")
        )
        recent_history.append(
            {
                "role": turn.role,
                "agent_role": agent_role,
                "content": str(turn.content),
                "timestamp": str(turn.metadata.get("timestamp", "")),
            }
        )
    return {
        "shared_memory": shared_memory,
        "role_memory": role_memory,
        "other_roles_published_conclusions": other_conclusions,
        "recent_history": recent_history,
    }


def _clip_memory_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _memory_context_for_prompt(state: AppState, role: str) -> Dict[str, Any]:
    raw_context = _build_role_memory_context(state, role)
    shared_raw = raw_context.get("shared_memory", {}) if isinstance(raw_context, dict) else {}
    role_raw = raw_context.get("role_memory", {}) if isinstance(raw_context, dict) else {}
    other_raw = (
        raw_context.get("other_roles_published_conclusions", {})
        if isinstance(raw_context, dict)
        else {}
    )
    history_raw = raw_context.get("recent_history", []) if isinstance(raw_context, dict) else []

    shared_memory = {
        "task_summary": _clip_memory_text(shared_raw.get("task_summary", ""), 900),
        "current_focus": _clip_memory_text(shared_raw.get("current_focus", ""), 900),
        "confirmed_issues": [
            _clip_memory_text(item, 280)
            for item in (shared_raw.get("confirmed_issues", []) if isinstance(shared_raw, dict) else [])
        ][-12:],
        "confirmed_plans": [
            _clip_memory_text(item, 280)
            for item in (shared_raw.get("confirmed_plans", []) if isinstance(shared_raw, dict) else [])
        ][-12:],
        "active_file": _clip_memory_text(shared_raw.get("active_file", "main.tex"), 120),
        "last_compile_status": _clip_memory_text(shared_raw.get("last_compile_status", ""), 320),
    }

    role_memory = {
        "working_summary": _clip_memory_text(role_raw.get("working_summary", ""), 1200),
        "open_items": [
            _clip_memory_text(item, 320)
            for item in (role_raw.get("open_items", []) if isinstance(role_raw, dict) else [])
        ][-12:],
        "private_notes": [
            _clip_memory_text(item, 320)
            for item in (role_raw.get("private_notes", []) if isinstance(role_raw, dict) else [])
        ][-12:],
        "published_conclusions": [
            _clip_memory_text(item, 320)
            for item in (role_raw.get("published_conclusions", []) if isinstance(role_raw, dict) else [])
        ][-12:],
    }

    other_roles_published_conclusions: Dict[str, List[str]] = {}
    if isinstance(other_raw, dict):
        for role_name, conclusions in other_raw.items():
            if not isinstance(conclusions, list):
                continue
            other_roles_published_conclusions[str(role_name)] = [
                _clip_memory_text(item, 320) for item in conclusions[-8:]
            ]

    recent_history: List[Dict[str, Any]] = []
    if isinstance(history_raw, list):
        for turn in history_raw[-10:]:
            if not isinstance(turn, dict):
                continue
            recent_history.append(
                {
                    "role": str(turn.get("role", "")),
                    "agent_role": _normalize_role_name(str(turn.get("agent_role", ""))),
                    "content": _clip_memory_text(turn.get("content", ""), 320),
                    "timestamp": str(turn.get("timestamp", "")),
                }
            )

    return {
        "shared_memory": shared_memory,
        "role_memory": role_memory,
        "other_roles_published_conclusions": other_roles_published_conclusions,
        "recent_history": recent_history,
    }


def _memory_aware_role_chat_reply(state: AppState, user_text: str) -> str:
    pending_action = {
        "action": "send_message",
        "submit_text": str(user_text or "").strip(),
        "selected_checks": [],
        "ui_checks_explicit": False,
    }
    request, decision = _decide_router_pending_action(state, pending_action)
    if decision.kind == "BLOCKED_PRECONDITION" and decision.blocked_response is not None:
        return decision.blocked_response.message
    if decision.kind == "SYSTEM_EXCEPTION" and decision.role_payload is not None:
        return _run_role_payload_response(state, decision.role_payload)
    if decision.kind == "CHAT" and decision.role_payload is not None:
        return _run_role_payload_response(state, decision.role_payload)
    if decision.kind in {"RUN_CHECKS_UI", "RUN_CHECKS_NL"}:
        return (
            "This compatibility helper no longer executes checks directly. "
            "Checks must be routed through the unified Router action."
        )
    return ""


def _update_memory_after_turn(
    state: AppState,
    role: str,
    user_text: str,
    assistant_text: str,
    mode: str,
) -> None:
    normalized_role = _normalize_role_name(role)
    if not normalized_role:
        return
    store = st.session_state.get("workspace_store", {})
    payload = store.get(state.state_doc_id)
    if not isinstance(payload, dict):
        return

    memory_payload = _normalize_memory_payload(payload.get("memory", {}), current_role=normalized_role)
    shared = memory_payload["shared_memory"]
    shared["task_summary"] = str(user_text).strip()[:1200]
    shared["current_focus"] = str(user_text).strip()[:1200]
    shared["active_file"] = _active_file_for_state(state)
    if state.issues:
        shared["confirmed_issues"] = [str(issue.message) for issue in state.issues[:12] if str(issue.message).strip()]

    role_memory = memory_payload["role_memories"][normalized_role]
    role_memory["working_summary"] = str(assistant_text).strip()[:1500]
    role_memory["open_items"] = _normalized_memory_list(
        [issue.message for issue in state.issues],
        item_limit=16,
        item_max_chars=360,
    )
    private_notes = _normalized_memory_list(
        role_memory.get("private_notes", []),
        item_limit=20,
        item_max_chars=420,
    )
    user_note = str(user_text).strip()
    if user_note:
        private_notes.append(f"{_utc_now_iso()} | {user_note[:320]}")
    role_memory["private_notes"] = _normalized_memory_list(
        private_notes,
        item_limit=20,
        item_max_chars=420,
    )
    if mode in {"analysis", "advisor_suggestions", "writing_patch", "editor_patch_applied"}:
        conclusions = [str(x) for x in role_memory.get("published_conclusions", []) if str(x).strip()]
        conclusions.append(str(assistant_text).strip()[:1000])
        role_memory["published_conclusions"] = conclusions[-20:]
    if mode in {"advisor_suggestions", "writing_patch", "editor_patch_applied"}:
        plans = _normalized_memory_list(
            shared.get("confirmed_plans", []),
            item_limit=20,
            item_max_chars=360,
        )
        plan_entry = str(assistant_text).strip()
        if plan_entry:
            plans.append(plan_entry[:360])
        shared["confirmed_plans"] = _normalized_memory_list(
            plans,
            item_limit=20,
            item_max_chars=360,
        )

    memory_payload["meta"]["current_role"] = normalized_role
    memory_payload["meta"]["last_updated"] = _utc_now_iso()
    payload["memory"] = memory_payload


def _persist_workspace_to_disk(workspace_id: str, touch_activity: bool = False) -> None:
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return
    if touch_activity:
        _mark_workspace_activity(workspace_id)
    active_workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if workspace_id == active_workspace_id:
        runtime_state = st.session_state.get("app_state")
        if isinstance(runtime_state, AppState):
            state = runtime_state
            payload["state"] = copy.deepcopy(runtime_state)
        else:
            state = payload.get("state")
    else:
        state = payload.get("state")
    if not isinstance(state, AppState):
        return

    ensure_project_dir(workspace_id)
    ui_payload = payload.get("ui", {}) if isinstance(payload.get("ui"), dict) else {}
    if workspace_id == active_workspace_id:
        active_source = st.session_state.get("active_file_path", "main.tex")
    else:
        active_source = ui_payload.get("active_file_path", "main.tex")
    try:
        active_file_path = _normalize_project_relative_path(active_source)
    except ValueError:
        active_file_path = "main.tex"
    if _is_internal_state_relative_path(active_file_path) or not _is_text_editable_project_file(active_file_path):
        active_file_path = "main.tex"
    save_project_text_file(workspace_id, active_file_path, state.current_text)

    role = _normalize_role_name(str(payload.get("ui", {}).get("current_role", "")))
    if role:
        state.role = role

    conversation_payload = _chat_turns_to_payload(state.chat_history, fallback_role=role)
    normalized_conversation = _normalize_agent_chat_conversation_payload(conversation_payload)
    agent_chats_payload = _normalize_agent_chats_payload(
        payload.get("agent_chats"),
        fallback_conversation=normalized_conversation,
    )
    chats_map = (
        dict(agent_chats_payload.get("chats", {}))
        if isinstance(agent_chats_payload.get("chats"), dict)
        else {}
    )
    active_chat_id = str(agent_chats_payload.get("active_chat_id", "")).strip()
    if active_chat_id not in chats_map:
        order_fallback = [str(item).strip() for item in list(agent_chats_payload.get("order", [])) if str(item).strip()]
        active_chat_id = order_fallback[0] if order_fallback else ""
    if active_chat_id not in chats_map and chats_map:
        active_chat_id = next(iter(chats_map.keys()))
    if active_chat_id in chats_map:
        active_chat_payload = dict(chats_map.get(active_chat_id, {}))
        existing_conversation = _normalize_agent_chat_conversation_payload(active_chat_payload.get("conversation", []))
        conversation_changed = existing_conversation != normalized_conversation
        active_chat_payload["conversation"] = normalized_conversation
        if touch_activity and conversation_changed:
            active_chat_payload["updated_at"] = _utc_now_iso()
        chats_map[active_chat_id] = active_chat_payload
        order = [str(item).strip() for item in list(agent_chats_payload.get("order", [])) if str(item).strip()]
        if touch_activity and conversation_changed:
            order = [active_chat_id] + [item for item in order if item != active_chat_id]
        agent_chats_payload = _normalize_agent_chats_payload(
            {
                "active_chat_id": active_chat_id,
                "order": order,
                "chats": chats_map,
            },
            fallback_conversation=normalized_conversation,
        )

    save_agent_chats(workspace_id, agent_chats_payload)
    save_conversation(workspace_id, normalized_conversation)

    memory_payload = _refresh_memory_from_state(
        payload.get("memory", _workspace_memory_defaults(current_role=role)),
        state=state,
        role=role,
        active_file_path=active_file_path,
    )
    save_memory(workspace_id, memory_payload)

    metadata = load_metadata(workspace_id)
    metadata["title"] = str(payload.get("name", metadata.get("title", "Conversation 1"))).strip() or "Conversation 1"
    metadata["current_role"] = role
    metadata["project_root"] = str(payload.get("ui", {}).get("project_root", str(Path.cwd())))
    metadata["active_file_path"] = active_file_path
    save_metadata(workspace_id, metadata, touch_updated_at=touch_activity)

    payload["memory"] = memory_payload
    payload["agent_chats"] = agent_chats_payload
    payload["metadata"] = metadata
    payload["name"] = metadata["title"]


def _force_save_active_editor_file(workspace_id: str) -> None:
    active_workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    state = st.session_state.get("app_state")
    if workspace_id != active_workspace_id or state is None:
        return

    try:
        active_file_path = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
    except ValueError:
        active_file_path = "main.tex"
    if _is_internal_state_relative_path(active_file_path) or not _is_text_editable_project_file(active_file_path):
        active_file_path = "main.tex"

    editor_component_key = f"interactive_latex_editor_{workspace_id}"
    latest_text = str(state.current_text or "")
    component_state = st.session_state.get(editor_component_key)
    if isinstance(component_state, dict):
        payload_matches_active = True
        try:
            payload_file_path = _normalize_project_relative_path(component_state.get("file_path", active_file_path))
        except ValueError:
            payload_matches_active = False
            payload_file_path = ""
        if payload_matches_active and payload_file_path != active_file_path:
            payload_matches_active = False
        if payload_matches_active:
            candidate_text = str(component_state.get("text", latest_text))
            candidate_trigger = str(component_state.get("trigger", "")).strip().lower()
            allow_clear_triggers = {
                "input",
                "confirm",
                "cancel",
                "confirm_edit",
                "preserve_selection",
                "invalidate_selection",
            }
            suspicious_empty_echo = (
                not candidate_text.strip()
                and bool(latest_text.strip())
                and candidate_trigger not in allow_clear_triggers
            )
            if (
                (not suspicious_empty_echo)
                and _should_accept_editor_component_text(
                    state,
                    candidate_text,
                    candidate_trigger=candidate_trigger,
                )
            ):
                latest_text = candidate_text

    save_project_text_file(workspace_id, active_file_path, latest_text)
    state.current_text = latest_text
    hashes = dict(st.session_state.get("workspace_text_hashes", {}))
    hashes[f"{workspace_id}:{active_file_path}"] = hashlib.sha256(latest_text.encode("utf-8")).hexdigest()
    st.session_state.workspace_text_hashes = hashes


def _editor_backend_sync_key(doc_id: str) -> str:
    normalized = str(doc_id or "default").strip() or "default"
    return f"interactive_editor_backend_sync_{normalized}"


def _arm_editor_backend_text_sync(state: AppState) -> None:
    doc_id = str(state.state_doc_id or st.session_state.get("active_workspace_id", "default")).strip() or "default"
    expected_text = str(state.current_text or "")
    st.session_state[_editor_backend_sync_key(doc_id)] = expected_text
    previous_event_id = int(st.session_state.get("editor_backend_sync_event_id", 0) or 0)
    st.session_state.editor_backend_sync_event_id = max(int(time.time() * 1000), previous_event_id + 1)


def _adopt_active_editor_component_text_for_compile(state: AppState, editor_component_key: str) -> None:
    component_state = st.session_state.get(editor_component_key)
    if not isinstance(component_state, dict):
        return

    try:
        active_file_path = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
    except ValueError:
        active_file_path = "main.tex"

    payload_matches_active = True
    try:
        payload_file_path = _normalize_project_relative_path(component_state.get("file_path", active_file_path))
    except ValueError:
        payload_matches_active = False
        payload_file_path = ""
    if payload_matches_active and payload_file_path != active_file_path:
        payload_matches_active = False
    if not payload_matches_active:
        return

    candidate_text = str(component_state.get("text", ""))
    candidate_trigger = str(component_state.get("trigger", "")).strip().lower()
    if (
        candidate_text
        and candidate_text != state.current_text
        and _should_accept_editor_component_text(
            state,
            candidate_text,
            candidate_trigger=candidate_trigger,
        )
    ):
        state.current_text = candidate_text


def _should_accept_editor_component_text(
    state: AppState,
    candidate_text: str,
    *,
    candidate_trigger: str = "",
) -> bool:
    del candidate_trigger  # Reserved for future disambiguation if needed.
    doc_id = str(state.state_doc_id or st.session_state.get("active_workspace_id", "default")).strip() or "default"
    sync_key = _editor_backend_sync_key(doc_id)
    expected_text = st.session_state.get(sync_key)
    if not isinstance(expected_text, str):
        return True
    if str(candidate_text or "") == expected_text:
        st.session_state.pop(sync_key, None)
        return True
    return False


def _bootstrap_workspaces_from_disk() -> None:
    st.session_state.workspace_order = []
    st.session_state.workspace_store = {}

    doc_ids = _workspace_doc_ids_from_disk()
    sortable: List[Tuple[str, str, str]] = []
    for doc_id in doc_ids:
        metadata = load_metadata(doc_id)
        created_at = str(metadata.get("created_at", "")).strip()
        last_active_at = str(metadata.get("updated_at", "")).strip() or created_at
        sortable.append((last_active_at, created_at, doc_id))
    sortable.sort(reverse=True)

    for _, _, doc_id in sortable:
        payload = _workspace_payload_from_disk(doc_id)
        st.session_state.workspace_order.append(doc_id)
        st.session_state.workspace_store[doc_id] = payload

    if not st.session_state.workspace_order:
        _create_workspace("Conversation 1")
        return

    active_candidate = str(st.session_state.get("active_workspace_id", "")).strip()
    if active_candidate not in st.session_state.workspace_store:
        active_candidate = st.session_state.workspace_order[0]
    _load_workspace(active_candidate)


def _save_active_workspace_snapshot(touch_activity: bool = False) -> None:
    active_id = st.session_state.get("active_workspace_id", "")
    store = st.session_state.get("workspace_store", {})
    state = st.session_state.get("app_state")
    if not active_id or active_id not in store or state is None:
        return
    editor_component_key = f"interactive_latex_editor_{active_id}"
    component_state = st.session_state.get(editor_component_key)
    if isinstance(component_state, dict):
        try:
            active_file_path = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
        except ValueError:
            active_file_path = "main.tex"
        payload_matches_active = True
        try:
            payload_file_path = _normalize_project_relative_path(component_state.get("file_path", active_file_path))
        except ValueError:
            payload_matches_active = False
            payload_file_path = ""
        if payload_matches_active and payload_file_path != active_file_path:
            payload_matches_active = False
        if payload_matches_active:
            latest_text = str(component_state.get("text", state.current_text))
            latest_trigger = str(component_state.get("trigger", "")).strip().lower()
            allow_clear_triggers = {
                "input",
                "confirm",
                "cancel",
                "confirm_edit",
                "preserve_selection",
                "invalidate_selection",
            }
            suspicious_empty_echo = (
                not latest_text.strip()
                and bool(str(state.current_text).strip())
                and latest_trigger not in allow_clear_triggers
            )
            if (
                (not suspicious_empty_echo)
                and latest_text != state.current_text
                and _should_accept_editor_component_text(
                    state,
                    latest_text,
                    candidate_trigger=latest_trigger,
                )
            ):
                state.current_text = latest_text
    role = _normalize_role_name(st.session_state.get("current_role", ""))
    if role:
        state.role = role
    store[active_id]["state"] = copy.deepcopy(state)
    store[active_id]["ui"] = _capture_workspace_ui()
    requested_touch = bool(touch_activity) or _consume_workspace_activity_mark(active_id)
    _persist_workspace_to_disk(active_id, touch_activity=requested_touch)


def _load_workspace(workspace_id: str) -> None:
    store = st.session_state.get("workspace_store", {})
    if workspace_id not in store:
        return
    payload = store[workspace_id]
    disk_payload = _workspace_payload_from_disk(workspace_id)
    disk_state = copy.deepcopy(disk_payload.get("state", AppState(current_text="")))
    disk_agent_chats = _normalize_agent_chats_payload(
        disk_payload.get("agent_chats"),
        fallback_conversation=_chat_turns_to_payload(disk_state.chat_history, fallback_role=disk_state.role),
    )
    active_chat_id = str(disk_agent_chats.get("active_chat_id", "")).strip()
    active_chat_payload = (
        disk_agent_chats.get("chats", {}).get(active_chat_id, {})
        if isinstance(disk_agent_chats.get("chats"), dict)
        else {}
    )
    loaded_state = copy.deepcopy(payload.get("state", disk_state))
    loaded_state.current_text = str(disk_state.current_text or "")
    loaded_state.chat_history = _payload_to_chat_turns(active_chat_payload.get("conversation", []))
    loaded_state.state_doc_id = workspace_id

    disk_name = str(disk_payload.get("name", "")).strip()
    if disk_name:
        payload["name"] = disk_name
    if isinstance(disk_payload.get("memory"), dict):
        payload["memory"] = copy.deepcopy(disk_payload["memory"])
    if isinstance(disk_payload.get("metadata"), dict):
        payload["metadata"] = copy.deepcopy(disk_payload["metadata"])
    payload["agent_chats"] = copy.deepcopy(disk_agent_chats)
    if not isinstance(payload.get("ui"), dict):
        payload["ui"] = copy.deepcopy(disk_payload.get("ui", _workspace_ui_defaults()))
    payload["state"] = copy.deepcopy(loaded_state)
    store[workspace_id] = payload

    st.session_state.active_workspace_id = workspace_id
    st.session_state.app_state = loaded_state
    _set_consistency_orchestrator(loaded_state)
    _restore_workspace_ui(payload.get("ui", {}))
    _close_resource_preview()
    # Role selection is driven by per-workspace UI state, not AppState default.
    role = _normalize_role_name(st.session_state.get("current_role", ""))
    st.session_state.current_role = role
    st.session_state.selected_role = role
    if role:
        loaded_state.role = role
    _sync_patch_preview_compat(loaded_state)
    _sync_patch_gate_flags(loaded_state)
    current_hashes = dict(st.session_state.get("workspace_text_hashes", {}))
    active_file_path = "main.tex"
    try:
        active_file_path = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
    except ValueError:
        active_file_path = "main.tex"
    if _is_internal_state_relative_path(active_file_path) or not _is_text_editable_project_file(active_file_path):
        active_file_path = "main.tex"
    current_hashes[f"{workspace_id}:{active_file_path}"] = hashlib.sha256(
        loaded_state.current_text.encode("utf-8")
    ).hexdigest()
    st.session_state.workspace_text_hashes = current_hashes


def _create_workspace(name: str | None = None) -> str:
    _save_active_workspace_snapshot()
    workspace_name = name or _next_conversation_title()
    should_seed_demo = (
        workspace_name == "Conversation 1"
        and not bool(list(st.session_state.get("workspace_order", [])))
    )
    workspace_id = ""
    while True:
        candidate = f"chat-{uuid4().hex[:8]}"
        if not get_project_dir(candidate).exists():
            workspace_id = candidate
            break
    ensure_project_dir(workspace_id)
    seeded_demo = False
    if should_seed_demo:
        seeded_demo = _seed_demo_workspace(workspace_id)
    save_metadata(
        workspace_id,
        {
            "doc_id": workspace_id,
            "title": workspace_name,
            "current_role": "",
            "project_root": str(Path.cwd()),
            "active_file_path": "main.tex",
        },
    )
    if not seeded_demo:
        save_main_tex(workspace_id, DEFAULT_MAIN_TEX_TEMPLATE)
    save_conversation(workspace_id, [])
    save_agent_chats(workspace_id, {"active_chat_id": "", "order": [], "chats": {}})
    save_memory(workspace_id, _workspace_memory_defaults())
    state = _create_state_from_project(workspace_id, active_file_path="main.tex")
    state.role = "Reviewer"
    st.session_state.workspace_order.insert(0, workspace_id)
    st.session_state.workspace_store[workspace_id] = {
        "name": workspace_name,
        "state": copy.deepcopy(state),
        "ui": _workspace_ui_defaults(),
        "memory": _workspace_memory_defaults(),
        "agent_chats": load_agent_chats(workspace_id),
        "metadata": load_metadata(workspace_id),
    }
    _persist_workspace_to_disk(workspace_id, touch_activity=False)
    _load_workspace(workspace_id)
    return workspace_id


def _delete_workspace(workspace_id: str) -> None:
    order = st.session_state.workspace_order
    store = st.session_state.workspace_store
    if workspace_id not in order:
        return
    if len(order) == 1:
        delete_project(workspace_id)
        st.session_state.workspace_order = []
        st.session_state.workspace_store = {}
        st.session_state.active_workspace_id = ""
        st.session_state.workspace_delete_confirm = {}
        st.session_state.files_delete_confirm = {
            key: value
            for key, value in dict(st.session_state.get("files_delete_confirm", {})).items()
            if not str(key).startswith(f"{workspace_id}:")
        }
        st.session_state.workspace_text_hashes = {}
        marks = dict(st.session_state.get("workspace_activity_marks", {}))
        marks.pop(workspace_id, None)
        st.session_state.workspace_activity_marks = marks
        _create_workspace()
        return

    is_active = workspace_id == st.session_state.get("active_workspace_id", "")
    index = order.index(workspace_id)
    delete_project(workspace_id)
    order.remove(workspace_id)
    store.pop(workspace_id, None)
    text_hashes = dict(st.session_state.get("workspace_text_hashes", {}))
    filtered_hashes = {
        key: value
        for key, value in text_hashes.items()
        if not str(key).startswith(f"{workspace_id}:")
    }
    st.session_state.workspace_text_hashes = filtered_hashes
    marks = dict(st.session_state.get("workspace_activity_marks", {}))
    marks.pop(workspace_id, None)
    st.session_state.workspace_activity_marks = marks
    st.session_state.files_delete_confirm = {
        key: value
        for key, value in dict(st.session_state.get("files_delete_confirm", {})).items()
        if not str(key).startswith(f"{workspace_id}:")
    }

    if is_active:
        next_index = max(0, index - 1)
        next_workspace = order[next_index]
        _load_workspace(next_workspace)


def _rename_workspace(workspace_id: str, new_name: str) -> None:
    cleaned = new_name.strip()
    if not cleaned:
        return
    if workspace_id in st.session_state.workspace_store:
        st.session_state.workspace_store[workspace_id]["name"] = cleaned
        metadata = load_metadata(workspace_id)
        metadata["title"] = cleaned
        save_metadata(workspace_id, metadata, touch_updated_at=False)


def _arm_workspace_delete(workspace_id: str) -> None:
    st.session_state.workspace_delete_confirm[workspace_id] = True


def _cancel_workspace_delete(workspace_id: str) -> None:
    st.session_state.workspace_delete_confirm.pop(workspace_id, None)


def _confirm_workspace_delete(workspace_id: str) -> None:
    st.session_state.workspace_delete_confirm.pop(workspace_id, None)
    _delete_workspace(workspace_id)


def _workspace_agent_chats_payload(workspace_id: str) -> Dict[str, Any]:
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return _normalize_agent_chats_payload({}, fallback_conversation=[])
    state = payload.get("state")
    ui_payload = payload.get("ui", {}) if isinstance(payload.get("ui"), dict) else {}
    role = _normalize_role_name(str(ui_payload.get("current_role", "")))
    fallback_conversation: List[Dict[str, Any]] = []
    if isinstance(state, AppState):
        fallback_conversation = _chat_turns_to_payload(state.chat_history, fallback_role=role)
    normalized = _normalize_agent_chats_payload(
        payload.get("agent_chats"),
        fallback_conversation=fallback_conversation,
    )
    payload["agent_chats"] = normalized
    return normalized


def _set_workspace_agent_chat_ui_field(workspace_id: str, key: str, value: Any) -> None:
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return
    ui_payload = dict(payload.get("ui", {}))
    ui_payload[key] = copy.deepcopy(value)
    payload["ui"] = ui_payload
    if str(st.session_state.get("active_workspace_id", "")).strip() == workspace_id:
        st.session_state[key] = copy.deepcopy(value)


def _agent_chat_delete_flags(workspace_id: str) -> Dict[str, bool]:
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return {}
    ui_payload = payload.get("ui", {}) if isinstance(payload.get("ui"), dict) else {}
    flags_raw = ui_payload.get("agent_chat_delete_confirm")
    flags = flags_raw if isinstance(flags_raw, dict) else {}
    normalized_flags = {str(k): bool(v) for k, v in flags.items() if str(k).strip()}
    if normalized_flags != flags:
        _set_workspace_agent_chat_ui_field(workspace_id, "agent_chat_delete_confirm", normalized_flags)
    return normalized_flags


def _set_agent_chat_history_open(workspace_id: str, is_open: bool) -> None:
    _set_workspace_agent_chat_ui_field(workspace_id, "agent_chat_history_open", bool(is_open))


def _arm_agent_chat_delete(workspace_id: str, chat_id: str) -> None:
    chat_key = str(chat_id or "").strip()
    if not chat_key:
        return
    flags = dict(_agent_chat_delete_flags(workspace_id))
    flags[chat_key] = True
    _set_workspace_agent_chat_ui_field(workspace_id, "agent_chat_delete_confirm", flags)


def _cancel_agent_chat_delete(workspace_id: str, chat_id: str) -> None:
    chat_key = str(chat_id or "").strip()
    if not chat_key:
        return
    flags = dict(_agent_chat_delete_flags(workspace_id))
    flags.pop(chat_key, None)
    _set_workspace_agent_chat_ui_field(workspace_id, "agent_chat_delete_confirm", flags)


def _switch_agent_chat(workspace_id: str, chat_id: str) -> None:
    chat_key = str(chat_id or "").strip()
    if not chat_key:
        return
    active_workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if workspace_id != active_workspace_id:
        return
    _save_active_workspace_snapshot(touch_activity=False)
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return
    agent_chats = _workspace_agent_chats_payload(workspace_id)
    chats_map = agent_chats.get("chats", {}) if isinstance(agent_chats.get("chats"), dict) else {}
    if chat_key not in chats_map:
        return
    agent_chats["active_chat_id"] = chat_key
    payload["agent_chats"] = _normalize_agent_chats_payload(
        agent_chats,
        fallback_conversation=[],
    )
    active_chat_payload = payload["agent_chats"]["chats"].get(chat_key, {})
    state = st.session_state.get("app_state")
    if isinstance(state, AppState):
        state.chat_history = _payload_to_chat_turns(active_chat_payload.get("conversation", []))
        payload["state"] = copy.deepcopy(state)
    _cancel_agent_chat_delete(workspace_id, chat_key)
    _persist_workspace_to_disk(workspace_id, touch_activity=False)


def _create_agent_chat(workspace_id: str) -> None:
    active_workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if workspace_id != active_workspace_id:
        return
    _save_active_workspace_snapshot(touch_activity=False)
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return
    agent_chats = _workspace_agent_chats_payload(workspace_id)
    new_chat_id = _new_agent_chat_id()
    while new_chat_id in agent_chats.get("chats", {}):
        new_chat_id = _new_agent_chat_id()
    now = _utc_now_iso()
    chats_map = dict(agent_chats.get("chats", {}))
    chats_map[new_chat_id] = {
        "name": _next_agent_chat_title(agent_chats),
        "created_at": now,
        "updated_at": now,
        "conversation": [],
    }
    order = [new_chat_id] + [item for item in list(agent_chats.get("order", [])) if item != new_chat_id]
    payload["agent_chats"] = _normalize_agent_chats_payload(
        {
            "active_chat_id": new_chat_id,
            "order": order,
            "chats": chats_map,
        },
        fallback_conversation=[],
    )
    state = st.session_state.get("app_state")
    if isinstance(state, AppState):
        state.chat_history = []
        payload["state"] = copy.deepcopy(state)
    _set_workspace_agent_chat_ui_field(workspace_id, "agent_chat_history_open", False)
    _set_workspace_agent_chat_ui_field(workspace_id, "agent_chat_delete_confirm", {})
    _persist_workspace_to_disk(workspace_id, touch_activity=True)


def _rename_agent_chat(workspace_id: str, chat_id: str, new_name: str) -> None:
    chat_key = str(chat_id or "").strip()
    if not chat_key:
        return
    cleaned = " ".join(str(new_name or "").split()).strip()
    if not cleaned:
        return
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return
    agent_chats = _workspace_agent_chats_payload(workspace_id)
    chats_map = dict(agent_chats.get("chats", {}))
    chat_payload = chats_map.get(chat_key)
    if not isinstance(chat_payload, dict):
        return
    chat_payload = dict(chat_payload)
    chat_payload["name"] = cleaned
    chats_map[chat_key] = chat_payload
    payload["agent_chats"] = _normalize_agent_chats_payload(
        {
            "active_chat_id": agent_chats.get("active_chat_id", ""),
            "order": list(agent_chats.get("order", [])),
            "chats": chats_map,
        },
        fallback_conversation=[],
    )
    _persist_workspace_to_disk(workspace_id, touch_activity=False)


def _delete_agent_chat(workspace_id: str, chat_id: str) -> None:
    chat_key = str(chat_id or "").strip()
    if not chat_key:
        return
    active_workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if workspace_id != active_workspace_id:
        return
    _save_active_workspace_snapshot(touch_activity=False)
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if not isinstance(payload, dict):
        return
    agent_chats = _workspace_agent_chats_payload(workspace_id)
    chats_map = dict(agent_chats.get("chats", {}))
    if chat_key not in chats_map:
        return
    chats_map.pop(chat_key, None)
    order = [item for item in list(agent_chats.get("order", [])) if item in chats_map and item != chat_key]
    active_chat_id = str(agent_chats.get("active_chat_id", "")).strip()

    if not chats_map:
        now = _utc_now_iso()
        fallback_id = _new_agent_chat_id()
        chats_map[fallback_id] = {
            "name": "Chat 1",
            "created_at": now,
            "updated_at": now,
            "conversation": [],
        }
        order = [fallback_id]
        active_chat_id = fallback_id
    elif active_chat_id == chat_key or active_chat_id not in chats_map:
        active_chat_id = order[0] if order else next(iter(chats_map.keys()))

    payload["agent_chats"] = _normalize_agent_chats_payload(
        {
            "active_chat_id": active_chat_id,
            "order": order,
            "chats": chats_map,
        },
        fallback_conversation=[],
    )

    state = st.session_state.get("app_state")
    if isinstance(state, AppState):
        active_chat_payload = payload["agent_chats"]["chats"].get(active_chat_id, {})
        state.chat_history = _payload_to_chat_turns(active_chat_payload.get("conversation", []))
        payload["state"] = copy.deepcopy(state)
    _cancel_agent_chat_delete(workspace_id, chat_key)
    _persist_workspace_to_disk(workspace_id, touch_activity=True)


def _confirm_agent_chat_delete(workspace_id: str, chat_id: str) -> None:
    _cancel_agent_chat_delete(workspace_id, chat_id)
    _delete_agent_chat(workspace_id, chat_id)


def _clear_selection_state_after_file_switch(state: AppState) -> None:
    st.session_state.selection_text = ""
    st.session_state.selection_start = -1
    st.session_state.selection_end = -1
    st.session_state.pending_selection_text = ""
    st.session_state.pending_selection_start = -1
    st.session_state.pending_selection_end = -1
    st.session_state.confirmed_selection_text = ""
    st.session_state.confirmed_selection_start = -1
    st.session_state.confirmed_selection_end = -1
    st.session_state.active_selection = ""
    st.session_state.last_component_action_fingerprint = ""
    state.active_selection = None


def _project_entries(doc_id: str, relative_dir: str = "") -> List[Dict[str, Any]]:
    project_dir = ensure_project_dir(doc_id)
    folder = project_dir
    if relative_dir:
        normalized_dir = _normalize_project_relative_path(relative_dir)
        folder = _resolve_project_path(doc_id, normalized_dir)
    if not folder.exists() or not folder.is_dir():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        children = list(folder.iterdir())
    except OSError:
        return []
    for child in children:
        try:
            rel = child.relative_to(project_dir).as_posix()
        except ValueError:
            continue
        if _is_internal_state_relative_path(rel):
            continue
        entries.append(
            {
                "name": child.name,
                "rel_path": rel,
                "is_dir": child.is_dir(),
            }
        )
    entries.sort(key=lambda item: (not bool(item["is_dir"]), str(item["name"]).casefold()))
    return entries


def _close_resource_preview() -> None:
    active_candidate = str(st.session_state.get("active_file_path", "main.tex")).strip() or "main.tex"
    try:
        normalized_active = _normalize_project_relative_path(active_candidate)
        if not _is_text_editable_project_file(normalized_active):
            normalized_active = "main.tex"
    except ValueError:
        normalized_active = "main.tex"
    st.session_state.resource_preview_visible = False
    st.session_state.resource_preview_path = ""
    st.session_state.resource_preview_doc_id = ""
    st.session_state.files_selected_path = normalized_active
    workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if workspace_id:
        _sync_workspace_ui_to_store(workspace_id)
        _persist_workspace_to_disk(workspace_id, touch_activity=False)


def _open_workspace_resource_preview(workspace_id: str, relative_path: str) -> None:
    _save_active_workspace_snapshot()
    normalized_path = _normalize_project_relative_path(relative_path)
    if _is_internal_state_relative_path(normalized_path):
        return
    target = _resolve_project_path(workspace_id, normalized_path)
    if not target.exists() or target.is_dir():
        return
    st.session_state.files_selected_path = normalized_path
    st.session_state.resource_preview_visible = True
    st.session_state.resource_preview_path = normalized_path
    st.session_state.resource_preview_doc_id = workspace_id
    _sync_workspace_ui_to_store(workspace_id)
    _persist_workspace_to_disk(workspace_id, touch_activity=True)


def _activate_workspace_file(workspace_id: str, normalized_path: str, save_before_switch: bool = True) -> None:
    if save_before_switch:
        _save_active_workspace_snapshot()
        _force_save_active_editor_file(workspace_id)
    text = load_project_text_file(workspace_id, normalized_path)
    state = st.session_state.get("app_state")
    if state is None:
        return
    state.current_text = text
    _clear_selection_state_after_file_switch(state)
    st.session_state.active_file_path = normalized_path
    st.session_state.files_selected_path = normalized_path
    payload = st.session_state.get("workspace_store", {}).get(workspace_id, {})
    if isinstance(payload, dict):
        ui_payload = dict(payload.get("ui", {}))
        ui_payload["active_file_path"] = normalized_path
        ui_payload["files_selected_path"] = normalized_path
        payload["ui"] = ui_payload
        payload["state"] = copy.deepcopy(state)
        metadata = load_metadata(workspace_id)
        metadata["active_file_path"] = normalized_path
        save_metadata(workspace_id, metadata)
        payload["metadata"] = metadata
    _close_resource_preview()
    hash_key = f"{workspace_id}:{normalized_path}"
    hashes = dict(st.session_state.get("workspace_text_hashes", {}))
    hashes[hash_key] = hashlib.sha256(state.current_text.encode("utf-8")).hexdigest()
    st.session_state.workspace_text_hashes = hashes
    _persist_workspace_to_disk(workspace_id, touch_activity=True)


def _open_workspace_file(workspace_id: str, relative_path: str) -> None:
    normalized_path = _normalize_project_relative_path(relative_path)
    if not _is_text_editable_project_file(normalized_path):
        return
    _activate_workspace_file(workspace_id, normalized_path, save_before_switch=True)


def _create_project_file(doc_id: str, relative_path: str, allow_nested: bool = True) -> str:
    raw_path = str(relative_path or "").strip()
    if not raw_path:
        raise ValueError("Please provide a file path.")
    if not allow_nested and ("/" in raw_path or "\\" in raw_path):
        raise ValueError("Only a file name is allowed here, for example: method.txt")
    if raw_path.endswith("/") or raw_path.endswith("\\"):
        raise ValueError("Please provide a file path, not a folder path.")
    normalized = _normalize_project_relative_path(raw_path)
    if _is_internal_state_relative_path(normalized):
        raise ValueError("This filename is reserved by system state.")
    path = _resolve_project_path(doc_id, normalized)
    if path.exists():
        if path.is_dir():
            raise ValueError("A folder with the same name already exists.")
        raise ValueError("File already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if normalized == "main.tex":
        path.write_text(DEFAULT_MAIN_TEX_TEMPLATE, encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")
    return normalized


def _create_project_folder(doc_id: str, relative_path: str) -> str:
    normalized = _normalize_project_relative_path(relative_path)
    if _is_internal_state_relative_path(normalized):
        raise ValueError("This folder name conflicts with system state.")
    path = _resolve_project_path(doc_id, normalized)
    if path.exists():
        raise ValueError("Path already exists.")
    path.mkdir(parents=True, exist_ok=False)
    return normalized


def _normalize_optional_project_dir(path_text: str | None) -> str:
    raw = str(path_text or "").strip()
    if not raw:
        return ""
    normalized = _normalize_project_relative_path(raw)
    if _is_internal_state_relative_path(normalized):
        raise ValueError("Reserved system path is not allowed.")
    return normalized


def _normalize_dropped_relative_file_path(path_text: str | None) -> str:
    raw = str(path_text or "").strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    raw = raw.lstrip("/")
    if not raw:
        raise ValueError("Invalid dropped file path.")
    if raw.endswith("/"):
        raise ValueError("Dropped path points to a folder, not a file.")
    normalized = _normalize_project_relative_path(raw)
    if _is_internal_state_relative_path(normalized):
        raise ValueError("This filename is reserved by system state.")
    return normalized


def _relocate_by_prefix(path_value: str, source_prefix: str, target_prefix: str) -> str:
    if path_value == source_prefix:
        return target_prefix
    if path_value.startswith(f"{source_prefix}/"):
        return f"{target_prefix}{path_value[len(source_prefix):]}"
    return path_value


def _dedupe_relative_path(doc_id: str, preferred_relative_path: str) -> str:
    normalized = _normalize_project_relative_path(preferred_relative_path)
    candidate = normalized
    parent = Path(normalized).parent.as_posix()
    if parent == ".":
        parent = ""
    stem = Path(normalized).stem
    suffix = Path(normalized).suffix
    counter = 1
    while True:
        if _is_internal_state_relative_path(candidate):
            pass
        else:
            candidate_path = _resolve_project_path(doc_id, candidate)
            if not candidate_path.exists():
                return candidate
        deduped_name = f"{stem}_{counter}{suffix}"
        candidate = deduped_name if not parent else f"{parent}/{deduped_name}"
        counter += 1
        if counter > 4096:
            raise ValueError("Unable to resolve a unique filename for upload.")


def _rename_project_entry(doc_id: str, source_relative_path: str, new_name: str) -> str:
    source_rel = _normalize_project_relative_path(source_relative_path)
    if _is_internal_state_relative_path(source_rel):
        raise ValueError("System state file cannot be renamed.")
    source_path = _resolve_project_path(doc_id, source_rel)
    if not source_path.exists():
        raise ValueError("Source path does not exist.")
    cleaned_name = str(new_name).strip()
    if not cleaned_name or "/" in cleaned_name or "\\" in cleaned_name or cleaned_name in {".", ".."}:
        raise ValueError("Invalid new name.")
    if cleaned_name == source_path.name:
        return source_rel
    target_path = source_path.with_name(cleaned_name)
    project_dir = ensure_project_dir(doc_id)
    try:
        target_rel = target_path.relative_to(project_dir).as_posix()
    except ValueError as exc:
        raise ValueError("Unsafe target path.") from exc
    if _is_internal_state_relative_path(target_rel):
        raise ValueError("Cannot rename into a reserved system filename.")
    if target_path.exists():
        raise ValueError("Target path already exists.")
    source_path.rename(target_path)
    return target_rel


def _delete_project_entry(doc_id: str, relative_path: str) -> None:
    normalized = _normalize_project_relative_path(relative_path)
    if _is_internal_state_relative_path(normalized):
        raise ValueError("System state file cannot be deleted.")
    path = _resolve_project_path(doc_id, normalized)
    if not path.exists():
        raise ValueError("Path does not exist.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _default_upload_target_dir(doc_id: str) -> str:
    try:
        selected_rel = _normalize_project_relative_path(st.session_state.get("files_selected_path", ""))
    except ValueError:
        return ""
    if not selected_rel:
        return ""
    try:
        selected_path = _resolve_project_path(doc_id, selected_rel)
    except ValueError:
        return ""
    if selected_path.exists() and selected_path.is_dir():
        return selected_rel
    parent = Path(selected_rel).parent.as_posix()
    return "" if parent == "." else parent


def _apply_tree_entry_relocation(doc_id: str, source_rel: str, target_rel: str) -> None:
    st.session_state.files_selected_path = _relocate_by_prefix(
        str(st.session_state.get("files_selected_path", "main.tex")),
        source_rel,
        target_rel,
    )

    expanded_dirs = []
    for directory in list(st.session_state.get("files_expanded_dirs", [])):
        try:
            normalized_dir = _normalize_project_relative_path(directory)
        except ValueError:
            continue
        expanded_dirs.append(_relocate_by_prefix(normalized_dir, source_rel, target_rel))
    st.session_state.files_expanded_dirs = sorted(dict.fromkeys(expanded_dirs))

    if (
        bool(st.session_state.get("resource_preview_visible", False))
        and str(st.session_state.get("resource_preview_doc_id", "")) == doc_id
    ):
        preview_rel = str(st.session_state.get("resource_preview_path", "")).strip()
        if preview_rel:
            mapped_preview = _relocate_by_prefix(preview_rel, source_rel, target_rel)
            st.session_state.resource_preview_path = mapped_preview

    try:
        active_file = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
    except ValueError:
        active_file = "main.tex"
    mapped_active = _relocate_by_prefix(active_file, source_rel, target_rel)
    if mapped_active != active_file and _is_text_editable_project_file(mapped_active):
        _activate_workspace_file(doc_id, mapped_active, save_before_switch=False)
        _sync_workspace_ui_to_store(doc_id)
    elif mapped_active != active_file:
        main_tex = _resolve_project_path(doc_id, "main.tex")
        if not main_tex.exists():
            main_tex.write_text(DEFAULT_MAIN_TEX_TEMPLATE, encoding="utf-8")
        _activate_workspace_file(doc_id, "main.tex", save_before_switch=False)
        _sync_workspace_ui_to_store(doc_id)
    else:
        _sync_workspace_ui_to_store(doc_id)
        _persist_workspace_to_disk(doc_id, touch_activity=True)


def _apply_tree_entry_delete(doc_id: str, target_rel: str) -> None:
    st.session_state.files_selected_path = "main.tex"

    filtered_dirs = []
    for directory in list(st.session_state.get("files_expanded_dirs", [])):
        try:
            normalized_dir = _normalize_project_relative_path(directory)
        except ValueError:
            continue
        if normalized_dir == target_rel or normalized_dir.startswith(f"{target_rel}/"):
            continue
        filtered_dirs.append(normalized_dir)
    st.session_state.files_expanded_dirs = sorted(dict.fromkeys(filtered_dirs))

    if (
        bool(st.session_state.get("resource_preview_visible", False))
        and str(st.session_state.get("resource_preview_doc_id", "")) == doc_id
    ):
        preview_rel = str(st.session_state.get("resource_preview_path", "")).strip()
        if preview_rel == target_rel or preview_rel.startswith(f"{target_rel}/"):
            _close_resource_preview()

    try:
        active_file = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
    except ValueError:
        active_file = "main.tex"
    main_tex = _resolve_project_path(doc_id, "main.tex")
    if not main_tex.exists():
        main_tex.write_text(DEFAULT_MAIN_TEX_TEMPLATE, encoding="utf-8")
    deleting_active = active_file == target_rel or active_file.startswith(f"{target_rel}/")
    if deleting_active:
        _activate_workspace_file(doc_id, "main.tex", save_before_switch=False)
        _sync_workspace_ui_to_store(doc_id)
    else:
        _sync_workspace_ui_to_store(doc_id)
        _persist_workspace_to_disk(doc_id, touch_activity=True)


def _apply_tree_entry_action(
    doc_id: str,
    relative_path: str,
    action: str,
    new_name: str = "",
) -> None:
    source_rel = _normalize_project_relative_path(relative_path)
    token = f"{doc_id}:{source_rel}"
    armed_map = dict(st.session_state.get("files_delete_confirm", {}))
    armed_map.pop(token, None)
    st.session_state.files_delete_confirm = armed_map
    _save_active_workspace_snapshot()

    if action == "rename":
        target_rel = _rename_project_entry(doc_id, source_rel, new_name)
        _apply_tree_entry_relocation(doc_id, source_rel, target_rel)
        return
    if action == "delete":
        _delete_project_entry(doc_id, source_rel)
        _apply_tree_entry_delete(doc_id, source_rel)
        return
    raise ValueError("Unsupported file-tree action.")


def _material_icon_for_file(relative_path: str) -> str:
    suffix = Path(relative_path).suffix.lower()
    if suffix == ".tex":
        return ":material/description:"
    if suffix in RESOURCE_IMAGE_SUFFIXES:
        return ":material/image:"
    if suffix in RESOURCE_PDF_SUFFIXES:
        return ":material/picture_as_pdf:"
    return ":material/insert_drive_file:"


def _render_files_inline_notice(message: str) -> None:
    st.warning(str(message or "").strip() or "Operation failed.", icon=":material/info:")


def _render_files_delete_control(doc_id: str, rel_path: str) -> None:
    token = f"{doc_id}:{rel_path}"
    armed_map = dict(st.session_state.get("files_delete_confirm", {}))
    armed = bool(armed_map.get(token, False))
    if not armed:
        if st.button(
            "Delete",
            key=f"files_tree_delete_btn_{doc_id}_{rel_path}",
            use_container_width=True,
        ):
            armed_map[token] = True
            st.session_state.files_delete_confirm = armed_map
            st.rerun()
        return

    st.warning("Confirm\u00A0deletion  \nof this item?")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        if st.button(
            "Yes",
            key=f"files_tree_delete_yes_{doc_id}_{rel_path}",
            type="secondary",
            use_container_width=True,
        ):
            try:
                _apply_tree_entry_action(doc_id, rel_path, "delete")
            except ValueError as exc:
                _render_files_inline_notice(str(exc))
            else:
                refreshed = dict(st.session_state.get("files_delete_confirm", {}))
                refreshed.pop(token, None)
                st.session_state.files_delete_confirm = refreshed
                st.rerun()
    with c2:
        if st.button(
            "No",
            key=f"files_tree_delete_no_{doc_id}_{rel_path}",
            type="secondary",
            use_container_width=True,
        ):
            refreshed = dict(st.session_state.get("files_delete_confirm", {}))
            refreshed.pop(token, None)
            st.session_state.files_delete_confirm = refreshed
            st.rerun()


def _visible_files_tree_nodes(doc_id: str, relative_dir: str = "") -> List[Dict[str, str]]:
    expanded_dirs = set(st.session_state.get("files_expanded_dirs", []))
    nodes: List[Dict[str, str]] = []
    for entry in _project_entries(doc_id, relative_dir):
        rel_path = str(entry["rel_path"])
        nodes.append(
            {
                "kind": "dir" if bool(entry["is_dir"]) else "file",
                "path": rel_path,
                "name": str(entry["name"]),
            }
        )
        if bool(entry["is_dir"]) and rel_path in expanded_dirs:
            nodes.extend(_visible_files_tree_nodes(doc_id, rel_path))
    return nodes


def _render_files_tree(doc_id: str, relative_dir: str = "", depth: int = 0) -> None:
    entries = _project_entries(doc_id, relative_dir)
    if not entries and depth == 0:
        st.caption("No files yet.")
        return

    try:
        active_file = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
    except ValueError:
        active_file = "main.tex"
    try:
        selected_path = _normalize_project_relative_path(st.session_state.get("files_selected_path", active_file))
    except ValueError:
        selected_path = active_file
    expanded_dirs = set(st.session_state.get("files_expanded_dirs", []))

    for entry in entries:
        rel_path = str(entry["rel_path"])
        name = str(entry["name"])
        # This threshold reflects the current Files panel button width more closely
        # than the old 28-char cutoff, so visually overflowing names enter the long-name path.
        is_long_name = len(name) >= 20
        if bool(entry["is_dir"]):
            is_open = rel_path in expanded_dirs
            left_col, right_col = st.columns([0.78, 0.22], gap="small", vertical_alignment="center")
            with left_col:
                button_col = left_col
                if depth > 0:
                    _, button_col = st.columns([max(1, depth), 24], gap="small")
                with button_col:
                    toggle_key = (
                        f"files_toggle_long_{doc_id}_{rel_path}"
                        if is_long_name
                        else f"files_toggle_{doc_id}_{rel_path}"
                    )
                    if st.button(
                        name,
                        key=toggle_key,
                        use_container_width=True,
                        icon=":material/folder_open:" if is_open else ":material/folder:",
                        type="primary" if (selected_path == rel_path and is_open) else "secondary",
                        help=name if is_long_name else None,
                    ):
                        st.session_state.files_selected_path = rel_path
                        if is_open:
                            expanded_dirs.discard(rel_path)
                        else:
                            expanded_dirs.add(rel_path)
                        st.session_state.files_expanded_dirs = sorted(expanded_dirs)
                        st.rerun()
            with right_col:
                with st.popover("⋯"):
                    new_file_name = st.text_input(
                        "New file in folder",
                        value="",
                        key=f"files_tree_new_file_value_{doc_id}_{rel_path}",
                        placeholder="method.tex",
                        label_visibility="collapsed",
                    )
                    if st.button(
                        "New file in folder",
                        key=f"files_tree_new_file_btn_{doc_id}_{rel_path}",
                        use_container_width=True,
                    ):
                        try:
                            file_name = str(new_file_name or "").strip()
                            if not file_name:
                                raise ValueError("Please enter a file name.")
                            if "/" in file_name or "\\" in file_name:
                                raise ValueError("Please enter a file name only, not a path.")
                            created_rel = _create_project_file(doc_id, f"{rel_path}/{file_name}")
                        except ValueError as exc:
                            _render_files_inline_notice(str(exc))
                        else:
                            st.session_state.files_selected_path = created_rel
                            if _is_text_editable_project_file(created_rel):
                                _open_workspace_file(doc_id, created_rel)
                            else:
                                _open_workspace_resource_preview(doc_id, created_rel)
                            st.rerun()
                    st.markdown(
                        "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                        unsafe_allow_html=True,
                    )
                    rename_value = st.text_input(
                        "Rename entry",
                        value=name,
                        key=f"files_tree_rename_value_{doc_id}_{rel_path}",
                        label_visibility="collapsed",
                    )
                    if st.button(
                        "Rename",
                        key=f"files_tree_rename_btn_{doc_id}_{rel_path}",
                        use_container_width=True,
                    ):
                        try:
                            _apply_tree_entry_action(doc_id, rel_path, "rename", new_name=rename_value)
                        except ValueError as exc:
                            _render_files_inline_notice(str(exc))
                        else:
                            st.rerun()
                    st.markdown(
                        "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                        unsafe_allow_html=True,
                    )
                    try:
                        folder_payload, folder_filename, folder_mime = _build_project_entry_download_payload(
                            doc_id,
                            rel_path,
                        )
                    except ValueError as exc:
                        _render_files_inline_notice(str(exc))
                    else:
                        st.download_button(
                            "Download ZIP",
                            data=folder_payload,
                            file_name=folder_filename,
                            mime=folder_mime,
                            key=f"files_tree_download_dir_{doc_id}_{rel_path}",
                            use_container_width=True,
                        )
                    st.markdown(
                        "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                        unsafe_allow_html=True,
                    )
                    _render_files_delete_control(doc_id, rel_path)
            if is_open:
                _render_files_tree(doc_id, rel_path, depth + 1)
            continue

        is_text_file = _is_text_editable_project_file(rel_path)
        is_selected = selected_path == rel_path
        is_active = active_file == rel_path
        left_col, right_col = st.columns([0.78, 0.22], gap="small", vertical_alignment="center")
        with left_col:
            button_col = left_col
            if depth > 0:
                _, button_col = st.columns([max(1, depth), 24], gap="small")
            with button_col:
                open_key = (
                    f"files_open_long_{doc_id}_{rel_path}"
                    if is_long_name
                    else f"files_open_{doc_id}_{rel_path}"
                )
                if st.button(
                    name,
                    key=open_key,
                    use_container_width=True,
                    icon=_material_icon_for_file(rel_path),
                    type="primary" if (is_active or is_selected) else "secondary",
                    help=name if is_long_name else None,
                ):
                    st.session_state.files_selected_path = rel_path
                    if is_text_file:
                        _open_workspace_file(doc_id, rel_path)
                    else:
                        _open_workspace_resource_preview(doc_id, rel_path)
                    st.rerun()
        with right_col:
            with st.popover("⋯"):
                rename_value = st.text_input(
                    "Rename entry",
                    value=name,
                    key=f"files_tree_rename_value_{doc_id}_{rel_path}",
                    label_visibility="collapsed",
                )
                if st.button(
                    "Rename",
                    key=f"files_tree_rename_btn_{doc_id}_{rel_path}",
                    use_container_width=True,
                ):
                    try:
                        _apply_tree_entry_action(doc_id, rel_path, "rename", new_name=rename_value)
                    except ValueError as exc:
                        _render_files_inline_notice(str(exc))
                    else:
                        st.rerun()
                st.markdown(
                    "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                    unsafe_allow_html=True,
                )
                try:
                    file_payload, file_filename, file_mime = _build_project_entry_download_payload(
                        doc_id,
                        rel_path,
                    )
                except ValueError as exc:
                    _render_files_inline_notice(str(exc))
                else:
                    st.download_button(
                        "Download",
                        data=file_payload,
                        file_name=file_filename,
                        mime=file_mime,
                        key=f"files_tree_download_file_{doc_id}_{rel_path}",
                        use_container_width=True,
                    )
                st.markdown(
                    "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                    unsafe_allow_html=True,
                )
                _render_files_delete_control(doc_id, rel_path)


def _save_project_file_bytes(
    doc_id: str,
    filename: str,
    payload: bytes,
    target_dir: str = "",
    relative_path: str = "",
) -> str:
    normalized_relative = ""
    raw_relative_path = str(relative_path or "").strip()
    if raw_relative_path:
        normalized_relative = _normalize_dropped_relative_file_path(raw_relative_path)
        clean_name = Path(normalized_relative).name.strip()
    else:
        raw_name = str(filename or "").strip()
        clean_name = Path(raw_name).name.strip()
        if not clean_name or clean_name in {".", ".."}:
            raise ValueError("Invalid upload filename.")

    normalized_dir = _normalize_optional_project_dir(target_dir)
    if normalized_dir:
        target_dir_path = _resolve_project_path(doc_id, normalized_dir)
        if not target_dir_path.exists() or not target_dir_path.is_dir():
            raise ValueError("Upload target folder does not exist.")

    if normalized_relative:
        target_rel = normalized_relative if not normalized_dir else f"{normalized_dir}/{normalized_relative}"
    else:
        target_rel = clean_name if not normalized_dir else f"{normalized_dir}/{clean_name}"
    target_rel = _normalize_project_relative_path(target_rel)
    normalized_target = _dedupe_relative_path(doc_id, target_rel)
    target_path = _resolve_project_path(doc_id, normalized_target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(bytes(payload))
    return normalized_target


def _move_project_entry(doc_id: str, source_relative_path: str, target_dir: str) -> str:
    source_rel = _normalize_project_relative_path(source_relative_path)
    if _is_internal_state_relative_path(source_rel):
        raise ValueError("System state file cannot be moved.")

    source_path = _resolve_project_path(doc_id, source_rel)
    if not source_path.exists():
        raise ValueError("Source path does not exist.")

    normalized_target_dir = _normalize_optional_project_dir(target_dir)
    if normalized_target_dir and _is_internal_state_relative_path(normalized_target_dir):
        raise ValueError("Cannot move into a reserved system folder.")
    if source_rel == "main.tex" and normalized_target_dir:
        return source_rel

    if normalized_target_dir:
        target_dir_path = _resolve_project_path(doc_id, normalized_target_dir)
        if not target_dir_path.exists() or not target_dir_path.is_dir():
            raise ValueError("Target folder does not exist.")
        target_rel = _normalize_project_relative_path(f"{normalized_target_dir}/{source_path.name}")
    else:
        target_dir_path = get_project_dir(doc_id)
        target_rel = _normalize_project_relative_path(source_path.name)

    if target_rel == source_rel:
        return source_rel
    if source_path.is_dir() and target_rel.startswith(f"{source_rel}/"):
        raise ValueError("Cannot move a folder into its own subfolder.")
    if _is_internal_state_relative_path(target_rel):
        raise ValueError("Cannot move into a reserved system path.")

    target_path = _resolve_project_path(doc_id, target_rel)
    if target_path.exists():
        raise ValueError("Target path already exists.")

    source_path.rename(target_path)
    return target_rel


def _clear_files_dnd_bridge_state() -> None:
    st.session_state.files_dnd_bridge_action = ""
    st.session_state.files_dnd_bridge_payload = ""
    st.session_state.files_dnd_bridge_nonce = ""


def _reset_files_selection_after_dnd(doc_id: str, preferred_dir: str = "") -> None:
    candidate = str(preferred_dir or "").strip()
    selected: str | None = None
    if candidate:
        try:
            normalized_dir = _normalize_project_relative_path(candidate)
            dir_path = _resolve_project_path(doc_id, normalized_dir)
            if dir_path.exists() and dir_path.is_dir():
                selected = normalized_dir
        except ValueError:
            selected = None
    if not selected:
        active_candidate = str(st.session_state.get("active_file_path", "main.tex")).strip() or "main.tex"
        try:
            normalized_active = _normalize_project_relative_path(active_candidate)
        except ValueError:
            normalized_active = "main.tex"
        selected = normalized_active
    st.session_state.files_selected_path = selected
    _sync_workspace_ui_to_store(doc_id)
    _persist_workspace_to_disk(doc_id, touch_activity=True)


def _handle_files_dnd_move(payload: Dict[str, Any]) -> None:
    doc_id = str(payload.get("doc_id", "")).strip()
    source_rel = str(payload.get("source_path", "")).strip()
    target_dir = str(payload.get("target_dir", "")).strip()
    if not doc_id or not source_rel:
        raise ValueError("Drag move payload is incomplete.")

    _save_active_workspace_snapshot()
    moved_rel = _move_project_entry(doc_id, source_rel, target_dir)
    _apply_tree_entry_relocation(doc_id, source_rel, moved_rel)
    _reset_files_selection_after_dnd(doc_id, target_dir)


def _handle_files_dnd_upload(payload: Dict[str, Any]) -> None:
    doc_id = str(payload.get("doc_id", "")).strip()
    if not doc_id:
        raise ValueError("Upload payload is missing workspace id.")

    target_dir = _normalize_optional_project_dir(payload.get("target_dir", ""))
    files_payload = payload.get("files", [])
    if not isinstance(files_payload, list) or not files_payload:
        raise ValueError("No files were dropped.")

    _save_active_workspace_snapshot()
    created_paths: List[str] = []
    expanded_dirs = set(st.session_state.get("files_expanded_dirs", []))
    for item in files_payload:
        if not isinstance(item, dict):
            continue
        file_name = Path(str(item.get("name", "")).strip()).name.strip()
        relative_path = str(item.get("relative_path", "")).strip()
        content_b64 = str(item.get("content_b64", "")).strip()
        if (not file_name and not relative_path) or not content_b64:
            continue
        try:
            file_bytes = base64.b64decode(content_b64, validate=True)
        except Exception as exc:  # pragma: no cover - browser payload decode guard
            raise ValueError(f"Failed to decode dropped file: {file_name}") from exc
        created_rel = (
            _save_project_file_bytes(
                doc_id,
                filename=file_name or Path(relative_path).name,
                payload=file_bytes,
                target_dir=target_dir,
                relative_path=relative_path,
            )
        )
        created_paths.append(created_rel)
        parent_dir = Path(created_rel).parent.as_posix()
        if parent_dir and parent_dir != ".":
            expanded_dirs.add(parent_dir)

    if not created_paths:
        raise ValueError("No valid files were dropped.")

    if target_dir:
        expanded_dirs.add(target_dir)
    st.session_state.files_expanded_dirs = sorted(expanded_dirs)

    _reset_files_selection_after_dnd(doc_id, target_dir)


def _reset_sidebar_delete_confirms() -> None:
    st.session_state.files_delete_confirm = {}
    st.session_state.workspace_delete_confirm = {}


def _handle_files_dnd_bridge_action() -> None:
    action = str(st.session_state.get("files_dnd_bridge_action", "")).strip().lower()
    raw_payload = str(st.session_state.get("files_dnd_bridge_payload", "")).strip()
    if not action or not raw_payload:
        _clear_files_dnd_bridge_state()
        return

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        _clear_files_dnd_bridge_state()
        st.toast("Dropped data is invalid.", icon=":material/info:")
        return

    try:
        if action == "move":
            _handle_files_dnd_move(payload if isinstance(payload, dict) else {})
        elif action == "upload":
            _handle_files_dnd_upload(payload if isinstance(payload, dict) else {})
        elif action == "pdf_dblclick":
            payload_map = payload if isinstance(payload, dict) else {}
            try:
                page = int(payload_map.get("page", 0) or 0)
            except (TypeError, ValueError):
                page = 0
            try:
                x = float(payload_map.get("x", 0.0) or 0.0)
            except (TypeError, ValueError):
                x = 0.0
            try:
                y = float(payload_map.get("y", 0.0) or 0.0)
            except (TypeError, ValueError):
                y = 0.0
            try:
                event_id = int(payload_map.get("event_id", 0) or 0)
            except (TypeError, ValueError):
                event_id = 0
            if page > 0:
                if event_id <= 0:
                    event_id = int(time.time() * 1000)
                st.session_state.pdf_pending_dblclick_event = {
                    "action": "pdf_dblclick",
                    "page": page,
                    "x": x,
                    "y": y,
                    "event_id": event_id,
                }
        elif action == "reset_delete_confirms":
            _reset_sidebar_delete_confirms()
        else:
            raise ValueError("Unsupported drag action.")
    except ValueError as exc:
        st.toast(str(exc), icon=":material/info:")
    finally:
        _clear_files_dnd_bridge_state()


def _handle_files_dnd_component_event(event: Dict[str, Any] | None) -> bool:
    if not isinstance(event, dict):
        return False

    action = str(event.get("action", "")).strip().lower()
    if not action or action == "idle":
        return False

    try:
        event_id = int(event.get("event_id", 0) or 0)
    except (TypeError, ValueError):
        event_id = 0
    if event_id <= 0:
        return False

    last_event_id = int(st.session_state.get("files_dnd_component_last_event_id", 0) or 0)
    if event_id == last_event_id:
        return False
    st.session_state.files_dnd_component_last_event_id = event_id

    payload = event.get("payload", {})
    try:
        if action == "move":
            _handle_files_dnd_move(payload if isinstance(payload, dict) else {})
        elif action == "upload":
            _handle_files_dnd_upload(payload if isinstance(payload, dict) else {})
        elif action == "pdf_dblclick":
            payload_map = payload if isinstance(payload, dict) else {}
            try:
                page = int(payload_map.get("page", 0) or 0)
            except (TypeError, ValueError):
                page = 0
            try:
                x = float(payload_map.get("x", 0.0) or 0.0)
            except (TypeError, ValueError):
                x = 0.0
            try:
                y = float(payload_map.get("y", 0.0) or 0.0)
            except (TypeError, ValueError):
                y = 0.0
            if page > 0:
                st.session_state.pdf_pending_dblclick_event = {
                    "action": "pdf_dblclick",
                    "page": page,
                    "x": x,
                    "y": y,
                    "event_id": event_id,
                }
        elif action == "reset_delete_confirms":
            _reset_sidebar_delete_confirms()
        else:
            raise ValueError("Unsupported drag action.")
    except ValueError as exc:
        st.toast(str(exc), icon=":material/info:")
    return True


def _on_files_dnd_bridge_nonce_change() -> None:
    current_nonce = str(st.session_state.get("files_dnd_bridge_nonce", "")).strip()
    last_nonce = str(st.session_state.get("files_dnd_bridge_last_nonce", "")).strip()
    if not current_nonce or current_nonce == last_nonce:
        return
    st.session_state.files_dnd_bridge_last_nonce = current_nonce
    _handle_files_dnd_bridge_action()


def _upload_project_file(doc_id: str, uploaded_file: Any, target_dir: str = "") -> str:
    if uploaded_file is None:
        raise ValueError("Please choose a file to upload.")
    raw_name = str(getattr(uploaded_file, "name", "")).strip()
    return _save_project_file_bytes(
        doc_id,
        filename=raw_name,
        payload=bytes(uploaded_file.getvalue()),
        target_dir=target_dir,
    )


def _upload_project_folder(doc_id: str, uploaded_files: Any, target_dir: str = "") -> List[str]:
    if uploaded_files is None:
        raise ValueError("Please choose a folder to upload.")
    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    if not files:
        raise ValueError("Please choose a folder to upload.")

    _save_active_workspace_snapshot()
    created_paths: List[str] = []
    expanded_dirs = set(st.session_state.get("files_expanded_dirs", []))
    normalized_target_dir = _normalize_optional_project_dir(target_dir)
    for item in files:
        if item is None:
            continue
        relative_name = str(getattr(item, "name", "")).strip()
        if not relative_name:
            continue
        created_rel = _save_project_file_bytes(
            doc_id,
            filename=Path(relative_name).name,
            payload=bytes(item.getvalue()),
            target_dir=normalized_target_dir,
            relative_path=relative_name,
        )
        created_paths.append(created_rel)
        parent_dir = Path(created_rel).parent.as_posix()
        if parent_dir and parent_dir != ".":
            expanded_dirs.add(parent_dir)

    if not created_paths:
        raise ValueError("No valid files were found in the selected folder.")

    if normalized_target_dir:
        expanded_dirs.add(normalized_target_dir)
    st.session_state.files_expanded_dirs = sorted(expanded_dirs)
    return created_paths


def _render_files_dnd_bridge_controls() -> bool:
    st.text_input(
        "Files DnD action",
        key="files_dnd_bridge_action",
        label_visibility="collapsed",
    )
    st.text_area(
        "Files DnD payload",
        key="files_dnd_bridge_payload",
        label_visibility="collapsed",
    )
    st.text_input(
        "Files DnD nonce",
        key="files_dnd_bridge_nonce",
        label_visibility="collapsed",
        on_change=_on_files_dnd_bridge_nonce_change,
    )
    return st.button(
        "Apply Files DnD",
        key="files_dnd_bridge_apply",
        use_container_width=False,
    )


def _render_agent_chat_history_list(workspace_id: str) -> None:
    st.markdown(
        "<div class='awc-agent-history-popover-marker' style='display:none;'></div>",
        unsafe_allow_html=True,
    )
    agent_chats = _workspace_agent_chats_payload(workspace_id)
    chats_map = agent_chats.get("chats", {}) if isinstance(agent_chats.get("chats"), dict) else {}
    order = [str(item).strip() for item in list(agent_chats.get("order", [])) if str(item).strip() in chats_map]
    if not order:
        st.caption("No chats yet.")
        return

    active_chat_id = str(agent_chats.get("active_chat_id", "")).strip()
    for chat_id in order:
        chat_payload = chats_map.get(chat_id, {})
        chat_name = str(chat_payload.get("name", "")).strip() or "Untitled Chat"
        is_active = chat_id == active_chat_id
        row_left, row_right = st.columns([0.83, 0.17], gap="small", vertical_alignment="center")
        with row_left:
            if st.button(
                chat_name,
                key=f"agent_chat_switch_{workspace_id}_{chat_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if not is_active:
                    _switch_agent_chat(workspace_id, chat_id)
                    st.rerun()
        with row_right:
            with st.popover("⋯", use_container_width=True):
                st.markdown(
                    "<div class='awc-agent-history-rename-popover-marker' style='display:none;'></div>",
                    unsafe_allow_html=True,
                )
                renamed = st.text_input(
                    "Rename agent chat",
                    value=chat_name,
                    key=f"agent_chat_rename_value_{workspace_id}_{chat_id}",
                    label_visibility="collapsed",
                )
                if st.button(
                    "Rename",
                    key=f"agent_chat_rename_confirm_{workspace_id}_{chat_id}",
                    type="secondary",
                    use_container_width=True,
                ):
                    _rename_agent_chat(workspace_id, chat_id, renamed)
                    st.rerun()

                st.markdown(
                    "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                    unsafe_allow_html=True,
                )
                armed = bool(_agent_chat_delete_flags(workspace_id).get(chat_id, False))
                if not armed:
                    st.button(
                        "Delete",
                        key=f"agent_chat_delete_arm_{workspace_id}_{chat_id}",
                        type="secondary",
                        use_container_width=True,
                        on_click=_arm_agent_chat_delete,
                        args=(workspace_id, chat_id),
                    )
                else:
                    st.warning("Confirm\u00A0deletion  \nof this chat?")
                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        st.button(
                            "Yes",
                            key=f"agent_chat_delete_yes_{workspace_id}_{chat_id}",
                            type="secondary",
                            use_container_width=True,
                            on_click=_confirm_agent_chat_delete,
                            args=(workspace_id, chat_id),
                        )
                    with c2:
                        st.button(
                            "No",
                            key=f"agent_chat_delete_no_{workspace_id}_{chat_id}",
                            type="secondary",
                            use_container_width=True,
                            on_click=_cancel_agent_chat_delete,
                            args=(workspace_id, chat_id),
                        )


def _render_chats_sidebar_panel() -> None:
    st.markdown("<div class='awc-left-panel-content-lift'></div>", unsafe_allow_html=True)
    st.markdown("### Chats")
    if st.button("+ New Chat", key="workspace_new_chat_btn", use_container_width=True):
        _create_workspace()
        st.rerun()

    active_id = st.session_state.get("active_workspace_id", "")
    for workspace_id in st.session_state.workspace_order:
        name = st.session_state.workspace_store[workspace_id]["name"]
        is_active = workspace_id == active_id
        row_left, row_right = st.columns([0.79, 0.21], gap="small", vertical_alignment="center")
        with row_left:
            if st.button(
                name,
                key=f"workspace_switch_{workspace_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    _save_active_workspace_snapshot()
                    _load_workspace(workspace_id)
                    st.rerun()
        with row_right:
            with st.popover("⋯"):
                st.markdown(
                    "<div style='font-size:0.76rem; color:#6b7280; margin:0 0 0.22rem 0;'>Rename</div>",
                    unsafe_allow_html=True,
                )
                renamed = st.text_input(
                    "Rename chat",
                    value=name,
                    key=f"workspace_rename_value_{workspace_id}",
                    label_visibility="collapsed",
                )
                if st.button(
                    "Confirm\u00A0Rename",
                    key=f"workspace_rename_confirm_{workspace_id}",
                    type="secondary",
                    use_container_width=True,
                ):
                    _rename_workspace(workspace_id, renamed)
                    st.rerun()

                st.markdown(
                    "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                    unsafe_allow_html=True,
                )
                try:
                    archive_payload, archive_name = _build_workspace_download_zip_payload(workspace_id, name)
                except ValueError as exc:
                    st.warning(str(exc), icon=":material/info:")
                else:
                    st.download_button(
                        "Download ZIP",
                        data=archive_payload,
                        file_name=archive_name,
                        mime="application/zip",
                        key=f"workspace_download_zip_{workspace_id}",
                        use_container_width=True,
                    )

                st.markdown(
                    "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                    unsafe_allow_html=True,
                )
                armed = bool(st.session_state.workspace_delete_confirm.get(workspace_id, False))
                if not armed:
                    st.button(
                        "Delete Chat",
                        key=f"workspace_delete_arm_{workspace_id}",
                        type="secondary",
                        use_container_width=True,
                        on_click=_arm_workspace_delete,
                        args=(workspace_id,),
                    )
                else:
                    st.warning("Confirm\u00A0deletion  \nof this chat?")
                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        st.button(
                            "Yes",
                            key=f"workspace_delete_yes_{workspace_id}",
                            type="secondary",
                            use_container_width=True,
                            on_click=_confirm_workspace_delete,
                            args=(workspace_id,),
                        )
                    with c2:
                        st.button(
                            "No",
                            key=f"workspace_delete_no_{workspace_id}",
                            type="secondary",
                            use_container_width=True,
                            on_click=_cancel_workspace_delete,
                            args=(workspace_id,),
                        )


def _render_files_sidebar_panel() -> None:
    st.markdown("<div class='awc-left-panel-content-lift'></div>", unsafe_allow_html=True)
    st.markdown("### Files")
    st.markdown("<div class='awc-files-panel-spacing-marker'></div>", unsafe_allow_html=True)
    active_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if not active_id:
        st.info("No active workspace.")
        return

    if not st.session_state.get("files_selected_path"):
        st.session_state.files_selected_path = st.session_state.get("active_file_path", "main.tex")

    action_cols = st.columns([1.0, 1.22, 1.0], gap="small", vertical_alignment="center")
    with action_cols[0]:
        st.markdown(
            "<div class='awc-files-actions-row-marker' style='height:0;min-height:0;margin:0;padding:0;overflow:hidden;'></div>",
            unsafe_allow_html=True,
        )
        with st.popover("+ File"):
            file_path = st.text_input(
                "New file name",
                key=f"files_new_file_input_{active_id}",
                placeholder="method.tex",
                label_visibility="collapsed",
            )
            if st.button("Create", key=f"files_new_file_btn_{active_id}", use_container_width=True):
                try:
                    created_rel = _create_project_file(active_id, file_path, allow_nested=False)
                except ValueError as exc:
                    _render_files_inline_notice(str(exc))
                else:
                    st.session_state.files_selected_path = created_rel
                    if _is_text_editable_project_file(created_rel):
                        _open_workspace_file(active_id, created_rel)
                    else:
                        _open_workspace_resource_preview(active_id, created_rel)
                    st.rerun()
    with action_cols[1]:
        st.markdown(
            "<div class='awc-files-actions-row-marker' style='height:0;min-height:0;margin:0;padding:0;overflow:hidden;'></div>",
            unsafe_allow_html=True,
        )
        with st.popover("+ Folder"):
            folder_path = st.text_input(
                "New folder path",
                key=f"files_new_folder_input_{active_id}",
                placeholder="figures",
                label_visibility="collapsed",
            )
            if st.button("Create", key=f"files_new_folder_btn_{active_id}", use_container_width=True):
                try:
                    created_dir = _create_project_folder(active_id, folder_path)
                except ValueError as exc:
                    _render_files_inline_notice(str(exc))
                else:
                    expanded_dirs = set(st.session_state.get("files_expanded_dirs", []))
                    expanded_dirs.add(created_dir)
                    st.session_state.files_expanded_dirs = sorted(expanded_dirs)
                    st.session_state.files_selected_path = created_dir
                    _mark_workspace_activity(active_id)
                    st.rerun()
    with action_cols[2]:
        st.markdown(
            "<div class='awc-files-actions-row-marker' style='height:0;min-height:0;margin:0;padding:0;overflow:hidden;'></div>",
            unsafe_allow_html=True,
        )
        with st.popover("Upload"):
            st.markdown("<div class='awc-upload-popover-marker'></div>", unsafe_allow_html=True)
            st.caption("File")
            upload_file = st.file_uploader(
                "Upload file",
                key=f"files_upload_picker_{active_id}",
                accept_multiple_files=False,
                label_visibility="collapsed",
            )
            if st.button("Upload", key=f"files_upload_btn_{active_id}", use_container_width=True):
                try:
                    uploaded_rel = _upload_project_file(
                        active_id,
                        upload_file,
                        "",
                    )
                except ValueError as exc:
                    _render_files_inline_notice(str(exc))
                else:
                    st.session_state.files_selected_path = uploaded_rel
                    if _is_text_editable_project_file(uploaded_rel):
                        _open_workspace_file(active_id, uploaded_rel)
                    else:
                        _open_workspace_resource_preview(active_id, uploaded_rel)
                    st.rerun()
            st.markdown(
                "<hr style='margin:0.26rem 0; border:0; border-top:1px solid #e5e7eb;' />",
                unsafe_allow_html=True,
            )
            st.caption("Folder")
            upload_folder_files = st.file_uploader(
                "Upload folder",
                key=f"files_upload_folder_picker_{active_id}",
                accept_multiple_files="directory",
                label_visibility="collapsed",
            )
            if st.button("Upload Folder", key=f"files_upload_folder_btn_{active_id}", use_container_width=True):
                try:
                    uploaded_paths = _upload_project_folder(
                        active_id,
                        upload_folder_files,
                        "",
                    )
                except ValueError as exc:
                    _render_files_inline_notice(str(exc))
                else:
                    first_uploaded = str(uploaded_paths[0]).strip()
                    st.session_state.files_selected_path = first_uploaded
                    if _is_text_editable_project_file(first_uploaded):
                        _open_workspace_file(active_id, first_uploaded)
                    else:
                        _open_workspace_resource_preview(active_id, first_uploaded)
                    st.rerun()

    st.markdown(
        "<hr style='margin:-0.16rem 0 0.04rem 0; border:0; border-top:1px solid #e5e7eb;' />",
        unsafe_allow_html=True,
    )
    _render_files_tree(active_id)


def _toggle_sidebar_panel(mode: str) -> None:
    target_mode = str(mode).strip().lower()
    current_mode = str(st.session_state.get("sidebar_panel_mode", "")).strip().lower()
    st.session_state.sidebar_panel_mode = "" if current_mode == target_mode else target_mode


def _render_workspace_sidebar() -> None:
    panel_mode = str(st.session_state.get("sidebar_panel_mode", "chats") or "").strip().lower()
    active_id = str(st.session_state.get("active_workspace_id", "")).strip()
    with st.sidebar:
        nav_col, panel_col = st.columns([0.18, 0.82], gap="small", vertical_alignment="top")

        with nav_col:
            st.markdown("<div class='awc-iconbar-marker'></div>", unsafe_allow_html=True)
            chats_pressed = st.button(
                " ",
                key="sidebar_nav_chats_btn",
                use_container_width=False,
                help="Chats",
                icon=":material/chat_bubble:",
            )
            if chats_pressed:
                _toggle_sidebar_panel("chats")
                st.rerun()
            files_pressed = st.button(
                " ",
                key="sidebar_nav_files_btn",
                use_container_width=False,
                help="Files",
                icon=":material/folder:",
            )
            if files_pressed:
                _toggle_sidebar_panel("files")
                st.rerun()
            st.markdown(
                f"""
<div class="awc-activity-bar" role="toolbar" aria-label="Workspace navigation">
  <button
    id="awc-activity-chats"
    class="awc-activity-btn{' is-active' if panel_mode == 'chats' else ''}"
    type="button"
    aria-label="Chat history"
    data-tooltip="Chat history"
    aria-pressed="{'true' if panel_mode == 'chats' else 'false'}"
  >
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M4 6.75C4 5.23 5.23 4 6.75 4h10.5C18.77 4 20 5.23 20 6.75v6.5C20 14.77 18.77 16 17.25 16H9.5l-3.74 2.8A.75.75 0 0 1 4.5 18.2V16A2.75 2.75 0 0 1 4 14.75v-8Z" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/>
    </svg>
  </button>
  <button
    id="awc-activity-files"
    class="awc-activity-btn{' is-active' if panel_mode == 'files' else ''}"
    type="button"
    aria-label="File tree"
    data-tooltip="File tree"
    aria-pressed="{'true' if panel_mode == 'files' else 'false'}"
  >
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M4 6.75A2.75 2.75 0 0 1 6.75 4h3.12l1.7 1.9h5.68A2.75 2.75 0 0 1 20 8.65v8.6A2.75 2.75 0 0 1 17.25 20H6.75A2.75 2.75 0 0 1 4 17.25V6.75Z" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/>
    </svg>
  </button>
</div>
""",
                unsafe_allow_html=True,
            )

        with panel_col:
            st.markdown("<div class='awc-sidepanel-marker'></div>", unsafe_allow_html=True)
            if panel_mode == "chats":
                _render_chats_sidebar_panel()
            elif panel_mode == "files":
                _render_files_sidebar_panel()
                if active_id and _resource_preview_is_open_for_workspace(active_id):
                    preview_height = max(320, int(st.session_state.get("viewport_available_height", 760)) - 250)
                    with st.container(key=f"files_resource_preview_mount_{active_id}"):
                        _render_resource_preview_panel(active_id, preview_height)


def _apply_left_nav_panel_style() -> None:
    panel_mode = str(st.session_state.get("sidebar_panel_mode", "") or "").strip().lower()
    panel_open = panel_mode in {"chats", "files"}
    icon_bar_width_px = 48
    panel_width_px = 244
    sidebar_top_padding_px = 0
    sidebar_width_px = icon_bar_width_px + panel_width_px if panel_open else icon_bar_width_px
    panel_display = "block" if panel_open else "none"
    st.markdown(
        f"""
<style>
section[data-testid="stSidebar"][aria-expanded="true"],
section[data-testid="stSidebar"][aria-expanded="false"] {{
  position: relative !important;
  min-width: {sidebar_width_px}px !important;
  max-width: {sidebar_width_px}px !important;
  overflow: visible !important;
  transition: min-width 0.22s cubic-bezier(0.22, 1, 0.36, 1), max-width 0.22s cubic-bezier(0.22, 1, 0.36, 1) !important;
}}
section[data-testid="stSidebar"][aria-expanded="true"] > div,
section[data-testid="stSidebar"][aria-expanded="false"] > div {{
  padding-top: {sidebar_top_padding_px}px !important;
  min-width: {sidebar_width_px}px !important;
  max-width: {sidebar_width_px}px !important;
  overflow: visible !important;
  transition: min-width 0.22s cubic-bezier(0.22, 1, 0.36, 1), max-width 0.22s cubic-bezier(0.22, 1, 0.36, 1) !important;
}}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
  padding-top: 0 !important;
  margin-top: 0 !important;
}}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div[data-testid="stVerticalBlock"] {{
  padding-top: 0 !important;
  margin-top: 0 !important;
}}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:has(.awc-iconbar-marker) {{
  margin-top: 0 !important;
  padding-top: 0 !important;
}}
section[data-testid="stSidebar"][aria-expanded="true"]::after,
section[data-testid="stSidebar"][aria-expanded="false"]::after {{
  content: "" !important;
  position: absolute !important;
  top: 0 !important;
  bottom: 0 !important;
  left: {icon_bar_width_px - 1}px !important;
  width: 1px !important;
  background: rgba(255, 255, 255, 0.88) !important;
  opacity: 1 !important;
  transition: opacity 0.18s ease !important;
  pointer-events: none !important;
  z-index: 4 !important;
}}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarNavCollapseButton"],
[data-testid="collapsedControl"] {{
  display: none !important;
}}
[data-testid="stSidebar"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:has(.awc-iconbar-marker) {{
  gap: 0 !important;
  align-items: flex-start !important;
  position: relative !important;
  min-height: 100% !important;
}}
[data-testid="stSidebar"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:has(.awc-iconbar-marker) > div:nth-child(1) {{
  flex: 0 0 {icon_bar_width_px}px !important;
  width: {icon_bar_width_px}px !important;
  min-width: {icon_bar_width_px}px !important;
  max-width: {icon_bar_width_px}px !important;
  padding-top: 0.16rem !important;
  position: absolute !important;
  left: 0 !important;
  top: 0 !important;
  bottom: 0 !important;
  z-index: 1200 !important;
  display: flex !important;
  justify-content: center !important;
}}
[data-testid="stSidebar"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:has(.awc-iconbar-marker) > div:nth-child(2) {{
  display: {panel_display} !important;
  flex: 1 1 auto !important;
  min-width: 0 !important;
  margin-left: {icon_bar_width_px}px !important;
  margin-top: 0 !important;
  width: calc(100% - {icon_bar_width_px}px) !important;
  max-width: calc(100% - {icon_bar_width_px}px) !important;
  padding-left: 0.56rem !important;
  padding-top: 0 !important;
  position: relative !important;
  top: 0 !important;
  z-index: 10 !important;
  transform: none !important;
  will-change: auto !important;
  transition: none !important;
  opacity: 0 !important;
  visibility: hidden !important;
  pointer-events: none !important;
  overflow: visible !important;
}}
[data-testid="stSidebar"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:has(.awc-iconbar-marker) > div:nth-child(2)[data-awc-top-pinned="1"] {{
  opacity: 1 !important;
  visibility: visible !important;
  pointer-events: auto !important;
}}
[data-testid="stSidebar"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:has(.awc-iconbar-marker) > div:nth-child(2) > div[data-testid="stVerticalBlock"] {{
  margin-top: 0 !important;
  padding-top: 0 !important;
  gap: 0.38rem !important;
}}
[data-testid="stSidebar"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:has(.awc-iconbar-marker) > div:nth-child(2) > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child,
[data-testid="stSidebar"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:has(.awc-iconbar-marker) > div:nth-child(2) > div[data-testid="stVerticalBlock"] > [data-testid="element-container"]:first-child {{
  margin-top: 0 !important;
  padding-top: 0 !important;
}}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) [data-testid="stMarkdownContainer"] h3 {{
  margin-top: 0 !important;
  margin-bottom: 0.38rem !important;
}}
[data-testid="stSidebar"] .awc-iconbar-marker,
[data-testid="stSidebar"] .awc-sidepanel-marker {{
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) div[data-testid="stVerticalBlock"] {{
  overflow: visible !important;
}}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) [class*="st-key-files_resource_preview_mount_"] {{
  --awc-files-preview-width: 388px;
  position: absolute !important;
  left: calc(100% + 0.62rem) !important;
  top: 0 !important;
  width: var(--awc-files-preview-width) !important;
  min-width: var(--awc-files-preview-width) !important;
  max-width: var(--awc-files-preview-width) !important;
  max-height: calc(100vh - 32px) !important;
  overflow: auto !important;
  padding: 0.62rem 0.72rem 0.78rem !important;
  border: 1px solid #d1d5db !important;
  border-radius: 12px !important;
  background: #f3f4f6 !important;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.16) !important;
  z-index: 36 !important;
}}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) [class*="st-key-files_resource_preview_mount_"] h4 {{
  margin: 0 !important;
}}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) [class*="st-key-files_resource_preview_mount_"] .awc-files-preview-resizer {{
  position: absolute !important;
  top: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  width: 10px !important;
  cursor: ew-resize !important;
  z-index: 8 !important;
  background: transparent !important;
}}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) [class*="st-key-files_resource_preview_mount_"] .awc-files-preview-resizer::before {{
  content: "" !important;
  position: absolute !important;
  left: 4px !important;
  top: 12px !important;
  bottom: 12px !important;
  width: 1px !important;
  background: #c7ced8 !important;
  opacity: 0 !important;
  transition: opacity 0.16s ease !important;
}}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) [class*="st-key-files_resource_preview_mount_"]:hover .awc-files-preview-resizer::before {{
  opacity: 1 !important;
}}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) [class*="st-key-files_resource_preview_mount_"].is-resizing {{
  user-select: none !important;
}}
[class*="st-key-sidebar_nav_chats_btn"],
[class*="st-key-sidebar_nav_files_btn"],
[class*="st-key-sidebar_nav_chats_btn"] > div,
[class*="st-key-sidebar_nav_files_btn"] > div,
[class*="st-key-sidebar_nav_chats_btn"] :is([data-testid="stButton"], .stButton),
[class*="st-key-sidebar_nav_files_btn"] :is([data-testid="stButton"], .stButton) {{
  position: absolute !important;
  inset: 0 auto auto 0 !important;
  width: 1px !important;
  height: 1px !important;
  min-width: 1px !important;
  max-width: 1px !important;
  min-height: 1px !important;
  max-height: 1px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}}
[class*="st-key-sidebar_nav_chats_btn"] :is([data-testid="stButton"], .stButton) > button,
[class*="st-key-sidebar_nav_files_btn"] :is([data-testid="stButton"], .stButton) > button {{
  width: 1px !important;
  height: 1px !important;
  min-width: 1px !important;
  max-width: 1px !important;
  min-height: 1px !important;
  max-height: 1px !important;
  padding: 0 !important;
  margin: 0 !important;
  opacity: 0 !important;
  pointer-events: none !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}}
[data-testid="stSidebar"] .awc-activity-bar {{
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 0.72rem !important;
  width: 30px !important;
  min-width: 30px !important;
  max-width: 30px !important;
  padding: 0.18rem 0 0 0 !important;
  position: absolute !important;
  top: -0.18rem !important;
  left: calc(-10px + 0.001rem) !important;
  transform: none !important;
  overflow: visible !important;
  z-index: 1300 !important;
}}
[data-testid="stSidebar"] .awc-activity-btn {{
  all: unset !important;
  width: 30px !important;
  min-width: 30px !important;
  max-width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  flex: 0 0 30px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  color: #4b5563 !important;
  cursor: pointer !important;
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  padding: 0 !important;
  margin: 0 !important;
  box-sizing: border-box !important;
  transition: color 0.16s ease !important;
  transform: none !important;
  overflow: visible !important;
  z-index: 1301 !important;
}}
[data-testid="stSidebar"] .awc-activity-btn::before {{
  content: none !important;
  display: none !important;
}}
[data-testid="stSidebar"] .awc-activity-btn::after {{
  content: "" !important;
  position: absolute !important;
  left: 50% !important;
  bottom: 2px !important;
  transform: translateX(-50%) !important;
  width: 16px !important;
  height: 2px !important;
  border-radius: 999px !important;
  background: transparent !important;
}}
[data-testid="stSidebar"] .awc-activity-btn:hover {{
  color: #0f172a !important;
}}
[data-testid="stSidebar"] .awc-activity-btn:focus,
[data-testid="stSidebar"] .awc-activity-btn:focus-visible,
[data-testid="stSidebar"] .awc-activity-btn:active {{
  outline: none !important;
  box-shadow: none !important;
  transform: none !important;
}}
[data-testid="stSidebar"] .awc-activity-btn svg {{
  width: 22px !important;
  height: 22px !important;
  display: block !important;
  flex: 0 0 22px !important;
}}
[data-testid="stSidebar"] #awc-activity-files svg {{
  transform: scaleY(0.88) !important;
  transform-origin: center center !important;
}}
[data-testid="stSidebar"] .awc-activity-btn.is-active {{
  color: #1e3a8a !important;
}}
[data-testid="stSidebar"] .awc-activity-btn.is-active::after {{
  background: currentColor !important;
}}
.awc-left-panel-content-lift {{
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}}
.awc-upload-popover-marker {{
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}}
.awc-files-panel-spacing-marker {{
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.awc-files-panel-spacing-marker),
[data-testid="stSidebar"] [data-testid="element-container"]:has(.awc-files-panel-spacing-marker) {{
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 0 -2.00rem 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def _apply_files_dnd_bridge_style() -> None:
    st.markdown(
        """
<style>
[class*="st-key-files_dnd_bridge_action"],
[class*="st-key-files_dnd_bridge_payload"],
[class*="st-key-files_dnd_bridge_nonce"],
[class*="st-key-files_dnd_bridge_apply"],
[class*="st-key-files_dnd_bridge_component"],
[class*="st-key-files_dnd_bridge_action"] > div,
[class*="st-key-files_dnd_bridge_payload"] > div,
[class*="st-key-files_dnd_bridge_nonce"] > div,
[class*="st-key-files_dnd_bridge_apply"] > div,
[class*="st-key-files_dnd_bridge_component"] > div {
  position: absolute !important;
  inset: 0 auto auto 0 !important;
  width: 1px !important;
  height: 1px !important;
  min-width: 1px !important;
  max-width: 1px !important;
  min-height: 1px !important;
  max-height: 1px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
[data-testid="stElementContainer"]:has([class*="st-key-files_dnd_bridge_action"]),
[data-testid="element-container"]:has([class*="st-key-files_dnd_bridge_action"]),
[data-testid="stElementContainer"]:has([class*="st-key-files_dnd_bridge_payload"]),
[data-testid="element-container"]:has([class*="st-key-files_dnd_bridge_payload"]),
[data-testid="stElementContainer"]:has([class*="st-key-files_dnd_bridge_nonce"]),
[data-testid="element-container"]:has([class*="st-key-files_dnd_bridge_nonce"]),
[data-testid="stElementContainer"]:has([class*="st-key-files_dnd_bridge_apply"]),
[data-testid="element-container"]:has([class*="st-key-files_dnd_bridge_apply"]),
[data-testid="stElementContainer"]:has([class*="st-key-files_dnd_bridge_component"]),
[data-testid="element-container"]:has([class*="st-key-files_dnd_bridge_component"]) {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
section[data-testid="stSidebar"] button[data-awc-files-drop-hover="1"] {
  border-color: #9cb5dc !important;
  background: #eef3fb !important;
  box-shadow: inset 0 0 0 1px #9cb5dc !important;
}
section[data-testid="stSidebar"] [data-awc-files-root-drop-hover="1"] {
  background: rgba(238, 243, 251, 0.88) !important;
  box-shadow: inset 0 0 0 1px #c8d6eb !important;
  min-height: calc(100vh - 176px) !important;
}
section[data-testid="stSidebar"] [data-awc-files-root-drop-hover="1"] > div[data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] [data-awc-files-root-drop-hover="1"] > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
section[data-testid="stSidebar"] [data-awc-files-root-drop-hover="1"] > div[data-testid="stVerticalBlock"] > [data-testid="element-container"] {
  background: transparent !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _locate_selection(full_text: str, selection_text: str) -> Tuple[int, int]:
    query = selection_text.strip()
    if not query:
        return -1, -1
    start = full_text.find(query)
    if start < 0:
        return -1, -1
    return start, start + len(query)


def _sanitize_latex_cell(cell: str) -> str:
    text = cell.strip()
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = text.replace("\\%", "%").replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _latex_table_block_to_csv(table_block: str) -> str:
    # Kept as a thin wrapper for backward compatibility inside app-level call sites.
    return latex_table_block_to_csv(table_block)


def _resolve_resource_path(path_text: str, project_root: Path) -> Path | None:
    raw = path_text.strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for base in [project_root, Path.cwd()]:
        merged = (base / raw).resolve()
        if merged.exists():
            return merged

    # If LaTeX path has no extension, try common image suffixes.
    if "." not in Path(raw).name:
        for ext in [".png", ".jpg", ".jpeg", ".pdf"]:
            for base in [project_root, Path.cwd()]:
                merged = (base / f"{raw}{ext}").resolve()
                if merged.exists():
                    return merged
    return None


def _extract_figure_id(label: str) -> str:
    match = re.search(r"(\d+)", label)
    return match.group(1) if match else "unknown"


def _load_bib_from_latex(full_text: str, project_root: Path) -> UploadedBib:
    bib_paths: List[str] = []
    for pattern in [
        r"\\bibliography\{([^}]+)\}",
        r"\\addbibresource\{([^}]+)\}",
    ]:
        for match in re.finditer(pattern, full_text):
            values = [x.strip() for x in match.group(1).split(",") if x.strip()]
            bib_paths.extend(values)

    merged_content: List[str] = []
    resolved_names: List[str] = []
    for bib_path in bib_paths:
        normalized = bib_path
        if normalized.endswith(".bib"):
            candidates = [normalized]
        else:
            candidates = [normalized, f"{normalized}.bib"]
        found = None
        for candidate in candidates:
            maybe = _resolve_resource_path(candidate, project_root)
            if maybe and maybe.exists():
                found = maybe
                break
        if not found:
            continue
        merged_content.append(found.read_text(encoding="utf-8", errors="ignore"))
        resolved_names.append(str(found))

    if not merged_content:
        fallback_paths: List[Path] = []
        try:
            for path in sorted(project_root.rglob("*.bib")):
                if not path.is_file():
                    continue
                try:
                    rel = path.relative_to(project_root).as_posix()
                except Exception:
                    rel = path.name
                if _is_internal_state_relative_path(rel):
                    continue
                fallback_paths.append(path)
                if len(fallback_paths) >= 12:
                    break
        except OSError:
            fallback_paths = []
        if not fallback_paths:
            return UploadedBib()
        fallback_content: List[str] = []
        fallback_names: List[str] = []
        for path in fallback_paths:
            try:
                fallback_content.append(path.read_text(encoding="utf-8", errors="ignore"))
                fallback_names.append(str(path))
            except OSError:
                continue
        if not fallback_content:
            return UploadedBib()
        return UploadedBib(
            content="\n\n".join(fallback_content),
            meta={"source": "workspace_bib_fallback", "paths": fallback_names},
        )

    return UploadedBib(
        content="\n\n".join(merged_content),
        meta={"source": "latex_bibliography", "paths": resolved_names},
    )


def _hydrate_assets_from_selection(
    state: AppState,
    selection_text: str,
    project_root: Path,
    selection_span: Tuple[int, int] | None = None,
    full_latex_text: str | None = None,
) -> Dict[str, Any]:
    analysis_text = str(full_latex_text if full_latex_text is not None else state.current_text or "")
    previous_metadata = {}
    if state.active_selection:
        previous_metadata = dict(state.active_selection.metadata)

    if selection_span and selection_span[0] >= 0 and selection_span[1] > selection_span[0]:
        start, end = selection_span
    else:
        start, end = _locate_selection(analysis_text, selection_text)
    if start >= 0 and end > start:
        metadata = dict(previous_metadata)
        metadata["located"] = True
        state.active_selection = SelectionContext(
            start=start,
            end=end,
            snippet=selection_text.strip(),
            sentence_indices=[],
            metadata=metadata,
        )
    else:
        metadata = dict(previous_metadata)
        metadata["located"] = False
        state.active_selection = SelectionContext(
            start=0,
            end=0,
            snippet=selection_text.strip(),
            sentence_indices=[],
            metadata=metadata,
        )

    resolved = st.session_state.resolver.resolve(
        selection_text=selection_text,
        full_latex_text=analysis_text,
    )

    uploaded_tables: List[UploadedTable] = []
    table_contexts: List[str] = []
    for table in resolved.get("tables", []):
        if not table.get("found"):
            continue
        block = str(table.get("block", ""))
        table_contexts.append(block)
        csv_content = _latex_table_block_to_csv(block)
        if not csv_content:
            continue
        uploaded_tables.append(
            UploadedTable(
                content=csv_content,
                meta={
                    "name": f"{table.get('label', 'table')}.csv",
                    "source": "latex_table",
                },
            )
        )
    state.uploaded_tables = uploaded_tables

    uploaded_figures: List[UploadedFigure] = []
    figure_contexts: List[str] = []
    seen_paths = set()
    for figure in resolved.get("figures", []):
        if not figure.get("found"):
            continue
        figure_contexts.append(str(figure.get("block", "")))
        label = str(figure.get("label", ""))
        for image_path in figure.get("image_paths", []):
            resolved_path = _resolve_resource_path(str(image_path), project_root)
            if not resolved_path or str(resolved_path) in seen_paths:
                continue
            seen_paths.add(str(resolved_path))
            try:
                content = resolved_path.read_bytes()
            except OSError:
                continue
            ext = resolved_path.suffix.lower().replace(".", "") or "png"
            uploaded_figures.append(
                UploadedFigure(
                    content=content,
                    meta={
                        "name": str(resolved_path),
                        "ext": ext,
                        "figure_id": _extract_figure_id(label),
                        "source": "latex_figure",
                    },
                )
            )
    state.uploaded_figures = uploaded_figures

    state.uploaded_bib = _load_bib_from_latex(analysis_text, project_root)

    citation_contexts = []
    if state.uploaded_bib.content:
        citation_contexts.append(state.uploaded_bib.content[:2000])

    selection_for_context = state.active_selection if state.active_selection.metadata.get("located") else None
    state.grounded_context = st.session_state.context_builder.build(
        full_text=analysis_text,
        selection=selection_for_context,
        table_contexts=table_contexts,
        figure_contexts=figure_contexts,
        citation_contexts=citation_contexts,
    )
    state.grounded_context.metadata.update(
        {
            "refs": resolved.get("refs", {}),
            "missing_labels": resolved.get("missing_labels", []),
            "auto_table_count": len(uploaded_tables),
            "auto_figure_count": len(uploaded_figures),
            "auto_bib_loaded": bool(state.uploaded_bib.content),
        }
    )
    return resolved


def _issues_json_payload(state: AppState) -> List[Dict[str, Any]]:
    return [issue.to_dict() for issue in state.issues]


def _format_patch_for_chat(diff_text: str) -> str:
    if not diff_text.strip():
        return "No patch diff is available."
    lines = diff_text.splitlines()
    if len(lines) > 220:
        lines = lines[:220] + ["...<diff truncated>"]
    return "```diff\n" + "\n".join(lines) + "\n```"


def _format_advisor_suggestions(state: AppState) -> str:
    execution = CheckExecutionArtifacts(issues=list(state.issues), metadata={"role": state.role})
    rendered = CheckArtifactBuilder(state.config.text_provider_name).build(state, execution).rendered_issues
    if not rendered:
        return "No issue-specific suggestion is available."
    lines = ["Here are targeted suggestions based on detected issues:"]
    for idx, item in enumerate(rendered[:5], start=1):
        fix = item.suggested_fix or "Review this issue manually."
        lines.append(f"{idx}. {item.issue.type}: {fix}")
    lines.append("If you want, I can next convert these into an Editor patch.")
    return "\n".join(lines)


def _render_diff_html(diff_text: str) -> None:
    if not diff_text.strip():
        st.info("No patch diff is currently available.")
        return
    lines = diff_text.splitlines()
    html_lines: List[str] = []
    for line in lines:
        escaped = html.escape(line)
        if line.startswith("+") and not line.startswith("+++"):
            html_lines.append(
                f'<div style="background:#e6ffed;color:#116329;padding:2px 10px;">{escaped}</div>'
            )
        elif line.startswith("-") and not line.startswith("---"):
            html_lines.append(
                f'<div style="background:#ffebe9;color:#82071e;padding:2px 10px;">{escaped}</div>'
            )
        else:
            html_lines.append(
                f'<div style="background:#f6f8fa;color:#24292f;padding:2px 10px;">{escaped}</div>'
            )
    block = (
        '<div style="border:1px solid #d0d7de;border-radius:8px;overflow:auto;max-height:280px;'
        'font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;">'
        + "".join(html_lines)
        + "</div>"
    )
    st.markdown(block, unsafe_allow_html=True)


def _render_assistant_avatar(role: str, key: str, height: int = 74, compact: bool = False) -> None:
    normalized = _normalize_role_name(role)
    if not normalized:
        return

    # Always use deterministic animated fallback avatar for stable UX across environments.
    _render_fallback_companion(normalized, key=key, compact=compact)


def _chat_role_for_message(state: AppState, turn: ChatTurn) -> str:
    metadata_role = str(turn.metadata.get("companion_role", "")) if turn.metadata else ""
    normalized_metadata = _normalize_role_name(metadata_role)
    if normalized_metadata:
        return normalized_metadata

    normalized_current = _normalize_role_name(st.session_state.get("current_role", ""))
    if normalized_current:
        return normalized_current

    normalized_state = _normalize_role_name(state.role)
    return normalized_state or "Reviewer"


def _on_composer_submit() -> None:
    st.session_state.composer_triggered = True


def _writing_help_reply(state: AppState, user_text: str) -> str:
    return _memory_aware_role_chat_reply(state, user_text)


def _persist_state(state: AppState) -> None:
    _save_active_workspace_snapshot(touch_activity=True)
    st.session_state.state_manager.persist_state(
        doc_id=state.state_doc_id,
        state=state,
        request_id=state.request_id,
    )


def _autosave_main_tex_if_needed(state: AppState) -> None:
    workspace_id = str(state.state_doc_id or "").strip()
    if not workspace_id:
        return
    active_file_path = _active_file_for_state(state)
    current_text = str(state.current_text or "")
    current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
    hash_key = f"{workspace_id}:{active_file_path}"
    hashes = dict(st.session_state.get("workspace_text_hashes", {}))
    if hashes.get(hash_key) == current_hash:
        return
    save_project_text_file(workspace_id, active_file_path, current_text)
    _mark_workspace_activity(workspace_id)
    hashes[hash_key] = current_hash
    st.session_state.workspace_text_hashes = hashes
    store = st.session_state.get("workspace_store", {})
    payload = store.get(workspace_id)
    if isinstance(payload, dict):
        payload["state"] = copy.deepcopy(state)
        ui_payload = dict(payload.get("ui", {}))
        ui_payload["active_file_path"] = active_file_path
        payload["ui"] = ui_payload


def _undo_last_change(state: AppState) -> str:
    snapshot = st.session_state.state_manager.pop_history_snapshot(state.state_doc_id)
    if not snapshot:
        return ""
    previous_text = str(snapshot.get("text", ""))
    state.current_text = previous_text
    return "Undo successful."


def _apply_patch_with_history(state: AppState) -> str:
    st.session_state.state_manager.append_history_snapshot(
        doc_id=state.state_doc_id,
        text=state.current_text,
        request_id=state.request_id,
    )
    state, message = _patch_manager_for_state(state).apply_patch(state)
    _arm_editor_backend_text_sync(state)
    workspace_id = str(state.state_doc_id or "").strip()
    if workspace_id:
        _force_save_active_editor_file(workspace_id)
    st.session_state.app_state = state
    return message


def _handle_analyze_selection(
    state: AppState,
    selection_text: str,
    project_root: Path,
    enabled_checks: List[str],
) -> None:
    # Compatibility wrapper. Legacy calls are forced through Router so checks do not bypass the mainline.
    pending_action = {
        "action": "run_checks",
        "submit_text": "",
        "selected_checks": [x for x in enabled_checks if x in CHECK_NAME_SET],
        "ui_checks_explicit": True,
    }
    _dispatch_router_pending_action(state, pending_action)


def _handle_chat_input(state: AppState, user_input: str, trigger_rerun: bool = True) -> None:
    # Compatibility wrapper. Legacy direct chat entry now always routes through Router.
    if not str(user_input or "").strip():
        return
    pending_action = {
        "action": "send_message",
        "submit_text": str(user_input or "").strip(),
        "selected_checks": [],
        "ui_checks_explicit": False,
    }
    _dispatch_router_pending_action(state, pending_action)
    if trigger_rerun:
        st.rerun()


def _render_chat_history(state: AppState, height: int = 620, border: bool = True) -> None:
    chat_box = st.container(height=height, border=border)
    with chat_box:
        st.markdown("<div class='awc-chat-history-marker'></div>", unsafe_allow_html=True)
        for turn in state.chat_history[-50:]:
            if turn.role == "user":
                with st.chat_message("user", avatar="😃"):
                    st.markdown(turn.content)
                    _render_chat_copy_marker(turn.content, "copy message")
                continue

            role_for_avatar = _chat_role_for_message(state, turn)
            avatar_path = ROLE_CHAT_AVATAR_FILES.get(role_for_avatar, ROLE_CHAT_AVATAR_FILES["Reviewer"])
            with st.chat_message("assistant", avatar=avatar_path):
                st.markdown(turn.content)
                st.caption(role_for_avatar)
                _render_chat_copy_marker(turn.content, "copy response")


def _render_chat_copy_marker(content: str, tooltip_label: str) -> None:
    payload = base64.b64encode(str(content or "").encode("utf-8")).decode("ascii")
    safe_label = html.escape(tooltip_label, quote=True)
    st.markdown(
        f"""
<div
  class="awc-chat-copy-marker"
  data-awc-copy-b64="{payload}"
  data-awc-copy-label="{safe_label}"
  aria-hidden="true"
></div>
""",
        unsafe_allow_html=True,
    )


def _render_floating_companion(role: str) -> None:
    normalized = _normalize_role_name(role)
    if not normalized:
        return
    accent_map = {
        "Reviewer": "#2563EB",
        "Advisor": "#0891B2",
        "Editor": "#16A34A",
    }
    surface_map = {
        "Reviewer": "#DBEAFE",
        "Advisor": "#CFFAFE",
        "Editor": "#DCFCE7",
    }
    avatar_path = Path(
        ROLE_CHAT_AVATAR_FILES.get(normalized, ROLE_CHAT_AVATAR_FILES["Reviewer"])
    )
    try:
        avatar_bytes = avatar_path.read_bytes()
    except OSError:
        return
    avatar_data_uri = (
        "data:image/svg+xml;base64," + base64.b64encode(avatar_bytes).decode("utf-8")
    )
    st.markdown(
        f"""
<style>
#floating-companion {{
  position: fixed;
  right: 22px;
  top: 248px;
  width: 96px;
  height: 112px;
  z-index: 1004;
  pointer-events: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  animation: floatingCompanionBob 2.6s ease-in-out infinite;
  cursor: grab;
  user-select: none;
  touch-action: none;
}}
#floating-companion.dragging {{
  animation-play-state: paused;
  cursor: grabbing;
}}
#floating-companion .bubble {{
  width: 76px;
  height: 76px;
  border-radius: 999px;
  background: {surface_map[normalized]};
  border: 2px solid {accent_map[normalized]};
  box-shadow: 0 10px 22px rgba(15,23,42,0.15);
  display: flex;
  align-items: center;
  justify-content: center;
}}
#floating-companion .bubble img {{
  width: 54px;
  height: 54px;
}}
#floating-companion .name {{
  margin-top: 6px;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  color: #0f172a;
  background: white;
  border: 1px solid #e2e8f0;
}}
body.awc-chat-drawer-open #floating-companion {{
  right: min(calc({CHAT_DRAWER_WIDTH_PX}px + 24px), calc(92vw + 24px));
}}
@keyframes floatingCompanionBob {{
  0%, 100% {{ transform: translateY(0px); }}
  50% {{ transform: translateY(-6px); }}
}}
</style>
<div id="floating-companion">
  <div class="bubble">
    <img src="{avatar_data_uri}" alt="{normalized} avatar"/>
  </div>
  <div class="name">{normalized}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st_components.html(
        """
<script>
(() => {
  const doc = window.parent && window.parent.document ? window.parent.document : null;
  if (!doc) return;
  const hostWin = window.parent || window;
  const companion = doc.getElementById("floating-companion");
  if (!companion) return;
  if (companion.dataset.dragReady === "1") return;
  companion.dataset.dragReady = "1";

  const parentWin = window.parent || window;
  const storageKey = "awc_floating_companion_top";
  const minTop = 72;
  const viewportPadding = 20;
  const clampTop = (candidate) => {
    const maxTop = Math.max(minTop, parentWin.innerHeight - companion.offsetHeight - viewportPadding);
    return Math.max(minTop, Math.min(candidate, maxTop));
  };
  const safeGetStoredTop = () => {
    try {
      return parentWin.localStorage.getItem(storageKey);
    } catch (_err) {
      return null;
    }
  };
  const safeSetStoredTop = (value) => {
    try {
      parentWin.localStorage.setItem(storageKey, value);
    } catch (_err) {
      // Ignore storage errors.
    }
  };

  const savedTop = Number.parseFloat(safeGetStoredTop() || "");
  if (!Number.isNaN(savedTop)) {
    companion.style.top = String(clampTop(savedTop)) + "px";
  }

  let dragging = false;
  let pointerOffsetY = 0;

  const onMouseMove = (event) => {
    if (!dragging) return;
    const targetTop = clampTop(event.clientY - pointerOffsetY);
    companion.style.top = String(targetTop) + "px";
  };
  const onMouseUp = () => {
    if (!dragging) return;
    dragging = false;
    companion.classList.remove("dragging");
    const finalTop = Number.parseFloat(companion.style.top || "");
    if (!Number.isNaN(finalTop)) {
      safeSetStoredTop(String(finalTop));
    }
    doc.removeEventListener("mousemove", onMouseMove, true);
    doc.removeEventListener("mouseup", onMouseUp, true);
  };

  companion.addEventListener("mousedown", (event) => {
    if (event.button !== 0) return;
    const rect = companion.getBoundingClientRect();
    dragging = true;
    pointerOffsetY = event.clientY - rect.top;
    companion.classList.add("dragging");
    doc.addEventListener("mousemove", onMouseMove, true);
    doc.addEventListener("mouseup", onMouseUp, true);
    event.preventDefault();
  });

  parentWin.addEventListener("resize", () => {
    const currentTop = Number.parseFloat(
      companion.style.top || String(companion.getBoundingClientRect().top)
    );
    if (Number.isNaN(currentTop)) return;
    const adjustedTop = clampTop(currentTop);
    companion.style.top = String(adjustedTop) + "px";
    safeSetStoredTop(String(adjustedTop));
  });
})();
</script>
""",
        height=0,
        scrolling=False,
    )

def _resolve_page_icon() -> Any:
    if PAGE_ICON_PATH.exists():
        return str(PAGE_ICON_PATH)
    return "🤖"


def _resolve_page_icon_data_url() -> str:
    if not PAGE_ICON_PATH.exists():
        return ""
    try:
        encoded = base64.b64encode(PAGE_ICON_PATH.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded}"


def _inject_favicon_override() -> None:
    icon_data_url = _resolve_page_icon_data_url()
    if not icon_data_url:
        return
    script_html = f"""
<script>
(() => {{
  const iconHref = {json.dumps(icon_data_url)};
  const parentDoc = (window.parent && window.parent.document) ? window.parent.document : document;
  const applyIcon = () => {{
    try {{
      let links = Array.from(parentDoc.querySelectorAll('link[rel*="icon"]'));
      if (!links.length) {{
        const link = parentDoc.createElement("link");
        link.setAttribute("rel", "icon");
        links = [link];
        parentDoc.head.appendChild(link);
      }}
      links.forEach((link) => {{
        link.setAttribute("type", "image/png");
        link.setAttribute("href", iconHref);
      }});
    }} catch (_error) {{
      // no-op
    }}
  }};

  applyIcon();
  const intervalId = window.setInterval(applyIcon, 100);
  window.setTimeout(() => window.clearInterval(intervalId), 4500);

  try {{
    const observer = new MutationObserver(applyIcon);
    observer.observe(parentDoc.documentElement, {{ childList: true, subtree: true }});
    window.setTimeout(() => observer.disconnect(), 4500);
  }} catch (_error) {{
    // no-op
  }}
}})();
</script>
"""
    st_components.html(script_html, height=0, scrolling=False)


def _inject_scroll_to_top_on_refresh() -> None:
    st_components.html(
        """
<script>
(() => {
  const parentWin = window.parent || window;
  const parentDoc = (parentWin && parentWin.document) ? parentWin.document : document;
  if (!parentDoc || !parentDoc.body) return;

  const controllerKey = "__awcScrollTopControllerV3";
  const existing = parentWin[controllerKey];
  if (existing && typeof existing.stop === "function") {
    try {
      existing.stop();
    } catch (_err) {
      // No-op.
    }
  }

  try {
    if (parentWin.history && "scrollRestoration" in parentWin.history) {
      parentWin.history.scrollRestoration = "manual";
    }
  } catch (_err) {
    // No-op.
  }

  const state = {
    stopped: false,
    rafId: 0,
    timers: [],
  };
  state.stop = () => {
    state.stopped = true;
    if (state.rafId) {
      try {
        parentWin.cancelAnimationFrame(state.rafId);
      } catch (_err) {
        // No-op.
      }
      state.rafId = 0;
    }
    while (state.timers.length) {
      const timerId = state.timers.pop();
      try {
        parentWin.clearTimeout(timerId);
      } catch (_err) {
        // No-op.
      }
    }
  };
  parentWin[controllerKey] = state;

  const scheduleTimeout = (callback, delay) => {
    const timerId = parentWin.setTimeout(() => {
      callback();
    }, delay);
    state.timers.push(timerId);
  };

  const forceScrollableElementsToTop = () => {
    const seen = new Set();
    const targets = [];
    const addTarget = (target) => {
      if (!target || seen.has(target)) return;
      seen.add(target);
      targets.push(target);
    };

    addTarget(parentDoc.scrollingElement);
    addTarget(parentDoc.documentElement);
    addTarget(parentDoc.body);
    addTarget(parentDoc.querySelector('[data-testid="stAppViewContainer"]'));
    addTarget(parentDoc.querySelector('[data-testid="stAppViewContainer"] .main'));
    addTarget(parentDoc.querySelector('[data-testid="stAppViewContainer"] .main .block-container'));
    addTarget(parentDoc.querySelector('section[data-testid="stSidebar"]'));
    addTarget(parentDoc.querySelector('section[data-testid="stSidebar"] > div'));

    const allNodes = parentDoc.querySelectorAll("*");
    allNodes.forEach((node) => {
      if (!(node instanceof parentWin.HTMLElement)) return;
      if (node.dataset && node.dataset.awcChatHistoryScrollShell === "1") return;
      const style = parentWin.getComputedStyle(node);
      const overflowY = `${style.overflowY || ""} ${style.overflow || ""}`;
      if (!/(auto|scroll|overlay)/i.test(overflowY)) return;
      if (node.scrollHeight <= node.clientHeight + 4) return;
      addTarget(node);
    });

    targets.forEach((target) => {
      try {
        if (typeof target.scrollTop === "number") {
          target.scrollTop = 0;
        }
      } catch (_err) {
        // No-op.
      }
    });
  };

  const forceTop = () => {
    if (state.stopped) return;
    try {
      parentWin.scrollTo(0, 0);
    } catch (_err) {
      // No-op.
    }
    forceScrollableElementsToTop();
  };

  let rafCount = 0;
  const maxRaf = 180;
  const tick = () => {
    forceTop();
    if (state.stopped) return;
    rafCount += 1;
    if (rafCount < maxRaf) {
      state.rafId = parentWin.requestAnimationFrame(tick);
    }
  };

  const stopOnUserScrollIntent = () => {
    state.stop();
  };
  try {
    parentWin.addEventListener("wheel", stopOnUserScrollIntent, { passive: true, once: true });
    parentWin.addEventListener("touchmove", stopOnUserScrollIntent, { passive: true, once: true });
    parentWin.addEventListener("keydown", stopOnUserScrollIntent, { passive: true, once: true });
  } catch (_err) {
    // No-op.
  }

  const beforeUnloadKey = "__awcScrollTopBeforeUnloadBoundV1";
  if (!parentWin[beforeUnloadKey]) {
    parentWin[beforeUnloadKey] = true;
    parentWin.addEventListener("beforeunload", () => {
      try {
        parentWin.scrollTo(0, 0);
      } catch (_err) {
        // No-op.
      }
      try {
        forceScrollableElementsToTop();
      } catch (_err) {
        // No-op.
      }
    });
  }

  state.rafId = parentWin.requestAnimationFrame(tick);
  scheduleTimeout(forceTop, 80);
  scheduleTimeout(forceTop, 220);
  scheduleTimeout(forceTop, 420);
  scheduleTimeout(forceTop, 760);
  scheduleTimeout(forceTop, 1200);
  scheduleTimeout(forceTop, 1800);
  scheduleTimeout(forceTop, 2600);
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _inject_chat_history_scroll_to_bottom() -> None:
    st_components.html(
        """
<script>
(() => {
  const parentWin = window.parent || window;
  const parentDoc = (parentWin && parentWin.document) ? parentWin.document : document;
  if (!parentDoc || !parentDoc.body) return;

  const controllerKey = "__awcChatHistoryBottomControllerV1";
  const existing = parentWin[controllerKey];
  if (existing && typeof existing.stop === "function") {
    try {
      existing.stop();
    } catch (_err) {
      // No-op.
    }
  }

  const state = {
    stopped: false,
    rafId: 0,
    timers: [],
  };
  state.stop = () => {
    state.stopped = true;
    if (state.rafId) {
      try {
        parentWin.cancelAnimationFrame(state.rafId);
      } catch (_err) {
        // No-op.
      }
      state.rafId = 0;
    }
    while (state.timers.length) {
      const timerId = state.timers.pop();
      try {
        parentWin.clearTimeout(timerId);
      } catch (_err) {
        // No-op.
      }
    }
  };
  parentWin[controllerKey] = state;

  const scheduleTimeout = (callback, delay) => {
    const timerId = parentWin.setTimeout(() => {
      callback();
    }, delay);
    state.timers.push(timerId);
  };

  const isScrollable = (node) => {
    if (!(node instanceof parentWin.HTMLElement)) return false;
    const style = parentWin.getComputedStyle(node);
    const overflowY = `${style.overflowY || ""} ${style.overflow || ""}`;
    if (!/(auto|scroll|overlay)/i.test(overflowY)) return false;
    return node.scrollHeight > node.clientHeight + 2;
  };

  const findScrollTargetFrom = (seed) => {
    let node = seed;
    for (let i = 0; i < 8 && node; i += 1) {
      if (isScrollable(node)) return node;
      node = node.parentElement;
    }
    if (!(seed instanceof parentWin.HTMLElement) || !seed.querySelectorAll) return null;
    const descendants = seed.querySelectorAll("*");
    for (const child of descendants) {
      if (isScrollable(child)) return child;
    }
    return null;
  };

  const collectChatTargets = () => {
    const markers = Array.from(parentDoc.querySelectorAll(".awc-chat-history-marker"));
    const targets = [];
    const seen = new Set();
    const add = (node) => {
      if (!node || seen.has(node)) return;
      seen.add(node);
      node.dataset.awcChatHistoryScrollShell = "1";
      targets.push(node);
    };

    markers.forEach((marker) => {
      const shell =
        marker.closest('[data-testid="stVerticalBlockBorderWrapper"]')
        || marker.closest('[data-testid="stVerticalBlock"]')
        || marker.parentElement;
      const target = findScrollTargetFrom(shell) || findScrollTargetFrom(marker.parentElement);
      if (target) add(target);
    });
    return targets;
  };

  const scrollChatsToBottom = () => {
    if (state.stopped) return;
    const targets = collectChatTargets();
    targets.forEach((target) => {
      try {
        target.scrollTop = target.scrollHeight;
      } catch (_err) {
        // No-op.
      }
    });
  };

  let rafCount = 0;
  const maxRaf = 160;
  const tick = () => {
    scrollChatsToBottom();
    if (state.stopped) return;
    rafCount += 1;
    if (rafCount < maxRaf) {
      state.rafId = parentWin.requestAnimationFrame(tick);
    }
  };

  state.rafId = parentWin.requestAnimationFrame(tick);
  scheduleTimeout(scrollChatsToBottom, 80);
  scheduleTimeout(scrollChatsToBottom, 220);
  scheduleTimeout(scrollChatsToBottom, 420);
  scheduleTimeout(scrollChatsToBottom, 760);
  scheduleTimeout(scrollChatsToBottom, 1200);
  scheduleTimeout(scrollChatsToBottom, 1800);
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _apply_workspace_layout_style() -> None:
    css_text = """
<style>
[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 100% !important;
  position: relative !important;
  left: -1.35rem !important;
  width: calc(100% + 1.35rem) !important;
  padding-left: 0.2rem !important;
  padding-top: 0 !important;
  padding-bottom: 0.15rem !important;
}
[data-testid="stAppViewContainer"] .main h1 {
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
[data-testid="stAppViewContainer"] {
  overflow-x: hidden !important;
}
[data-testid="stAppViewContainer"] .main .stElementContainer:has(iframe[title^="viewport_probe_v1"]) {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stAppViewContainer"] .main .stElementContainer:has(iframe[title^="export_hint_mount_v1"]) {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
/* Control the real Streamlit row gap at the top-level vertical layout. */
[data-testid="stAppViewContainer"] .main .block-container > div[data-testid="stVerticalBlock"],
[data-testid="stAppViewContainer"] .main .block-container > div > div[data-testid="stVerticalBlock"] {
  gap: __TOP_LEVEL_ROW_GAP_REM__rem !important;
}
/* Composer container anchors its own down-expanding panel without affecting chat scroll shell. */
[data-testid="stElementContainer"]:has(iframe[title^="chat_composer_v"]) {
  position: relative !important;
  overflow: visible !important;
  z-index: 20 !important;
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
.awc-chat-history-marker {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
[data-testid="stElementContainer"]:has(.awc-chat-history-marker),
[data-testid="element-container"]:has(.awc-chat-history-marker) {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
/* Remove default Streamlit vertical gaps inside the right chat column. */
[data-testid="stVerticalBlock"]:has(iframe[title^="chat_composer_v"]) > [data-testid="stElementContainer"] {
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
[data-testid="stAppViewContainer"] .main div[data-testid="stMarkdownContainer"] h4 {
  margin-top: __PANEL_TITLE_MARGIN_TOP_REM__rem !important;
  margin-bottom: __PANEL_TITLE_MARGIN_BOTTOM_REM__rem !important;
}
[data-testid="stVerticalBlock"]:has(.awc-editor-panel-marker) > [data-testid="stHorizontalBlock"]:first-of-type,
[data-testid="stVerticalBlock"]:has(.awc-pdf-panel-marker) > [data-testid="stHorizontalBlock"]:first-of-type {
  height: 2.38rem !important;
  min-height: 2.38rem !important;
  align-items: center !important;
}
[data-testid="stVerticalBlock"]:has(.awc-editor-panel-marker) > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stMarkdownContainer"],
[data-testid="stVerticalBlock"]:has(.awc-pdf-panel-marker) > [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stMarkdownContainer"] {
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stVerticalBlock"]:has(.awc-editor-panel-marker) > [data-testid="stHorizontalBlock"]:first-of-type .stButton > button,
[data-testid="stVerticalBlock"]:has(.awc-pdf-panel-marker) > [data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
  height: 2.32rem !important;
  min-height: 2.32rem !important;
  margin: 0 !important;
}

[data-testid="stAppViewContainer"] .main [data-testid="stHorizontalBlock"]:first-of-type {
  margin-bottom: -0.58rem !important;
  padding-bottom: 0 !important;
}
[data-testid="stAppViewContainer"] .main .block-container > div[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]:nth-of-type(2),
[data-testid="stAppViewContainer"] .main .block-container > div > div[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]:nth-of-type(2) {
  margin-top: -__HEADER_TO_PANEL_PULLUP_REM__rem !important;
}
[data-testid="stHorizontalBlock"] {
  align-items: flex-start;
}
/* Reduce extra bottom spacing in the first header row controls. */
[data-testid="stAppViewContainer"] .main [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stSelectbox"],
[data-testid="stAppViewContainer"] .main [data-testid="stHorizontalBlock"]:first-of-type [data-testid="stTextInput"] {
  margin-bottom: 0 !important;
}

/* Fix sidebar width only when expanded and disable manual resizing */
section[data-testid="stSidebar"][aria-expanded="true"] {
  min-width: 286px !important;
  max-width: 286px !important;
}
section[data-testid="stSidebar"][aria-expanded="true"] > div {
  min-width: 286px !important;
  max-width: 286px !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
  min-width: 0 !important;
  max-width: 0 !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] > div {
  min-width: 0 !important;
  max-width: 0 !important;
}
[data-testid="stSidebarResizeHandle"] {
  display: none !important;
  width: 0 !important;
}

/* Sidebar panel buttons (exclude icon bar): active light gray, inactive white */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) .stButton > button {
  border-radius: 12px !important;
  border: 1px solid #e5e7eb !important;
  background: #ffffff !important;
  color: #111827 !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) .stButton > button[kind="primary"] {
  background: #f3f4f6 !important;
  border-color: #d1d5db !important;
  color: #111827 !important;
}
section[data-testid="stSidebar"] button[data-awc-files-node-kind][data-awc-files-node-path] {
  overflow: hidden !important;
}
section[data-testid="stSidebar"] button[data-awc-files-node-kind="file"][data-awc-files-node-path] {
  text-align: left !important;
}
section[data-testid="stSidebar"] button[data-awc-files-node-kind="file"][data-awc-files-node-path] > div {
  width: 100% !important;
  min-width: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 0.68rem !important;
  padding-left: 0.82rem !important;
  padding-right: 0.42rem !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
}
section[data-testid="stSidebar"] button[data-awc-files-node-kind="file"][data-awc-files-node-path] > div > span:first-child {
  flex: 0 0 auto !important;
}
section[data-testid="stSidebar"] button[data-awc-files-node-kind="file"][data-awc-files-node-path] p {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  max-width: 100% !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  text-align: left !important;
  margin: 0 !important;
}
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button {
  border-radius: 12px !important;
  border: 1px solid #e5e7eb !important;
  background: #ffffff !important;
  color: #111827 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button[kind="primary"],
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button[kind="primary"] {
  background: #f3f4f6 !important;
  border-color: #d1d5db !important;
  color: #111827 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button:hover,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button:hover,
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button:focus,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button:focus,
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button:focus-visible,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button:focus-visible,
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button:active,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button:active {
  border-color: #d1d5db !important;
  color: #111827 !important;
  box-shadow: none !important;
  outline: none !important;
}
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button[kind="primary"]:hover,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button[kind="primary"]:hover,
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button[kind="primary"]:focus,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button[kind="primary"]:focus,
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button[kind="primary"]:focus-visible,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button[kind="primary"]:focus-visible,
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button[kind="primary"]:active,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button[kind="primary"]:active {
  background: #f3f4f6 !important;
  border-color: #d1d5db !important;
  color: #111827 !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button {
  text-align: left !important;
}
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button > div,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button > div {
  width: 100% !important;
  min-width: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 0.68rem !important;
  padding-left: 0.82rem !important;
  padding-right: 0.42rem !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
}
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button > div > span:first-child,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button > div > span:first-child {
  flex: 0 0 auto !important;
  margin-left: 0 !important;
}
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button p,
section[data-testid="stSidebar"] [class*="st-key-files_toggle_long_"] button p {
  flex: 1 1 auto !important;
  min-width: 0 !important;
  max-width: 100% !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  text-align: left !important;
}
/* Long file-name row only: keep icon column aligned with normal file rows. */
section[data-testid="stSidebar"] [class*="st-key-files_open_long_"] button > div {
  padding-left: 0.82rem !important;
  justify-content: flex-start !important;
  gap: 0.68rem !important;
}
/* Long file-name file rows only: shift icon + text left by 0.2rem. */
section[data-testid="stSidebar"] button[data-awc-files-node-kind="file"][data-awc-files-long-name="1"] > div {
  padding-left: 0.15rem !important;
}
/* Exclude icon-bar nav buttons from generic white card button styling. */
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_chats_btn"],
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_files_btn"],
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_chats_btn"] > div,
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_files_btn"] > div,
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_chats_btn"] .stButton,
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_files_btn"] .stButton,
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_chats_btn"] .stButton > button,
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_files_btn"] .stButton > button {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_chats_btn"] .stButton > button,
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_files_btn"] .stButton > button {
  min-height: 8px !important;
  height: 8px !important;
  min-width: 8px !important;
  width: 8px !important;
  max-width: 8px !important;
  padding: 0 !important;
}

/* Input and multiselect in neutral gray theme */
div[data-baseweb="input"] > div {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
div[data-baseweb="input"] > div:focus-within {
  border-color: #9ca3af !important;
  box-shadow: 0 0 0 1px #9ca3af !important;
}
div[data-baseweb="input"] > div[aria-invalid="true"] {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
div[data-baseweb="input"] > div[aria-invalid="true"]:focus-within {
  border-color: #9ca3af !important;
  box-shadow: 0 0 0 1px #9ca3af !important;
}
div[data-testid="stTextInput"] > div > div {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
div[data-testid="stTextInput"] > div > div:focus-within {
  border-color: #9ca3af !important;
  box-shadow: 0 0 0 1px #9ca3af !important;
}
/* LaTeX Project Root: gray fill and no red/invalid highlight while editing. */
[class*="st-key-project_root_input"] [data-testid="stTextInput"] > div > div,
[class*="st-key-project_root_input"] [data-testid="stTextInput"] > div > div:focus-within,
[class*="st-key-project_root_input"] [data-testid="stTextInput"] div[data-baseweb="base-input"],
[class*="st-key-project_root_input"] [data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
[class*="st-key-project_root_input"] [data-testid="stTextInput"] div[data-baseweb="base-input"][data-invalid="true"],
[class*="st-key-project_root_input"] [data-testid="stTextInput"] div[data-baseweb="input"] > div,
[class*="st-key-project_root_input"] [data-testid="stTextInput"] div[data-baseweb="input"] > div:focus-within,
[class*="st-key-project_root_input"] [data-testid="stTextInput"] div[data-baseweb="input"] > div[aria-invalid="true"] {
  background: #f3f4f6 !important;
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
[class*="st-key-project_root_input"] [data-testid="stTextInput"] input {
  background: transparent !important;
  color: #111827 !important;
  caret-color: #111827 !important;
}
[class*="st-key-project_root_input"] [aria-invalid="true"],
[class*="st-key-project_root_input"] [data-invalid="true"],
[class*="st-key-project_root_input"] [aria-invalid="true"]:focus-within,
[class*="st-key-project_root_input"] [data-invalid="true"]:focus-within {
  border-color: #d1d5db !important;
  box-shadow: none !important;
  outline: none !important;
}
[class*="st-key-project_root_input"] [data-baseweb="base-input"] svg {
  display: none !important;
}
[class*="st-key-project_root_input"] [data-baseweb="base-input"] {
  --st-color-error: #d1d5db !important;
  --red70: #d1d5db !important;
}
section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-baseweb="base-input"]:focus-within {
  border-color: #9ca3af !important;
  box-shadow: 0 0 0 1px #9ca3af !important;
}
section[data-testid="stSidebar"] div[data-baseweb="base-input"][data-invalid="true"] {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
div[data-baseweb="tag"] {
  background: #f3f4f6 !important;
  color: #374151 !important;
  border: 1px solid #e5e7eb !important;
}
div[data-baseweb="select"] > div {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
div[data-baseweb="select"] > div:focus-within {
  border-color: #9ca3af !important;
  box-shadow: 0 0 0 1px #9ca3af !important;
}
/* Role select keeps selectbox appearance but disallows text editing/typing. */
[class*="st-key-current_role"] div[data-baseweb="select"] input {
  pointer-events: none !important;
  user-select: none !important;
  caret-color: transparent !important;
}
[class*="st-key-current_role"] div[data-baseweb="select"] input::selection {
  background: transparent !important;
}
[role="listbox"] [aria-selected="true"] {
  background: #f3f4f6 !important;
  color: #111827 !important;
}
[role="listbox"] [aria-selected="true"] * {
  color: #111827 !important;
}
[role="listbox"] [aria-selected="false"] {
  background: #ffffff !important;
}

/* Files uploader in sidebar popovers: keep compact, hide drag-drop guidance block. */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzoneInstructions"] {
  display: none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] {
  min-height: auto !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] section,
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] section {
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] button {
  margin: 0 auto !important;
  width: 8.25rem !important;
  min-width: 8.25rem !important;
  height: 2.4rem !important;
  min-height: 2.4rem !important;
  padding: 0.35rem 0.85rem !important;
  display: block !important;
  justify-content: center !important;
}
[class*="st-key-files_upload_btn_"] .stButton > button {
  width: 8.25rem !important;
  min-width: 8.25rem !important;
  height: 2.4rem !important;
  min-height: 2.4rem !important;
  padding: 0.35rem 0.85rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
  align-self: center !important;
  display: block !important;
  justify-content: center !important;
}

/* Checkbox/radio accent keep neutral gray rather than warm highlight */
[data-baseweb="checkbox"] div[aria-checked="true"] {
  background-color: #6b7280 !important;
  border-color: #6b7280 !important;
}
[data-baseweb="checkbox"] div[aria-checked="false"] {
  border-color: #9ca3af !important;
}
[data-baseweb="radio"] div[aria-checked="true"] {
  border-color: #6b7280 !important;
}
[data-baseweb="radio"] div[aria-checked="true"] > div {
  background-color: #6b7280 !important;
}

/* Keep action buttons gray */
.stButton > button[kind="primary"] {
  background: #f3f4f6 !important;
  color: #111827 !important;
  border: 1px solid #d1d5db !important;
}

/* Composer text area keeps GPT-like gray focus and supports long text */
div[data-testid="stTextArea"] textarea {
  border: 1px solid #d1d5db !important;
  border-radius: 14px !important;
  box-shadow: none !important;
  background: #ffffff !important;
  color: #111827 !important;
  line-height: 1.5 !important;
}
div[data-testid="stTextArea"] textarea:focus {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
  color: #9ca3af !important;
  opacity: 1 !important;
}

/* Sidebar panel dot button visual centering (exclude icon bar) */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) .stButton > button {
  min-height: 2.34rem !important;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_chats_btn"] .stButton > button,
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_files_btn"] .stButton > button {
  min-height: 8px !important;
  height: 8px !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) .stButton > button:has(span:only-child) {
  padding-left: 0 !important;
  padding-right: 0 !important;
  min-width: 42px !important;
  text-align: center !important;
}
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_chats_btn"] .stButton > button:has(span:only-child),
[data-testid="stSidebar"] [class*="st-key-sidebar_nav_files_btn"] .stButton > button:has(span:only-child) {
  min-width: 8px !important;
  width: 8px !important;
  max-width: 8px !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverButton"] {
  min-height: 2.34rem !important;
  min-width: 42px !important;
  border-radius: 12px !important;
  border: 1px solid #e5e7eb !important;
  background: #ffffff !important;
  color: #111827 !important;
  padding-left: 0.56rem !important;
  padding-right: 0.56rem !important;
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  overflow: visible !important;
  position: relative !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin-left: 0 !important;
  margin-right: auto !important;
  transition: transform 0.14s ease, background-color 0.18s ease !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverButton"]::before {
  content: none !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverButton"] > div {
  margin-right: 0 !important;
  width: 100% !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0 !important;
  opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverButton"] p {
  margin: 0 !important;
  width: 100% !important;
  text-align: center !important;
}
/* Keep row-menu (⋯) popover trigger compact and centered in right column rows. */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) div[data-testid="stHorizontalBlock"]:not(:has(.awc-files-actions-row-marker)) > div:nth-child(2) [data-testid="stPopoverButton"] {
  width: 2.65rem !important;
  min-width: 2.65rem !important;
  max-width: 2.65rem !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  margin-left: auto !important;
  margin-right: auto !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) div[data-testid="stHorizontalBlock"]:not(:has(.awc-files-actions-row-marker)) > div:nth-child(2) [data-testid="stPopoverButton"] > div {
  width: auto !important;
  min-width: 0 !important;
  margin: 0 auto !important;
  flex: 0 0 auto !important;
  transform: translateX(0.05rem) !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) div[data-testid="stHorizontalBlock"]:not(:has(.awc-files-actions-row-marker)) > div:nth-child(2) [data-testid="stPopoverButton"] p,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) div[data-testid="stHorizontalBlock"]:not(:has(.awc-files-actions-row-marker)) > div:nth-child(2) [data-testid="stPopoverButton"] span {
  width: auto !important;
  min-width: 0 !important;
  margin: 0 !important;
  transform: translateX(0.05rem) !important;
  text-align: center !important;
  line-height: 1 !important;
}
/* Keep Chats row-menu (⋯) at original center position. */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) div[data-testid="stHorizontalBlock"]:has([class*="st-key-workspace_switch_"]) > div:nth-child(2) [data-testid="stPopoverButton"] > div,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) div[data-testid="stHorizontalBlock"]:has([class*="st-key-workspace_switch_"]) > div:nth-child(2) [data-testid="stPopoverButton"] p,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) div[data-testid="stHorizontalBlock"]:has([class*="st-key-workspace_switch_"]) > div:nth-child(2) [data-testid="stPopoverButton"] span {
  transform: none !important;
}
/* Files top action row (+ File / + Folder / Upload): equal width, equal height, no wrapping. */
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) {
  column-gap: 10px !important;
  align-items: stretch !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div {
  min-width: 0 !important;
  display: flex !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"] {
  width: 100% !important;
  min-width: 100% !important;
  max-width: 100% !important;
  height: 2.34rem !important;
  min-height: 2.34rem !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
  padding: 0.3rem 0.72rem !important;
  box-sizing: border-box !important;
  white-space: nowrap !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"] > div,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"] p,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"] span {
  white-space: nowrap !important;
  overflow-wrap: normal !important;
  word-break: keep-all !important;
  text-align: center !important;
  width: auto !important;
  min-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"] > div {
  width: auto !important;
  min-width: 0 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 auto !important;
  gap: 6px !important;
  transform: none !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"] p,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"] span {
  transform: none !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(1) [data-testid="stPopoverButton"] > div,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(1) [data-testid="stPopoverButton"] p,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(1) [data-testid="stPopoverButton"] span {
  transform: translateX(0.14rem) !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(2) [data-testid="stPopoverButton"] > div,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(2) [data-testid="stPopoverButton"] p,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(2) [data-testid="stPopoverButton"] span {
  transform: translateX(-0.01rem) !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(3) [data-testid="stPopoverButton"] > div,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(3) [data-testid="stPopoverButton"] p,
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) > div:nth-child(3) [data-testid="stPopoverButton"] span {
  transform: translateX(-0.09rem) !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"] svg {
  display: none !important;
}
[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-files-actions-row-marker) [data-testid="stPopoverButton"]:hover {
  transform: none !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverButton"]:hover {
  transform: translateY(-1px);
  background: #f9fafb !important;
}
/* Sidebar close button: float on top-right and avoid taking panel content space. */
[class*="st-key-sidebar_panel_close_btn"] {
  position: relative !important;
  min-height: 0 !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
[class*="st-key-sidebar_panel_close_btn"] .stButton {
  position: absolute !important;
  top: 0.04rem;
  right: 0.08rem;
  width: 2rem !important;
  z-index: 20 !important;
}
[class*="st-key-sidebar_panel_close_btn"] .stButton > button {
  min-height: 1.52rem !important;
  height: 1.52rem !important;
  min-width: 2rem !important;
  width: 2rem !important;
  border-radius: 9px !important;
  border: 1px solid #d1d5db !important;
  background: #ffffff !important;
  color: #111827 !important;
  padding: 0 !important;
  box-shadow: none !important;
  line-height: 1 !important;
}
[class*="st-key-sidebar_panel_close_btn"] + [data-testid="stVerticalBlock"] {
  margin-top: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] {
  width: 180px !important;
  min-width: 180px !important;
  max-width: 180px !important;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12) !important;
  animation: sidebarPopoverIn 0.16s cubic-bezier(0.22, 1, 0.36, 1);
  transform-origin: top right;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker),
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) {
  width: 188px !important;
  min-width: 188px !important;
  max-width: 188px !important;
  padding: 0.86rem 0.56rem !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stElementContainer"]:has(.awc-upload-popover-marker),
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="element-container"]:has(.awc-upload-popover-marker),
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stElementContainer"]:has(.awc-upload-popover-marker),
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="element-container"]:has(.awc-upload-popover-marker) {
  display: none !important;
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) > div[data-testid="stVerticalBlock"],
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) > div[data-testid="stVerticalBlock"] {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0.74rem !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) .stFileUploader,
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) .stButton,
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) .stFileUploader,
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) .stButton {
  width: 9rem !important;
  min-width: 9rem !important;
  max-width: 9rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stFileUploaderDropzone"],
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stFileUploaderDropzone"] section,
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stFileUploaderDropzone"] section,
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stFileUploaderDropzone"] button,
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [class*="st-key-files_upload_btn_"] .stButton > button,
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [class*="st-key-files_upload_btn_"] .stButton > button {
  width: 9rem !important;
  min-width: 9rem !important;
  max-width: 9rem !important;
  box-sizing: border-box !important;
  margin-left: auto !important;
  margin-right: auto !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stFileUploaderDropzone"] button,
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] [data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [class*="st-key-files_upload_btn_"] .stButton > button,
[data-testid="stPopoverBody"]:has(.awc-upload-popover-marker) [class*="st-key-files_upload_btn_"] .stButton > button {
  height: 2.4rem !important;
  min-height: 2.4rem !important;
  padding: 0.35rem 0.85rem !important;
  text-align: center !important;
  display: block !important;
}
/* Popover may be rendered in a portal outside sidebar DOM; force width there too. */
[data-testid="stPopoverBody"]:has(div[data-testid="stTextInput"]) {
  width: 180px !important;
  min-width: 180px !important;
  max-width: 180px !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] .stButton > button,
[data-testid="stPopoverBody"]:has(div[data-testid="stTextInput"]) .stButton > button {
  white-space: nowrap !important;
  font-size: 0.84rem !important;
  line-height: 1.15 !important;
  padding-left: 0.48rem !important;
  padding-right: 0.48rem !important;
}
[class*="st-key-workspace_rename_confirm_"] button {
  white-space: nowrap !important;
  font-size: 0.8rem !important;
  line-height: 1.12 !important;
  padding-left: 0.34rem !important;
  padding-right: 0.34rem !important;
}
[class*="st-key-files_tree_download_file_"] :is(.stButton, .stDownloadButton) > button,
[class*="st-key-files_tree_download_dir_"] :is(.stButton, .stDownloadButton) > button,
[class*="st-key-workspace_download_zip_"] :is(.stButton, .stDownloadButton) > button {
  margin-top: 0.1rem !important;
}
/* Keep rename/delete popover controls white-background + black-text in dark mode too. */
[data-testid="stSidebar"] [data-testid="stPopoverBody"] .stButton > button,
[data-testid="stPopoverBody"]:has(div[data-testid="stTextInput"]) .stButton > button {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #d1d5db !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] .stButton > button:hover,
[data-testid="stPopoverBody"]:has(div[data-testid="stTextInput"]) .stButton > button:hover {
  background: #f9fafb !important;
  color: #111827 !important;
  border-color: #cfd5dd !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] .stButton > button:focus,
[data-testid="stSidebar"] [data-testid="stPopoverBody"] .stButton > button:focus-visible,
[data-testid="stPopoverBody"]:has(div[data-testid="stTextInput"]) .stButton > button:focus,
[data-testid="stPopoverBody"]:has(div[data-testid="stTextInput"]) .stButton > button:focus-visible {
  outline: none !important;
  box-shadow: 0 0 0 1px #9ca3af !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-baseweb="base-input"],
[data-testid="stPopoverBody"]:has(div[data-testid="stTextInput"]) div[data-baseweb="base-input"] {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #d1d5db !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-baseweb="base-input"] input,
[data-testid="stPopoverBody"]:has(div[data-testid="stTextInput"]) div[data-baseweb="base-input"] input {
  background: #ffffff !important;
  color: #111827 !important;
  caret-color: #111827 !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-testid="stTextInput"] > div > div {
  border: 1px solid #d1d5db !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  background: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-testid="stTextInput"] > div > div:focus-within {
  border-color: #d1d5db !important;
  box-shadow: none !important;
  background: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-baseweb="base-input"] {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-baseweb="base-input"]:focus-within {
  border-color: #9ca3af !important;
  box-shadow: 0 0 0 1px #9ca3af !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-baseweb="base-input"][data-invalid="true"] {
  border-color: #d1d5db !important;
  box-shadow: none !important;
}
/* Popover text input keeps a single-layer frame: outer gray border + inner white field. */
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stPopoverBody"] div[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stPopoverBody"] div[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-testid="stTextInput"] div[data-baseweb="input"] > div,
[data-testid="stPopoverBody"] div[data-testid="stTextInput"] div[data-baseweb="input"] > div {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverBody"] div[data-testid="stTextInput"] input,
[data-testid="stPopoverBody"] div[data-testid="stTextInput"] input {
  background: #ffffff !important;
  border: 0 !important;
  box-shadow: none !important;
  color: #111827 !important;
  caret-color: #111827 !important;
}
/* Delete confirmation warning (chat-history popover only): keep text alignment stable. */
[data-testid="stPopoverBody"]:has(.awc-agent-history-rename-popover-marker) [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
  margin-left: -0.16rem !important;
}
/* Chat-history delete warning style; conversation warning is left as default/original. */
[data-testid="stPopoverBody"]:has(.awc-agent-history-rename-popover-marker) [data-testid="stAlert"] {
  background: #FAF8EE !important;
  border: 0 !important;
  box-shadow: none !important;
  border-radius: 0.5rem !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-rename-popover-marker) [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
  color: #8A6F2A !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-rename-popover-marker) [data-testid="stAlert"] [data-testid="stIconMaterial"] {
  color: #8A6F2A !important;
}
@keyframes sidebarPopoverIn {
  from {
    opacity: 0;
    transform: translateY(-5px) scale(0.985);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

/* Hide popover chevron for three-dots and plus triggers */
[data-testid="stPopover"] > button svg {
  display: none !important;
}
[data-testid="stPopover"] button svg {
  display: none !important;
}
[data-testid="stPopover"] button [data-testid="stIconMaterial"] {
  display: none !important;
}
[data-testid="stPopover"] button::after {
  display: none !important;
}
[data-testid="stPopover"] button {
  background-image: none !important;
}

/* History popover: shift only the row-menu trigger (⋯) right. */
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [data-testid="stPopoverButton"] {
  margin-left: 0.40rem !important;
  width: calc(100% - 0.40rem) !important;
  max-width: calc(100% - 0.40rem) !important;
  box-sizing: border-box !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [data-testid="stPopoverButton"] > div,
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [data-testid="stPopoverButton"] p,
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [data-testid="stPopoverButton"] span {
  transform: translateX(0.16rem) !important;
}
/* History list: highlight active chat with a darker selected background (like sidebar active conversation). */
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [class*="st-key-agent_chat_switch_"] .stButton > button {
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  color: #111827 !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [class*="st-key-agent_chat_switch_"] .stButton > button[kind="primary"] {
  background: #f3f4f6 !important;
  border-color: #d1d5db !important;
  color: #111827 !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [class*="st-key-agent_chat_switch_"] .stButton > button[kind="primary"]:hover {
  background: #f3f4f6 !important;
  border-color: #d1d5db !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [class*="st-key-agent_chat_switch_"] .stButton > button[kind="primary"]:focus,
[data-testid="stPopoverBody"]:has(.awc-agent-history-popover-marker) [class*="st-key-agent_chat_switch_"] .stButton > button[kind="primary"]:focus-visible {
  box-shadow: 0 0 0 1px #9ca3af !important;
}

/* History row-menu popover: make rename text field framed and sized like Rename/Delete buttons. */
[data-testid="stPopoverBody"]:has(.awc-agent-history-rename-popover-marker) div[data-testid="stTextInput"] {
  width: 100% !important;
  min-width: 100% !important;
  max-width: 100% !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-rename-popover-marker) div[data-testid="stTextInput"] > div {
  width: 100% !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-rename-popover-marker) div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
  width: 100% !important;
  min-height: 2.38rem !important;
  border: 1px solid #d1d5db !important;
  border-radius: 10px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  padding-left: 0.48rem !important;
  padding-right: 0.48rem !important;
  box-sizing: border-box !important;
}
[data-testid="stPopoverBody"]:has(.awc-agent-history-rename-popover-marker) div[data-testid="stTextInput"] input {
  height: 2.38rem !important;
  line-height: normal !important;
  font-size: 1rem !important;
  font-weight: 400 !important;
  background: transparent !important;
  color: #111827 !important;
  caret-color: #111827 !important;
}

/* Avoid ugly mid-word wrapping such as "Terminology" */
label[data-testid="stWidgetLabel"] p,
div[data-testid="stMarkdownContainer"] p {
  word-break: normal !important;
  overflow-wrap: normal !important;
  white-space: normal !important;
}

/* Workspace shell */
div[data-testid="stVerticalBlock"]:has(.awc-main-shell-marker) {
  min-height: auto !important;
  border: none !important;
  border-radius: 0 !important;
  margin-top: -__MAIN_WORKSPACE_LIFT_REM__rem !important;
  padding: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
div[data-testid="stElementContainer"]:has(.awc-main-shell-marker) {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
div[data-testid="stElementContainer"]:has(.awc-main-workspace-marker) {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-pdf-panel-marker) {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 0.35rem 0.45rem 0.45rem 0.45rem;
  background: #ffffff;
}
div[data-testid="stVerticalBlock"]:has(.awc-editor-panel-marker) {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 0.35rem 0.45rem 0.45rem 0.45rem;
  background: #ffffff;
}
/* Right companion chat drawer (fixed + smooth transition) */
[data-testid="stElementContainer"]:has(.awc-chat-drawer-marker) {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) {
  position: fixed !important;
  top: 3.78rem !important;
  bottom: 12px !important;
  right: 14px !important;
  width: min(__CHAT_DRAWER_WIDTH_PX__px, 92vw) !important;
  height: auto !important;
  z-index: 1002 !important;
  border: 1px solid #d1d5db !important;
  border-radius: 16px !important;
  background: #ffffff !important;
  box-shadow: 0 18px 36px rgba(15, 23, 42, 0.18) !important;
  padding: 0.42rem 0.62rem 0.64rem 0.62rem !important;
  overflow: hidden !important;
  transform: translateX(calc(100% + 34px)) !important;
  transition: transform 0.30s cubic-bezier(0.22, 1, 0.36, 1) !important;
  will-change: transform;
}
body.awc-chat-drawer-open div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) {
  transform: translateX(0) !important;
}

/* Keep Companion Chat message style consistent with light mode, even in dark mode. */
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessage"],
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .stChatMessage {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageContent"],
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .stChatMessageContent,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageContent"] p,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageContent"] li,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageContent"] span {
  color: #111827 !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stCaptionContainer"],
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stCaptionContainer"] p {
  color: #6b7280 !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid*="stChatMessageAvatar"],
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [class*="stChatMessageAvatar"] {
  background: #ffffff !important;
  color: #111827 !important;
  border-color: #e5e7eb !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid^="chatAvatarIcon-"],
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [class*="chatAvatarIcon"] {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #e5e7eb !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageAvatarUser"],
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageAvatarAssistant"] {
  background: #ffffff !important;
  border-radius: 999px !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageAvatarUser"] > div,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageAvatarAssistant"] > div {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 999px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  overflow: hidden !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageAvatarUser"]::before,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageAvatarUser"]::after,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageAvatarAssistant"]::before,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessageAvatarAssistant"]::after {
  background: #ffffff !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessage"] > :first-child {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 10px !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessage"] > :first-child *,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessage"] > :first-child svg {
  color: #111827 !important;
  fill: currentColor !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessage"][background="true"] {
  background: rgba(243, 244, 246, 0.92) !important;
  background-color: rgba(243, 244, 246, 0.92) !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessage"] .eeusbqq2 {
  background: #ffffff !important;
  background-color: #ffffff !important;
  border: 1px solid #d1d5db !important;
  color: #111827 !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-marker {
  display: none !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stChatMessage"].awc-chat-copy-host {
  position: relative !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-btn {
  position: absolute !important;
  right: 8px !important;
  bottom: 8px !important;
  width: 24px !important;
  height: 24px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  border: 1px solid #d1d5db !important;
  border-radius: 7px !important;
  background: rgba(255, 255, 255, 0.95) !important;
  color: #6b7280 !important;
  padding: 0 !important;
  margin: 0 !important;
  box-shadow: none !important;
  cursor: pointer !important;
  transition: border-color 0.14s ease, color 0.14s ease, background-color 0.14s ease !important;
  z-index: 2 !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-btn:hover,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-btn:focus,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-btn:focus-visible {
  border-color: #9ca3af !important;
  background: #ffffff !important;
  color: #374151 !important;
  outline: none !important;
  box-shadow: none !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-btn svg {
  width: 14px !important;
  height: 14px !important;
  fill: none !important;
  stroke: currentColor !important;
  stroke-width: 1.8 !important;
  stroke-linecap: round !important;
  stroke-linejoin: round !important;
  pointer-events: none !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-btn::after {
  content: attr(data-awc-tooltip) !important;
  position: absolute !important;
  right: 0 !important;
  bottom: calc(100% + 6px) !important;
  background: rgba(17, 24, 39, 0.94) !important;
  color: #ffffff !important;
  border-radius: 7px !important;
  padding: 0.26rem 0.42rem !important;
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  white-space: nowrap !important;
  pointer-events: none !important;
  opacity: 0 !important;
  transform: translateY(2px) !important;
  transition: opacity 0.14s ease, transform 0.14s ease !important;
  z-index: 1006 !important;
}
/* History trigger button: suppress thick focus ring on the popover trigger itself. */
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"],
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"]:hover,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"]:focus,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"]:focus-visible,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"]:active,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"][aria-expanded="true"],
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"][aria-expanded="false"] {
  outline: none !important;
  box-shadow: none !important;
  outline-offset: 0 !important;
  border-width: 1px !important;
  border-style: solid !important;
  border-color: #e5e7eb !important;
  filter: none !important;
  transform: none !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"] > div,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"] p,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover [data-testid="stPopoverButton"] span {
  outline: none !important;
  box-shadow: none !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover:focus-within,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover:focus,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover:active {
  outline: none !important;
  box-shadow: none !important;
  filter: none !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover *,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover *:focus,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover *:focus-visible,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) [data-testid="stPopover"].awc-history-trigger-popover *:active {
  outline: none !important;
  box-shadow: none !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-btn:hover::after,
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-copy-btn:focus-visible::after {
  opacity: 1 !important;
  transform: translateY(0) !important;
}

/* Floating robot toggle button */
#awc-chat-drawer-fab {
  position: fixed;
  top: 46%;
  right: 12px;
  z-index: 1003;
  width: 56px;
  height: 56px;
  border-radius: 999px;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.20);
  color: #0f172a;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.20s ease, box-shadow 0.20s ease;
  font-size: 23px;
}
#awc-chat-drawer-fab:hover {
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 14px 28px rgba(15, 23, 42, 0.24);
}
#awc-chat-drawer-fab .awc-chat-fab-label {
  display: none;
}
body.awc-chat-drawer-open #awc-chat-drawer-fab {
  right: min(calc(__CHAT_DRAWER_WIDTH_PX__px + 20px), calc(92vw + 20px));
}

/* Replace Streamlit running-man status with local robot compile loader. */
[data-testid="stStatusWidget"] {
  display: none !important;
}

.awc-compile-loader {
  height: 2.32rem;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  background: #f8fafc;
  color: #334155;
  font-size: 0.82rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.34rem;
}

.awc-compile-loader-bot {
  display: inline-block;
  transform-origin: center;
  animation: awcCompileBotBounce 0.9s ease-in-out infinite;
}

@keyframes awcCompileBotBounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-2px); }
}

.awc-chat-busy-loader {
  margin: 0.2rem 0 0.36rem 0 !important;
  padding: 0.34rem 0.5rem !important;
  border: 1px solid #d1d5db !important;
  border-radius: 10px !important;
  background: #f8fafc !important;
  color: #374151 !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  display: inline-flex !important;
  align-items: center !important;
  gap: 0.42rem !important;
}
.awc-chat-busy-loader .awc-chat-busy-spinner {
  width: 14px !important;
  height: 14px !important;
  border-radius: 999px !important;
  border: 2px solid #cbd5e1 !important;
  border-top-color: #64748b !important;
  animation: awcChatBusySpin 0.8s linear infinite !important;
}
@keyframes awcChatBusySpin {
  to { transform: rotate(360deg); }
}
/* Keep busy loader floating above composer so it never pushes input downward. */
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) > [data-testid="stElementContainer"]:has(.awc-chat-busy-loader),
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) > [data-testid="element-container"]:has(.awc-chat-busy-loader) {
  position: absolute !important;
  left: 0.62rem !important;
  right: 0.62rem !important;
  bottom: calc(72px + 8px) !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  z-index: 26 !important;
  pointer-events: none !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker)) .awc-chat-busy-loader {
  position: absolute !important;
  right: 14px !important;
  bottom: 0 !important;
  margin: 0 !important;
  pointer-events: none !important;
}

[data-stale="true"],
[stale_data="true"] {
  opacity: 1 !important;
  filter: none !important;
}

/* Keep editor and PDF columns always same outer height. */
[data-testid="stHorizontalBlock"]:has(.awc-editor-panel-marker):has(.awc-pdf-panel-marker) {
  align-items: stretch !important;
}
div[data-testid="stVerticalBlock"]:has(.awc-editor-panel-marker),
div[data-testid="stVerticalBlock"]:has(.awc-pdf-panel-marker) {
  height: 100% !important;
  min-height: 100% !important;
}
</style>
"""
    st.markdown(
        css_text.replace("__TOP_LEVEL_ROW_GAP_REM__", str(TOP_LEVEL_ROW_GAP_REM))
        .replace("__HEADER_TO_PANEL_PULLUP_REM__", str(HEADER_TO_PANEL_PULLUP_REM))
        .replace("__PANEL_TITLE_MARGIN_TOP_REM__", str(PANEL_TITLE_MARGIN_TOP_REM))
        .replace("__PANEL_TITLE_MARGIN_BOTTOM_REM__", str(PANEL_TITLE_MARGIN_BOTTOM_REM))
        .replace("__MAIN_WORKSPACE_LIFT_REM__", str(MAIN_WORKSPACE_LIFT_REM))
        .replace("__CHAT_DRAWER_WIDTH_PX__", str(CHAT_DRAWER_WIDTH_PX)),
        unsafe_allow_html=True,
    )


def _compute_panel_height(viewport_height: int, available_height: int) -> int:
    raw_height = _compute_panel_height_raw(viewport_height, available_height)

    def _clamp(candidate: int) -> int:
        return max(PANEL_MIN_HEIGHT, min(PANEL_MAX_HEIGHT, int(candidate)))

    return _clamp(raw_height)


def _compute_panel_height_raw(viewport_height: int, available_height: int) -> int:
    candidates: List[int] = []
    if available_height > 0:
        candidates.append(available_height - PANEL_COLUMNS_OVERHEAD + PANEL_HEIGHT_BOOST)
    if viewport_height > 0:
        candidates.append(viewport_height - PANEL_VIEWPORT_RESERVED + PANEL_HEIGHT_BOOST)
    if not candidates:
        return int(PANEL_FALLBACK_HEIGHT)
    return int(max(candidates))


def _line_span_for_index(text: str, line_number: int) -> Tuple[int, int]:
    if line_number <= 0:
        return -1, -1
    lines = text.splitlines(keepends=True)
    if not lines:
        return -1, -1
    index = min(max(1, line_number), len(lines)) - 1
    start = sum(len(line) for line in lines[:index])
    raw_line = lines[index]
    line_text = raw_line.rstrip("\n")
    end = start + len(line_text)
    if end <= start:
        end = min(len(text), start + 1)
    return start, end


def _schedule_editor_focus_for_line(doc_id: str, relative_path: str, line_number: int) -> None:
    if line_number <= 0:
        return
    try:
        normalized = _normalize_project_relative_path(relative_path)
    except ValueError:
        return
    if _is_internal_state_relative_path(normalized) or not _is_text_editable_project_file(normalized):
        return
    try:
        file_text = load_project_text_file(doc_id, normalized)
    except OSError:
        return
    focus_start, focus_end = _line_span_for_index(file_text, line_number)
    if focus_start < 0 or focus_end <= focus_start:
        return
    st.session_state.editor_focus_start = focus_start
    st.session_state.editor_focus_end = focus_end
    previous_event_id = int(st.session_state.get("editor_focus_event_id", 0) or 0)
    st.session_state.editor_focus_event_id = max(int(time.time() * 1000), previous_event_id + 1)


def _jump_to_compile_error(doc_id: str, error_item: Dict[str, Any]) -> bool:
    target_file = str(error_item.get("file", "")).strip() or "main.tex"
    try:
        normalized_target = _normalize_project_relative_path(target_file)
    except ValueError:
        normalized_target = "main.tex"
    if _is_internal_state_relative_path(normalized_target) or not _is_text_editable_project_file(normalized_target):
        return False
    try:
        target_path = _resolve_project_path(doc_id, normalized_target)
    except ValueError:
        return False
    if not target_path.exists() or not target_path.is_file():
        return False

    try:
        current_rel = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
    except ValueError:
        current_rel = "main.tex"

    if normalized_target != current_rel:
        _open_workspace_file(doc_id, normalized_target)

    try:
        line_no = int(error_item.get("line", -1))
    except (TypeError, ValueError):
        line_no = -1
    if line_no > 0:
        _schedule_editor_focus_for_line(doc_id, normalized_target, line_no)
    return True


def _resolve_pdf_export_directory(project_root: Path) -> Path:
    export_dir = project_root.expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def _sanitize_export_filename(raw_name: str) -> str:
    candidate = str(raw_name or "").strip()
    if not candidate:
        return PDF_EXPORT_NAME_PREFIX
    candidate = re.sub(r'[\\/:*?"<>|]+', "_", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .")
    return candidate[:80] or PDF_EXPORT_NAME_PREFIX


def _active_workspace_name() -> str:
    active_id = str(st.session_state.get("active_workspace_id", "")).strip()
    workspace_store = st.session_state.get("workspace_store", {})
    if not active_id or not isinstance(workspace_store, dict):
        return ""
    payload = workspace_store.get(active_id)
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("name", "")).strip()


def _show_transient_success_toast(message: str, duration_ms: int = 3000) -> None:
    text = str(message or "").strip() or "Saved"
    safe_duration_ms = max(500, min(10000, int(duration_ms)))
    st.session_state.export_hint_message = text
    st.session_state.export_hint_duration_ms = safe_duration_ms
    st.session_state.export_hint_nonce = int(st.session_state.get("export_hint_nonce", 0)) + 1


def _render_export_success_toast_mount() -> None:
    message = str(st.session_state.get("export_hint_message", "") or "")
    duration_ms = max(500, min(10000, int(st.session_state.get("export_hint_duration_ms", 3000))))
    nonce = int(st.session_state.get("export_hint_nonce", 0))
    message_literal = json.dumps(message)
    st_components.html(
        f"""
<script>
(() => {{
  const doc = window.parent && window.parent.document ? window.parent.document : document;
  if (!doc || !doc.body) return;
  const nonce = {nonce};
  const message = {message_literal};
  const durationMs = {duration_ms};
  if (!nonce || !message) return;

  const hostWin = window.parent || window;
  const lastNonce = Number(hostWin.__awcExportHintNonce || 0);
  if (nonce <= lastNonce) return;
  hostWin.__awcExportHintNonce = nonce;

  const toastId = "awc-export-success-toast";
  const oldToast = doc.getElementById(toastId);
  if (oldToast) oldToast.remove();

  const toast = doc.createElement("div");
  toast.id = toastId;
  toast.textContent = message;
  toast.style.position = "fixed";
  toast.style.right = "20px";
  toast.style.bottom = "20px";
  toast.style.zIndex = "10080";
  toast.style.padding = "9px 13px";
  toast.style.borderRadius = "10px";
  toast.style.background = "#ffffff";
  toast.style.color = "#111827";
  toast.style.border = "1px solid #d1d5db";
  toast.style.fontFamily = "inherit";
  toast.style.fontSize = "13px";
  toast.style.fontWeight = "600";
  toast.style.lineHeight = "1.2";
  toast.style.boxShadow = "0 10px 24px rgba(15, 23, 42, 0.16)";
  toast.style.opacity = "0";
  toast.style.transform = "translateY(6px)";
  toast.style.transition = "opacity 180ms ease, transform 180ms ease";

  const exportBtn = doc.querySelector(".st-key-export_pdf_btn_v2 button");
  if (exportBtn) {{
    const btnStyle = window.getComputedStyle(exportBtn);
    toast.style.fontFamily = btnStyle.fontFamily;
    toast.style.fontWeight = btnStyle.fontWeight;
    toast.style.fontStyle = btnStyle.fontStyle;
    toast.style.letterSpacing = btnStyle.letterSpacing;
  }}

  doc.body.appendChild(toast);
  window.requestAnimationFrame(() => {{
    toast.style.opacity = "1";
    toast.style.transform = "translateY(0)";
  }});

  const hideToast = () => {{
    if (!toast.isConnected) return;
    toast.style.opacity = "0";
    toast.style.transform = "translateY(6px)";
    window.setTimeout(() => {{
      if (toast.isConnected) toast.remove();
    }}, 220);
  }};

  window.setTimeout(hideToast, durationMs);
}})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _export_compiled_pdf(project_root: Path) -> str:
    pdf_bytes = bytes(st.session_state.get("pdf_compiled_bytes") or b"")
    if not pdf_bytes:
        return ""
    export_dir = _resolve_pdf_export_directory(project_root)
    base_name = _sanitize_export_filename(_active_workspace_name() or PDF_EXPORT_NAME_PREFIX)
    output_path = export_dir / f"{base_name}.pdf"
    suffix = 2
    while output_path.exists():
        output_path = export_dir / f"{base_name}_{suffix}.pdf"
        suffix += 1
    output_path.write_bytes(pdf_bytes)
    return str(output_path)


def _request_browser_pdf_export() -> bool:
    pdf_bytes = bytes(st.session_state.get("pdf_compiled_bytes") or b"")
    if not pdf_bytes:
        return False
    export_name = _sanitize_export_filename(_active_workspace_name() or PDF_EXPORT_NAME_PREFIX)
    st.session_state.browser_pdf_export_name = f"{export_name}.pdf"
    st.session_state.browser_pdf_export_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    st.session_state.browser_pdf_export_nonce = int(st.session_state.get("browser_pdf_export_nonce", 0)) + 1
    return True


def _render_browser_pdf_export_mount() -> None:
    nonce = int(st.session_state.get("browser_pdf_export_nonce", 0) or 0)
    file_name = str(st.session_state.get("browser_pdf_export_name", "") or "").strip()
    b64_payload = str(st.session_state.get("browser_pdf_export_b64", "") or "").strip()
    if not nonce or not file_name or not b64_payload:
        return
    safe_name = json.dumps(file_name)
    safe_payload = json.dumps(b64_payload)
    st_components.html(
        f"""
<script>
(() => {{
  const hostWin = window.parent || window;
  const hostDoc = hostWin.document || document;
  const nonce = {nonce};
  const fileName = {safe_name};
  const payloadB64 = {safe_payload};
  if (!nonce || !fileName || !payloadB64) return;
  const nonceKey = "__awcBrowserPdfExportNonce";
  const lastNonce = Number(hostWin[nonceKey] || 0);
  if (nonce <= lastNonce) return;
  hostWin[nonceKey] = nonce;

  const decodeBase64 = (text) => {{
    const binary = hostWin.atob(text);
    const size = binary.length;
    const bytes = new Uint8Array(size);
    for (let i = 0; i < size; i += 1) {{
      bytes[i] = binary.charCodeAt(i);
    }}
    return bytes;
  }};

  const blob = new Blob([decodeBase64(payloadB64)], {{ type: "application/pdf" }});
  const showSavedToast = () => {{
    const toastId = "awc-browser-export-saved-toast";
    const oldToast = hostDoc.getElementById(toastId);
    if (oldToast) oldToast.remove();

    const toast = hostDoc.createElement("div");
    toast.id = toastId;
    toast.textContent = "Saved successfully!";
    toast.style.position = "fixed";
    toast.style.right = "20px";
    toast.style.bottom = "20px";
    toast.style.zIndex = "10080";
    toast.style.padding = "9px 13px";
    toast.style.borderRadius = "10px";
    toast.style.background = "#ffffff";
    toast.style.color = "#111827";
    toast.style.border = "1px solid #d1d5db";
    toast.style.fontFamily = "inherit";
    toast.style.fontSize = "13px";
    toast.style.fontWeight = "600";
    toast.style.lineHeight = "1.2";
    toast.style.boxShadow = "0 10px 24px rgba(15, 23, 42, 0.16)";
    toast.style.opacity = "0";
    toast.style.transform = "translateY(6px)";
    toast.style.transition = "opacity 180ms ease, transform 180ms ease";

    const exportBtn = hostDoc.querySelector(".st-key-export_pdf_btn_v2 button");
    if (exportBtn) {{
      const btnStyle = hostWin.getComputedStyle(exportBtn);
      toast.style.fontFamily = btnStyle.fontFamily;
      toast.style.fontWeight = btnStyle.fontWeight;
      toast.style.fontStyle = btnStyle.fontStyle;
      toast.style.letterSpacing = btnStyle.letterSpacing;
    }}

    hostDoc.body.appendChild(toast);
    hostWin.requestAnimationFrame(() => {{
      toast.style.opacity = "1";
      toast.style.transform = "translateY(0)";
    }});
    hostWin.setTimeout(() => {{
      toast.style.opacity = "0";
      toast.style.transform = "translateY(6px)";
      hostWin.setTimeout(() => {{
        if (toast.isConnected) toast.remove();
      }}, 220);
    }}, 3000);
  }};
  const fallbackDownload = () => {{
    const objectUrl = hostWin.URL.createObjectURL(blob);
    const anchor = hostDoc.createElement("a");
    anchor.href = objectUrl;
    anchor.download = fileName;
    anchor.style.display = "none";
    hostDoc.body.appendChild(anchor);
    anchor.click();
    hostWin.setTimeout(() => {{
      hostWin.URL.revokeObjectURL(objectUrl);
      if (anchor.isConnected) anchor.remove();
    }}, 1200);
  }};

  const saveWithPicker = async () => {{
    if (typeof hostWin.showSaveFilePicker !== "function") {{
      fallbackDownload();
      return;
    }}
    try {{
      const handle = await hostWin.showSaveFilePicker({{
        suggestedName: fileName,
        types: [{{ description: "PDF document", accept: {{ "application/pdf": [".pdf"] }} }}],
      }});
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      showSavedToast();
    }} catch (error) {{
      if (error && String(error.name || "") === "AbortError") {{
        return;
      }}
      fallbackDownload();
    }}
  }};

  saveWithPicker();
}})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _extract_latex_errors(log_text: str, source_text: str, max_items: int = 12) -> List[Dict[str, Any]]:
    return _extract_latex_errors_with_file(
        doc_id=str(st.session_state.get("active_workspace_id", "")).strip() or "default",
        log_text=log_text,
        source_text=source_text,
        max_items=max_items,
    )


def _extract_latex_errors_with_file(
    doc_id: str,
    log_text: str,
    source_text: str,
    max_items: int = 12,
    build_dir: str = "",
) -> List[Dict[str, Any]]:
    log_lines = str(log_text or "").splitlines()
    src_lines = source_text.splitlines()
    items: List[Dict[str, Any]] = []
    seen = set()
    file_lines_cache: Dict[str, List[str]] = {}
    normalized_doc_id = _validate_doc_id(doc_id or "default")
    project_dir = ensure_project_dir(normalized_doc_id)
    if build_dir:
        build_root = Path(build_dir)
    else:
        build_root = PDF_BUILDS_ROOT_DIR / normalized_doc_id

    def _resolve_error_file(raw_path: str) -> str:
        raw = str(raw_path or "").strip().strip('"').strip("'")
        if not raw:
            return ""
        resolved = _resolve_synctex_input_to_project_tex(normalized_doc_id, raw, build_root)
        if resolved:
            return resolved
        try:
            normalized = _normalize_project_relative_path(raw)
            path = _resolve_project_path(normalized_doc_id, normalized)
        except (ValueError, OSError):
            normalized = ""
        else:
            if (
                normalized
                and not _is_internal_state_relative_path(normalized)
                and _is_text_editable_project_file(normalized)
                and path.exists()
                and path.is_file()
            ):
                return normalized
        basename = Path(raw).name.strip()
        if not basename:
            return ""
        try:
            matches = list(project_dir.rglob(basename))
        except OSError:
            matches = []
        possible: List[str] = []
        for match in matches:
            if not match.is_file():
                continue
            try:
                rel = match.relative_to(project_dir).as_posix()
                normalized = _normalize_project_relative_path(rel)
            except (ValueError, OSError):
                continue
            if _is_internal_state_relative_path(normalized) or not _is_text_editable_project_file(normalized):
                continue
            possible.append(normalized)
        if not possible:
            return ""
        possible = sorted(dict.fromkeys(possible), key=lambda item: (0 if item == "main.tex" else 1, item.casefold()))
        return possible[0]

    def _line_snippet(file_rel: str, line_no: int) -> str:
        if line_no <= 0:
            return ""
        if file_rel:
            if file_rel not in file_lines_cache:
                try:
                    file_text = load_project_text_file(normalized_doc_id, file_rel)
                except Exception:
                    file_text = ""
                file_lines_cache[file_rel] = file_text.splitlines()
            lines = file_lines_cache.get(file_rel, [])
            if 0 < line_no <= len(lines):
                return lines[line_no - 1].strip()
        if 0 < line_no <= len(src_lines):
            return src_lines[line_no - 1].strip()
        return ""

    def _append(line_no: int, message: str, file_rel: str = "") -> None:
        try:
            parsed_line = int(line_no)
        except (TypeError, ValueError):
            parsed_line = -1
        normalized_line = parsed_line if parsed_line > 0 else -1
        normalized_file = ""
        if file_rel:
            try:
                normalized_file = _normalize_project_relative_path(file_rel)
            except ValueError:
                normalized_file = ""
        normalized_msg = str(message or "LaTeX error").strip()
        key = (normalized_file, normalized_line, normalized_msg)
        if key in seen:
            return
        seen.add(key)
        snippet = _line_snippet(normalized_file, normalized_line)
        items.append(
            {
                "file": normalized_file,
                "line": normalized_line,
                "message": normalized_msg,
                "snippet": snippet,
            }
        )

    def _merge_wrapped_text(current: str, extra: str) -> str:
        left = str(current or "").rstrip()
        right = str(extra or "").strip()
        if not right:
            return left
        if not left:
            return right
        if left[-1].isalnum() and right[:1].islower():
            return f"{left}{right}"
        return f"{left} {right}"

    def _collect_wrapped_error_message(start_idx: int) -> str:
        base = log_lines[start_idx].strip().lstrip("!").strip()
        if not base:
            return "LaTeX error"
        message = base
        for probe in range(start_idx + 1, min(start_idx + 8, len(log_lines))):
            candidate = log_lines[probe].rstrip()
            stripped = candidate.strip()
            if not stripped:
                break
            if stripped.startswith("!"):
                break
            if re.match(r"^l\.\d+\b", stripped):
                break
            if re.match(r"^[A-Za-z0-9_./-]+\.tex:\d+:", stripped):
                break
            if stripped.startswith("<") or stripped.startswith("("):
                break
            if stripped.startswith("Type ") or stripped.startswith("See the ") or stripped.startswith("? "):
                break
            message = _merge_wrapped_text(message, stripped)
        return message.strip() or "LaTeX error"

    def _collect_wrapped_fileline_message(start_idx: int, first_text: str) -> str:
        message = str(first_text or "").strip()
        if not message:
            return "LaTeX error"
        for probe in range(start_idx + 1, min(start_idx + 8, len(log_lines))):
            candidate = log_lines[probe].rstrip()
            stripped = candidate.strip()
            if not stripped:
                break
            if stripped.startswith("!"):
                break
            if re.match(r"^l\.\d+\b", stripped):
                break
            if re.match(r"^[A-Za-z0-9_./-]+\.tex:\d+:", stripped):
                break
            if stripped.startswith("<") or stripped.startswith("("):
                break
            if stripped.startswith("Type ") or stripped.startswith("See the ") or stripped.startswith("? "):
                break
            message = _merge_wrapped_text(message, stripped)
        return message.strip() or "LaTeX error"

    for idx, line in enumerate(log_lines):
        match = re.search(r"(?:^|\s)([^:\s]+\.tex):(\d+):\s*(.+)$", line)
        if not match:
            continue
        file_rel = _resolve_error_file(match.group(1))
        message = _collect_wrapped_fileline_message(idx, match.group(3))
        _append(int(match.group(2)), message, file_rel=file_rel)
        if len(items) >= max_items:
            return items

    for idx, line in enumerate(log_lines):
        stripped = line.strip()
        if not stripped.startswith("!"):
            continue
        message = _collect_wrapped_error_message(idx)
        line_no = -1
        file_rel = ""
        for probe in range(idx + 1, min(idx + 8, len(log_lines))):
            marker = re.search(r"l\.(\d+)", log_lines[probe])
            if marker:
                line_no = int(marker.group(1))
                break
        for probe in range(max(0, idx - 12), min(idx + 8, len(log_lines))):
            file_marker = re.search(r"(?:^|\s)([^:\s]+\.tex):(\d+):", log_lines[probe])
            if file_marker:
                file_rel = _resolve_error_file(file_marker.group(1))
                if file_rel:
                    break
            alt_marker = re.search(r"\(([^()\s]+\.tex)", log_lines[probe])
            if alt_marker and not file_rel:
                file_rel = _resolve_error_file(alt_marker.group(1))
                if file_rel:
                    break
        _append(line_no, message, file_rel=file_rel)
        if len(items) >= max_items:
            return items

    if not items and log_lines:
        error_like_patterns = [
            r"^!",
            r"\berror\b",
            r"undefined control sequence",
            r"emergency stop",
            r"missing \$ inserted",
            r"runaway argument",
            r"fatal error",
        ]
        for line in log_lines:
            text = line.strip()
            lowered = text.lower()
            if text and any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in error_like_patterns):
                _append(-1, text, file_rel="")
            if len(items) >= min(6, max_items):
                break
    if not items:
        _append(-1, "Compilation failed. See raw LaTeX log for details.", file_rel="")
    return items


def _compile_error_sort_key(item: Dict[str, Any]) -> Tuple[int, str, int, str]:
    raw_file = str(item.get("file", "")).strip()
    try:
        normalized_file = _normalize_project_relative_path(raw_file) if raw_file else ""
    except ValueError:
        normalized_file = ""
    if normalized_file and normalized_file != "main.tex":
        file_bucket = 0
    elif normalized_file == "main.tex":
        file_bucket = 1
    else:
        file_bucket = 2
    try:
        raw_line = int(item.get("line", -1))
    except (TypeError, ValueError):
        raw_line = -1
    normalized_line = raw_line if raw_line > 0 else 10**9
    message = str(item.get("message", "")).strip().casefold()
    return (file_bucket, normalized_file.casefold(), normalized_line, message)


def _compile_latex_with_synctex(doc_id: str) -> Dict[str, Any]:
    compiler = detect_latex_compiler()
    if not compiler:
        return {
            "ok": False,
            "log": "No LaTeX compiler found. Please install latexmk or pdflatex.",
            "pdf_bytes": b"",
            "pdf_path": "",
            "synctex_path": "",
            "tex_path": "",
        }

    normalized_doc_id = _validate_doc_id(doc_id or "default")
    project_dir = ensure_project_dir(normalized_doc_id)
    source_tex_path = project_dir / "main.tex"

    build_dir = PDF_BUILDS_ROOT_DIR / normalized_doc_id
    build_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = build_dir / "main.pdf"
    synctex_path = build_dir / "main.synctex.gz"
    log_path = build_dir / "main.log"

    # Clear stale log to avoid mixing previous failures into current diagnostics.
    for stale in [log_path]:
        if stale.exists():
            try:
                stale.unlink()
            except OSError:
                pass

    merged_log = ""
    if compiler == "latexmk":
        cmd = [
            "latexmk",
            "-pdf",
            "-g",
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            f"-outdir={str(build_dir)}",
            "main.tex",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        merged_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    else:
        cmd = [
            "pdflatex",
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-output-directory",
            str(build_dir),
            "main.tex",
        ]
        pdflatex_logs: List[str] = []
        last_proc = None
        for _ in range(2):
            last_proc = subprocess.run(
                cmd,
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            pdflatex_logs.append((last_proc.stdout or "") + "\n" + (last_proc.stderr or ""))
            if last_proc.returncode != 0:
                break
        proc = last_proc if last_proc is not None else subprocess.CompletedProcess(cmd, returncode=1)
        merged_log = "\n".join(pdflatex_logs)
    if log_path.exists():
        merged_log += "\n" + log_path.read_text(encoding="utf-8", errors="ignore")

    ok = pdf_path.exists() and proc.returncode == 0
    synctex_candidates = [
        build_dir / "main.synctex.gz",
        build_dir / "main.synctex",
        project_dir / "main.synctex.gz",
        project_dir / "main.synctex",
    ]
    resolved_synctex = ""
    for candidate in synctex_candidates:
        if candidate.exists():
            resolved_synctex = str(candidate)
            break

    if ok:
        return {
            "ok": True,
            "log": merged_log,
            "pdf_bytes": pdf_path.read_bytes(),
            "pdf_path": str(pdf_path),
            "synctex_path": resolved_synctex,
            "tex_path": str(source_tex_path),
        }
    return {
        "ok": False,
        "log": merged_log,
        "pdf_bytes": b"",
        "pdf_path": "",
        "synctex_path": resolved_synctex,
        "tex_path": str(source_tex_path),
    }


def _build_pdf_first_page_preview_png(pdf_bytes: bytes) -> bytes:
    if not pdf_bytes:
        return b""
    gs_bin = shutil.which("gs")
    if not gs_bin:
        for candidate in ["/usr/local/bin/gs", "/opt/homebrew/bin/gs", "/usr/bin/gs"]:
            if Path(candidate).exists():
                gs_bin = candidate
                break
    if not gs_bin:
        return b""

    with tempfile.TemporaryDirectory(prefix="awc_pdf_preview_") as tmpdir:
        tmp_path = Path(tmpdir)
        pdf_path = tmp_path / "preview.pdf"
        png_path = tmp_path / "preview.png"
        pdf_path.write_bytes(pdf_bytes)
        cmd = [
            gs_bin,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{PDF_PREVIEW_DPI}",
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            "-dFirstPage=1",
            "-dLastPage=1",
            f"-sOutputFile={png_path}",
            str(pdf_path),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not png_path.exists():
            return b""
        return png_path.read_bytes()


def _build_pdf_preview_png_pages(pdf_bytes: bytes, max_pages: int = 48) -> List[bytes]:
    if not pdf_bytes:
        return []
    gs_bin = shutil.which("gs")
    if not gs_bin:
        for candidate in ["/usr/local/bin/gs", "/opt/homebrew/bin/gs", "/usr/bin/gs"]:
            if Path(candidate).exists():
                gs_bin = candidate
                break
    if not gs_bin:
        return []

    with tempfile.TemporaryDirectory(prefix="awc_pdf_preview_pages_") as tmpdir:
        tmp_path = Path(tmpdir)
        pdf_path = tmp_path / "preview.pdf"
        output_pattern = tmp_path / "page-%03d.png"
        pdf_path.write_bytes(pdf_bytes)
        cmd = [
            gs_bin,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=png16m",
            f"-r{PDF_PREVIEW_DPI}",
            "-dTextAlphaBits=4",
            "-dGraphicsAlphaBits=4",
            "-dFirstPage=1",
            f"-dLastPage={max(1, int(max_pages))}",
            f"-sOutputFile={output_pattern}",
            str(pdf_path),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return []
        page_paths = sorted(tmp_path.glob("page-*.png"))
        if not page_paths:
            return []
        pages: List[bytes] = []
        for page_path in page_paths:
            try:
                pages.append(page_path.read_bytes())
            except OSError:
                continue
        return pages


def _run_compile_pipeline(state: AppState) -> None:
    active_file_path = _active_file_for_state(state)
    save_project_text_file(state.state_doc_id, active_file_path, state.current_text)
    source_text = load_main_tex(state.state_doc_id)
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()

    if not source_text.strip():
        st.session_state.pdf_compile_ok = False
        st.session_state.pdf_compiled_text_hash = source_hash
        st.session_state.pdf_compiled_bytes = b""
        st.session_state.pdf_compiled_base64 = ""
        st.session_state.pdf_compiled_pdf_path = ""
        st.session_state.pdf_compiled_synctex_path = ""
        st.session_state.pdf_compile_log = "main.tex is empty. Please add LaTeX source before compiling."
        st.session_state.pdf_compile_errors = [
            {
                "file": "main.tex",
                "line": -1,
                "message": "main.tex is empty. Please add LaTeX source before compiling.",
                "snippet": "",
            }
        ]
        st.session_state.pdf_preview_png_hash = ""
        st.session_state.pdf_preview_png_bytes = b""
        st.session_state.pdf_preview_png_pages_hash = ""
        st.session_state.pdf_preview_png_pages_bytes = []
        st.session_state.pdf_compiled_extracted_text_hash = ""
        st.session_state.pdf_compiled_extracted_text = ""
        store = st.session_state.get("workspace_store", {})
        payload = store.get(state.state_doc_id)
        if isinstance(payload, dict):
            memory_payload = _workspace_memory_payload(state.state_doc_id)
            memory_payload["shared_memory"]["last_compile_status"] = "failed: empty_editor"
            payload["memory"] = memory_payload
        _save_active_workspace_snapshot(touch_activity=True)
        return

    result = _compile_latex_with_synctex(state.state_doc_id)
    st.session_state.pdf_compile_log = str(result.get("log", ""))
    if result.get("ok"):
        pdf_bytes = bytes(result.get("pdf_bytes") or b"")
        st.session_state.pdf_compile_ok = True
        st.session_state.pdf_compiled_text_hash = source_hash
        st.session_state.pdf_compiled_bytes = pdf_bytes
        st.session_state.pdf_compiled_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.session_state.pdf_compiled_pdf_path = str(result.get("pdf_path", ""))
        st.session_state.pdf_compiled_synctex_path = str(result.get("synctex_path", ""))
        st.session_state.pdf_compiled_tex_path = str(result.get("tex_path", ""))
        st.session_state.pdf_compile_errors = []
        st.session_state.pdf_compiled_extracted_text_hash = ""
        st.session_state.pdf_compiled_extracted_text = ""
        if st.session_state.get("pdf_preview_png_hash") != source_hash:
            st.session_state.pdf_preview_png_hash = ""
            st.session_state.pdf_preview_png_bytes = b""
        if st.session_state.get("pdf_preview_png_pages_hash") != source_hash:
            st.session_state.pdf_preview_png_pages_hash = ""
            st.session_state.pdf_preview_png_pages_bytes = []
        store = st.session_state.get("workspace_store", {})
        payload = store.get(state.state_doc_id)
        if isinstance(payload, dict):
            memory_payload = _workspace_memory_payload(state.state_doc_id)
            memory_payload["shared_memory"]["last_compile_status"] = "success"
            payload["memory"] = memory_payload
    else:
        if not st.session_state.pdf_compile_log.strip():
            st.session_state.pdf_compile_log = "Compilation failed with no log output."
        st.session_state.pdf_compile_errors = _extract_latex_errors_with_file(
            doc_id=state.state_doc_id,
            log_text=st.session_state.pdf_compile_log,
            source_text=source_text,
            build_dir=str(PDF_BUILDS_ROOT_DIR / _validate_doc_id(state.state_doc_id)),
        )
        if not st.session_state.pdf_compile_errors:
            st.session_state.pdf_compile_errors = [
                {
                    "file": "",
                    "line": -1,
                    "message": "Compilation failed, but no structured LaTeX error was parsed. See raw log.",
                    "snippet": "",
                }
            ]
        st.session_state.pdf_compile_ok = False
        st.session_state.pdf_compiled_text_hash = source_hash
        st.session_state.pdf_compiled_bytes = b""
        st.session_state.pdf_compiled_base64 = ""
        st.session_state.pdf_compiled_pdf_path = ""
        st.session_state.pdf_compiled_synctex_path = ""
        st.session_state.pdf_compiled_tex_path = str(result.get("tex_path", ""))
        st.session_state.pdf_preview_png_hash = ""
        st.session_state.pdf_preview_png_bytes = b""
        st.session_state.pdf_preview_png_pages_hash = ""
        st.session_state.pdf_preview_png_pages_bytes = []
        st.session_state.pdf_compiled_extracted_text_hash = ""
        st.session_state.pdf_compiled_extracted_text = ""
        store = st.session_state.get("workspace_store", {})
        payload = store.get(state.state_doc_id)
        if isinstance(payload, dict):
            memory_payload = _workspace_memory_payload(state.state_doc_id)
            first_error = ""
            if st.session_state.pdf_compile_errors:
                first_error = str(st.session_state.pdf_compile_errors[0].get("message", "")).strip()
            memory_payload["shared_memory"]["last_compile_status"] = (
                f"failed: {first_error}" if first_error else "failed"
            )
            payload["memory"] = memory_payload
    _save_active_workspace_snapshot(touch_activity=True)


def _resolve_synctex_input_to_project_tex(doc_id: str, input_path: str, build_dir: Path) -> str:
    raw_path = str(input_path or "").strip().strip('"').strip("'")
    if not raw_path:
        return ""

    project_dir = ensure_project_dir(doc_id)
    try:
        project_root = project_dir.resolve()
    except OSError:
        return ""
    try:
        build_root = build_dir.resolve()
    except OSError:
        build_root = build_dir

    def _to_project_relative(candidate: Path) -> str:
        try:
            candidate_resolved = candidate.resolve(strict=False)
        except OSError:
            return ""
        if candidate_resolved == project_root or project_root not in candidate_resolved.parents:
            return ""
        try:
            rel = candidate_resolved.relative_to(project_root).as_posix()
            normalized = _normalize_project_relative_path(rel)
            resolved = _resolve_project_path(doc_id, normalized)
        except (ValueError, OSError):
            return ""
        if (
            _is_internal_state_relative_path(normalized)
            or not _is_text_editable_project_file(normalized)
            or not resolved.exists()
            or not resolved.is_file()
        ):
            return ""
        return normalized

    def _from_build_relative(candidate: Path) -> str:
        try:
            candidate_resolved = candidate.resolve(strict=False)
        except OSError:
            return ""
        if candidate_resolved != build_root and build_root not in candidate_resolved.parents:
            return ""
        try:
            rel_to_build = candidate_resolved.relative_to(build_root).as_posix()
            normalized = _normalize_project_relative_path(rel_to_build)
            resolved = _resolve_project_path(doc_id, normalized)
        except (ValueError, OSError):
            return ""
        if (
            _is_internal_state_relative_path(normalized)
            or not _is_text_editable_project_file(normalized)
            or not resolved.exists()
            or not resolved.is_file()
        ):
            return ""
        return normalized

    candidate_paths: List[Path] = []
    raw_candidate = Path(raw_path)
    if raw_candidate.is_absolute():
        candidate_paths.append(raw_candidate)
    else:
        candidate_paths.append(project_dir / raw_candidate)
        candidate_paths.append(build_dir / raw_candidate)

    for candidate in candidate_paths:
        resolved_rel = _to_project_relative(candidate)
        if resolved_rel:
            return resolved_rel

    for candidate in candidate_paths:
        resolved_rel = _from_build_relative(candidate)
        if resolved_rel:
            return resolved_rel

    basename = Path(raw_path).name.strip()
    if not basename:
        return ""
    possible: List[str] = []
    try:
        matches = list(project_dir.rglob(basename))
    except OSError:
        matches = []
    for match in matches:
        if not match.is_file():
            continue
        try:
            rel = match.relative_to(project_dir).as_posix()
            normalized = _normalize_project_relative_path(rel)
        except (ValueError, OSError):
            continue
        if _is_internal_state_relative_path(normalized) or not _is_text_editable_project_file(normalized):
            continue
        possible.append(normalized)
    if not possible:
        return ""
    possible = sorted(dict.fromkeys(possible), key=lambda item: (0 if item == "main.tex" else 1, item.casefold()))
    return possible[0]


def _resolve_existing_path_candidates(
    doc_id: str,
    raw_path: str,
    build_dir: Path,
) -> List[Path]:
    candidate_paths: List[Path] = []
    path_like = Path(str(raw_path or "").strip().strip('"').strip("'"))
    if not str(path_like):
        return candidate_paths
    if path_like.is_absolute():
        candidate_paths.append(path_like)
    else:
        project_dir = ensure_project_dir(doc_id)
        candidate_paths.append(project_dir / path_like)
        candidate_paths.append(build_dir / path_like)
    resolved_existing: List[Path] = []
    seen: set[str] = set()
    for candidate in candidate_paths:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        marker = str(resolved)
        if marker in seen:
            continue
        seen.add(marker)
        if resolved.exists() and resolved.is_file():
            resolved_existing.append(resolved)
    return resolved_existing


def _extract_bibitem_key_for_bbl_line(bbl_text: str, line_number: int) -> str:
    if line_number <= 0:
        return ""
    lines = str(bbl_text or "").splitlines()
    if not lines:
        return ""
    target_index = max(0, min(line_number - 1, len(lines) - 1))
    bibitem_pattern = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")
    for idx in range(target_index, -1, -1):
        match = bibitem_pattern.search(lines[idx])
        if match:
            return str(match.group(1) or "").strip()
    return ""


def _find_bib_entry_line_for_key(doc_id: str, cite_key: str) -> Tuple[str, int]:
    key = str(cite_key or "").strip()
    if not key:
        return ("", -1)
    project_dir = ensure_project_dir(doc_id)
    try:
        main_text = load_project_text_file(doc_id, "main.tex")
    except OSError:
        main_text = ""
    bib_payload = _load_bib_from_latex(main_text, project_dir)
    meta = bib_payload.meta if isinstance(getattr(bib_payload, "meta", None), dict) else {}
    raw_paths = meta.get("paths", []) if isinstance(meta, dict) else []
    candidate_paths: List[Path] = []
    seen: set[str] = set()
    for raw in raw_paths if isinstance(raw_paths, list) else []:
        try:
            path = Path(str(raw)).resolve(strict=False)
        except OSError:
            continue
        marker = str(path)
        if marker in seen:
            continue
        seen.add(marker)
        if path.exists() and path.is_file() and path.suffix.lower() == ".bib":
            candidate_paths.append(path)
    if not candidate_paths:
        try:
            for path in sorted(project_dir.rglob("*.bib")):
                if not path.is_file():
                    continue
                marker = str(path.resolve(strict=False))
                if marker in seen:
                    continue
                seen.add(marker)
                candidate_paths.append(path)
        except OSError:
            candidate_paths = []

    entry_pattern = re.compile(rf"@\w+\s*\{{\s*{re.escape(key)}\s*,", re.IGNORECASE)
    for path in candidate_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if entry_pattern.search(line):
                try:
                    rel = path.relative_to(project_dir).as_posix()
                    normalized = _normalize_project_relative_path(rel)
                except (ValueError, OSError):
                    continue
                if _is_internal_state_relative_path(normalized) or not _is_text_editable_project_file(normalized):
                    continue
                return (normalized, idx)
    return ("", -1)


def _resolve_synctex_bibliography_target(
    doc_id: str,
    input_path: str,
    line_number: int,
    build_dir: Path,
) -> Tuple[str, int]:
    basename = Path(str(input_path or "").strip().strip('"').strip("'")).name.lower()
    if not basename.endswith(".bbl"):
        return ("", -1)
    for candidate in _resolve_existing_path_candidates(doc_id, input_path, build_dir):
        if candidate.suffix.lower() != ".bbl":
            continue
        try:
            bbl_text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        cite_key = _extract_bibitem_key_for_bbl_line(bbl_text, line_number)
        if not cite_key:
            continue
        return _find_bib_entry_line_for_key(doc_id, cite_key)
    return ("", -1)


def _synctex_pdf_to_source(doc_id: str, page: int, x: float, y: float) -> Tuple[str, int]:
    pdf_path_text = str(st.session_state.get("pdf_compiled_pdf_path", "")).strip()
    synctex_path_text = str(st.session_state.get("pdf_compiled_synctex_path", "")).strip()
    if not pdf_path_text:
        return ("", -1)
    if not shutil.which("synctex"):
        return ("", -1)

    pdf_path = Path(pdf_path_text)
    if not pdf_path.exists():
        return ("", -1)

    synctex_dir = ""
    if synctex_path_text:
        synctex_candidate = Path(synctex_path_text)
        if synctex_candidate.exists():
            synctex_dir = str(synctex_candidate.parent)
    if not synctex_dir:
        synctex_dir = str(pdf_path.parent)
    def _run_probe(query_page: int, query_x: float, query_y: float) -> Tuple[List[Tuple[str, int]], int]:
        probe = f"{max(1, int(query_page))}:{float(query_x):.2f}:{float(query_y):.2f}:{str(pdf_path)}"
        query_cmd = ["synctex", "edit", "-o", probe, "-d", synctex_dir]
        proc = subprocess.run(
            query_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed_candidates: List[Tuple[str, int]] = []
        first_line = -1
        current_input = ""
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            input_match = re.match(r"Input:(.+)$", line)
            if input_match:
                current_input = input_match.group(1).strip()
                continue
            line_match = re.match(r"Line:(\d+)$", line)
            if not line_match:
                continue
            try:
                parsed_line = int(line_match.group(1))
            except ValueError:
                parsed_line = -1
            if parsed_line <= 0:
                continue
            if first_line <= 0:
                first_line = parsed_line
            parsed_candidates.append((current_input, parsed_line))
        return parsed_candidates, first_line

    try:
        requested_page = max(1, int(page))
    except (TypeError, ValueError):
        requested_page = 1
    try:
        base_x = max(0.0, float(x))
    except (TypeError, ValueError):
        base_x = 0.0
    try:
        base_y = max(0.0, float(y))
    except (TypeError, ValueError):
        base_y = 0.0

    probe_offsets: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (0.0, -18.0),
        (0.0, 18.0),
        (0.0, -36.0),
        (0.0, 36.0),
        (0.0, -72.0),
        (0.0, 72.0),
        (0.0, -140.0),
        (0.0, 140.0),
        (0.0, -220.0),
        (0.0, 220.0),
        (0.0, -320.0),
        (0.0, 320.0),
        (0.0, -420.0),
        (0.0, 420.0),
        (-18.0, 0.0),
        (18.0, 0.0),
        (-36.0, 0.0),
        (36.0, 0.0),
        (-72.0, 0.0),
        (72.0, 0.0),
        (-140.0, 0.0),
        (140.0, 0.0),
        (-220.0, 0.0),
        (220.0, 0.0),
        (-24.0, -24.0),
        (24.0, -24.0),
        (-24.0, 24.0),
        (24.0, 24.0),
    ]
    probe_points: List[Tuple[float, float]] = [(dx, dy) for dx, dy in probe_offsets]
    if requested_page >= 1:
        for page_height in (792.0, 842.0):
            mirrored_y = max(0.0, page_height - base_y)
            for delta in (0.0, -36.0, 36.0, -72.0, 72.0, -140.0, 140.0):
                probe_points.append((0.0, (mirrored_y - base_y) + delta))

    hits: List[Tuple[int, int, str, int]] = []
    fallback_line = -1
    fallback_dist = 10**9
    seen = set()

    for probe_rank, (dx, dy) in enumerate(probe_points):
        query_x = max(0.0, base_x + dx)
        query_y = max(0.0, base_y + dy)
        candidates, first_line = _run_probe(requested_page, query_x, query_y)
        distance_score = int(round(abs(dx) + abs(dy)))
        if first_line > 0 and distance_score < fallback_dist:
            fallback_line = first_line
            fallback_dist = distance_score
        for candidate_rank, (candidate_input, candidate_line) in enumerate(candidates):
            resolved_rel = _resolve_synctex_input_to_project_tex(doc_id, candidate_input, pdf_path.parent)
            resolved_line = candidate_line
            if not resolved_rel:
                resolved_rel, resolved_line = _resolve_synctex_bibliography_target(
                    doc_id,
                    candidate_input,
                    candidate_line,
                    pdf_path.parent,
                )
            if not resolved_rel:
                continue
            key = (distance_score, candidate_rank, resolved_rel, resolved_line)
            if key in seen:
                continue
            seen.add(key)
            hits.append(key)

    if hits:
        non_main_hits = [item for item in hits if item[2] != "main.tex"]
        pool = non_main_hits if non_main_hits else hits
        _best_distance, _best_rank, best_rel, best_line = min(
            pool, key=lambda item: (item[0], item[1], item[3])
        )
        if best_line > 0:
            return (best_rel, best_line)

    if fallback_line > 0:
        try:
            fallback_rel = _normalize_project_relative_path(st.session_state.get("active_file_path", "main.tex"))
        except ValueError:
            fallback_rel = "main.tex"
        if _is_text_editable_project_file(fallback_rel):
            return (fallback_rel, fallback_line)
    return ("", -1)


def _handle_composer_panel(state: AppState, composer_payload: Dict[str, Any], composer_disabled: bool) -> None:
    active_workspace_id = str(st.session_state.get("active_workspace_id", "")).strip()
    if active_workspace_id:
        state.state_doc_id = active_workspace_id
    _save_processing_debug_payload(
        state.state_doc_id,
        "composer_enter",
        {
            "raw_payload": dict(composer_payload),
            "composer_disabled": bool(composer_disabled),
            "processing_busy": bool(st.session_state.get("processing_busy", False)),
            "composer_last_event_id": int(st.session_state.get("composer_last_event_id", 0)),
            "active_workspace_id": active_workspace_id,
        },
    )
    if bool(st.session_state.get("processing_busy", False)):
        _save_processing_debug_payload(
            state.state_doc_id,
            "composer_early_return_busy",
            {
                "raw_payload": dict(composer_payload),
            },
        )
        return

    action = str(composer_payload.get("action", "idle"))
    draft = str(composer_payload.get("draft", ""))
    submit_text = str(composer_payload.get("submit_text", "")).strip()
    event_id = int(composer_payload.get("event_id", 0))
    payload_tools_open = bool(composer_payload.get("tools_open", st.session_state.composer_tools_open))

    if event_id and event_id == int(st.session_state.get("composer_last_event_id", 0)):
        action = "idle"
    elif event_id:
        st.session_state.composer_last_event_id = event_id

    _save_processing_debug_payload(
        state.state_doc_id,
        "composer_normalized",
        {
            "action": action,
            "draft": draft,
            "submit_text": submit_text,
            "event_id": event_id,
            "payload_tools_open": payload_tools_open,
            "composer_disabled": bool(composer_disabled),
            "selected_checks": list(composer_payload.get("selected_checks", [])),
        },
    )

    if draft != st.session_state.composer_text:
        st.session_state.composer_text = draft

    if not composer_disabled:
        st.session_state.composer_tools_open = payload_tools_open

    if action == "toggle_tools" and not composer_disabled:
        st.session_state.composer_tools_open = payload_tools_open
    elif action == "set_checks":
        enabled_map = _normalize_check_enabled_map(st.session_state.get("check_enabled_map"))
        selected_checks = [
            x for x in list(composer_payload.get("selected_checks", []))
            if x in CHECK_NAME_SET and enabled_map.get(x, True)
        ]
        st.session_state.selected_checks = selected_checks
        st.session_state.selected_checks_explicit = True
    elif action == "run_checks" and not composer_disabled:
        enabled_map = _normalize_check_enabled_map(st.session_state.get("check_enabled_map"))
        requested_checks = [
            x for x in list(composer_payload.get("selected_checks", st.session_state.get("selected_checks", [])))
            if x in CHECK_NAME_SET
        ]
        selected_checks = [
            x for x in requested_checks
            if x in CHECK_NAME_SET and enabled_map.get(x, True)
        ]
        _save_processing_debug_payload(
            state.state_doc_id,
            "composer_run_checks_filtered",
            {
                "enabled_map": dict(enabled_map),
                "requested_checks": list(requested_checks),
                "filtered_checks": list(selected_checks),
                "composer_disabled": bool(composer_disabled),
            },
        )
        st.session_state.selected_checks = selected_checks
        st.session_state.selected_checks_explicit = True
        checks_to_dispatch = selected_checks or requested_checks
        if checks_to_dispatch:
            _set_processing_busy(
                kind="checks",
                pending_action={
                    "action": "run_checks",
                    "selected_checks": checks_to_dispatch,
                    "ui_checks_explicit": True,
                },
            )
            st.rerun()
    elif action == "undo" and not composer_disabled:
        message = _undo_last_change(state)
        if message:
            _append_chat(
                state,
                role="assistant",
                content=message,
                companion_role=state.role,
                mode="undo",
            )
            _persist_state(state)
            st.rerun()

    if action == "send" and submit_text:
        if active_workspace_id and state.state_doc_id != active_workspace_id:
            return
        st.session_state.composer_text = ""
        _set_processing_busy(
            kind="chat",
            pending_action={
                "action": "send_message",
                "submit_text": submit_text,
                "selected_checks": list(st.session_state.get("selected_checks", [])),
                "ui_checks_explicit": bool(st.session_state.get("selected_checks_explicit", False)),
            },
        )
        st.rerun()


def _inject_left_sidebar_toggle() -> None:
    st_components.html(
        """
<script>
(() => {
  const doc = window.parent && window.parent.document ? window.parent.document : null;
  if (!doc) return;
  const tooltipId = "awc-activity-tooltip";
  const tooltipOffsetX = 10;
  const tooltipOffsetY = 0;
  const ensureTooltip = () => {
    let tip = doc.getElementById(tooltipId);
    if (tip) return tip;
    tip = doc.createElement("div");
    tip.id = tooltipId;
    tip.setAttribute("aria-hidden", "true");
    Object.assign(tip.style, {
      position: "fixed",
      left: "0px",
      top: "0px",
      transform: "translateY(-50%)",
      whiteSpace: "nowrap",
      fontSize: "13px",
      lineHeight: "1.24",
      fontWeight: "600",
      color: "#111827",
      background: "#ffffff",
      border: "1px solid #d1d5db",
      borderRadius: "6px",
      padding: "6px 12px",
      boxShadow: "0 6px 18px rgba(15, 23, 42, 0.12)",
      opacity: "0",
      visibility: "hidden",
      pointerEvents: "none",
      zIndex: "2147483647",
      transition: "opacity 120ms ease, visibility 120ms ease"
    });
    doc.body.appendChild(tip);
    return tip;
  };
  const tooltip = ensureTooltip();
  const setTooltipVisible = (visible) => {
    tooltip.style.opacity = visible ? "1" : "0";
    tooltip.style.visibility = visible ? "visible" : "hidden";
  };
  const hideTooltip = () => setTooltipVisible(false);
  const placeTooltip = (el) => {
    const rect = el.getBoundingClientRect();
    tooltip.style.left = `${rect.right + tooltipOffsetX}px`;
    tooltip.style.top = `${rect.top + (rect.height / 2) + tooltipOffsetY}px`;
  };
  const bindTooltip = (visual) => {
    if (!visual || visual.dataset.awcTooltipBound === "1") return;
    const readLabel = () =>
      visual.getAttribute("data-tooltip") ||
      visual.getAttribute("aria-label") ||
      "";
    const show = () => {
      const label = readLabel();
      if (!label) return;
      tooltip.textContent = label;
      placeTooltip(visual);
      setTooltipVisible(true);
    };
    visual.addEventListener("mouseenter", show);
    visual.addEventListener("mousemove", () => {
      if (tooltip.style.visibility === "visible") {
        placeTooltip(visual);
      }
    });
    visual.addEventListener("mouseleave", hideTooltip);
    visual.addEventListener("focus", show, true);
    visual.addEventListener("blur", hideTooltip, true);
    visual.dataset.awcTooltipBound = "1";
  };
  const getPanelColumnEl = () => {
    const sideMarker = doc.querySelector(".awc-sidepanel-marker");
    if (!sideMarker) return null;
    const verticalBlock = sideMarker.closest("div[data-testid='stVerticalBlock']");
    if (!verticalBlock) return null;
    return verticalBlock.parentElement;
  };
  const pinPanelTopOnce = () => {
    const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
    const panelColEl = getPanelColumnEl();
    if (!sidebar || !panelColEl) return false;
    const panelDisplay = window.getComputedStyle(panelColEl).display;
    if (panelDisplay === "none") {
      panelColEl.dataset.awcTopPinned = "";
      panelColEl.style.setProperty("transform", "none", "important");
      panelColEl.style.setProperty("will-change", "auto", "important");
      return false;
    }
    if (panelColEl.dataset.awcTopPinned === "1") {
      return true;
    }
    panelColEl.style.setProperty("transform", "none", "important");
    panelColEl.style.setProperty("will-change", "auto", "important");
    const sidebarTop = sidebar.getBoundingClientRect().top;
    const panelTop = panelColEl.getBoundingClientRect().top;
    const delta = Math.round((sidebarTop + 2) - panelTop);
    panelColEl.style.setProperty("transform", `translateY(${delta}px)`, "important");
    panelColEl.style.setProperty("will-change", "transform", "important");
    panelColEl.dataset.awcTopPinned = "1";
    return true;
  };
  let pinRaf = null;
  const schedulePinPanelTop = () => {
    if (pinRaf) {
      window.cancelAnimationFrame(pinRaf);
      pinRaf = null;
    }
    let attempts = 0;
    const tick = () => {
      if (pinPanelTopOnce()) return;
      attempts += 1;
      if (attempts < 28) {
        pinRaf = window.requestAnimationFrame(tick);
        return;
      }
      const panelColEl = getPanelColumnEl();
      if (panelColEl) {
        panelColEl.style.setProperty("transform", "none", "important");
        panelColEl.style.setProperty("will-change", "auto", "important");
        panelColEl.dataset.awcTopPinned = "1";
      }
    };
    pinRaf = window.requestAnimationFrame(tick);
  };
  const latestHiddenToggleButton = (mode) => {
    const nodes = Array.from(doc.querySelectorAll(`[class*="st-key-sidebar_nav_${mode}_btn"] button`));
    for (let index = nodes.length - 1; index >= 0; index -= 1) {
      const node = nodes[index];
      if (node && node.isConnected) return node;
    }
    return null;
  };
  const bridge = (mode, visualId) => {
    const visual = doc.getElementById(visualId);
    if (!visual) return false;
    if (visual.__awcBridgeClickHandler) {
      try {
        visual.removeEventListener("click", visual.__awcBridgeClickHandler);
      } catch (_err) {
        // No-op.
      }
    }
    const triggerHidden = (attempt = 0) => {
      const hidden = latestHiddenToggleButton(mode);
      if (hidden) {
        hidden.click();
        window.setTimeout(schedulePinPanelTop, 0);
        return;
      }
      if (attempt < 12) {
        window.requestAnimationFrame(() => triggerHidden(attempt + 1));
      }
    };
    const clickHandler = (event) => {
      event.preventDefault();
      event.stopPropagation();
      hideTooltip();
      const panelColEl = getPanelColumnEl();
      if (panelColEl) {
        panelColEl.dataset.awcTopPinned = "";
      }
      triggerHidden();
    };
    visual.addEventListener("click", clickHandler);
    visual.__awcBridgeClickHandler = clickHandler;
    return true;
  };
  const bind = () => {
    const chats = doc.getElementById("awc-activity-chats");
    const files = doc.getElementById("awc-activity-files");
    bindTooltip(chats);
    bindTooltip(files);
    bridge("chats", "awc-activity-chats");
    bridge("files", "awc-activity-files");
    schedulePinPanelTop();
  };
  bind();

  try {
    if (!hostWin.__awcLeftSidebarObserver) {
      const observer = new MutationObserver(() => {
        bind();
        const active = doc.activeElement;
        if (!active || !active.classList || !active.classList.contains("awc-activity-btn")) {
          hideTooltip();
        }
        schedulePinPanelTop();
      });
      observer.observe(doc.body, { childList: true, subtree: true });
      hostWin.__awcLeftSidebarObserver = observer;
    }
  } catch (_err) {
    // No-op.
  }
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _inject_files_preview_resizer() -> None:
    st_components.html(
        """
<script>
(() => {
  const doc = window.parent && window.parent.document ? window.parent.document : null;
  if (!doc) return;

  const previewSelector =
    'section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.awc-sidepanel-marker) > div:nth-child(2) [class*="st-key-files_resource_preview_mount_"]';
  const widthStorageKey = "awc_files_preview_width_px";
  const minWidth = 320;
  const maxWidth = 760;

  const clampWidth = (value) => Math.max(minWidth, Math.min(maxWidth, value));
  const readWidth = () => {
    try {
      const raw = window.localStorage.getItem(widthStorageKey) || "";
      const parsed = Number.parseFloat(raw);
      if (!Number.isNaN(parsed)) return clampWidth(parsed);
    } catch (_err) {
      // No-op.
    }
    return 388;
  };
  const saveWidth = (value) => {
    try {
      window.localStorage.setItem(widthStorageKey, String(clampWidth(value)));
    } catch (_err) {
      // No-op.
    }
  };

  const applyWidth = (panel, width) => {
    if (!panel) return;
    const normalized = clampWidth(width);
    panel.style.setProperty("--awc-files-preview-width", `${normalized}px`);
  };

  let dragging = false;
  let activePanel = null;
  let activePointerId = null;
  let panelLeft = 0;
  let currentWidth = readWidth();

  const ensureResizer = (panel) => {
    if (!panel) return;
    applyWidth(panel, currentWidth);
    if (panel.querySelector(".awc-files-preview-resizer")) return;

    const handle = doc.createElement("div");
    handle.className = "awc-files-preview-resizer";
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", "vertical");
    handle.setAttribute("aria-label", "Resize preview panel");

    handle.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      dragging = true;
      activePanel = panel;
      activePointerId = event.pointerId;
      panelLeft = panel.getBoundingClientRect().left;
      panel.classList.add("is-resizing");
      if (typeof handle.setPointerCapture === "function") {
        try { handle.setPointerCapture(event.pointerId); } catch (_err) {}
      }
      event.preventDefault();
    });

    handle.addEventListener("pointermove", (event) => {
      if (!dragging || event.pointerId !== activePointerId || !activePanel) return;
      const nextWidth = clampWidth(event.clientX - panelLeft);
      currentWidth = nextWidth;
      applyWidth(activePanel, nextWidth);
      event.preventDefault();
    });

    const stopDrag = (event) => {
      if (!dragging || event.pointerId !== activePointerId) return;
      dragging = false;
      if (activePanel) {
        activePanel.classList.remove("is-resizing");
      }
      if (typeof handle.releasePointerCapture === "function") {
        try { handle.releasePointerCapture(event.pointerId); } catch (_err) {}
      }
      saveWidth(currentWidth);
      activePointerId = null;
      activePanel = null;
    };

    handle.addEventListener("pointerup", stopDrag);
    handle.addEventListener("pointercancel", stopDrag);

    panel.appendChild(handle);
  };

  const bind = () => {
    const panels = doc.querySelectorAll(previewSelector);
    if (!panels || panels.length === 0) return;
    panels.forEach((panel) => ensureResizer(panel));
  };

  bind();
  try {
    const observer = new MutationObserver(() => {
      bind();
    });
    observer.observe(doc.body, { childList: true, subtree: true });
    window.setTimeout(() => observer.disconnect(), 90000);
  } catch (_err) {
    // No-op.
  }
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _inject_files_dnd_bridge() -> None:
    panel_mode = str(st.session_state.get("sidebar_panel_mode", "") or "").strip().lower()
    doc_id = str(st.session_state.get("active_workspace_id", "")).strip()
    visible_nodes = _visible_files_tree_nodes(doc_id) if panel_mode == "files" and doc_id else []
    bridge_payload = json.dumps(
        {
            "panel_mode": panel_mode,
            "doc_id": doc_id,
            "nodes": visible_nodes,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    st_components.html(
        f"""
<script>
(() => {{
  const doc = window.parent && window.parent.document ? window.parent.document : null;
  if (!doc) return;
  const parentView = doc.defaultView || window.parent || window;

  const config = {bridge_payload};
  const bridgeFrameSelector = '[class*="st-key-files_dnd_bridge_component"] iframe';
  const panelHoverAttr = "data-awc-files-root-drop-hover";
  const nodeHoverAttr = "data-awc-files-drop-hover";
  const longNameTooltipId = "awc-files-longname-tooltip";
  const excludedLabels = new Set([
    "+ File",
    "+ Folder",
    "Upload",
    "Create",
    "Rename",
    "Delete",
    "Download",
    "Download ZIP",
    "Yes",
    "No",
    "Close",
    "⋯",
    "..."
  ]);
  const bridgeStateKey = "__awcFilesDndBridge";
  const bridgeState = parentView[bridgeStateKey] || {{
    config: {{
      panel_mode: "",
      doc_id: "",
      nodes: [],
    }},
    draggingEntry: null,
    deleteConfirmVisible: false,
  }};
  parentView[bridgeStateKey] = bridgeState;
  bridgeState.config = config;

  const normalizeText = (value) => String(value || "").replace(/\\s+/g, " ").trim();
  const isVisible = (element) => {{
    if (!element) return false;
    const style = parentView.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden") return false;
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }};
  let eventSeq = 0;
  const nextEventId = () => {{
    eventSeq += 1;
    return Date.now() * 100 + eventSeq;
  }};
  const latestElement = (selector) => {{
    const nodes = Array.from(doc.querySelectorAll(selector));
    for (let index = nodes.length - 1; index >= 0; index -= 1) {{
      const node = nodes[index];
      if (node && node.isConnected) return node;
    }}
    return null;
  }};
  const getBridgeFrame = () => {{
    const iframe = latestElement(bridgeFrameSelector);
    if (!iframe || !iframe.contentWindow) return null;
    return iframe;
  }};
  const submitBridgeAction = (action, payload) => {{
    const frame = getBridgeFrame();
    if (!frame) return false;
    frame.contentWindow.postMessage(
      {{
        isStreamlitMessage: true,
        type: "awc-files-dnd-component-event",
        payload: {{
          action,
          payload,
          event_id: nextEventId(),
        }},
      }},
      "*",
    );
    return true;
  }};
  if (parentView.__awcPdfDblclickToFilesBridgeHandler) {{
    try {{
      parentView.removeEventListener("message", parentView.__awcPdfDblclickToFilesBridgeHandler, false);
    }} catch (_err) {{
      // No-op.
    }}
  }}
  const pdfDblclickToBridge = (event) => {{
    const data = event && event.data ? event.data : {{}};
    if (!data || data.type !== "awc-pdf-dblclick") return;
    const payload = data.payload && typeof data.payload === "object" ? data.payload : {{}};
    const page = Number(payload.page || 0) || 0;
    if (page <= 0) return;
    const x = Number(payload.x || 0) || 0;
    const y = Number(payload.y || 0) || 0;
    submitBridgeAction("pdf_dblclick", {{ page, x, y }});
  }};
  parentView.__awcPdfDblclickToFilesBridgeHandler = pdfDblclickToBridge;
  parentView.addEventListener("message", pdfDblclickToBridge, false);
  const normalizeDroppedRelativePath = (value) => {{
    let text = String(value || "").replace(/\\\\/g, "/").trim();
    while (text.startsWith("./")) {{
      text = text.slice(2);
    }}
    text = text.replace(/^\/+/, "");
    return text;
  }};
  const readFilePayload = async (file, relativePath = "") => {{
    const normalizedRelative = normalizeDroppedRelativePath(relativePath);
    const fileName = String(file && file.name ? file.name : "").trim();
    const fallbackRelative = normalizeDroppedRelativePath(fileName);
    return new Promise((resolve, reject) => {{
      const reader = new parentView.FileReader();
      reader.onload = () => {{
        const result = String(reader.result || "");
        const commaIndex = result.indexOf(",");
        resolve({{
          name: fileName,
          mime: String(file && file.type ? file.type : ""),
          relative_path: normalizedRelative || fallbackRelative,
          content_b64: commaIndex >= 0 ? result.slice(commaIndex + 1) : "",
        }});
      }};
      reader.onerror = () => reject(new Error(`Failed to read ${{fileName || "dropped file"}}`));
      reader.readAsDataURL(file);
    }});
  }};
  const readDroppedFiles = async (fileList) => {{
    const files = Array.from(fileList || []).filter((file) => {{
      return !!file && typeof file.name === "string" && typeof file.size === "number";
    }});
    const reads = files.map((file) => {{
      const relative = normalizeDroppedRelativePath(file.webkitRelativePath || file.name || "");
      return readFilePayload(file, relative);
    }});
    return Promise.all(reads);
  }};
  const readDirectoryEntries = (directoryEntry) => new Promise((resolve, reject) => {{
    try {{
      const reader = directoryEntry.createReader();
      const output = [];
      const pump = () => {{
        reader.readEntries(
          (batch) => {{
            const entries = Array.from(batch || []);
            if (!entries.length) {{
              resolve(output);
              return;
            }}
            output.push(...entries);
            pump();
          }},
          (error) => reject(error || new Error("Failed to read dropped directory.")),
        );
      }};
      pump();
    }} catch (error) {{
      reject(error);
    }}
  }});
  const readEntryTree = async (entry, prefix = "") => {{
    if (!entry) return [];
    if (entry.isFile) {{
      const file = await new Promise((resolve, reject) => {{
        try {{
          entry.file(resolve, reject);
        }} catch (error) {{
          reject(error);
        }}
      }});
      const basePrefix = normalizeDroppedRelativePath(prefix);
      const relative = normalizeDroppedRelativePath(basePrefix ? `${{basePrefix}}/${{file.name}}` : file.name);
      return [await readFilePayload(file, relative)];
    }}
    if (entry.isDirectory) {{
      const dirName = normalizeDroppedRelativePath(entry.name || "");
      const basePrefix = normalizeDroppedRelativePath(prefix);
      const nextPrefix = normalizeDroppedRelativePath(basePrefix ? `${{basePrefix}}/${{dirName}}` : dirName);
      const children = await readDirectoryEntries(entry);
      let files = [];
      for (const child of children) {{
        const childFiles = await readEntryTree(child, nextPrefix);
        if (childFiles.length) files = files.concat(childFiles);
      }}
      return files;
    }}
    return [];
  }};
  const readDroppedDataTransfer = async (dataTransfer) => {{
    const transfer = dataTransfer || null;
    if (!transfer) return [];
    const items = Array.from(transfer.items || []);
    const canReadEntries = items.some((item) => item && typeof item.webkitGetAsEntry === "function");
    if (canReadEntries) {{
      let files = [];
      for (const item of items) {{
        if (!item || String(item.kind || "").toLowerCase() !== "file") continue;
        let entry = null;
        try {{
          entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
        }} catch (_err) {{
          entry = null;
        }}
        if (!entry) continue;
        const entryFiles = await readEntryTree(entry, "");
        if (entryFiles.length) files = files.concat(entryFiles);
      }}
      if (files.length) return files;
    }}
    return readDroppedFiles(transfer.files || []);
  }};
  const hasDeleteConfirmPopover = () => {{
    const popoverBodies = Array.from(doc.querySelectorAll('[data-testid="stPopoverBody"]'));
    return popoverBodies.some((body) => {{
      const text = normalizeText(body.innerText || body.textContent || "").toLowerCase();
      return text.includes("confirm deletion");
    }});
  }};
  const getPanelColumn = () => {{
    const sideMarker = doc.querySelector(".awc-sidepanel-marker");
    if (!sideMarker) return null;
    const verticalBlock = sideMarker.closest("div[data-testid='stVerticalBlock']");
    if (!verticalBlock) return null;
    const panelColumn = verticalBlock.parentElement;
    if (!panelColumn) return null;
    const headings = Array.from(panelColumn.querySelectorAll("h3"));
    if (!headings.some((heading) => normalizeText(heading.textContent) === "Files")) return null;
    return panelColumn;
  }};
  const setRootDropHover = (active) => {{
    const panelColumn = getPanelColumn();
    if (!panelColumn) return;
    const verticalBlock = panelColumn.querySelector('div[data-testid="stVerticalBlock"]');
    if (active) {{
      panelColumn.setAttribute(panelHoverAttr, "1");
      if (verticalBlock) {{
        verticalBlock.setAttribute(panelHoverAttr, "1");
      }}
      return;
    }}
    panelColumn.removeAttribute(panelHoverAttr);
    if (verticalBlock) {{
      verticalBlock.removeAttribute(panelHoverAttr);
    }}
  }};
  const clearDropHover = () => {{
    setRootDropHover(false);
    doc
      .querySelectorAll(`section[data-testid="stSidebar"] button[${{nodeHoverAttr}}="1"]`)
      .forEach((button) => button.removeAttribute(nodeHoverAttr));
  }};
  const ensureLongNameTooltip = () => {{
    let tip = doc.getElementById(longNameTooltipId);
    if (tip) return tip;
    tip = doc.createElement("div");
    tip.id = longNameTooltipId;
    tip.style.position = "fixed";
    tip.style.left = "0";
    tip.style.top = "0";
    tip.style.transform = "translate(-50%, calc(-100% - 8px))";
    tip.style.zIndex = "30000";
    tip.style.padding = "0.36rem 0.58rem";
    tip.style.borderRadius = "8px";
    tip.style.background = "#ffffff";
    tip.style.color = "#111827";
    tip.style.border = "1px solid #d1d5db";
    tip.style.boxShadow = "0 10px 30px rgba(15, 23, 42, 0.18)";
    tip.style.fontSize = "0.82rem";
    tip.style.fontWeight = "500";
    tip.style.lineHeight = "1.2";
    tip.style.whiteSpace = "nowrap";
    tip.style.pointerEvents = "none";
    tip.style.opacity = "0";
    tip.style.visibility = "hidden";
    doc.body.appendChild(tip);
    return tip;
  }};
  const longNameTooltip = ensureLongNameTooltip();
  const setLongNameTooltipVisible = (visible) => {{
    longNameTooltip.style.opacity = visible ? "1" : "0";
    longNameTooltip.style.visibility = visible ? "visible" : "hidden";
  }};
  const hideLongNameTooltip = () => {{
    setLongNameTooltipVisible(false);
  }};
  const placeLongNameTooltip = (button) => {{
    if (!button) return;
    const rect = button.getBoundingClientRect();
    longNameTooltip.style.left = `${{rect.left + (rect.width / 2)}}px`;
    longNameTooltip.style.top = `${{rect.top}}px`;
  }};
  const bindLongNameTooltip = (button, fullName) => {{
    if (!button || !fullName) return;
    button.dataset.awcFullName = fullName;
    if (button.__awcLongNameTooltipHandlers) {{
      button.removeEventListener("mouseenter", button.__awcLongNameTooltipHandlers.show);
      button.removeEventListener("mouseover", button.__awcLongNameTooltipHandlers.over);
      button.removeEventListener("pointerenter", button.__awcLongNameTooltipHandlers.show);
      button.removeEventListener("mousemove", button.__awcLongNameTooltipHandlers.move);
      button.removeEventListener("mouseleave", button.__awcLongNameTooltipHandlers.hide);
      button.removeEventListener("pointerleave", button.__awcLongNameTooltipHandlers.hide);
      button.removeEventListener("focus", button.__awcLongNameTooltipHandlers.show, true);
      button.removeEventListener("blur", button.__awcLongNameTooltipHandlers.hide, true);
    }}
    const show = () => {{
      const value = normalizeText(button.dataset.awcFullName || "");
      if (!value) return;
      longNameTooltip.textContent = value;
      placeLongNameTooltip(button);
      setLongNameTooltipVisible(true);
    }};
    const move = () => {{
      if (longNameTooltip.style.visibility === "visible") {{
        placeLongNameTooltip(button);
      }}
    }};
    const over = () => show();
    button.addEventListener("mouseenter", show);
    button.addEventListener("mouseover", over);
    button.addEventListener("pointerenter", show);
    button.addEventListener("mousemove", move);
    button.addEventListener("mouseleave", hideLongNameTooltip);
    button.addEventListener("pointerleave", hideLongNameTooltip);
    button.addEventListener("focus", show, true);
    button.addEventListener("blur", hideLongNameTooltip, true);
    button.__awcLongNameTooltipHandlers = {{
      show,
      over,
      move,
      hide: hideLongNameTooltip,
    }};
  }};
  const isLocalFileDrag = (event) => {{
    const transfer = event.dataTransfer;
    if (!transfer) return false;
    const items = Array.from(transfer.items || []);
    if (items.some((item) => item && String(item.kind || "").toLowerCase() === "file")) {{
      return true;
    }}
    const types = Array.from(transfer.types || []);
    if (types.includes("Files") || types.includes("application/x-moz-file")) return true;
    return types.some((value) => String(value || "").toLowerCase().includes("file"));
  }};
  const resolveDraggingEntry = (event) => {{
    if (isLocalFileDrag(event)) {{
      return null;
    }}
    if (bridgeState.draggingEntry && bridgeState.draggingEntry.docId === bridgeState.config.doc_id) {{
      return bridgeState.draggingEntry;
    }}
    const transfer = event && event.dataTransfer;
    if (!transfer || typeof transfer.getData !== "function") return null;
    const raw = String(transfer.getData("application/x-awc-project-entry") || "").trim();
    if (!raw) return null;
    try {{
      const parsed = JSON.parse(raw);
      const path = String(parsed && parsed.path ? parsed.path : "").trim();
      if (!path) return null;
      return {{
        docId: String(parsed && parsed.docId ? parsed.docId : bridgeState.config.doc_id),
        path,
        kind: String(parsed && parsed.kind ? parsed.kind : ""),
      }};
    }} catch (_error) {{
      return null;
    }}
  }};

  const closestNodeButton = (target) => {{
    if (!target || !target.closest) return null;
    const button = target.closest("button[data-awc-files-node-kind][data-awc-files-node-path]");
    return button && button.isConnected ? button : null;
  }};
  const closestDirButton = (target) => {{
    const button = closestNodeButton(target);
    if (!button) return null;
    return button.getAttribute("data-awc-files-node-kind") === "dir" ? button : null;
  }};
  const setHoveredTarget = (button) => {{
    doc
      .querySelectorAll(`section[data-testid="stSidebar"] button[${{nodeHoverAttr}}="1"]`)
      .forEach((node) => {{
        if (button && node === button) return;
        node.removeAttribute(nodeHoverAttr);
      }});
    if (button) {{
      button.setAttribute(nodeHoverAttr, "1");
      setRootDropHover(false);
    }} else {{
      setRootDropHover(true);
    }}
  }};

  const bindTreeButtons = () => {{
    if (bridgeState.config.panel_mode !== "files" || !bridgeState.config.doc_id) {{
      clearDropHover();
      return;
    }}
    const panelColumn = getPanelColumn();
    if (!panelColumn) return;

    const allButtons = Array.from(
      panelColumn.querySelectorAll(
        '[class*="st-key-files_open_"] button, [class*="st-key-files_toggle_"] button',
      ),
    ).filter((button) => {{
      if (!isVisible(button)) return false;
      if (button.closest('[data-testid="stPopoverBody"]')) return false;
      const label = normalizeText(button.innerText || button.textContent || "");
      if (!label) return false;
      if (excludedLabels.has(label)) return false;
      return true;
    }});

    const nodes = Array.isArray(bridgeState.config.nodes) ? bridgeState.config.nodes : [];
    const mapCount = Math.min(nodes.length, allButtons.length);
    const mapped = [];
    for (let index = 0; index < mapCount; index += 1) {{
      mapped.push({{ node: nodes[index], button: allButtons[index] }});
    }}

    mapped.forEach((entry) => {{
      const node = entry.node || {{}};
      const button = entry.button;
      const isFixedRootMainTex = String(node.path || "") === "main.tex";
	      const fullName = String(node.name || "");
	      const fullTitle = String(node.path || node.name || "");
	      const isLongFileName = fullName.length >= 20;
      button.setAttribute("data-awc-files-node-kind", String(node.kind || ""));
	      button.setAttribute("data-awc-files-node-path", String(node.path || ""));
	      button.setAttribute("data-awc-files-full-label", fullName);
	      button.setAttribute("data-awc-files-full-title", fullTitle);
	      button.setAttribute("data-awc-files-long-name", isLongFileName ? "1" : "0");
	      button.setAttribute("aria-label", fullName || fullTitle);
	      if (isLongFileName) {{
	        button.setAttribute("title", fullName || fullTitle);
	      }} else {{
	        button.removeAttribute("title");
	      }}
	      if (button.__awcLongNameTooltipHandlers) {{
	        button.removeEventListener("mouseenter", button.__awcLongNameTooltipHandlers.show);
	        button.removeEventListener("mouseover", button.__awcLongNameTooltipHandlers.over);
	        button.removeEventListener("pointerenter", button.__awcLongNameTooltipHandlers.show);
	        button.removeEventListener("mousemove", button.__awcLongNameTooltipHandlers.move);
	        button.removeEventListener("mouseleave", button.__awcLongNameTooltipHandlers.hide);
	        button.removeEventListener("pointerleave", button.__awcLongNameTooltipHandlers.hide);
	        button.removeEventListener("focus", button.__awcLongNameTooltipHandlers.show, true);
	        button.removeEventListener("blur", button.__awcLongNameTooltipHandlers.hide, true);
	        button.__awcLongNameTooltipHandlers = null;
	      }}
	      delete button.dataset.awcFullName;
	      const labelNode = button.querySelector("p");
	      if (labelNode) {{
	        if (isLongFileName) {{
	          labelNode.setAttribute("title", fullName || fullTitle);
	        }} else {{
	          labelNode.removeAttribute("title");
	        }}
	      }}
	      button.setAttribute("draggable", isFixedRootMainTex ? "false" : "true");
      button.style.webkitUserDrag = "element";
      if (button.__awcFilesDndSourceHandlers) {{
        button.removeEventListener("dragstart", button.__awcFilesDndSourceHandlers.dragstart);
        button.removeEventListener("dragend", button.__awcFilesDndSourceHandlers.dragend);
      }}
      if (isFixedRootMainTex) {{
        button.__awcFilesDndSourceHandlers = null;
        return;
      }}
      const dragstart = (event) => {{
        bridgeState.draggingEntry = {{
          docId: bridgeState.config.doc_id,
          path: String(node.path || ""),
          kind: String(node.kind || ""),
        }};
        if (event.dataTransfer) {{
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData(
            "application/x-awc-project-entry",
            JSON.stringify(bridgeState.draggingEntry),
          );
          event.dataTransfer.setData("text/plain", String(node.path || ""));
        }}
      }};
      const dragend = () => {{
        bridgeState.draggingEntry = null;
        clearDropHover();
        hideLongNameTooltip();
      }};
      button.addEventListener("dragstart", dragstart);
      button.addEventListener("dragend", dragend);
      button.__awcFilesDndSourceHandlers = {{ dragstart, dragend }};
    }});

    if (panelColumn.__awcFilesDndDelegatedHandlers) {{
      const handlers = panelColumn.__awcFilesDndDelegatedHandlers;
      panelColumn.removeEventListener("dragenter", handlers.dragenter);
      panelColumn.removeEventListener("dragover", handlers.dragover);
      panelColumn.removeEventListener("dragleave", handlers.dragleave);
      panelColumn.removeEventListener("drop", handlers.drop);
    }}

    const dragenter = (event) => {{
      const movingEntry = resolveDraggingEntry(event);
      const localFiles = isLocalFileDrag(event);
      if (!movingEntry && !localFiles) return;
      event.preventDefault();
      event.stopPropagation();
      setHoveredTarget(closestDirButton(event.target));
    }};

    const dragover = (event) => {{
      const movingEntry = resolveDraggingEntry(event);
      const localFiles = isLocalFileDrag(event);
      if (!movingEntry && !localFiles) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) {{
        event.dataTransfer.dropEffect = movingEntry ? "move" : "copy";
      }}
      setHoveredTarget(closestDirButton(event.target));
    }};

    const dragleave = (event) => {{
      const related = event.relatedTarget;
      if (related && panelColumn.contains(related)) return;
      clearDropHover();
      hideLongNameTooltip();
    }};

    const drop = async (event) => {{
      const movingEntry = resolveDraggingEntry(event);
      const localFiles = isLocalFileDrag(event);
      if (!movingEntry && !localFiles) return;
      event.preventDefault();
      event.stopPropagation();

      const targetDirButton = closestDirButton(event.target);
      const targetDir = String(targetDirButton?.getAttribute("data-awc-files-node-path") || "").trim();
      clearDropHover();

      if (movingEntry) {{
        if (!movingEntry.path) {{
          bridgeState.draggingEntry = null;
          return;
        }}
        submitBridgeAction("move", {{
          doc_id: bridgeState.config.doc_id,
          source_path: movingEntry.path,
          target_dir: targetDir,
        }});
        bridgeState.draggingEntry = null;
        return;
      }}

      try {{
        const files = await readDroppedDataTransfer(event.dataTransfer);
        if (!files.length) return;
        submitBridgeAction("upload", {{
          doc_id: bridgeState.config.doc_id,
          target_dir: targetDir,
          files,
        }});
        bridgeState.draggingEntry = null;
        hideLongNameTooltip();
      }} catch (_error) {{
        clearDropHover();
        hideLongNameTooltip();
      }}
    }};

    panelColumn.addEventListener("dragenter", dragenter);
    panelColumn.addEventListener("dragover", dragover);
    panelColumn.addEventListener("dragleave", dragleave);
    panelColumn.addEventListener("drop", drop);
    panelColumn.__awcFilesDndDelegatedHandlers = {{ dragenter, dragover, dragleave, drop }};
  }};

  bindTreeButtons();
  const syncDeleteConfirmState = () => {{
    const visibleNow = hasDeleteConfirmPopover();
    if (bridgeState.deleteConfirmVisible && !visibleNow) {{
      submitBridgeAction("reset_delete_confirms", {{}});
    }}
    bridgeState.deleteConfirmVisible = visibleNow;
  }};
  syncDeleteConfirmState();
  try {{
    const observer = new MutationObserver(() => {{
      bindTreeButtons();
      syncDeleteConfirmState();
    }});
    observer.observe(doc.body, {{ childList: true, subtree: true }});
    window.setTimeout(() => observer.disconnect(), 90000);
  }} catch (_err) {{
    // No-op.
  }}
}})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _inject_long_filename_button_fix() -> None:
    return


def _inject_chat_drawer_toggle() -> None:
    st_components.html(
        """
<script>
(() => {
  const doc = window.parent && window.parent.document ? window.parent.document : null;
  if (!doc) return;
  const body = doc.body;
  const storageKey = "awc_chat_drawer_open_session";
  const drawerClass = "awc-chat-drawer-open";
  const buttonId = "awc-chat-drawer-fab";

  const getStore = () => {
    try { return window.parent.sessionStorage; } catch (_err) { return null; }
  };
  const safeGet = () => {
    const store = getStore();
    if (!store) return null;
    try { return store.getItem(storageKey); } catch (_err) { return null; }
  };
  const safeSet = (value) => {
    const store = getStore();
    if (!store) return;
    try { store.setItem(storageKey, value); } catch (_err) {}
  };

  const setOpen = (open) => {
    body.classList.toggle(drawerClass, Boolean(open));
    const btn = doc.getElementById(buttonId);
    if (btn) {
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      btn.title = open ? "Hide Companion Chat" : "Open Companion Chat";
    }
    safeSet(open ? "1" : "0");
  };

  let btn = doc.getElementById(buttonId);
  if (!btn) {
    btn = doc.createElement("button");
    btn.id = buttonId;
    btn.type = "button";
    btn.innerHTML = "<span aria-hidden='true'>🤖</span><span class='awc-chat-fab-label'>Companion</span>";
    doc.body.appendChild(btn);
  }
  if (btn.__awcDrawerClickHandler) {
    try {
      btn.removeEventListener("click", btn.__awcDrawerClickHandler);
    } catch (_err) {
      // No-op.
    }
  }
  const drawerClickHandler = () => {
    const isOpen = body.classList.contains(drawerClass);
    setOpen(!isOpen);
  };
  btn.addEventListener("click", drawerClickHandler);
  btn.__awcDrawerClickHandler = drawerClickHandler;

  const saved = safeGet();
  // Default open on every fresh page entry; keep state only within current tab session.
  setOpen(saved === null ? true : saved !== "0");

  const syncedProps = [
    "background",
    "background-color",
    "background-image",
    "color",
    "border",
    "border-color",
    "border-style",
    "border-width",
    "box-shadow",
    "font-weight",
    "opacity",
  ];
  const normalizeLabel = (text) => String(text || "").replace(/\\s+/g, " ").trim();
  const findButtonByRegex = (pattern) => {
    const buttons = Array.from(doc.querySelectorAll("button"));
    for (const button of buttons) {
      if (pattern.test(normalizeLabel(button.textContent))) {
        return button;
      }
    }
    return null;
  };
  const resolveCompileButton = () => (
    doc.querySelector(".st-key-compile_latex_btn_v2 button")
    || findButtonByRegex(/^(Compile|Recompile)$/i)
  );
  const resolveExportButton = () => (
    doc.querySelector(".st-key-export_pdf_btn_v2 button")
    || findButtonByRegex(/^Export PDF$/i)
  );
  const applyDarkExportFallback = (btn) => {
    if (!btn) return;
    btn.style.setProperty("background", "#111827", "important");
    btn.style.setProperty("background-color", "#111827", "important");
    btn.style.setProperty("color", "#ffffff", "important");
    btn.style.setProperty("border", "1px solid #111827", "important");
    btn.style.setProperty("border-color", "#111827", "important");
    btn.style.setProperty("opacity", btn.disabled ? "0.55" : "1", "important");
  };
  const syncExportButtonStyle = () => {
    const compileBtn = resolveCompileButton();
    const exportBtn = resolveExportButton();
    if (!exportBtn) return;
    if (compileBtn && compileBtn.matches(":hover")) return;
    if (!compileBtn) {
      applyDarkExportFallback(exportBtn);
      return;
    }
    const source = window.getComputedStyle(compileBtn);
    for (const prop of syncedProps) {
      const value = source.getPropertyValue(prop);
      if (!value) continue;
      exportBtn.style.setProperty(prop, value, "important");
    }
    exportBtn.style.setProperty("opacity", exportBtn.disabled ? "0.55" : "1", "important");
  };
  syncExportButtonStyle();
  if (body.dataset.awcExportStyleSyncBound !== "1") {
    body.dataset.awcExportStyleSyncBound = "1";
    const scheduleSync = () => {
      window.requestAnimationFrame(syncExportButtonStyle);
    };
    const observer = new MutationObserver(scheduleSync);
    observer.observe(body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["class", "style", "disabled", "aria-disabled"],
    });
  }
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _inject_chat_drawer_layout_sync() -> None:
    st_components.html(
        """
<script>
(() => {
  const doc = window.parent && window.parent.document ? window.parent.document : null;
  if (!doc) return;
  const hostWin = window.parent || window;
  const drawerSelector = 'div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker))';
  const composerSelector = 'iframe[title^="chat_composer_v"]';

  const getDrawer = () => doc.querySelector(drawerSelector);
  const getComposerContainer = (drawer) => {
    if (!drawer) return null;
    const iframe = drawer.querySelector(composerSelector);
    if (!iframe) return null;
    return iframe.closest('[data-testid="stElementContainer"]') || iframe.parentElement;
  };
  const getChatShell = (drawer) => {
    if (!drawer) return null;
    const marker = drawer.querySelector(".awc-chat-history-marker");
    if (!marker) return null;
    return (
      marker.closest('[data-testid="stVerticalBlockBorderWrapper"]')
      || marker.closest('[data-testid="stVerticalBlock"]')
      || marker.parentElement
    );
  };

  let rafId = 0;
  const syncLayout = () => {
    const drawer = getDrawer();
    const composerContainer = getComposerContainer(drawer);
    const chatShell = getChatShell(drawer);
    if (!drawer || !composerContainer || !chatShell) return;

    const chatRect = chatShell.getBoundingClientRect();
    const composerRect = composerContainer.getBoundingClientRect();
    if (chatRect.width <= 0 || composerRect.width <= 0) return;

    const gapPx = 8;
    const nextHeight = Math.max(220, Math.floor(composerRect.top - chatRect.top - gapPx));
    const nextHeightPx = `${nextHeight}px`;
    chatShell.style.setProperty("height", nextHeightPx, "important");
    chatShell.style.setProperty("max-height", nextHeightPx, "important");
    chatShell.style.setProperty("min-height", "0px", "important");
    chatShell.style.setProperty("overflow", "hidden", "important");
  };

  const scheduleSync = () => {
    if (rafId) hostWin.cancelAnimationFrame(rafId);
    rafId = hostWin.requestAnimationFrame(() => {
      rafId = 0;
      syncLayout();
    });
  };

  scheduleSync();
  hostWin.addEventListener("resize", scheduleSync, { passive: true });

  try {
    if (!hostWin.__awcChatDrawerLayoutObserver) {
      const observer = new MutationObserver(() => {
        scheduleSync();
      });
      observer.observe(doc.body, { childList: true, subtree: true, attributes: true });
      hostWin.__awcChatDrawerLayoutObserver = observer;
    }
  } catch (_err) {
    // No-op.
  }
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _inject_chat_drawer_history_arrow() -> None:
    st_components.html(
        """
<script>
(() => {
  const doc = window.parent && window.parent.document ? window.parent.document : null;
  if (!doc) return;
  const hostWin = window.parent || window;
  const drawerSelector = 'div[data-testid="stVerticalBlock"]:has(.awc-chat-drawer-marker):not(:has(.awc-main-workspace-marker))';
  const normalize = (text) => String(text || "").replace(/\\s+/g, " ").trim();
  const hasHistoryPopoverBody = () => {
    const popoverBodies = Array.from(doc.querySelectorAll('[data-testid="stPopoverBody"]'));
    return popoverBodies.some((node) => {
      if (!node) return false;
      if (node.querySelector(".awc-agent-history-popover-marker")) return true;
      if (node.querySelector('[class*="st-key-agent_chat_switch_"]')) return true;
      if (node.querySelector('[class*="st-key-agent_chat_rename_confirm_"]')) return true;
      const text = normalize(node.textContent);
      return text.includes("No chats yet.");
    });
  };

  const isExpanded = (button) => {
    if (!button) return false;
    return hasHistoryPopoverBody();
  };

  const bindHistoryClick = (button) => {
    if (!button || button.dataset.awcHistoryArrowBound === "1") return;
    button.dataset.awcHistoryArrowBound = "1";
    button.addEventListener("click", () => {
      const wasExpanded = isExpanded(button);
      const nextExpanded = !wasExpanded;
      button.dataset.awcHistoryOpen = nextExpanded ? "1" : "0";
      const instantArrow = button.querySelector(".awc-history-arrow");
      if (instantArrow) instantArrow.textContent = nextExpanded ? "▴" : "▾";

      hostWin.requestAnimationFrame(() => {
        const expandedNow = isExpanded(button);
        button.dataset.awcHistoryOpen = expandedNow ? "1" : "0";
        const arrow = button.querySelector(".awc-history-arrow");
        if (arrow) arrow.textContent = expandedNow ? "▴" : "▾";
      });
      hostWin.setTimeout(() => {
        const expandedNow = isExpanded(button);
        button.dataset.awcHistoryOpen = expandedNow ? "1" : "0";
        const arrow = button.querySelector(".awc-history-arrow");
        if (arrow) arrow.textContent = expandedNow ? "▴" : "▾";
      }, 120);
      hostWin.setTimeout(() => {
        const expandedNow = isExpanded(button);
        button.dataset.awcHistoryOpen = expandedNow ? "1" : "0";
        const arrow = button.querySelector(".awc-history-arrow");
        if (arrow) arrow.textContent = expandedNow ? "▴" : "▾";
      }, 320);
      hostWin.setTimeout(() => {
        try { button.blur(); } catch (_) {}
      }, 0);
    });
  };

  const clearRingStyles = (node) => {
    if (!node || !node.style) return;
    node.style.setProperty("outline", "none", "important");
    node.style.setProperty("box-shadow", "none", "important");
    node.style.setProperty("filter", "none", "important");
  };

  const enforceHistoryTriggerNoRing = (button) => {
    if (!button) return;
    clearRingStyles(button);
    button.style.setProperty("border-width", "1px", "important");
    button.style.setProperty("border-style", "solid", "important");
    button.style.setProperty("border-color", "#e5e7eb", "important");
    button.style.setProperty("-webkit-tap-highlight-color", "transparent", "important");

    const p1 = button.parentElement;
    const p2 = p1 ? p1.parentElement : null;
    clearRingStyles(p1);
    clearRingStyles(p2);

    const popoverHost = button.closest('[data-testid="stPopover"]');
    if (popoverHost) {
      clearRingStyles(popoverHost);
      const popoverButton = popoverHost.querySelector('[data-testid="stPopoverButton"]');
      clearRingStyles(popoverButton);
      if (popoverButton && popoverButton !== button) {
        popoverButton.style.setProperty("border-width", "1px", "important");
        popoverButton.style.setProperty("border-style", "solid", "important");
        popoverButton.style.setProperty("border-color", "#e5e7eb", "important");
      }
      const focused = doc.activeElement;
      if (focused && popoverHost.contains(focused) && typeof focused.blur === "function") {
        try { focused.blur(); } catch (_) {}
      }
    }
  };

  const bindHistoryFocusGuards = (button) => {
    if (!button || button.dataset.awcHistoryFocusBound === "1") return;
    button.dataset.awcHistoryFocusBound = "1";
    const sync = () => enforceHistoryTriggerNoRing(button);
    const preventFocusOnPointer = (evt) => {
      try { evt.preventDefault(); } catch (_) {}
      sync();
    };
    button.addEventListener("mousedown", preventFocusOnPointer, true);
    button.addEventListener("pointerdown", preventFocusOnPointer, true);
    button.addEventListener("focus", sync);
    button.addEventListener("blur", sync);
    button.addEventListener("mouseup", sync);
    button.addEventListener("pointerup", sync);
    button.addEventListener("mouseenter", sync);
    button.addEventListener("mouseleave", sync);
  };

  const syncArrow = () => {
    const drawer = doc.querySelector(drawerSelector);
    if (!drawer) return;
    const buttons = Array.from(drawer.querySelectorAll('button'))
      .filter((button) => !button.closest('[data-testid="stPopoverBody"]'))
      .filter((button) => normalize(button.textContent).startsWith("History"));
    for (const button of buttons) {
      button.dataset.awcHistoryTrigger = "1";
      button.classList.add("awc-history-trigger-btn");
      const popoverHost = button.closest('[data-testid="stPopover"]');
      if (popoverHost) popoverHost.classList.add("awc-history-trigger-popover");
      bindHistoryFocusGuards(button);
      enforceHistoryTriggerNoRing(button);
      button.style.setProperty("position", "relative", "important");
      button.style.setProperty("overflow", "visible", "important");
      button.style.setProperty("box-sizing", "border-box", "important");
      bindHistoryClick(button);

      let arrow = button.querySelector(".awc-history-arrow");
      if (!arrow) {
        arrow = doc.createElement("span");
        arrow.className = "awc-history-arrow";
        arrow.setAttribute("aria-hidden", "true");
        button.appendChild(arrow);
      }
      arrow.style.position = "absolute";
      arrow.style.right = "0.80rem";
      arrow.style.top = "50%";
      arrow.style.transform = "translateY(-50%)";
      arrow.style.fontSize = "1.10rem";
      arrow.style.lineHeight = "1";
      arrow.style.color = "#6b7280";
      arrow.style.pointerEvents = "none";
      arrow.style.userSelect = "none";
      arrow.style.display = "inline-block";
      arrow.style.zIndex = "1";
      const expanded = isExpanded(button);
      button.dataset.awcHistoryOpen = expanded ? "1" : "0";
      arrow.textContent = expanded ? "▴" : "▾";
    }
  };

  const bindGlobalHistoryRingGuard = () => {
    if (hostWin.__awcHistoryRingGuardBound === "1") return;
    hostWin.__awcHistoryRingGuardBound = "1";
    doc.addEventListener("focusin", () => {
      const drawer = doc.querySelector(drawerSelector);
      if (!drawer) return;
      const buttons = Array.from(drawer.querySelectorAll("button.awc-history-trigger-btn"));
      for (const button of buttons) enforceHistoryTriggerNoRing(button);
    }, true);
  };

  let rafId = 0;
  const schedule = () => {
    if (rafId) hostWin.cancelAnimationFrame(rafId);
    rafId = hostWin.requestAnimationFrame(() => {
      rafId = 0;
      syncArrow();
    });
  };

  schedule();
  bindGlobalHistoryRingGuard();
  if (!hostWin.__awcChatDrawerHistoryArrowObserver) {
    const observer = new MutationObserver(schedule);
    observer.observe(doc.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "style", "aria-expanded"],
    });
    hostWin.__awcChatDrawerHistoryArrowObserver = observer;
  }
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _inject_chat_copy_buttons() -> None:
    st_components.html(
        """
<script>
(() => {
  const doc = window.parent && window.parent.document ? window.parent.document : null;
  if (!doc) return;
  const hostWin = window.parent || window;
  if (hostWin.__awcChatCopyBound === "1") return;
  hostWin.__awcChatCopyBound = "1";

  const decodePayload = (payload) => {
    if (!payload) return "";
    try {
      const binary = hostWin.atob(payload);
      const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
      if (typeof TextDecoder !== "undefined") {
        return new TextDecoder("utf-8").decode(bytes);
      }
      let text = "";
      for (const b of bytes) text += String.fromCharCode(b);
      try {
        return decodeURIComponent(escape(text));
      } catch (_err) {
        return text;
      }
    } catch (_err) {
      return "";
    }
  };

  const fallbackCopy = (text) => {
    try {
      const textarea = doc.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "readonly");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      textarea.style.pointerEvents = "none";
      textarea.style.left = "-9999px";
      doc.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      const ok = doc.execCommand("copy");
      textarea.remove();
      return ok;
    } catch (_err) {
      return false;
    }
  };

  const copyText = async (text) => {
    const clipboard = hostWin.navigator && hostWin.navigator.clipboard;
    if (clipboard && typeof clipboard.writeText === "function") {
      try {
        await clipboard.writeText(text);
        return true;
      } catch (_err) {
        return fallbackCopy(text);
      }
    }
    return fallbackCopy(text);
  };

  const setTooltip = (button, text) => {
    button.setAttribute("data-awc-tooltip", text);
  };

  const ensureButtonForMarker = (marker) => {
    const message = marker.closest('[data-testid="stChatMessage"]');
    if (!message) return;
    message.classList.add("awc-chat-copy-host");

    let button = message.querySelector(".awc-chat-copy-btn");
    if (!button) {
      button = doc.createElement("button");
      button.type = "button";
      button.className = "awc-chat-copy-btn";
      button.innerHTML = `
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <rect x="9" y="9" width="11" height="11" rx="2"></rect>
          <path d="M6 15H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v1"></path>
        </svg>
      `;
      message.appendChild(button);
    }

    const label = marker.getAttribute("data-awc-copy-label") || "copy message";
    const payload = marker.getAttribute("data-awc-copy-b64") || "";
    button.setAttribute("data-awc-copy-b64", payload);
    button.setAttribute("data-awc-copy-label", label);
    button.setAttribute("aria-label", label);
    setTooltip(button, label);
  };

  const ensureButtons = () => {
    const markers = doc.querySelectorAll(".awc-chat-copy-marker");
    for (const marker of markers) {
      ensureButtonForMarker(marker);
    }
  };

  doc.addEventListener("click", async (event) => {
    const target = event.target;
    if (!target || target.nodeType !== 1) return;
    const button = target.closest && target.closest(".awc-chat-copy-btn");
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();

    const text = decodePayload(button.getAttribute("data-awc-copy-b64") || "");
    if (!text) return;
    const originalTooltip = button.getAttribute("data-awc-copy-label") || "copy message";
    const ok = await copyText(text);
    if (!ok) return;

    setTooltip(button, "copied");
    if (button.__awcCopyTooltipTimer) {
      hostWin.clearTimeout(button.__awcCopyTooltipTimer);
    }
    button.__awcCopyTooltipTimer = hostWin.setTimeout(() => {
      setTooltip(button, originalTooltip);
      button.__awcCopyTooltipTimer = null;
    }, 1200);
  }, true);

  ensureButtons();
  try {
    const observer = new MutationObserver(() => {
      hostWin.requestAnimationFrame(ensureButtons);
    });
    observer.observe(doc.body, { childList: true, subtree: true });
    hostWin.__awcChatCopyObserver = observer;
  } catch (_err) {
    // No-op.
  }
})();
</script>
""",
        height=0,
        scrolling=False,
    )


def _render_pdf_frame(
    content_html: str,
    height: int,
    scrollable: bool,
    overlay_html: str = "",
    script_html: str = "",
    hint_text: str = "Double-click PDF text to locate source",
) -> None:
    safe_height = max(280, int(height))
    body_overflow = "auto" if scrollable else "hidden"
    st_components.html(
        f"""
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}
  *, *::before, *::after {{
    box-sizing: border-box;
  }}
</style>
<div style="height:{safe_height}px; border:1px solid #d0d7de; border-radius:10px; overflow:hidden; background:#ffffff; margin:0; padding:0; display:flex; flex-direction:column;">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px; border-bottom:1px solid #d0d7de; background:linear-gradient(120deg, #f8fbff 0%, #f7f7f7 100%); min-height:45px;">
    <div style="margin:0; font-size:14px; font-weight:600; color:#24292f;">PDF Preview</div>
    <div style="display:flex; align-items:center; gap:8px; margin-left:auto; min-width:0;">
      <div style="font-size:12px; color:#57606a; max-width:280px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:right;">{html.escape(hint_text)}</div>
      {overlay_html}
    </div>
  </div>
  <div id="awc-pdf-body" style="flex:1; min-height:0; overflow:{body_overflow}; background:#ffffff; margin:0; padding:0;">
    {content_html}
  </div>
</div>
{script_html}
""",
        height=safe_height + 8,
        scrolling=False,
    )


def _render_png_preview_frame(preview_png_base64: str, height: int) -> None:
    if not preview_png_base64:
        return
    overlay_html = """
<div style="display:inline-flex; align-items:center; gap:4px;">
  <button id="awc-zoom-out-btn" type="button" style="width:26px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:14px; font-weight:700; cursor:pointer;">-</button>
  <button id="awc-zoom-reset-btn" type="button" style="min-width:58px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:12px; font-weight:600; cursor:pointer;">100%</button>
  <button id="awc-zoom-in-btn" type="button" style="width:26px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:14px; font-weight:700; cursor:pointer;">+</button>
</div>
"""
    script_html = """
<script>
(() => {
  const getTarget = () => document.getElementById("awc-preview-img");
  const body = document.getElementById("awc-pdf-body");
  const zoomOutBtn = document.getElementById("awc-zoom-out-btn");
  const zoomInBtn = document.getElementById("awc-zoom-in-btn");
  const zoomResetBtn = document.getElementById("awc-zoom-reset-btn");
  if (!zoomOutBtn || !zoomInBtn || !zoomResetBtn) return;
  let zoom = 1;
  const ZOOM_MIN = 0.6;
  const ZOOM_MAX = 3.0;
  const ZOOM_STEP = 0.15;
  const PREVIEW_DPI = __PREVIEW_DPI__;
  let dblclickSeq = 0;
  const dispatchPdfDblclick = (payload) => {
    let widgetBridgeQueued = false;
    try {
      const parentDoc = window.parent && window.parent.document ? window.parent.document : null;
      if (parentDoc) {
        const latest = (selector) => {
          const nodes = Array.from(parentDoc.querySelectorAll(selector));
          for (let index = nodes.length - 1; index >= 0; index -= 1) {
            const node = nodes[index];
            if (node && node.isConnected) return node;
          }
          return null;
        };
        const actionInput = latest('[class*="st-key-files_dnd_bridge_action"] input');
        const payloadInput = latest('[class*="st-key-files_dnd_bridge_payload"] textarea');
        const nonceInput = latest('[class*="st-key-files_dnd_bridge_nonce"] input');
        if (actionInput && payloadInput && nonceInput) {
          const setValue = (element, value) => {
            const elementWindow = element.ownerDocument && element.ownerDocument.defaultView
              ? element.ownerDocument.defaultView
              : window;
            const proto = element.tagName === "TEXTAREA"
              ? elementWindow.HTMLTextAreaElement.prototype
              : elementWindow.HTMLInputElement.prototype;
            const descriptor = Object.getOwnPropertyDescriptor(proto, "value");
            if (descriptor && typeof descriptor.set === "function") {
              descriptor.set.call(element, value);
            } else {
              element.value = value;
            }
            element.dispatchEvent(new elementWindow.Event("input", { bubbles: true }));
            element.dispatchEvent(new elementWindow.Event("change", { bubbles: true }));
          };
          const nonceValue = String(Number(payload.event_id || 0) || Date.now());
          setValue(actionInput, "pdf_dblclick");
          setValue(payloadInput, JSON.stringify(payload || {}));
          const commitNonce = () => setValue(nonceInput, nonceValue);
          if (typeof window.requestAnimationFrame === "function") {
            window.requestAnimationFrame(commitNonce);
          } else {
            window.setTimeout(commitNonce, 0);
          }
          widgetBridgeQueued = true;
        }
      }
    } catch (_err) {
      // Fall through to the component bridge.
    }
    try {
      const parentDoc = window.parent && window.parent.document ? window.parent.document : null;
      const bridgeFrame = parentDoc
        ? parentDoc.querySelector('[class*="st-key-files_dnd_bridge_component"] iframe')
        : null;
      if (bridgeFrame && bridgeFrame.contentWindow) {
        bridgeFrame.contentWindow.postMessage(
          {
            isStreamlitMessage: true,
            type: "awc-files-dnd-component-event",
            payload: {
              action: "pdf_dblclick",
              payload,
              event_id: Number(payload.event_id || 0) || 0,
            },
          },
          "*"
        );
        return;
      }
    } catch (_err) {
      // Fall back to the parent relay below.
    }
    if (widgetBridgeQueued) {
      return;
    }
    try {
      (window.parent || window).postMessage({ type: "awc-pdf-dblclick", payload }, "*");
    } catch (_err) {
      // no-op
    }
  };
  const clamp = (v) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v));
  const updateButtons = () => {
    const pct = Math.round(zoom * 100);
    zoomResetBtn.textContent = pct + "%";
    zoomOutBtn.disabled = zoom <= ZOOM_MIN + 0.001;
    zoomInBtn.disabled = zoom >= ZOOM_MAX - 0.001;
    zoomOutBtn.style.opacity = zoomOutBtn.disabled ? "0.45" : "1";
    zoomInBtn.style.opacity = zoomInBtn.disabled ? "0.45" : "1";
  };
  const applyToTarget = () => {
    const target = getTarget();
    if (!target) return false;
    const pct = Math.round(zoom * 100);
    target.style.width = pct + "%";
    target.style.maxWidth = "none";
    target.style.height = "auto";
    return true;
  };
  const bindDblclick = () => {
    const target = getTarget();
    if (!target || target.dataset.awcPdfDblclickBound === "1") return;
    target.dataset.awcPdfDblclickBound = "1";
    target.addEventListener("dblclick", (event) => {
      const rect = target.getBoundingClientRect();
      const displayWidth = rect.width || 1;
      const displayHeight = rect.height || 1;
      const naturalWidth = target.naturalWidth || displayWidth;
      const naturalHeight = target.naturalHeight || displayHeight;
      const px = Math.max(0, Math.min(event.clientX - rect.left, displayWidth));
      const py = Math.max(0, Math.min(event.clientY - rect.top, displayHeight));
      const sourceX = (px / displayWidth) * naturalWidth;
      const sourceY = (py / displayHeight) * naturalHeight;
      const synctexX = sourceX * (72 / PREVIEW_DPI);
      const synctexY = sourceY * (72 / PREVIEW_DPI);
      dblclickSeq += 1;
      const payload = {
        action: "pdf_dblclick",
        page: 1,
        x: Number(synctexX || 0),
        y: Number(synctexY || 0),
        event_id: Date.now() * 100 + dblclickSeq,
      };
      dispatchPdfDblclick(payload);
    });
  };
  const apply = () => {
    updateButtons();
    applyToTarget();
    bindDblclick();
  };
  const setZoom = (nextZoom) => {
    zoom = clamp(nextZoom);
    apply();
  };
  zoomOutBtn.addEventListener("click", () => {
    setZoom(zoom - ZOOM_STEP);
  });
  zoomInBtn.addEventListener("click", () => {
    setZoom(zoom + ZOOM_STEP);
  });
  zoomResetBtn.addEventListener("click", () => {
    setZoom(1);
  });

  if (body) {
    body.addEventListener(
      "wheel",
      (event) => {
        if (!event.ctrlKey) return;
        event.preventDefault();
        const factor = Math.exp(-event.deltaY * 0.0015);
        setZoom(zoom * factor);
      },
      { passive: false }
    );
  }

  let retries = 0;
  const boot = () => {
    updateButtons();
    if (applyToTarget()) {
      bindDblclick();
      return;
    }
    retries += 1;
    if (retries < 40) window.requestAnimationFrame(boot);
  };
  boot();
})();
</script>
"""
    script_html = script_html.replace("__PREVIEW_DPI__", str(int(PDF_PREVIEW_DPI)))
    _render_pdf_frame(
        (
            f"<img id='awc-preview-img' alt='PDF Preview' src='data:image/png;base64,{preview_png_base64}' "
            "style='display:block; width:100%; height:auto; background:#ffffff; margin:0; padding:0;' />"
        ),
        height=height,
        scrollable=True,
        overlay_html=overlay_html,
        script_html=script_html,
    )


def _render_png_preview_pages_frame(preview_pages_base64: List[str], height: int) -> None:
    if not preview_pages_base64:
        return
    overlay_html = """
<div style="display:inline-flex; align-items:center; gap:4px;">
  <button id="awc-zoom-out-btn" type="button" style="width:26px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:14px; font-weight:700; cursor:pointer;">-</button>
  <button id="awc-zoom-reset-btn" type="button" style="min-width:58px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:12px; font-weight:600; cursor:pointer;">100%</button>
  <button id="awc-zoom-in-btn" type="button" style="width:26px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:14px; font-weight:700; cursor:pointer;">+</button>
</div>
"""
    pages_json = json.dumps(list(preview_pages_base64), ensure_ascii=False).replace("</", "<\\/")
    script_html = """
<script>
(() => {
  const pages = __PAGES_JSON__;
  const pagesRoot = document.getElementById("awc-preview-pages-wrap");
  const body = document.getElementById("awc-pdf-body");
  const zoomOutBtn = document.getElementById("awc-zoom-out-btn");
  const zoomInBtn = document.getElementById("awc-zoom-in-btn");
  const zoomResetBtn = document.getElementById("awc-zoom-reset-btn");
  if (!pagesRoot || !Array.isArray(pages) || pages.length === 0 || !zoomOutBtn || !zoomInBtn || !zoomResetBtn) return;
  let zoom = 1;
  const ZOOM_MIN = 0.6;
  const ZOOM_MAX = 3.0;
  const ZOOM_STEP = 0.15;
  const PREVIEW_DPI = __PREVIEW_DPI__;
  let dblclickSeq = 0;
  const dispatchPdfDblclick = (payload) => {
    let widgetBridgeQueued = false;
    try {
      const parentDoc = window.parent && window.parent.document ? window.parent.document : null;
      if (parentDoc) {
        const latest = (selector) => {
          const nodes = Array.from(parentDoc.querySelectorAll(selector));
          for (let index = nodes.length - 1; index >= 0; index -= 1) {
            const node = nodes[index];
            if (node && node.isConnected) return node;
          }
          return null;
        };
        const actionInput = latest('[class*="st-key-files_dnd_bridge_action"] input');
        const payloadInput = latest('[class*="st-key-files_dnd_bridge_payload"] textarea');
        const nonceInput = latest('[class*="st-key-files_dnd_bridge_nonce"] input');
        if (actionInput && payloadInput && nonceInput) {
          const setValue = (element, value) => {
            const elementWindow = element.ownerDocument && element.ownerDocument.defaultView
              ? element.ownerDocument.defaultView
              : window;
            const proto = element.tagName === "TEXTAREA"
              ? elementWindow.HTMLTextAreaElement.prototype
              : elementWindow.HTMLInputElement.prototype;
            const descriptor = Object.getOwnPropertyDescriptor(proto, "value");
            if (descriptor && typeof descriptor.set === "function") {
              descriptor.set.call(element, value);
            } else {
              element.value = value;
            }
            element.dispatchEvent(new elementWindow.Event("input", { bubbles: true }));
            element.dispatchEvent(new elementWindow.Event("change", { bubbles: true }));
          };
          const nonceValue = String(Number(payload.event_id || 0) || Date.now());
          setValue(actionInput, "pdf_dblclick");
          setValue(payloadInput, JSON.stringify(payload || {}));
          const commitNonce = () => setValue(nonceInput, nonceValue);
          if (typeof window.requestAnimationFrame === "function") {
            window.requestAnimationFrame(commitNonce);
          } else {
            window.setTimeout(commitNonce, 0);
          }
          widgetBridgeQueued = true;
        }
      }
    } catch (_err) {
      // Fall through to the component bridge.
    }
    try {
      const parentDoc = window.parent && window.parent.document ? window.parent.document : null;
      const bridgeFrame = parentDoc
        ? parentDoc.querySelector('[class*="st-key-files_dnd_bridge_component"] iframe')
        : null;
      if (bridgeFrame && bridgeFrame.contentWindow) {
        bridgeFrame.contentWindow.postMessage(
          {
            isStreamlitMessage: true,
            type: "awc-files-dnd-component-event",
            payload: {
              action: "pdf_dblclick",
              payload,
              event_id: Number(payload.event_id || 0) || 0,
            },
          },
          "*"
        );
        return;
      }
    } catch (_err) {
      // Fall back to the parent relay below.
    }
    if (widgetBridgeQueued) {
      return;
    }
    try {
      (window.parent || window).postMessage({ type: "awc-pdf-dblclick", payload }, "*");
    } catch (_err) {
      // no-op
    }
  };

  const clamp = (v) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v));
  const updateButtons = () => {
    const pct = Math.round(zoom * 100);
    zoomResetBtn.textContent = pct + "%";
    zoomOutBtn.disabled = zoom <= ZOOM_MIN + 0.001;
    zoomInBtn.disabled = zoom >= ZOOM_MAX - 0.001;
    zoomOutBtn.style.opacity = zoomOutBtn.disabled ? "0.45" : "1";
    zoomInBtn.style.opacity = zoomInBtn.disabled ? "0.45" : "1";
  };
  const buildPagesOnce = () => {
    if (pagesRoot.dataset.awcRendered === "1") return;
    pagesRoot.dataset.awcRendered = "1";
    pagesRoot.innerHTML = "";
    pages.forEach((encoded, index) => {
      const pageNumber = index + 1;
      const wrap = document.createElement("div");
      wrap.style.margin = "0 0 12px 0";
      wrap.style.width = "100%";
      wrap.style.display = "block";

      const img = document.createElement("img");
      img.className = "awc-preview-page-img";
      img.alt = "PDF Page " + pageNumber;
      img.dataset.page = String(pageNumber);
      img.src = "data:image/png;base64," + String(encoded || "");
      img.style.display = "block";
      img.style.width = "100%";
      img.style.maxWidth = "none";
      img.style.height = "auto";
      img.style.background = "#ffffff";
      img.style.margin = "0";
      img.style.padding = "0";
      img.addEventListener("dblclick", (event) => {
        const rect = img.getBoundingClientRect();
        const displayWidth = rect.width || 1;
        const displayHeight = rect.height || 1;
        const naturalWidth = img.naturalWidth || displayWidth;
        const naturalHeight = img.naturalHeight || displayHeight;
        const px = Math.max(0, Math.min(event.clientX - rect.left, displayWidth));
        const py = Math.max(0, Math.min(event.clientY - rect.top, displayHeight));
        const sourceX = (px / displayWidth) * naturalWidth;
        const sourceY = (py / displayHeight) * naturalHeight;
        const synctexX = sourceX * (72 / PREVIEW_DPI);
        const synctexY = sourceY * (72 / PREVIEW_DPI);
        dblclickSeq += 1;
        dispatchPdfDblclick({
          action: "pdf_dblclick",
          page: pageNumber,
          x: Number(synctexX || 0),
          y: Number(synctexY || 0),
          event_id: Date.now() * 100 + dblclickSeq,
        });
      });
      wrap.appendChild(img);
      pagesRoot.appendChild(wrap);
    });
  };
  const applyZoom = () => {
    const pct = Math.round(zoom * 100);
    const images = pagesRoot.querySelectorAll(".awc-preview-page-img");
    images.forEach((img) => {
      img.style.width = pct + "%";
      img.style.maxWidth = "none";
      img.style.height = "auto";
    });
  };
  const apply = () => {
    buildPagesOnce();
    applyZoom();
    updateButtons();
  };
  const setZoom = (nextZoom) => {
    zoom = clamp(nextZoom);
    apply();
  };
  zoomOutBtn.addEventListener("click", () => setZoom(zoom - ZOOM_STEP));
  zoomInBtn.addEventListener("click", () => setZoom(zoom + ZOOM_STEP));
  zoomResetBtn.addEventListener("click", () => setZoom(1));
  if (body) {
    body.addEventListener(
      "wheel",
      (event) => {
        if (!event.ctrlKey) return;
        event.preventDefault();
        const factor = Math.exp(-event.deltaY * 0.0015);
        setZoom(zoom * factor);
      },
      { passive: false }
    );
  }
  apply();
})();
</script>
"""
    script_html = script_html.replace("__PREVIEW_DPI__", str(int(PDF_PREVIEW_DPI)))
    script_html = script_html.replace("__PAGES_JSON__", pages_json)
    _render_pdf_frame(
        "<div id='awc-preview-pages-wrap' style='padding:0 0 8px 0; margin:0;'></div>",
        height=height,
        scrollable=True,
        overlay_html=overlay_html,
        script_html=script_html,
    )


def _render_pdf_embed_frame(pdf_base64: str, height: int) -> None:
    if not pdf_base64:
        return
    overlay_html = """
<div style="display:inline-flex; align-items:center; gap:4px;">
  <button id="awc-zoom-out-btn" type="button" style="width:26px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:14px; font-weight:700; cursor:pointer;">-</button>
  <button id="awc-zoom-reset-btn" type="button" style="min-width:58px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:12px; font-weight:600; cursor:pointer;">100%</button>
  <button id="awc-zoom-in-btn" type="button" style="width:26px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:14px; font-weight:700; cursor:pointer;">+</button>
</div>
"""
    script_html = """
<script>
(() => {
  const target = document.getElementById("awc-preview-embed");
  const body = document.getElementById("awc-pdf-body");
  const zoomOutBtn = document.getElementById("awc-zoom-out-btn");
  const zoomInBtn = document.getElementById("awc-zoom-in-btn");
  const zoomResetBtn = document.getElementById("awc-zoom-reset-btn");
  if (!target || !zoomOutBtn || !zoomInBtn || !zoomResetBtn) return;
  let zoom = 1;
  const ZOOM_MIN = 0.6;
  const ZOOM_MAX = 3.0;
  const ZOOM_STEP = 0.15;
  const clamp = (v) => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, v));
  const updateButtons = () => {
    const pct = Math.round(zoom * 100);
    zoomResetBtn.textContent = pct + "%";
    zoomOutBtn.disabled = zoom <= ZOOM_MIN + 0.001;
    zoomInBtn.disabled = zoom >= ZOOM_MAX - 0.001;
    zoomOutBtn.style.opacity = zoomOutBtn.disabled ? "0.45" : "1";
    zoomInBtn.style.opacity = zoomInBtn.disabled ? "0.45" : "1";
  };
  const apply = () => {
    const pct = Math.round(zoom * 100);
    const baseSrc = target.getAttribute("data-base-src") || "";
    if (baseSrc) {
      target.setAttribute("src", baseSrc + "#zoom=" + pct);
    }
    updateButtons();
  };
  const setZoom = (nextZoom) => {
    zoom = clamp(nextZoom);
    apply();
  };
  zoomOutBtn.addEventListener("click", () => {
    setZoom(zoom - ZOOM_STEP);
  });
  zoomInBtn.addEventListener("click", () => {
    setZoom(zoom + ZOOM_STEP);
  });
  zoomResetBtn.addEventListener("click", () => {
    setZoom(1);
  });
  if (body) {
    body.addEventListener(
      "wheel",
      (event) => {
        if (!event.ctrlKey) return;
        event.preventDefault();
        const factor = Math.exp(-event.deltaY * 0.0015);
        setZoom(zoom * factor);
      },
      { passive: false }
    );
  }
  apply();
})();
</script>
"""
    _render_pdf_frame(
        (
            f"<embed id='awc-preview-embed' data-base-src='data:application/pdf;base64,{pdf_base64}' src='data:application/pdf;base64,{pdf_base64}#zoom=100' type='application/pdf' "
            "style='display:block; width:100%; height:100%; border:0; margin:0; padding:0;' />"
        ),
        height=height,
        scrollable=False,
        overlay_html=overlay_html,
        script_html=script_html,
    )


def _render_pdf_empty_frame(height: int) -> None:
    _render_pdf_frame(
        (
            "<div style='height:100%; display:flex; align-items:center; justify-content:center; "
            "background:#ffffff; color:#6b7280; font-size:13px;'>"
            "Click Compile to generate PDF preview"
            "</div>"
        ),
        height=height,
        scrollable=False,
    )


def _render_pdf_error_frame(
    height: int,
    error_title: str,
    error_items: List[Dict[str, Any]],
    raw_log: str,
) -> None:
    overlay_html = """
<div style="display:inline-flex; align-items:center; gap:4px;">
  <button type="button" disabled style="width:26px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:14px; font-weight:700; opacity:0.45; cursor:default;">-</button>
  <button type="button" disabled style="min-width:58px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:12px; font-weight:600; opacity:0.45; cursor:default;">100%</button>
  <button type="button" disabled style="width:26px; height:24px; border:1px solid #cbd5e1; border-radius:6px; background:#ffffff; color:#334155; font-size:14px; font-weight:700; opacity:0.45; cursor:default;">+</button>
</div>
"""
    sorted_items = sorted(list(error_items or []), key=_compile_error_sort_key)
    cards: List[str] = []
    for item in sorted_items[:10]:
        raw_file = str(item.get("file", "")).strip()
        display_file = ""
        if raw_file:
            try:
                display_file = _normalize_project_relative_path(raw_file)
            except ValueError:
                display_file = raw_file
        try:
            line_no = int(item.get("line", -1))
        except (TypeError, ValueError):
            line_no = -1
        message = html.escape(str(item.get("message", "LaTeX compile error")).strip())
        location_bits: List[str] = []
        if line_no > 0:
            location_bits.append(f"Line {line_no}")
        if display_file:
            location_bits.append(display_file)
        location_html = (
            f"<div style='font-size:12px; font-weight:700; color:#7f1d1d; margin-bottom:3px;'>{html.escape(' | '.join(location_bits))}</div>"
            if location_bits
            else ""
        )
        cards.append(
            "<div style='border:1px solid #fecaca; background:#fff1f2; border-radius:10px; padding:10px 12px; margin-bottom:8px;'>"
            f"{location_html}"
            f"<div style='font-size:13px; color:#7f1d1d;'>{message}</div>"
            "</div>"
        )
    if not cards:
        cards.append(
            "<div style='border:1px solid #fecaca; background:#fff1f2; border-radius:10px; padding:10px 12px; margin-bottom:8px;'>"
            "<div style='font-size:13px; color:#7f1d1d;'>Compilation failed. See raw log for details.</div>"
            "</div>"
        )
    raw_log = str(raw_log or "").strip()
    raw_log_html = ""
    if raw_log:
        raw_log_html = (
            "<details style='margin-top:10px; border:1px solid #e5e7eb; border-radius:8px; background:#ffffff;'>"
            "<summary style='cursor:pointer; padding:8px 10px; font-size:12px; font-weight:600; color:#334155;'>Raw LaTeX Log</summary>"
            "<div style='border-top:1px solid #e5e7eb; max-height:290px; overflow:auto; padding:8px 10px 10px 10px; "
            "font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color:#1f2937; white-space:pre-wrap;'>"
            f"{html.escape(raw_log[-4000:])}"
            "</div></details>"
        )
    content_html = (
        "<div style='height:100%; overflow:auto; background:#ffffff; padding:20px;'>"
        f"<div style='font-size:18px; font-weight:800; color:#991b1b; margin-bottom:10px;'>{html.escape(error_title or 'Cannot generate PDF, please recheck')}</div>"
        f"{''.join(cards)}"
        f"{raw_log_html}"
        "</div>"
    )
    _render_pdf_frame(
        content_html,
        height=height,
        scrollable=False,
        overlay_html=overlay_html,
        hint_text="Compile failed. Fix errors and recompile.",
    )


def _format_file_size(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB"]
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _sanitize_download_stem(raw_name: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw_name or "").strip()).strip("._-")
    return normalized or fallback


def _build_zip_archive_bytes(
    file_entries: List[Tuple[Path, str]],
    empty_dir_name: str | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if file_entries:
            for source_path, archive_name in file_entries:
                archive.write(source_path, archive_name)
        elif empty_dir_name:
            safe_dir = str(empty_dir_name).strip().strip("/")
            if safe_dir:
                archive.writestr(f"{safe_dir}/", b"")
    return buffer.getvalue()


def _build_project_entry_download_payload(
    doc_id: str,
    relative_path: str,
) -> Tuple[bytes, str, str]:
    normalized_rel = _normalize_project_relative_path(relative_path)
    if _is_internal_state_relative_path(normalized_rel):
        raise ValueError("System state files cannot be downloaded.")
    source_path = _resolve_project_path(doc_id, normalized_rel)
    if not source_path.exists():
        raise ValueError("Path does not exist.")

    if source_path.is_file():
        payload = source_path.read_bytes()
        mime, _encoding = mimetypes.guess_type(source_path.name)
        return payload, source_path.name, (mime or "application/octet-stream")

    project_root = ensure_project_dir(doc_id)
    archive_entries: List[Tuple[Path, str]] = []
    for child in sorted(source_path.rglob("*")):
        if not child.is_file():
            continue
        relative_child = child.relative_to(project_root).as_posix()
        if _is_internal_state_relative_path(relative_child):
            continue
        archive_name = child.relative_to(source_path.parent).as_posix()
        archive_entries.append((child, archive_name))

    archive_name = f"{source_path.name}.zip"
    payload = _build_zip_archive_bytes(archive_entries, empty_dir_name=source_path.name)
    return payload, archive_name, "application/zip"


def _build_workspace_download_zip_payload(workspace_id: str, workspace_name: str) -> Tuple[bytes, str]:
    project_root = ensure_project_dir(workspace_id)
    archive_entries: List[Tuple[Path, str]] = []
    for child in sorted(project_root.rglob("*")):
        if not child.is_file():
            continue
        rel_path = child.relative_to(project_root).as_posix()
        if _is_internal_state_relative_path(rel_path):
            continue
        archive_entries.append((child, rel_path))

    fallback_stem = f"workspace_{workspace_id}"
    archive_stem = _sanitize_download_stem(workspace_name, fallback_stem)
    archive_name = f"{archive_stem}.zip"
    payload = _build_zip_archive_bytes(archive_entries)
    return payload, archive_name


def _is_probably_text_resource(file_bytes: bytes, suffix: str) -> bool:
    if suffix in RESOURCE_TEXT_PREVIEW_SUFFIXES:
        return True
    sample = file_bytes[:8192]
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    printable_count = 0
    for byte in sample:
        if byte in (9, 10, 13) or 32 <= byte <= 126:
            printable_count += 1
    return printable_count / max(len(sample), 1) >= 0.82


def _resource_preview_is_open_for_workspace(workspace_id: str) -> bool:
    return bool(
        st.session_state.get("resource_preview_visible", False)
        and str(st.session_state.get("resource_preview_doc_id", "")) == str(workspace_id)
        and str(st.session_state.get("resource_preview_path", "")).strip()
    )


def _render_resource_preview_panel(workspace_id: str, height: int) -> None:
    preview_path_raw = str(st.session_state.get("resource_preview_path", "")).strip()
    if not preview_path_raw:
        st.info("No resource selected.")
        return
    try:
        preview_rel = _normalize_project_relative_path(preview_path_raw)
        preview_abs = _resolve_project_path(workspace_id, preview_rel)
    except ValueError:
        _close_resource_preview()
        st.warning("Resource preview path is invalid.", icon=":material/info:")
        return
    if not preview_abs.exists() or preview_abs.is_dir():
        _close_resource_preview()
        st.warning("Resource file no longer exists.", icon=":material/info:")
        return

    head_left, head_right = st.columns([0.78, 0.22], gap="small")
    with head_left:
        st.markdown("<h4 style='color:#111827;'>Resource Preview</h4>", unsafe_allow_html=True)
        st.caption(preview_rel)
    with head_right:
        if st.button(
            "Close",
            key=f"resource_preview_close_{workspace_id}",
            use_container_width=True,
        ):
            _close_resource_preview()
            st.rerun()

    suffix = preview_abs.suffix.lower()
    if suffix in RESOURCE_IMAGE_SUFFIXES:
        try:
            image_bytes = preview_abs.read_bytes()
        except OSError:
            st.warning("Unable to read image resource.", icon=":material/info:")
            return
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }
        image_mime = mime_map.get(suffix, "application/octet-stream")
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        st.markdown(
            (
                "<div style='width:100%; border:1px solid #d0d7de; border-radius:10px; "
                "background:#ffffff; padding:8px; display:flex; align-items:center; justify-content:center;'>"
                f"<img src='data:{image_mime};base64,{image_base64}' "
                "style='display:block; max-width:100%; max-height:68vh; width:auto; height:auto; object-fit:contain;'/>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return
    if suffix in RESOURCE_PDF_SUFFIXES:
        try:
            pdf_bytes = preview_abs.read_bytes()
        except OSError:
            st.warning("Unable to read PDF resource.", icon=":material/info:")
            return
        preview_box_height = max(460, int(height) - 20)
        page_images = _build_pdf_preview_png_pages(pdf_bytes, max_pages=24)
        if page_images:
            page_blocks: List[str] = []
            for index, page_bytes in enumerate(page_images, start=1):
                page_b64 = base64.b64encode(page_bytes).decode("utf-8")
                page_blocks.append(
                    (
                        "<div style='border:1px solid #e5e7eb; border-radius:8px; background:#ffffff; padding:6px;'>"
                        f"<div style='font-size:12px; color:#6b7280; margin:0 0 4px 0;'>Page {index}</div>"
                        f"<img src='data:image/png;base64,{page_b64}' "
                        "style='display:block; width:100%; height:auto; object-fit:contain;'/>"
                        "</div>"
                    )
                )
            pages_html = "".join(page_blocks)
            st.markdown(
                (
                    "<div style='border:1px solid #d0d7de; border-radius:10px; background:#ffffff; "
                    f"height:{preview_box_height}px; overflow:auto; padding:10px; display:flex; flex-direction:column; gap:10px;'>"
                    f"{pages_html}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            try:
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
            except Exception:
                pdf_base64 = ""
            if pdf_base64:
                st_components.html(
                    (
                        "<div style='height:100%; min-height:"
                        f"{max(280, int(height))}px; border:1px solid #d0d7de; border-radius:10px; overflow:hidden; background:#ffffff;'>"
                        f"<embed src='data:application/pdf;base64,{pdf_base64}#zoom=100' type='application/pdf' "
                        "style='display:block; width:100%; height:100%; min-height:100%; border:0;' />"
                        "</div>"
                    ),
                    height=max(300, int(height) + 8),
                    scrolling=False,
                )
            else:
                st.warning("Unable to render PDF preview.", icon=":material/info:")
        return

    if suffix == ".docx":
        try:
            doc_bytes = preview_abs.read_bytes()
        except OSError:
            st.warning("Unable to read document resource.", icon=":material/info:")
            return
        docx_text = _extract_text_from_docx_bytes(doc_bytes)
        if not docx_text.strip():
            docx_text = _extract_text_via_textutil(preview_abs)
        if docx_text.strip():
            preview_box_height = max(460, int(height) - 20)
            st.markdown(
                (
                    "<div style='border:1px solid #d0d7de; border-radius:10px; background:#ffffff; "
                    f"height:{preview_box_height}px; overflow:auto; padding:12px;'>"
                    "<pre style='margin:0; white-space:pre-wrap; word-break:break-word; "
                    "font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color:#111827;'>"
                    f"{html.escape(docx_text)}"
                    "</pre></div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "DOCX text preview is unavailable for this file.",
                icon=":material/info:",
            )
        return

    if suffix == ".doc":
        doc_text = _extract_text_via_textutil(preview_abs)
        if doc_text.strip():
            preview_box_height = max(460, int(height) - 20)
            st.markdown(
                (
                    "<div style='border:1px solid #d0d7de; border-radius:10px; background:#ffffff; "
                    f"height:{preview_box_height}px; overflow:auto; padding:12px;'>"
                    "<pre style='margin:0; white-space:pre-wrap; word-break:break-word; "
                    "font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color:#111827;'>"
                    f"{html.escape(doc_text)}"
                    "</pre></div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "DOC text preview is unavailable for this file.",
                icon=":material/info:",
            )
        return

    try:
        file_bytes = preview_abs.read_bytes()
        file_size = _format_file_size(preview_abs.stat().st_size)
    except OSError:
        st.warning("Unable to read resource file.", icon=":material/info:")
        return

    if _is_probably_text_resource(file_bytes, suffix):
        preview_bytes = file_bytes[:RESOURCE_TEXT_PREVIEW_MAX_BYTES]
        try:
            preview_text = preview_bytes.decode("utf-8")
        except UnicodeDecodeError:
            preview_text = preview_bytes.decode("utf-8", errors="replace")
        if len(file_bytes) > len(preview_bytes):
            preview_text = f"{preview_text}\n\n...<preview truncated>"
        preview_box_height = max(460, int(height) - 20)
        preview_html = (
            "<div style='border:1px solid #d0d7de; border-radius:10px; background:#ffffff; "
            f"height:{preview_box_height}px; overflow:auto; padding:12px;'>"
            "<pre style='margin:0; white-space:pre-wrap; word-break:break-word; "
            "font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color:#111827;'>"
            f"{html.escape(preview_text)}"
            "</pre></div>"
        )
        st.markdown(preview_html, unsafe_allow_html=True)
        return

    st.markdown(f"**File:** `{preview_abs.name}`")
    st.markdown(f"**Size:** `{file_size}`")
    st.download_button(
        "Download",
        data=file_bytes,
        file_name=preview_abs.name,
        mime="application/octet-stream",
        key=f"resource_preview_download_{workspace_id}_{preview_rel}",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Academic Writing Companion Agent",
        page_icon=_resolve_page_icon(),
        layout="wide",
    )
    _apply_workspace_layout_style()
    init_state()
    _apply_left_nav_panel_style()
    _apply_files_dnd_bridge_style()
    _render_workspace_sidebar()
    _sync_role_state_from_session(write_widget_key=True)

    state: AppState = st.session_state.app_state
    if st.session_state.get("active_workspace_id"):
        state.state_doc_id = st.session_state.active_workspace_id

    st.markdown("<div class='awc-main-workspace-marker'></div>", unsafe_allow_html=True)
    workspace_shell = st.container()
    workspace_shell.markdown("<div class='awc-main-shell-marker'></div>", unsafe_allow_html=True)

    panel_height = _compute_panel_height(
        int(st.session_state.viewport_height),
        int(st.session_state.viewport_available_height),
    )
    editor_height = max(300, panel_height - EDITOR_BOTTOM_RAISE_PX)
    pdf_height = editor_height + PDF_HEIGHT_COMPENSATION_PX

    _sync_role_state_from_session()
    selected_role = st.session_state.current_role
    if selected_role:
        state.role = selected_role

    # Drain pending chat/check/apply actions before rendering interactive panels.
    # Otherwise, editor-side reruns can repeatedly preempt the pending action and
    # leave the UI stuck in "Processing response...".
    _drain_processing_pending(state)

    editor_col, pdf_col = workspace_shell.columns(EDITOR_PDF_RATIO, gap="medium")

    with editor_col:
        st.markdown("<div class='awc-editor-panel-marker'></div>", unsafe_allow_html=True)
        editor_head_left, editor_head_right = st.columns([0.66, 0.34], gap="small")
        compile_button_slot = None
        with editor_head_left:
            st.markdown("<h4 style='color:#111827;'>Interactive LaTeX Editor</h4>", unsafe_allow_html=True)
        with editor_head_right:
            compile_button_slot = st.empty()

        confirmed_preview = st.session_state.confirmed_selection_text.strip()
        pending_preview = st.session_state.pending_selection_text.strip()
        if confirmed_preview:
            selection_status = "confirmed"
        elif pending_preview:
            selection_status = "pending"
        else:
            selection_status = "none"

        display_start = st.session_state.confirmed_selection_start
        display_end = st.session_state.confirmed_selection_end
        editor_component_key = f"interactive_latex_editor_{state.state_doc_id}"
        try:
            current_active_file = _normalize_project_relative_path(
                st.session_state.get("active_file_path", "main.tex")
            )
        except ValueError:
            current_active_file = "main.tex"
        if _is_internal_state_relative_path(current_active_file) or not _is_text_editable_project_file(
            current_active_file
        ):
            current_active_file = "main.tex"

        payload = interactive_editor(
            text=state.current_text,
            file_path=current_active_file,
            height=editor_height,
            highlight_start=display_start,
            highlight_end=display_end,
            selection_status=selection_status,
            confirmed_preview=confirmed_preview[:140],
            focus_start=int(st.session_state.get("editor_focus_start", -1)),
            focus_end=int(st.session_state.get("editor_focus_end", -1)),
            focus_event_id=int(st.session_state.get("editor_focus_event_id", 0)),
            backend_sync_event_id=int(st.session_state.get("editor_backend_sync_event_id", 0)),
            key=editor_component_key,
        )
        payload_matches_active_file = True
        payload_file_raw = str(payload.get("file_path", "")).strip()
        if payload_file_raw:
            try:
                payload_file_path = _normalize_project_relative_path(payload_file_raw)
            except ValueError:
                payload_matches_active_file = False
            else:
                if payload_file_path != current_active_file:
                    payload_matches_active_file = False
        previous_text = str(state.current_text)
        incoming_text = str(payload.get("text", previous_text))
        incoming_trigger = str(payload.get("trigger", "")).strip().lower()
        allow_clear_triggers = {
            "input",
            "confirm",
            "cancel",
            "confirm_edit",
            "preserve_selection",
            "invalidate_selection",
        }
        suspicious_empty_echo = (
            not incoming_text.strip()
            and bool(previous_text.strip())
            and incoming_trigger not in allow_clear_triggers
        )
        if (
            payload_matches_active_file
            and not suspicious_empty_echo
            and _should_accept_editor_component_text(
                state,
                incoming_text,
                candidate_trigger=incoming_trigger,
            )
        ):
            state.current_text = incoming_text
        _autosave_main_tex_if_needed(state)
        if payload_matches_active_file and bool(payload.get("focus_applied", False)):
            st.session_state.editor_focus_start = -1
            st.session_state.editor_focus_end = -1
            st.session_state.editor_focus_event_id = 0
        if payload_matches_active_file:
            st.session_state.selection_text = str(payload.get("selection", "")).strip()
            st.session_state.selection_start = int(payload.get("selection_start", -1))
            st.session_state.selection_end = int(payload.get("selection_end", -1))
            selection_trigger = str(payload.get("trigger", ""))
            selection_action = str(payload.get("confirm_action", "")).strip().lower()
        else:
            selection_trigger = ""
            selection_action = ""
        action_fingerprint = (
            f"{selection_action}|{selection_trigger}|"
            f"{st.session_state.selection_start}|{st.session_state.selection_end}|"
            f"{st.session_state.selection_text}"
        )
        action_state_changed = False

        if st.session_state.suppress_next_selection_payload:
            st.session_state.selection_text = ""
            st.session_state.selection_start = -1
            st.session_state.selection_end = -1
            st.session_state.suppress_next_selection_payload = False

        st.session_state.active_selection = st.session_state.selection_text

        if (
            selection_action == "invalidate_selection"
            and action_fingerprint != st.session_state.last_component_action_fingerprint
        ):
            st.session_state.pending_selection_text = ""
            st.session_state.pending_selection_start = -1
            st.session_state.pending_selection_end = -1
            st.session_state.confirmed_selection_text = ""
            st.session_state.confirmed_selection_start = -1
            st.session_state.confirmed_selection_end = -1
            st.session_state.active_selection = ""
            state.active_selection = None
            st.session_state.last_component_action_fingerprint = action_fingerprint
            action_state_changed = True
        elif (
            selection_action in {"confirm_edit", "preserve_selection"}
            and action_fingerprint != st.session_state.last_component_action_fingerprint
        ):
            candidate_text = st.session_state.selection_text.strip()
            candidate_start = st.session_state.selection_start
            candidate_end = st.session_state.selection_end
            if candidate_text and candidate_start >= 0 and candidate_end > candidate_start:
                st.session_state.confirmed_selection_text = candidate_text
                st.session_state.confirmed_selection_start = candidate_start
                st.session_state.confirmed_selection_end = candidate_end
                st.session_state.pending_selection_text = candidate_text
                st.session_state.pending_selection_start = candidate_start
                st.session_state.pending_selection_end = candidate_end
                st.session_state.active_selection = candidate_text
                st.session_state.last_component_action_fingerprint = action_fingerprint
                if selection_action == "confirm_edit":
                    action_state_changed = True
        elif selection_trigger == "input" and selection_action not in {
            "confirm_edit",
            "preserve_selection",
            "invalidate_selection",
        }:
            st.session_state.pending_selection_text = ""
            st.session_state.pending_selection_start = -1
            st.session_state.pending_selection_end = -1
            st.session_state.confirmed_selection_text = ""
            st.session_state.confirmed_selection_start = -1
            st.session_state.confirmed_selection_end = -1
            st.session_state.active_selection = ""
            state.active_selection = None
            st.session_state.last_component_action_fingerprint = ""
        elif (
            selection_action == "cancel"
            and action_fingerprint != st.session_state.last_component_action_fingerprint
        ):
            st.session_state.pending_selection_text = ""
            st.session_state.pending_selection_start = -1
            st.session_state.pending_selection_end = -1
            st.session_state.selection_text = ""
            st.session_state.selection_start = -1
            st.session_state.selection_end = -1
            if st.session_state.confirmed_selection_text.strip():
                st.session_state.active_selection = st.session_state.confirmed_selection_text.strip()
            else:
                st.session_state.active_selection = ""
                state.active_selection = None
            st.session_state.last_component_action_fingerprint = action_fingerprint
            action_state_changed = True
        elif (
            st.session_state.selection_text
            and st.session_state.selection_start >= 0
            and st.session_state.selection_end > st.session_state.selection_start
        ):
            st.session_state.pending_selection_text = st.session_state.selection_text
            st.session_state.pending_selection_start = st.session_state.selection_start
            st.session_state.pending_selection_end = st.session_state.selection_end

        if (
            selection_action == "confirm"
            and action_fingerprint != st.session_state.last_component_action_fingerprint
        ):
            candidate_text = st.session_state.selection_text.strip() or st.session_state.pending_selection_text.strip()
            candidate_start = st.session_state.selection_start
            candidate_end = st.session_state.selection_end
            if candidate_start < 0 or candidate_end <= candidate_start:
                candidate_start = st.session_state.pending_selection_start
                candidate_end = st.session_state.pending_selection_end
            if candidate_text and candidate_start >= 0 and candidate_end > candidate_start:
                st.session_state.confirmed_selection_text = candidate_text
                st.session_state.confirmed_selection_start = candidate_start
                st.session_state.confirmed_selection_end = candidate_end
                st.session_state.pending_selection_text = candidate_text
                st.session_state.pending_selection_start = candidate_start
                st.session_state.pending_selection_end = candidate_end
                st.session_state.active_selection = candidate_text
                st.session_state.last_component_action_fingerprint = action_fingerprint
                action_state_changed = True

        if st.session_state.confirmed_selection_text:
            start = st.session_state.confirmed_selection_start
            end = st.session_state.confirmed_selection_end
            if start >= 0 and end > start:
                state.active_selection = SelectionContext(
                    start=start,
                    end=end,
                    snippet=st.session_state.confirmed_selection_text,
                    sentence_indices=[],
                    metadata={"source": "interactive_editor", "confirmed": True},
                )
            else:
                state.active_selection = SelectionContext(
                    start=0,
                    end=0,
                    snippet=st.session_state.confirmed_selection_text,
                    sentence_indices=[],
                    metadata={"source": "interactive_editor", "confirmed": True, "offset_unknown": True},
                )
            st.session_state.active_selection = st.session_state.confirmed_selection_text
        else:
            state.active_selection = None
            st.session_state.active_selection = ""

        if action_state_changed:
            if selection_action in {"confirm", "confirm_edit", "preserve_selection"}:
                _save_active_workspace_snapshot(touch_activity=True)
            st.rerun()

        compile_button_label = "Recompile" if st.session_state.get("pdf_compiled_text_hash") else "Compile"
        if st.session_state.get("compile_in_progress", False):
            compile_button_slot.markdown(
                "<div class='awc-compile-loader'><span class='awc-compile-loader-bot'>🤖</span><span>Compiling...</span></div>",
                unsafe_allow_html=True,
            )
            _run_compile_pipeline(state)
            st.session_state.compile_in_progress = False
            st.rerun()
        compile_clicked = compile_button_slot.button(
            compile_button_label,
            key="compile_latex_btn_v2",
            use_container_width=True,
        )
        if compile_clicked:
            _adopt_active_editor_component_text_for_compile(state, editor_component_key)
            _autosave_main_tex_if_needed(state)
            st.session_state.compile_in_progress = True
            st.rerun()

    with pdf_col:
        st.markdown("<div class='awc-pdf-panel-marker'></div>", unsafe_allow_html=True)
        pdf_head_left, pdf_head_right = st.columns([0.66, 0.34], gap="small")
        with pdf_head_left:
            st.markdown("<h4 style='color:#111827;'>PDF Preview</h4>", unsafe_allow_html=True)
        with pdf_head_right:
            export_disabled = not bool(st.session_state.get("pdf_compiled_bytes", b""))
            if st.button(
                "Export PDF",
                key="export_pdf_btn_v2",
                use_container_width=True,
                disabled=export_disabled,
                help="Save compiled PDF to a local folder",
            ):
                if _request_browser_pdf_export():
                    st.session_state.pdf_last_export_path = ""

        preview_png_base64 = ""
        preview_pages_base64: List[str] = []
        if st.session_state.get("pdf_compile_ok", False):
            compiled_pdf_bytes = bytes(st.session_state.get("pdf_compiled_bytes", b""))
            compiled_hash = (
                hashlib.sha256(compiled_pdf_bytes).hexdigest()
                if compiled_pdf_bytes
                else str(st.session_state.get("pdf_compiled_text_hash", ""))
            )

            cached_hash = str(st.session_state.get("pdf_preview_png_hash", ""))
            cached_preview = bytes(st.session_state.get("pdf_preview_png_bytes", b""))
            if compiled_hash and (compiled_hash != cached_hash or not cached_preview):
                generated_preview = _build_pdf_first_page_preview_png(
                    compiled_pdf_bytes
                )
                st.session_state.pdf_preview_png_bytes = generated_preview
                st.session_state.pdf_preview_png_hash = compiled_hash if generated_preview else ""
                cached_preview = generated_preview
            if cached_preview:
                preview_png_base64 = base64.b64encode(cached_preview).decode("utf-8")

            cached_pages_hash = str(st.session_state.get("pdf_preview_png_pages_hash", ""))
            cached_pages_list = st.session_state.get("pdf_preview_png_pages_bytes", [])
            cached_pages_ok = isinstance(cached_pages_list, list) and len(cached_pages_list) > 0
            if compiled_hash and (compiled_hash != cached_pages_hash or not cached_pages_ok):
                generated_pages = _build_pdf_preview_png_pages(
                    compiled_pdf_bytes
                )
                st.session_state.pdf_preview_png_pages_bytes = generated_pages
                st.session_state.pdf_preview_png_pages_hash = compiled_hash if generated_pages else ""
                cached_pages_list = generated_pages
            if isinstance(cached_pages_list, list) and cached_pages_list:
                preview_pages_base64 = [
                    base64.b64encode(bytes(page_bytes)).decode("utf-8")
                    for page_bytes in cached_pages_list
                    if isinstance(page_bytes, (bytes, bytearray)) and page_bytes
                ]

        pdf_payload: Dict[str, Any] = {"action": "idle", "page": 0, "x": 0.0, "y": 0.0, "event_id": 0}
        has_compiled_once = bool(st.session_state.get("pdf_compiled_text_hash"))
        if not has_compiled_once:
            _render_pdf_empty_frame(pdf_height)
        elif st.session_state.get("pdf_compile_ok", False):
            if preview_pages_base64:
                _render_png_preview_pages_frame(preview_pages_base64, pdf_height)
            elif preview_png_base64:
                _render_png_preview_frame(preview_png_base64, pdf_height)
            else:
                _render_pdf_embed_frame(
                    str(st.session_state.get("pdf_compiled_base64", "")),
                    pdf_height,
                )
        else:
            _render_pdf_error_frame(
                height=pdf_height,
                error_title="Cannot generate PDF, please recheck",
                error_items=list(st.session_state.get("pdf_compile_errors", [])),
                raw_log=str(st.session_state.get("pdf_compile_log", ""))[-4000:],
            )
        pending_pdf_payload = st.session_state.get("pdf_pending_dblclick_event")
        if isinstance(pending_pdf_payload, dict):
            try:
                pending_event_id = int(pending_pdf_payload.get("event_id", 0) or 0)
            except (TypeError, ValueError):
                pending_event_id = 0
            if pending_event_id != int(st.session_state.get("pdf_viewer_last_event_id", 0) or 0):
                pdf_payload = {
                    "action": "pdf_dblclick",
                    "page": int(pending_pdf_payload.get("page", 0) or 0),
                    "x": float(pending_pdf_payload.get("x", 0.0) or 0.0),
                    "y": float(pending_pdf_payload.get("y", 0.0) or 0.0),
                    "event_id": pending_event_id,
                }
            st.session_state.pdf_pending_dblclick_event = None
        pdf_event_id = int(pdf_payload.get("event_id", 0))
        if pdf_event_id and pdf_event_id != int(st.session_state.get("pdf_viewer_last_event_id", 0)):
            st.session_state.pdf_viewer_last_event_id = pdf_event_id
            action = str(pdf_payload.get("action", "idle"))
            if action == "pdf_dblclick":
                target_rel, line_number = _synctex_pdf_to_source(
                    state.state_doc_id,
                    int(pdf_payload.get("page", 0)),
                    float(pdf_payload.get("x", 0.0)),
                    float(pdf_payload.get("y", 0.0)),
                )
                try:
                    current_rel = _normalize_project_relative_path(
                        st.session_state.get("active_file_path", "main.tex")
                    )
                except ValueError:
                    current_rel = "main.tex"
                if (
                    target_rel
                    and _is_text_editable_project_file(target_rel)
                ):
                    if target_rel != current_rel:
                        _open_workspace_file(state.state_doc_id, target_rel)
                    _schedule_editor_focus_for_line(state.state_doc_id, target_rel, line_number)
                    st.rerun()

    with st.container():
        st.markdown("<div class='awc-chat-drawer-marker'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#111827;'>Companion Chat</h4>", unsafe_allow_html=True)
        role_col, chat_manage_col = st.columns([0.42, 0.58], gap="small")
        with role_col:
            st.markdown(
                "<div style='font-size:0.78rem; color:#6b7280; line-height:1; margin:0 0 0.06rem 0;'>Role</div>",
                unsafe_allow_html=True,
            )
            role_options = ["", "Reviewer", "Advisor", "Editor"]
            st.selectbox(
                "Role",
                options=role_options,
                key="current_role",
                on_change=_on_role_change,
                format_func=lambda x: "Select a role..." if x == "" else x,
                label_visibility="collapsed",
            )
        with chat_manage_col:
            st.markdown(
                "<div style='height:1.0rem; line-height:1; margin:0 0 0.06rem 0;'></div>",
                unsafe_allow_html=True,
            )
            active_id = str(st.session_state.get("active_workspace_id", "")).strip()

            manage_btn_col1, manage_btn_col2 = st.columns(2, gap="small")
            with manage_btn_col1:
                if st.button(
                    "+ New Chat",
                    key=f"agent_chat_new_btn_{active_id}",
                    use_container_width=True,
                    disabled=not bool(active_id),
                ):
                    _create_agent_chat(active_id)
                    st.rerun()
            with manage_btn_col2:
                if active_id:
                    with st.popover("History", use_container_width=True):
                        _render_agent_chat_history_list(active_id)
                else:
                    st.button(
                        "History",
                        key=f"agent_chat_history_disabled_btn_{active_id}",
                        use_container_width=True,
                        disabled=True,
                    )

        _sync_role_state_from_session()
        selected_role = st.session_state.current_role
        if selected_role:
            state.role = selected_role

        processing_busy = bool(st.session_state.get("processing_busy", False))
        composer_disabled = (not bool(st.session_state.current_role)) or processing_busy
        if composer_disabled and st.session_state.composer_tools_open:
            st.session_state.composer_tools_open = False

        confirmed_scope = _confirmed_scope_payload(state)
        analysis_doc = _analysis_document_text_for_scope(state, confirmed_scope)
        availability_document_text = str(analysis_doc.get("check_document_text", "") or "")
        availability_probe_text = (
            availability_document_text
            if bool(analysis_doc.get("expanded_main", False))
            else (str(confirmed_scope.text or "") if confirmed_scope is not None else "")
        )
        check_availability = _compute_check_availability(
            selection_text=availability_probe_text,
            workspace_resources=_workspace_resource_snapshot(state),
            document_text=availability_document_text,
        )
        check_enabled_map = _normalize_check_enabled_map(check_availability.get("enabled"))
        st.session_state.check_enabled_map = dict(check_enabled_map)
        selected_checks_runtime = [
            name
            for name in list(st.session_state.get("selected_checks", list(CHECK_NAMES)))
            if name in CHECK_NAME_SET and check_enabled_map.get(name, False)
        ]
        if selected_checks_runtime != list(st.session_state.get("selected_checks", [])):
            st.session_state.selected_checks = list(selected_checks_runtime)

        reserved_input_height = (
            COMPOSER_CLOSED_HEIGHT + CHAT_PANEL_BOTTOM_BUFFER + 112 - CHAT_COMPOSER_VERTICAL_SHIFT_PX
        )
        raw_panel_height = _compute_panel_height_raw(
            int(st.session_state.get("viewport_height", 0)),
            int(st.session_state.get("viewport_available_height", 0)),
        )
        chat_height = max(220, raw_panel_height - reserved_input_height)
        composer_component_key = f"chat_composer_component_{state.state_doc_id}"
        _render_chat_history(state, height=chat_height, border=True)
        busy_kind = str(st.session_state.get("processing_kind", "")).strip().lower()
        busy_text = "Running checks..." if busy_kind == "checks" else "Processing response..."
        if processing_busy:
            st.markdown(
                f"<div class='awc-chat-busy-loader'><span class='awc-chat-busy-spinner'></span><span>{busy_text}</span></div>",
                unsafe_allow_html=True,
            )
        composer_payload = chat_composer(
            draft=st.session_state.composer_text,
            disabled=composer_disabled,
            tools_open=st.session_state.composer_tools_open,
            selected_checks=selected_checks_runtime,
            check_enabled=check_enabled_map,
            placeholder="Discuss findings, ask for revisions, or request grounded papers.",
            key=composer_component_key,
        )
        _save_component_debug_payload(state.state_doc_id, "chat_composer", composer_payload)
        _handle_composer_panel(state, composer_payload, composer_disabled)
        _drain_processing_pending(state)
        _save_active_workspace_snapshot()

    viewport_metrics = viewport_probe(key="viewport_probe_v5")
    measured_viewport = int(viewport_metrics.get("height", 0))
    measured_available = int(viewport_metrics.get("available_height", 0))
    viewport_changed = False
    if measured_viewport > 0 and measured_viewport != int(st.session_state.viewport_height):
        st.session_state.viewport_height = measured_viewport
        viewport_changed = True
    if measured_available > 0 and measured_available != int(st.session_state.viewport_available_height):
        st.session_state.viewport_available_height = measured_available
        viewport_changed = True
    if viewport_changed:
        st.rerun()

    panel_mode = str(st.session_state.get("sidebar_panel_mode", "") or "").strip().lower()
    active_workspace_id = str(st.session_state.get("active_workspace_id", "") or "").strip()
    _render_files_dnd_bridge_controls()
    dnd_event = render_files_dnd_bridge(
        panel_mode=panel_mode,
        doc_id=active_workspace_id,
        nodes=_visible_files_tree_nodes(active_workspace_id) if panel_mode == "files" and active_workspace_id else [],
        key="files_dnd_bridge_component",
    )
    if _handle_files_dnd_component_event(dnd_event):
        st.rerun()

    _render_export_success_toast_mount()
    _render_browser_pdf_export_mount()
    _inject_favicon_override()
    _inject_scroll_to_top_on_refresh()
    _inject_chat_history_scroll_to_bottom()
    _inject_left_sidebar_toggle()
    _inject_files_preview_resizer()
    _inject_files_dnd_bridge()
    _inject_long_filename_button_fix()
    _inject_chat_drawer_toggle()
    _inject_chat_drawer_layout_sync()
    _inject_chat_drawer_history_arrow()
    _inject_chat_copy_buttons()
    if selected_role:
        _render_floating_companion(selected_role)


if __name__ == "__main__":
    main()
