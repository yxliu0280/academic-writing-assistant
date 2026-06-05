from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from core.schemas import (
    BlockedResponse,
    RequestedCheckName,
    RolePayload,
    RouterDecision,
    RouterRequest,
    ScopePayload,
    SystemExceptionPayload,
)


CHECK_ORDER: List[RequestedCheckName] = ["table", "figure", "citation", "terminology"]


class RouterV1:
    CHECK_KEYWORDS: Dict[RequestedCheckName, List[str]] = {
        "table": ["table", "tables", "tabular", "tab", "表", "表格"],
        "figure": ["figure", "fig", "image", "images", "chart", "plot", "graph", "图", "图片", "图表"],
        "citation": ["citation", "citations", "cite", "reference", "references", "bib", "bibtex", "文献", "引用", "参考文献"],
        "terminology": ["terminology", "term", "terms", "glossary", "acronym", "acronyms", "术语", "缩写", "专有名词"],
    }
    CHECK_INTENT_KEYWORDS: List[str] = [
        "check",
        "checks",
        "checking",
        "verify",
        "verification",
        "consistency",
        "inconsistency",
        "inspect",
        "run",
        "review",
        "看看",
        "检查",
        "核查",
        "一致性",
        "不一致",
        "有问题",
        "进行",
        "执行",
        "做",
        "帮我",
    ]
    IMAGE_INTENT_KEYWORDS: List[str] = [
        "image",
        "images",
        "figure",
        "fig",
        "chart",
        "plot",
        "graph",
        "diagram",
        "图",
        "图片",
        "图像",
        "图表",
        "曲线",
        "趋势",
    ]
    IMAGE_NUMERIC_HINT_KEYWORDS: List[str] = [
        "%",
        "percent",
        "percentage",
        "pp",
        "increase",
        "decrease",
        "improve",
        "improvement",
        "drop",
        "from",
        "to",
        "value",
        "数值",
        "百分比",
        "提升",
        "下降",
        "从",
        "到",
    ]
    TABLE_INTENT_KEYWORDS: List[str] = [
        "table",
        "tables",
        "tabular",
        "table data",
        "表",
        "表格",
        "表中",
        "表里的",
        "表格里",
    ]
    FIGURE_REF_PATTERN = re.compile(r"\\(?:ref|autoref|cref|Cref)\{([^}]+)\}")
    TABLE_REF_CMD_PATTERN = re.compile(r"\\(?:ref|autoref|cref|Cref)\{[^}]*tab:[^}]*\}", flags=re.IGNORECASE)
    FIGURE_REF_CMD_PATTERN = re.compile(r"\\(?:ref|autoref|cref|Cref)\{[^}]*fig:[^}]*\}", flags=re.IGNORECASE)
    TABLE_NATURAL_REF_PATTERN = re.compile(
        r"\btable(?:s)?\s*(?:~|no\.?\s*)?\d+[A-Za-z]?\b",
        flags=re.IGNORECASE,
    )
    FIGURE_NATURAL_REF_PATTERN = re.compile(
        r"\bfigure(?:s)?\s*(?:~|no\.?\s*)?\d+[A-Za-z]?\b|\bfig\.?\s*\d+[A-Za-z]?\b",
        flags=re.IGNORECASE,
    )
    CITATION_CMD_PATTERN = re.compile(r"\\cite\w*\{[^}]+\}", flags=re.IGNORECASE)
    FIGURE_ENV_PATTERN = re.compile(
        r"\\begin\{figure\}[\s\S]*?\\end\{figure\}",
        flags=re.IGNORECASE,
    )
    FIGURE_LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
    INCLUDEGRAPHICS_PATTERN = re.compile(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
        flags=re.IGNORECASE,
    )
    REWRITE_INTENT_KEYWORDS: List[str] = [
        "rewrite",
        "polish",
        "improve",
        "refine",
        "rephrase",
        "make this",
        "润色",
        "改写",
        "修改",
        "优化",
    ]
    AFFIRMATIVE_KEYWORDS: List[str] = [
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "go ahead",
        "please do",
        "do it",
        "apply",
        "accept",
        "proceed",
        "可以",
        "好的",
        "同意",
        "确认",
    ]
    NEGATIVE_KEYWORDS: List[str] = [
        "no",
        "n",
        "stop",
        "cancel",
        "reject",
        "not now",
        "do not",
        "不要",
        "取消",
        "不用",
        "不同意",
    ]

    def decide(self, request: RouterRequest) -> RouterDecision:
        if request.busy:
            return RouterDecision(
                kind="BLOCKED_PRECONDITION",
                blocked_response=BlockedResponse(
                    reason="busy",
                    message="Please wait for the current operation to finish before starting a new one.",
                    role=request.role,
                ),
            )

        normalized_role = str(request.role or "").strip()
        if not normalized_role:
            return RouterDecision(
                kind="BLOCKED_PRECONDITION",
                blocked_response=BlockedResponse(
                    reason="no_role",
                    message="Please select a role before sending messages, running checks, or using Undo.",
                    role="",
                ),
            )

        if request.action == "undo":
            return RouterDecision(kind="UNDO")

        if request.action == "send_message" and request.pending_patch_negotiation and normalized_role == "Editor":
            return RouterDecision(
                kind="PATCH_NEGOTIATION",
                metadata={
                    "pending_patch_stage": "negotiation",
                    "trigger_source": "patch_feedback",
                },
            )

        if request.action == "send_message" and request.pending_patch_preview_consent and normalized_role == "Editor":
            return RouterDecision(
                kind="PATCH_PREVIEW_CONSENT",
                metadata=self._consent_metadata(request.user_text),
            )

        if request.action == "send_message" and request.pending_patch_apply_consent and normalized_role == "Editor":
            return RouterDecision(
                kind="PATCH_APPLY_CONSENT",
                metadata=self._consent_metadata(request.user_text),
            )

        if request.action == "send_message" and request.pending_advisor_suggestion_consent and normalized_role == "Advisor":
            return RouterDecision(
                kind="ADVISOR_SUGGESTION_CONSENT",
                metadata=self._consent_metadata(request.user_text),
            )

        if not str(request.document_text or "").strip():
            return RouterDecision(
                kind="BLOCKED_PRECONDITION",
                blocked_response=self._blocked_for_missing_document(normalized_role, request.action),
            )

        if request.scope is None or not str(request.scope.text or "").strip():
            return RouterDecision(
                kind="BLOCKED_PRECONDITION",
                blocked_response=self._blocked_for_missing_selection(normalized_role, request.action),
            )

        if request.action == "run_checks":
            requested_checks = self._ordered_unique_checks(request.ui_selected_checks)
            if not requested_checks:
                return RouterDecision(
                    kind="BLOCKED_PRECONDITION",
                    blocked_response=BlockedResponse(
                        reason="invalid_scope",
                        message="Please select at least one consistency check type first.",
                        role=normalized_role,
                    ),
                )
            blocked = self._validate_requested_checks(request, requested_checks, normalized_role)
            if blocked is not None:
                return RouterDecision(kind="BLOCKED_PRECONDITION", blocked_response=blocked)
            return RouterDecision(
                kind="RUN_CHECKS_UI",
                requested_checks=requested_checks,
                metadata={"trigger_source": "ui"},
            )

        if request.action != "send_message":
            return RouterDecision(
                kind="BLOCKED_PRECONDITION",
                blocked_response=BlockedResponse(
                    reason="invalid_scope",
                    message="Unsupported action for Router v1.",
                    role=normalized_role,
                ),
            )

        nl_requested_checks = self._extract_requested_checks(request.user_text)
        ui_requested_checks = request.ui_selected_checks if request.ui_checks_explicit else []
        combined_checks = self._ordered_unique_checks(ui_requested_checks, nl_requested_checks)
        if combined_checks and self._looks_like_check_request(request.user_text, combined_checks):
            return RouterDecision(
                kind="RUN_CHECKS_NL",
                requested_checks=combined_checks,
                metadata={
                    "trigger_source": "natural_language",
                    "ui_checks_explicit": request.ui_checks_explicit,
                },
            )

        if self._looks_like_image_related_request(request.user_text):
            image_intent = "numeric_consistency" if self._looks_like_image_numeric_request(request.user_text) else "general"
            image_exception, image_target = self._resolve_image_request(request)
            if image_exception is not None:
                return RouterDecision(
                    kind="SYSTEM_EXCEPTION",
                    system_exception=image_exception,
                    role_payload=self._build_role_payload(
                        request=request,
                        mode="system_exception",
                        system_exception=image_exception,
                        metadata={"route": "image_related", "image_intent": image_intent},
                    ),
                    metadata={"route": "image_related", "image_intent": image_intent},
                )

            role_payload = self._build_role_payload(
                request=request,
                mode="grounded_chat",
                metadata={"route": "image_related", "image_intent": image_intent},
            )
            return RouterDecision(
                kind="CHAT",
                role_payload=role_payload,
                metadata={
                    "route": "image_related",
                    "image_intent": image_intent,
                    "image_target": image_target,
                },
            )

        if self._looks_like_table_related_request(request.user_text):
            table_exception, table_target = self._resolve_table_request(request)
            if table_exception is not None:
                return RouterDecision(
                    kind="SYSTEM_EXCEPTION",
                    system_exception=table_exception,
                    role_payload=self._build_role_payload(
                        request=request,
                        mode="system_exception",
                        system_exception=table_exception,
                        metadata={"route": "table_related"},
                    ),
                    metadata={"route": "table_related"},
                )

            role_payload = self._build_role_payload(
                request=request,
                mode="grounded_chat",
                metadata={"route": "table_related"},
            )
            return RouterDecision(
                kind="CHAT",
                role_payload=role_payload,
                metadata={
                    "route": "table_related",
                    "table_target": table_target,
                },
            )

        if normalized_role == "Editor" and self._looks_like_rewrite_request(request.user_text):
            role_payload = self._build_role_payload(
                request=request,
                mode="grounded_chat",
                metadata={"route": "editor_rewrite"},
            )
            return RouterDecision(
                kind="CHAT",
                role_payload=role_payload,
                metadata={"route": "editor_rewrite"},
            )
        role_payload = self._build_role_payload(
            request=request,
            mode="grounded_chat",
            metadata={"route": "grounded_chat"},
        )
        return RouterDecision(kind="CHAT", role_payload=role_payload, metadata={"route": "grounded_chat"})

    def _validate_requested_checks(
        self,
        request: RouterRequest,
        requested_checks: List[RequestedCheckName],
        role: str,
    ) -> Optional[BlockedResponse]:
        scope_text = str(request.scope.text or "") if request.scope is not None else ""
        workspace = request.workspace_resources if isinstance(request.workspace_resources, dict) else {}

        for check_name in requested_checks:
            if check_name == "table" and not self._selection_has_table_signal(scope_text):
                return BlockedResponse(
                    reason="module_inapplicable",
                    message="Table consistency is not applicable because the confirmed selection has no table signal (for example \\ref{tab:...}).",
                    role=role,
                    required_scope="table_selection",
                )
            if check_name == "figure" and not self._selection_has_figure_signal(scope_text):
                return BlockedResponse(
                    reason="module_inapplicable",
                    message="Figure consistency is not applicable because the confirmed selection has no figure signal (for example \\ref{fig:...}).",
                    role=role,
                    required_scope="figure_selection",
                )
            if check_name == "citation":
                if not self._selection_has_citation_signal(scope_text):
                    return BlockedResponse(
                        reason="module_inapplicable",
                        message="Citation consistency is not applicable because the confirmed selection has no citation command (for example \\cite{...}).",
                        role=role,
                        required_scope="citation_selection",
                    )
                has_bib = bool(workspace.get("has_bib", False))
                if not has_bib:
                    return BlockedResponse(
                        reason="missing_bib",
                        message="Citation consistency requires a local .bib file in the workspace file tree.",
                        role=role,
                        required_scope="workspace_bib",
                    )
        return None

    def _build_role_payload(
        self,
        request: RouterRequest,
        mode: str,
        system_exception: Optional[SystemExceptionPayload] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RolePayload:
        assert request.scope is not None
        return RolePayload(
            mode=mode,  # type: ignore[arg-type]
            role=request.role,
            scope=request.scope,
            user_request=request.user_text,
            system_exception=system_exception,
            memory=dict(request.memory or {}),
            metadata=dict(metadata or {}),
        )

    def _resolve_image_request(
        self,
        request: RouterRequest,
    ) -> tuple[Optional[SystemExceptionPayload], str]:
        multimodal_enabled = bool(request.tool_availability.get("multimodal_provider", False))
        if not multimodal_enabled:
            return (
                SystemExceptionPayload(
                    code="tool_unavailable",
                    message="image analysis is unavailable because the multimodal provider is not configured.",
                    target_type="image",
                    details={"requested_text": request.user_text},
                ),
                "",
            )
        images = list(request.workspace_resources.get("images", [])) if isinstance(request.workspace_resources, dict) else []
        scope_text = str(request.scope.text or "") if request.scope is not None else ""
        scope_figure_labels = self._extract_figure_labels(scope_text)
        if scope_figure_labels:
            scope_target, scope_issue = self._resolve_image_from_scope(
                labels=scope_figure_labels,
                full_text=str(request.document_text or ""),
                images=images,
            )
            if scope_target:
                return None, scope_target
            if scope_issue is not None:
                return scope_issue, ""

        image_target = self._match_image_target(request.user_text, images)
        if image_target:
            return None, str(image_target.get("path", ""))
        if not images:
            return (
                SystemExceptionPayload(
                    code="image_missing",
                    message="the requested image or figure is not available in the current workspace.",
                    target_type="image",
                    details={"requested_text": request.user_text},
                ),
                "",
            )
        if len(images) > 1:
            return (
                SystemExceptionPayload(
                    code="unsupported_request",
                    message="multiple image resources are available, so please mention the target figure or filename explicitly.",
                    target_type="image",
                    details={"available_images": [item.get("path", "") for item in images[:8]]},
                ),
                "",
            )
        return None, str(images[0].get("path", ""))

    def _resolve_image_from_scope(
        self,
        *,
        labels: List[str],
        full_text: str,
        images: List[Dict[str, Any]],
    ) -> tuple[str, Optional[SystemExceptionPayload]]:
        if not labels:
            return "", None

        image_path_candidates: List[str] = []
        for label in labels:
            image_path_candidates.extend(self._resolve_figure_image_paths_by_label(full_text, label))
        if not image_path_candidates:
            if not images:
                return "", SystemExceptionPayload(
                    code="image_missing",
                    message="the selected scope references figure labels, but no image resource is available in the current workspace.",
                    target=labels[0],
                    target_type="image",
                    details={"figure_labels": labels},
                )
            if len(images) == 1:
                return str(images[0].get("path", "")), None
            return "", SystemExceptionPayload(
                code="unsupported_request",
                message="the selected scope references figure labels, but multiple images are available; please specify the target figure filename.",
                target=labels[0],
                target_type="image",
                details={"figure_labels": labels, "available_images": [item.get("path", "") for item in images[:8]]},
            )

        matched: List[str] = []
        for image_raw in image_path_candidates:
            normalized = self._normalize_text(image_raw)
            normalized_stem = self._normalize_text(str(image_raw).split("/")[-1].split(".")[0])
            for item in images:
                item_path = self._normalize_text(str(item.get("path", "")))
                item_name = self._normalize_text(str(item.get("name", "")))
                item_stem = self._normalize_text(str(item.get("stem", "")))
                if normalized and (normalized in item_path or normalized in item_name):
                    matched.append(str(item.get("path", "")))
                    continue
                if normalized_stem and normalized_stem in {item_stem, item_name}:
                    matched.append(str(item.get("path", "")))

        unique_matched = []
        seen = set()
        for value in matched:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_matched.append(normalized)

        if len(unique_matched) == 1:
            return unique_matched[0], None
        if len(unique_matched) > 1:
            return "", SystemExceptionPayload(
                code="unsupported_request",
                message="the selected scope maps to multiple figure images; please specify which figure image to analyze.",
                target_type="image",
                details={"resolved_images": unique_matched[:8], "figure_labels": labels},
            )
        return "", SystemExceptionPayload(
            code="image_missing",
            message="the selected figure label was resolved, but its image file is missing in the workspace.",
            target_type="image",
            details={"requested_image_paths": image_path_candidates[:8], "figure_labels": labels},
        )

    def _resolve_table_request(
        self,
        request: RouterRequest,
    ) -> tuple[Optional[SystemExceptionPayload], str]:
        scope_text = str(request.scope.text or "") if request.scope is not None else ""
        full_text = str(request.document_text or "")
        table_labels = self._extract_table_labels(scope_text)
        if table_labels:
            for label in table_labels:
                if self._document_has_table_label(full_text, label):
                    return None, label
            return (
                SystemExceptionPayload(
                    code="table_missing",
                    message="the referenced table could not be resolved from the current document.",
                    target=table_labels[0],
                    target_type="table",
                    details={"requested_text": request.user_text, "table_labels": table_labels},
                ),
                "",
            )

        table_count = self._count_table_resources(full_text)
        if table_count <= 0:
            return (
                SystemExceptionPayload(
                    code="table_missing",
                    message="no table resource is available in the current document for this request.",
                    target_type="table",
                    details={"requested_text": request.user_text},
                ),
                "",
            )
        if table_count > 1:
            return (
                SystemExceptionPayload(
                    code="unsupported_request",
                    message="multiple table resources are available, so please mention the target table explicitly.",
                    target_type="table",
                    details={"requested_text": request.user_text},
                ),
                "",
            )
        return None, "__single__"

    def _match_image_target(
        self,
        user_text: str,
        images: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        lowered = self._normalize_text(user_text)
        for item in images:
            candidates = {
                self._normalize_text(str(item.get("path", ""))),
                self._normalize_text(str(item.get("name", ""))),
                self._normalize_text(str(item.get("stem", ""))),
            }
            candidates = {candidate for candidate in candidates if candidate}
            if any(candidate and candidate in lowered for candidate in candidates):
                return item
        return None

    def _extract_table_labels(self, text: str) -> List[str]:
        labels = re.findall(r"\\(?:ref|autoref|cref|Cref)\{([^}]+)\}", text)
        return [label.strip() for label in labels if str(label).strip().startswith("tab:")]

    def _extract_figure_labels(self, text: str) -> List[str]:
        labels = [m.group(1).strip() for m in self.FIGURE_REF_PATTERN.finditer(str(text or ""))]
        return [label for label in labels if label.startswith("fig:")]

    def _resolve_figure_image_paths_by_label(self, full_text: str, label: str) -> List[str]:
        paths: List[str] = []
        for block_match in self.FIGURE_ENV_PATTERN.finditer(str(full_text or "")):
            block = block_match.group(0)
            labels = [m.group(1).strip() for m in self.FIGURE_LABEL_PATTERN.finditer(block)]
            if label not in labels:
                continue
            for include_match in self.INCLUDEGRAPHICS_PATTERN.finditer(block):
                raw = str(include_match.group(1) or "").strip()
                if raw:
                    paths.append(raw)
        deduped: List[str] = []
        seen = set()
        for value in paths:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    @staticmethod
    def _document_has_table_label(full_text: str, label: str) -> bool:
        pattern = re.compile(
            rf"\\begin\{{table\}}[\s\S]*?\\label\{{{re.escape(label)}\}}[\s\S]*?\\end\{{table\}}",
            re.IGNORECASE,
        )
        if pattern.search(full_text):
            return True
        return re.search(rf"\\label\{{{re.escape(label)}\}}", full_text) is not None

    @staticmethod
    def _count_table_resources(full_text: str) -> int:
        table_env_count = len(re.findall(r"\\begin\{table\}", full_text, flags=re.IGNORECASE))
        if table_env_count > 0:
            return table_env_count
        return len(re.findall(r"\\begin\{tabular\}", full_text, flags=re.IGNORECASE))

    def _selection_has_table_signal(self, scope_text: str) -> bool:
        text = str(scope_text or "")
        return bool(
            self.TABLE_REF_CMD_PATTERN.search(text) is not None
            or self.TABLE_NATURAL_REF_PATTERN.search(text) is not None
        )

    def _selection_has_figure_signal(self, scope_text: str) -> bool:
        text = str(scope_text or "")
        return bool(
            self.FIGURE_REF_CMD_PATTERN.search(text) is not None
            or self.FIGURE_NATURAL_REF_PATTERN.search(text) is not None
        )

    def _selection_has_citation_signal(self, scope_text: str) -> bool:
        text = str(scope_text or "")
        return self.CITATION_CMD_PATTERN.search(text) is not None

    def _blocked_for_missing_document(self, role: str, action: str) -> BlockedResponse:
        if action == "run_checks":
            messages = {
                "Reviewer": "Before running checks, please provide manuscript text and confirm the span to review.",
                "Advisor": "Before running checks, please provide manuscript text and confirm the span to advise on.",
                "Editor": "Before running checks, please provide manuscript text and confirm the span to edit.",
            }
        else:
            messages = {
                "Reviewer": "Please provide manuscript text and confirm the passage you want me to review first.",
                "Advisor": "Please provide manuscript text and confirm the passage you want me to advise on first.",
                "Editor": "Please provide manuscript text and confirm the passage you want me to edit first.",
            }
        return BlockedResponse(reason="no_document", message=messages.get(role, messages["Reviewer"]), role=role)

    def _blocked_for_missing_selection(self, role: str, action: str) -> BlockedResponse:
        if action == "run_checks":
            messages = {
                "Reviewer": "Before running checks, please confirm the text span you want me to review.",
                "Advisor": "Before running checks, please confirm the text span you want me to advise on.",
                "Editor": "Before running checks, please confirm the text span you want me to edit.",
            }
        else:
            messages = {
                "Reviewer": "Please confirm the text span you want me to review first.",
                "Advisor": "Please confirm the text span you want me to advise on first.",
                "Editor": "Please confirm the text span you want me to edit first.",
            }
        return BlockedResponse(
            reason="no_confirmed_selection",
            message=messages.get(role, messages["Reviewer"]),
            role=role,
        )

    def _ordered_unique_checks(
        self,
        *groups: Iterable[RequestedCheckName],
    ) -> List[RequestedCheckName]:
        merged: List[RequestedCheckName] = []
        for check_name in CHECK_ORDER:
            for group in groups:
                if check_name in group and check_name not in merged:
                    merged.append(check_name)
                    break
        return merged

    def _extract_requested_checks(self, user_text: str) -> List[RequestedCheckName]:
        lowered = self._normalize_text(user_text)
        found: List[RequestedCheckName] = []
        for check_name in CHECK_ORDER:
            for keyword in self.CHECK_KEYWORDS[check_name]:
                if self._keyword_matches(lowered, keyword):
                    found.append(check_name)
                    break
        return self._ordered_unique_checks(found)

    def _looks_like_check_request(
        self,
        user_text: str,
        parsed_checks: List[RequestedCheckName],
    ) -> bool:
        if not parsed_checks:
            return False
        lowered = self._normalize_text(user_text)
        if any(self._keyword_matches(lowered, keyword) for keyword in self.CHECK_INTENT_KEYWORDS):
            return True
        if any(marker in lowered for marker in ["有没有", "有无", "是否"]):
            return True
        return False

    def _looks_like_image_related_request(self, user_text: str) -> bool:
        lowered = self._normalize_text(user_text)
        return any(self._keyword_matches(lowered, keyword) for keyword in self.IMAGE_INTENT_KEYWORDS)

    def _looks_like_image_numeric_request(self, user_text: str) -> bool:
        lowered = self._normalize_text(user_text)
        return any(self._keyword_matches(lowered, keyword) for keyword in self.IMAGE_NUMERIC_HINT_KEYWORDS)

    def _looks_like_table_related_request(self, user_text: str) -> bool:
        lowered = self._normalize_text(user_text)
        return any(self._keyword_matches(lowered, keyword) for keyword in self.TABLE_INTENT_KEYWORDS)

    def _looks_like_rewrite_request(self, user_text: str) -> bool:
        lowered = self._normalize_text(user_text)
        return any(self._keyword_matches(lowered, keyword) for keyword in self.REWRITE_INTENT_KEYWORDS)

    def _consent_metadata(self, user_text: str) -> Dict[str, Any]:
        lowered = self._normalize_text(user_text)
        return {
            "user_text": user_text,
            "affirmative": any(self._keyword_matches(lowered, token) for token in self.AFFIRMATIVE_KEYWORDS),
            "negative": any(self._keyword_matches(lowered, token) for token in self.NEGATIVE_KEYWORDS),
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    @staticmethod
    def _keyword_matches(text: str, keyword: str) -> bool:
        normalized_keyword = str(keyword or "").strip().lower()
        if not normalized_keyword:
            return False
        if re.fullmatch(r"[a-z0-9_:+.-]+", normalized_keyword):
            pattern = r"(?<![a-z0-9_])" + re.escape(normalized_keyword) + r"(?![a-z0-9_])"
            return re.search(pattern, text) is not None
        return normalized_keyword in text
