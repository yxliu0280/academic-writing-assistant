from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from core.schemas import AppState, Issue, Location
from tools.bib_utils import (
    detect_suspected_missing_citation_sentences,
    extract_cite_keys,
    parse_bibtex_keys,
)
from tools.citation_catalog_resolver import CitationCatalogResolver
from tools.citation_evidence_verifier import CitationEvidenceVerifier
from tools.text_utils import char_span_to_line_range


class CitationAgent(BaseAgent):
    name = "citation"

    _ENTRY_PATTERN = re.compile(
        r"@(\w+)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\s*\}",
        re.IGNORECASE,
    )
    _FIELD_PATTERN = re.compile(r"(\w+)\s*=\s*(\{[^{}]*\}|\"[^\"]*\"|[^,\n]+)", re.IGNORECASE)

    def __init__(self) -> None:
        timeout = float(os.getenv("CITATION_APP_TIMEOUT_SEC", "2.5"))
        self.enable_advanced = os.getenv("CITATION_APP_ADVANCED", "1").strip() != "0"
        self.max_advanced_cites = max(1, int(os.getenv("CITATION_APP_MAX_ADVANCED_CITES", "8")))
        self.catalog_resolver = CitationCatalogResolver(
            timeout_sec=timeout,
            use_semantic_scholar=False,
            max_candidates=6,
        )
        self.evidence_verifier = CitationEvidenceVerifier(
            resolver=self.catalog_resolver,
            timeout_sec=timeout,
            top_k=2,
            use_semantic_scholar=False,
        )

    def run(self, state: AppState) -> List[Issue]:
        text = state.current_text
        bib_content = state.uploaded_bib.content if state.uploaded_bib else ""

        allowlist = parse_bibtex_keys(bib_content)
        cite_blocks = extract_cite_keys(text)

        issues: List[Issue] = []

        for block in cite_blocks:
            for key in block["keys"]:
                if key not in allowlist:
                    start, end = int(block["start"]), int(block["end"])
                    issues.append(
                        Issue(
                            id=f"citation_missing_{key}_{start}",
                            type="citation_missing",
                            severity="high",
                            status="detected",
                            location=Location(
                                sentence_index=-1,
                                char_span=(start, end),
                                snippet=str(block["snippet"]),
                                line_range=char_span_to_line_range(text, start, end),
                            ),
                            evidence={
                                "missing_key": key,
                                "allowlist_size": len(allowlist),
                                "allowlist_preview": sorted(list(allowlist))[:20],
                            },
                            message=f"Citation key '{key}' is not found in uploaded BibTeX.",
                            provenance={"policy": "allowlist_only"},
                    )
                )

        if self.enable_advanced and bib_content.strip() and cite_blocks:
            entry_map = self._parse_bib_entries_with_raw(bib_content)
            issues.extend(
                self._advanced_citation_consistency_issues(
                    text=text,
                    cite_blocks=cite_blocks,
                    entry_map=entry_map,
                    allowlist=allowlist,
                )
            )

        for sent in detect_suspected_missing_citation_sentences(text):
            start, end = int(sent["start"]), int(sent["end"])
            issues.append(
                Issue(
                    id=f"citation_suspected_missing_{start}",
                    type="citation_suspected_missing",
                    severity="low",
                    status="detected",
                    location=Location(
                        sentence_index=int(sent["index"]),
                        char_span=(start, end),
                        snippet=str(sent["text"]),
                        line_range=char_span_to_line_range(text, start, end),
                    ),
                    evidence={
                        "reason": "trigger_terms_without_citation",
                        "allowlist_size": len(allowlist),
                    },
                    message="This sentence looks like related-work or factual claim but has no citation.",
                    provenance={"policy": "heuristic_missing_citation"},
                )
            )

        return issues

    def safe_citation_suggestion(
        self,
        key: str,
        sentence_start: int,
        sentence_end: int,
        snippet: str,
        allowlist: set[str],
        provenance: Optional[Dict[str, str]],
    ) -> Optional[Issue]:
        if key in allowlist and provenance and provenance.get("source"):
            return None

        return Issue(
            id=f"citation_unverified_suggestion_{sentence_start}",
            type="citation_unverified_suggestion",
            severity="medium",
            status="uncertain",
            location=Location(
                sentence_index=-1,
                char_span=(sentence_start, sentence_end),
                snippet=snippet,
            ),
            evidence={
                "candidate_key": key,
                "allowlisted": key in allowlist,
                "has_provenance": bool(provenance and provenance.get("source")),
            },
            message="Abstained from citation suggestion because key/provenance is unverified.",
            provenance={"policy": "must_have_allowlist_and_provenance"},
        )

    def _advanced_citation_consistency_issues(
        self,
        *,
        text: str,
        cite_blocks: List[Dict[str, object]],
        entry_map: Dict[str, Dict[str, Any]],
        allowlist: set[str],
    ) -> List[Issue]:
        issues: List[Issue] = []
        checked = 0

        for block in cite_blocks:
            if checked >= self.max_advanced_cites:
                break
            keys = [str(k).strip() for k in list(block.get("keys") or []) if str(k).strip()]
            start = int(block.get("start", -1))
            end = int(block.get("end", -1))
            snippet = str(block.get("snippet", ""))
            claim_text = self._claim_text_for_block(text=text, start=start, end=end, fallback=snippet)
            for key in keys:
                if checked >= self.max_advanced_cites:
                    break
                if key not in allowlist:
                    continue
                entry = entry_map.get(key)
                if not entry:
                    continue
                checked += 1
                issues.extend(
                    self._advanced_issue_for_key(
                        text=text,
                        key=key,
                        start=start,
                        end=end,
                        claim_text=claim_text,
                        entry=entry,
                    )
                )
        return issues

    def _advanced_issue_for_key(
        self,
        *,
        text: str,
        key: str,
        start: int,
        end: int,
        claim_text: str,
        entry: Dict[str, Any],
    ) -> List[Issue]:
        out: List[Issue] = []
        payload = {
            "sample_id": f"app_{key}_{max(0, start)}",
            "target_layer": "L1",
            "claim_text": claim_text,
            "citation_scope": "single",
            "primary_citation": {
                "cite_key": key,
                "local_bib_raw": str(entry.get("raw", "")),
                "local_title": str(entry.get("title", "")),
                "local_authors": list(entry.get("authors", [])),
                "local_year": entry.get("year"),
                "local_venue": str(entry.get("venue", "")),
                "local_doi": str(entry.get("doi", "")),
                "local_arxiv_id": str(entry.get("arxiv_id", "")),
            },
            "source": {
                "has_local_bib": True,
                "candidate_task_type": "app_interactive",
            },
        }
        try:
            l1 = self.catalog_resolver.resolve_from_payload(payload).to_dict()
        except Exception:
            return out

        l1_verdict = str(l1.get("verdict", ""))
        l1_reasons = [str(x).strip() for x in list(l1.get("reason_codes") or []) if str(x).strip()]
        if l1_verdict in {"invalid", "incomplete", "ambiguous"}:
            status = "detected" if l1_verdict == "invalid" else "uncertain"
            severity = "high" if l1_verdict == "invalid" else "medium"
            message = (
                f"Citation '{key}' catalog verification is {l1_verdict}. "
                f"Reason codes: {', '.join(l1_reasons) if l1_reasons else 'n/a'}."
            )
            out.append(
                Issue(
                    id=f"citation_unverified_suggestion_{key}_{max(0, start)}_l1",
                    type="citation_unverified_suggestion",
                    severity=severity,  # type: ignore[arg-type]
                    status=status,  # type: ignore[arg-type]
                    location=Location(
                        sentence_index=-1,
                        char_span=(max(0, start), max(max(0, start), end)),
                        snippet=claim_text,
                        line_range=char_span_to_line_range(text, max(0, start), max(max(0, start), end)),
                    ),
                    evidence={
                        "layer": "L1",
                        "cite_key": key,
                        "verdict": l1_verdict,
                        "reason_codes": l1_reasons,
                        "matched_source": str(l1.get("matched_source", "")),
                        "canonical_work_id": str(l1.get("canonical_work_id", "")),
                        "trace": {
                            "query_trace": list(l1.get("query_trace") or []),
                            "match_status": str(l1.get("match_status", "")),
                        },
                    },
                    message=message,
                    provenance={"policy": "catalog_resolution"},
                )
            )
            return out

        payload["target_layer"] = "L2"
        try:
            l2 = self.evidence_verifier.verify_from_payload(payload).to_dict()
        except Exception:
            return out
        l2_verdict = str(l2.get("verdict", ""))
        l2_reasons = [str(x).strip() for x in list(l2.get("reason_codes") or []) if str(x).strip()]
        if l2_verdict in {"contradicted", "unsupported"}:
            out.append(
                Issue(
                    id=f"citation_unverified_suggestion_{key}_{max(0, start)}_l2",
                    type="citation_unverified_suggestion",
                    severity="high",
                    status="detected",
                    location=Location(
                        sentence_index=-1,
                        char_span=(max(0, start), max(max(0, start), end)),
                        snippet=claim_text,
                        line_range=char_span_to_line_range(text, max(0, start), max(max(0, start), end)),
                    ),
                    evidence={
                        "layer": "L2",
                        "cite_key": key,
                        "verdict": l2_verdict,
                        "reason_codes": l2_reasons,
                        "evidence_doc_id": str(l2.get("evidence_doc_id", "")),
                        "evidence_spans": list(l2.get("evidence_spans") or [])[:2],
                        "decision_trace": dict(l2.get("decision_trace") or {}),
                    },
                    message=(
                        f"Citation '{key}' support verification is {l2_verdict}. "
                        f"Reason codes: {', '.join(l2_reasons) if l2_reasons else 'n/a'}."
                    ),
                    provenance={"policy": "evidence_grounding"},
                )
            )
        return out

    def _parse_bib_entries_with_raw(self, bib_content: str) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for match in self._ENTRY_PATTERN.finditer(bib_content or ""):
            key = str(match.group(2) or "").strip()
            if not key:
                continue
            body = str(match.group(3) or "")
            raw_entry = str(match.group(0) or "").strip()
            fields: Dict[str, str] = {}
            for field_match in self._FIELD_PATTERN.finditer(body):
                field_name = str(field_match.group(1) or "").strip().lower()
                raw_value = str(field_match.group(2) or "").strip().strip(",").strip()
                normalized = raw_value.strip("{}\"")
                if field_name:
                    fields[field_name] = normalized
            year: Optional[int] = None
            try:
                year_raw = str(fields.get("year", "")).strip()
                if year_raw:
                    year = int(re.sub(r"[^0-9]", "", year_raw)[:4])
            except Exception:
                year = None
            author_field = str(fields.get("author", "")).strip()
            authors = [a.strip() for a in re.split(r"\s+and\s+", author_field) if a.strip()]
            entries[key] = {
                "raw": raw_entry,
                "title": str(fields.get("title", "")).strip(),
                "authors": authors,
                "year": year,
                "venue": str(fields.get("journal", fields.get("booktitle", ""))).strip(),
                "doi": str(fields.get("doi", "")).strip(),
                "arxiv_id": str(fields.get("eprint", fields.get("archiveprefix", ""))).strip(),
            }
        return entries

    @staticmethod
    def _claim_text_for_block(text: str, start: int, end: int, fallback: str) -> str:
        if start < 0 or end <= start or not text:
            return fallback.strip()
        left = text.rfind(".", 0, start)
        right = text.find(".", end)
        if left == -1:
            left = 0
        else:
            left += 1
        if right == -1:
            right = len(text)
        snippet = text[left:right].strip()
        return snippet or fallback.strip()
