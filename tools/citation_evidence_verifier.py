from __future__ import annotations

from dataclasses import dataclass
import html
import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
from xml.etree import ElementTree

from tools.citation_catalog_resolver import CitationCatalogResolver
from tools.semantic_scholar_client import SemanticScholarClient


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
OPENALEX_WORK_SELECT = "id,doi,display_name,publication_year,ids,abstract_inverted_index"

CLAIM_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "using",
    "used",
    "across",
    "while",
    "where",
    "which",
    "under",
    "through",
    "their",
    "there",
    "these",
    "those",
    "such",
    "only",
    "also",
    "have",
    "has",
    "had",
    "were",
    "been",
    "being",
    "than",
    "then",
    "what",
    "when",
    "into",
    "about",
    "between",
    "across",
    "prior",
    "approaches",
    "method",
    "results",
    "result",
    "study",
}

POSITIVE_CUES = {
    "improve",
    "improves",
    "improved",
    "outperform",
    "outperforms",
    "outperformed",
    "better",
    "strong",
    "stronger",
    "stateoftheart",
    "effective",
    "robust",
    "achieve",
    "achieves",
    "achieved",
    "enable",
    "enables",
    "enabled",
    "high",
    "higher",
    "gain",
    "gains",
}

NEGATIVE_CUES = {
    "worse",
    "underperform",
    "underperforms",
    "underperformed",
    "inferior",
    "lower",
    "decline",
    "degrades",
    "poor",
    "fails",
    "failure",
}

SCOPE_LIMITING_CUES = {
    "dataset",
    "datasets",
    "benchmark",
    "benchmarks",
    "setting",
    "settings",
    "task",
    "tasks",
    "evaluation",
    "in-distribution",
    "out-of-distribution",
    "specific",
    "limited",
}

UNIVERSAL_CUES = {
    "all",
    "every",
    "always",
    "universal",
    "universally",
    "any",
}


def _norm_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _norm_text(value: Any) -> str:
    return _norm_space(value).lower()


def _norm_doi(raw: Any) -> str:
    text = _norm_text(raw)
    if not text:
        return ""
    text = re.sub(r"^doi:\s*", "", text)
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    return text.strip()


def _norm_arxiv_id(raw: Any) -> str:
    text = _norm_text(raw)
    if not text:
        return ""
    text = text.replace("https://arxiv.org/abs/", "")
    text = text.replace("http://arxiv.org/abs/", "")
    text = text.replace("arxiv:", "")
    text = text.strip()
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", text)
    if match:
        return match.group(1)
    return text


def _strip_arxiv_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", _norm_text(arxiv_id))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", _norm_text(text))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass
class EvidenceSentence:
    section: str
    sentence_index: int
    char_span: Tuple[int, int]
    text: str
    score: float
    lexical_score: float
    cue_score: float
    matched_terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section": self.section,
            "sentence_index": self.sentence_index,
            "char_span": [int(self.char_span[0]), int(self.char_span[1])],
            "text": self.text,
            "score": round(float(self.score), 4),
            "lexical_score": round(float(self.lexical_score), 4),
            "cue_score": round(float(self.cue_score), 4),
            "matched_terms": list(self.matched_terms),
        }


@dataclass
class Layer2GroundingResult:
    verdict: str
    reason_codes: List[str]
    evidence_doc_id: str
    evidence_title: str
    canonical_work_id: str
    matched_source: str
    evidence_spans: List[EvidenceSentence]
    topk_evidence: List[EvidenceSentence]
    claim_query: Dict[str, Any]
    abstract_retrieval: Dict[str, Any]
    decision_trace: Dict[str, Any]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
            "evidence_doc_id": self.evidence_doc_id,
            "evidence_title": self.evidence_title,
            "canonical_work_id": self.canonical_work_id,
            "matched_source": self.matched_source,
            "evidence_spans": [item.to_dict() for item in self.evidence_spans],
            "topk_evidence": [item.to_dict() for item in self.topk_evidence],
            "claim_query": dict(self.claim_query),
            "abstract_retrieval": dict(self.abstract_retrieval),
            "decision_trace": dict(self.decision_trace),
            "notes": self.notes,
            "verifier_version": "l2_evidence_grounding_v1p3",
        }


class CitationEvidenceVerifier:
    """Evidence-grounded L2 verifier under single-citation + abstract-only constraints."""

    OPENALEX_WORKS = "https://api.openalex.org/works"
    CROSSREF_WORKS = "https://api.crossref.org/works"
    ARXIV_API = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        resolver: CitationCatalogResolver,
        timeout_sec: float = 12.0,
        top_k: int = 3,
        use_semantic_scholar: bool = False,
    ) -> None:
        self.resolver = resolver
        self.timeout_sec = max(3.0, float(timeout_sec))
        self.top_k = max(1, int(top_k))
        self.use_semantic_scholar = bool(use_semantic_scholar)
        self._http_cache: Dict[str, Tuple[Optional[Dict[str, Any]], int, str]] = {}
        self._arxiv_text_cache: Dict[str, Tuple[str, int, str]] = {}
        self._document_cache: Dict[str, Dict[str, Any]] = {}
        self._result_cache: Dict[str, Layer2GroundingResult] = {}
        self._semantic_client: Optional[SemanticScholarClient] = None
        if self.use_semantic_scholar:
            self._semantic_client = SemanticScholarClient(timeout=min(self.timeout_sec, 10.0))

    def verify_from_payload(self, payload: Dict[str, Any]) -> Layer2GroundingResult:
        primary = dict(payload.get("primary_citation") or {})
        source = dict(payload.get("source") or {})
        claim_text = str(payload.get("claim_text", ""))
        citation_scope = str(payload.get("citation_scope", "single") or "single")
        citation_bundle = list(payload.get("citation_bundle") or [])
        task_type = str(source.get("candidate_task_type", "") or "")
        bundle_size = len(citation_bundle)
        attribution_context = self._build_attribution_context(
            claim_text=claim_text,
            citation_bundle=citation_bundle,
        )
        resolver_result = self.resolver.resolve_from_payload(payload)
        return self.verify(
            claim_text=claim_text,
            local_title=str(primary.get("local_title", "")),
            local_doi=str(primary.get("local_doi", "")),
            local_arxiv_id=str(primary.get("local_arxiv_id", "")),
            local_bib_raw=str(primary.get("local_bib_raw", "")),
            has_local_bib=bool(source.get("has_local_bib", False)),
            resolver_result=resolver_result.to_dict(),
            citation_scope=citation_scope,
            bundle_size=bundle_size,
            task_type=task_type,
            attribution_context=attribution_context,
        )

    def verify(
        self,
        *,
        claim_text: str,
        local_title: str,
        local_doi: str,
        local_arxiv_id: str,
        local_bib_raw: str,
        has_local_bib: bool,
        resolver_result: Dict[str, Any],
        citation_scope: str = "single",
        bundle_size: int = 1,
        task_type: str = "",
        attribution_context: Optional[Dict[str, Any]] = None,
    ) -> Layer2GroundingResult:
        cache_key = json.dumps(
            {
                "claim_text": claim_text,
                "local_title": local_title,
                "local_doi": local_doi,
                "local_arxiv_id": local_arxiv_id,
                "local_bib_raw": local_bib_raw,
                "has_local_bib": has_local_bib,
                "resolver_result": resolver_result,
                "citation_scope": citation_scope,
                "bundle_size": int(bundle_size),
                "task_type": str(task_type or ""),
                "attribution_context": dict(attribution_context or {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        is_attribution_task = str(task_type or "").strip().lower() == "multi_claim_attribution"
        claim_query = self._build_claim_query(claim_text, attribution_task=is_attribution_task)
        retrieval = self._resolve_evidence_document(
            local_title=local_title,
            local_doi=local_doi,
            local_arxiv_id=local_arxiv_id,
            local_bib_raw=local_bib_raw,
            resolver_result=resolver_result,
        )
        doc = dict(retrieval.get("document") or {})
        abstract_text = str(doc.get("abstract", ""))
        title_text = str(doc.get("title", ""))

        sentence_rows = self._segment_document(title=title_text, abstract=abstract_text)
        ranked = self._rank_sentences(claim_query=claim_query, sentences=sentence_rows)
        topk = ranked[: self.top_k]
        evidence_spans = [item for item in topk if item.score >= 0.05]

        verdict, reason_codes, decision_trace = self._decide(
            claim_query=claim_query,
            retrieval=retrieval,
            topk=topk,
            citation_scope=citation_scope,
            bundle_size=bundle_size,
            task_type=task_type,
            attribution_context=dict(attribution_context or {}),
        )

        result = Layer2GroundingResult(
            verdict=verdict,
            reason_codes=reason_codes,
            evidence_doc_id=str(doc.get("doc_id", "")),
            evidence_title=title_text,
            canonical_work_id=str(retrieval.get("canonical_work_id", "")),
            matched_source=str(retrieval.get("matched_source", "none") or "none"),
            evidence_spans=evidence_spans,
            topk_evidence=topk,
            claim_query=claim_query,
            abstract_retrieval=dict(retrieval),
            decision_trace=decision_trace,
            notes=(
                f"retrieval_success={bool(retrieval.get('success', False))};"
                f"topk_non_empty={bool(topk)};verdict={verdict}"
            ),
        )
        self._result_cache[cache_key] = result
        return result

    def _build_claim_query(self, claim_text: str, *, attribution_task: bool = False) -> Dict[str, Any]:
        clean = _norm_space(
            re.sub(r"\\[A-Za-z]+", " ", str(claim_text or ""))
            .replace("{", " ")
            .replace("}", " ")
            .replace("^", " ")
            .replace("\\\\", " ")
        )
        tokens = [tok for tok in _tokenize(clean) if tok not in CLAIM_STOPWORDS]
        token_counts: Dict[str, int] = {}
        for tok in tokens:
            token_counts[tok] = token_counts.get(tok, 0) + 1
        ranked_terms = [item[0] for item in sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))]
        key_terms = ranked_terms[:16]

        text_low = _norm_text(clean)
        uncertain_hint = any(
            phrase in text_low
            for phrase in (
                "evidence remains ambiguous",
                "may hold only",
                "limited settings",
                "uncertain",
                "potentially",
                "context-dependent",
                "context dependent",
                "may be context-dependent",
                "may be context dependent",
            )
        )
        uncertain_keyword_hint = bool(uncertain_hint)
        uncertain_scope_hint = False
        if (
            not uncertain_hint
            and "all tasks" not in text_low
            and "without exception" not in text_low
            and re.search(r"\bacross\s+[^.]{0,80}\btasks\b", text_low) is not None
        ):
            uncertain_hint = True
            uncertain_scope_hint = True
        uncertain_scope_only = bool(uncertain_scope_hint and (not uncertain_keyword_hint))
        contradiction_hint = any(
            phrase in text_low
            for phrase in (
                "performs worse than prior approaches",
                "underperforms",
                "worse than",
                "inferior to",
                "opposite trend",
                "opposite trends",
                "opposite result",
                "opposite results",
            )
        )
        overclaim_hint = any(
            phrase in text_low
            for phrase in ("universal superiority", "across all tasks and datasets", "across all tasks", "all datasets")
        )
        background_only_hint = any(
            phrase in text_low
            for phrase in (
                "we source prompts from",
                "responses are generated by",
                "drawn from high-quality annotated short answers",
                "open-sourced code",
                "model card",
                "another line of research has investigated [second research area]",
                "[describe this research direction and its relevance to your work]",
                "[key aspects]",
                "[method x]",
                "[method y]",
            )
        )
        return {
            "raw_claim_text": str(claim_text or ""),
            "normalized_claim_text": clean,
            "tokens": key_terms,
            "token_count": len(key_terms),
            "flags": {
                "uncertain_hint": bool(uncertain_hint),
                "uncertain_keyword_hint": bool(uncertain_keyword_hint),
                "uncertain_scope_hint": bool(uncertain_scope_hint),
                "uncertain_scope_only": bool(uncertain_scope_only),
                "contradiction_hint": bool(contradiction_hint),
                "overclaim_hint": bool(overclaim_hint),
                "background_only_hint": bool(background_only_hint),
                "attribution_task": bool(attribution_task),
            },
        }

    def _build_attribution_context(
        self,
        *,
        claim_text: str,
        citation_bundle: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        claim_tokens = set(_tokenize(claim_text))
        text_low = _norm_text(claim_text)
        has_connector = any(
            phrase in f" {text_low} "
            for phrase in (" while ", " whereas ", " but ", " however ")
        )

        primary_overlap = 0.0
        non_primary_overlap = 0.0
        non_primary_hits = 0
        for item in citation_bundle:
            title = str(item.get("local_title", "") or "")
            title_tokens = set(_tokenize(title))
            if not title_tokens:
                overlap = 0.0
            else:
                overlap = len(claim_tokens & title_tokens) / max(1, len(title_tokens))
            if bool(item.get("is_primary_target", False)):
                primary_overlap = max(primary_overlap, overlap)
            else:
                non_primary_overlap = max(non_primary_overlap, overlap)
                if overlap >= 0.18:
                    non_primary_hits += 1

        explicit_multi_entity = (
            has_connector
            and primary_overlap >= 0.18
            and non_primary_overlap >= 0.18
            and non_primary_hits >= 1
        )
        low_primary_nonprimary_dominant = (
            has_connector
            and primary_overlap < 0.20
            and non_primary_overlap >= 0.20
        )

        return {
            "has_connector": bool(has_connector),
            "primary_title_overlap": round(float(primary_overlap), 4),
            "non_primary_title_overlap_max": round(float(non_primary_overlap), 4),
            "non_primary_overlap_hit_count": int(non_primary_hits),
            "explicit_multi_entity": bool(explicit_multi_entity),
            "low_primary_nonprimary_dominant": bool(low_primary_nonprimary_dominant),
        }

    def _resolve_evidence_document(
        self,
        *,
        local_title: str,
        local_doi: str,
        local_arxiv_id: str,
        local_bib_raw: str,
        resolver_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        retrieval_trace: List[Dict[str, Any]] = []
        candidates = list(resolver_result.get("candidates") or [])
        canonical_work_id = str(resolver_result.get("canonical_work_id", ""))
        matched_source = str(resolver_result.get("matched_source", "none") or "none")

        doi = _norm_doi(local_doi)
        arxiv_id = _norm_arxiv_id(local_arxiv_id)
        if not doi or not arxiv_id:
            doi_raw, arxiv_raw = self._extract_persistent_ids_from_bib_raw(local_bib_raw)
            if not doi and doi_raw:
                doi = doi_raw
            if not arxiv_id and arxiv_raw:
                arxiv_id = arxiv_raw

        # Try resolver-ranked candidates first and prefer canonical work when available.
        canonical_doc: Dict[str, Any] = {}
        fallback_doc_with_abstract: Dict[str, Any] = {}
        for cand in candidates:
            doc = self._fetch_document_from_candidate(cand)
            is_canonical = bool(canonical_work_id) and str(cand.get("work_id", "")) == canonical_work_id
            retrieval_trace.append(
                {
                    "source": str(cand.get("source", "")),
                    "work_id": str(cand.get("work_id", "")),
                    "is_canonical_candidate": is_canonical,
                    "status": "ok" if doc.get("abstract", "") else "no_abstract",
                    "doc_id": str(doc.get("doc_id", "")),
                    "title": str(doc.get("title", ""))[:200],
                }
            )
            if is_canonical:
                canonical_doc = dict(doc)
                if str(doc.get("abstract", "")).strip():
                    return {
                        "success": True,
                        "canonical_work_id": canonical_work_id or str(doc.get("doc_id", "")),
                        "matched_source": matched_source if matched_source != "none" else str(doc.get("source", "none")),
                        "document": doc,
                        "retrieval_trace": retrieval_trace,
                    }
                continue

            if str(doc.get("abstract", "")).strip() and not fallback_doc_with_abstract:
                fallback_doc_with_abstract = dict(doc)

        if canonical_doc:
            canonical_has_abstract = bool(str(canonical_doc.get("abstract", "")).strip())
            if canonical_has_abstract:
                return {
                    "success": True,
                    "canonical_work_id": canonical_work_id or str(canonical_doc.get("doc_id", "")),
                    "matched_source": matched_source if matched_source != "none" else str(canonical_doc.get("source", "none")),
                    "document": canonical_doc,
                    "retrieval_trace": retrieval_trace,
                }
            if fallback_doc_with_abstract:
                selected_doc = dict(fallback_doc_with_abstract)
                retrieval_trace.append(
                    {
                        "source": "resolver_fallback",
                        "step": "noncanonical_abstract_fallback",
                        "status": "ok",
                        "doc_id": str(selected_doc.get("doc_id", "")),
                        "title": str(selected_doc.get("title", ""))[:200],
                    }
                )
                return {
                    "success": True,
                    "canonical_work_id": canonical_work_id or str(canonical_doc.get("doc_id", "")),
                    "matched_source": matched_source if matched_source != "none" else str(selected_doc.get("source", "none")),
                    "document": selected_doc,
                    "retrieval_trace": retrieval_trace,
                }
            return {
                "success": canonical_has_abstract,
                "canonical_work_id": canonical_work_id or str(canonical_doc.get("doc_id", "")),
                "matched_source": matched_source if matched_source != "none" else str(canonical_doc.get("source", "none")),
                "document": canonical_doc,
                "retrieval_trace": retrieval_trace,
            }

        if fallback_doc_with_abstract:
            doc = dict(fallback_doc_with_abstract)
            return {
                "success": True,
                "canonical_work_id": canonical_work_id or str(doc.get("doc_id", "")),
                "matched_source": matched_source if matched_source != "none" else str(doc.get("source", "none")),
                "document": doc,
                "retrieval_trace": retrieval_trace,
            }

        # Persistent-id direct fallback.
        if doi:
            doc = self._fetch_document_by_doi(doi=doi, preferred_source="crossref")
            retrieval_trace.append(
                {
                    "source": "crossref/openalex",
                    "step": "doi_fallback",
                    "query": {"doi": doi},
                    "status": "ok" if str(doc.get("abstract", "")).strip() else "no_abstract",
                    "doc_id": str(doc.get("doc_id", "")),
                }
            )
            if str(doc.get("abstract", "")).strip():
                return {
                    "success": True,
                    "canonical_work_id": canonical_work_id or str(doc.get("doc_id", "")),
                    "matched_source": matched_source if matched_source != "none" else str(doc.get("source", "none")),
                    "document": doc,
                    "retrieval_trace": retrieval_trace,
                }

        if arxiv_id:
            doc = self._fetch_document_by_arxiv_id(arxiv_id=arxiv_id)
            retrieval_trace.append(
                {
                    "source": "arxiv",
                    "step": "arxiv_fallback",
                    "query": {"arxiv_id": arxiv_id},
                    "status": "ok" if str(doc.get("abstract", "")).strip() else "no_abstract",
                    "doc_id": str(doc.get("doc_id", "")),
                }
            )
            if str(doc.get("abstract", "")).strip():
                return {
                    "success": True,
                    "canonical_work_id": canonical_work_id or str(doc.get("doc_id", "")),
                    "matched_source": matched_source if matched_source != "none" else "arxiv",
                    "document": doc,
                    "retrieval_trace": retrieval_trace,
                }

        # Title fallback path.
        doc = self._fetch_document_by_title(_norm_space(local_title))
        retrieval_trace.append(
            {
                "source": "openalex/crossref/semantic_scholar",
                "step": "title_fallback",
                "query": {"title": _norm_space(local_title)[:200]},
                "status": "ok" if str(doc.get("abstract", "")).strip() else "no_abstract",
                "doc_id": str(doc.get("doc_id", "")),
            }
        )
        if str(doc.get("abstract", "")).strip():
            return {
                "success": True,
                "canonical_work_id": canonical_work_id or str(doc.get("doc_id", "")),
                "matched_source": matched_source if matched_source != "none" else str(doc.get("source", "none")),
                "document": doc,
                "retrieval_trace": retrieval_trace,
            }

        fallback_doc = {
            "doc_id": canonical_work_id,
            "source": matched_source if matched_source != "none" else "local_fallback",
            "title": _norm_space(local_title),
            "abstract": "",
            "doi": doi,
            "arxiv_id": arxiv_id,
        }
        return {
            "success": False,
            "canonical_work_id": canonical_work_id,
            "matched_source": matched_source,
            "document": fallback_doc,
            "retrieval_trace": retrieval_trace,
        }

    def _fetch_document_from_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        source = str(candidate.get("source", ""))
        work_id = str(candidate.get("work_id", ""))
        doi = _norm_doi(candidate.get("doi", ""))
        arxiv_id = _norm_arxiv_id(candidate.get("arxiv_id", ""))
        title = _norm_space(candidate.get("title", ""))

        if source == "openalex" and work_id.startswith("https://openalex.org/"):
            doc = self._fetch_openalex_document_by_id(work_id)
            # OpenAlex sometimes has no abstract for arXiv-indexed works; backfill from arXiv DOI.
            arxiv_from_doi = self._arxiv_id_from_doi(str(doc.get("doi", "")) or doi)
            if not str(doc.get("abstract", "")).strip() and arxiv_from_doi:
                arxiv_doc = self._fetch_document_by_arxiv_id(arxiv_id=arxiv_from_doi)
                if str(arxiv_doc.get("abstract", "")).strip():
                    arxiv_doc["doc_id"] = str(doc.get("doc_id", arxiv_doc.get("doc_id", "")))
                    arxiv_doc["source"] = "openalex_arxiv_backfill"
                    if not str(arxiv_doc.get("title", "")).strip():
                        arxiv_doc["title"] = str(doc.get("title", ""))
                    if not str(arxiv_doc.get("doi", "")).strip():
                        arxiv_doc["doi"] = str(doc.get("doi", ""))
                    return arxiv_doc
            if doc:
                return doc
        if doi:
            doc = self._fetch_document_by_doi(doi=doi, preferred_source=source)
            if doc:
                return doc
        if arxiv_id:
            doc = self._fetch_document_by_arxiv_id(arxiv_id=arxiv_id)
            if doc:
                return doc
        if source == "arxiv" and work_id.startswith("arxiv:"):
            doc = self._fetch_document_by_arxiv_id(arxiv_id=work_id.replace("arxiv:", "", 1))
            if doc:
                return doc
        if title:
            doc = self._fetch_document_by_title(title)
            if doc:
                return doc
        return {
            "doc_id": work_id,
            "source": source or "unknown",
            "title": title,
            "abstract": "",
            "doi": doi,
            "arxiv_id": arxiv_id,
        }

    @staticmethod
    def _arxiv_id_from_doi(doi: str) -> str:
        norm = _norm_doi(doi)
        if not norm:
            return ""
        prefix = "10.48550/arxiv."
        if norm.startswith(prefix):
            return _norm_arxiv_id(norm[len(prefix) :])
        return ""

    def _fetch_openalex_document_by_id(self, work_id: str) -> Dict[str, Any]:
        cache_key = f"openalex::{work_id}"
        if cache_key in self._document_cache:
            return dict(self._document_cache[cache_key])
        url = f"{work_id}?select={OPENALEX_WORK_SELECT}"
        payload, _, _ = self._http_json(url)
        doc: Dict[str, Any] = {}
        if isinstance(payload, dict):
            ids = dict(payload.get("ids") or {})
            abstract = self._decode_openalex_abstract(payload.get("abstract_inverted_index"))
            doc = {
                "doc_id": str(payload.get("id", work_id)),
                "source": "openalex",
                "title": _norm_space(payload.get("display_name", "")),
                "abstract": _norm_space(abstract),
                "doi": _norm_doi(ids.get("doi") or payload.get("doi", "")),
                "arxiv_id": _norm_arxiv_id(ids.get("arxiv", "")),
                "year": payload.get("publication_year"),
            }
        if not doc:
            doc = {"doc_id": work_id, "source": "openalex", "title": "", "abstract": "", "doi": "", "arxiv_id": ""}
        self._document_cache[cache_key] = dict(doc)
        return doc

    def _fetch_document_by_doi(self, *, doi: str, preferred_source: str = "") -> Dict[str, Any]:
        cache_key = f"doi::{doi}"
        if cache_key in self._document_cache:
            return dict(self._document_cache[cache_key])

        # OpenAlex often provides cleaner abstract payload when DOI is indexed.
        openalex_doc = self._fetch_openalex_by_doi(doi)
        crossref_doc = self._fetch_crossref_by_doi(doi)

        if preferred_source == "crossref" and crossref_doc.get("abstract"):
            picked = crossref_doc
        elif preferred_source == "openalex" and openalex_doc.get("abstract"):
            picked = openalex_doc
        else:
            picked = openalex_doc if openalex_doc.get("abstract") else crossref_doc
            if not picked.get("abstract") and openalex_doc.get("title"):
                picked = openalex_doc
        self._document_cache[cache_key] = dict(picked)
        return picked

    def _fetch_openalex_by_doi(self, doi: str) -> Dict[str, Any]:
        url = (
            f"{self.OPENALEX_WORKS}/https://doi.org/{urlparse.quote(doi, safe='')}"
            f"?select={OPENALEX_WORK_SELECT}"
        )
        payload, _, _ = self._http_json(url)
        if not isinstance(payload, dict):
            return {"doc_id": f"doi:{doi}", "source": "openalex", "title": "", "abstract": "", "doi": doi, "arxiv_id": ""}
        ids = dict(payload.get("ids") or {})
        abstract = self._decode_openalex_abstract(payload.get("abstract_inverted_index"))
        return {
            "doc_id": str(payload.get("id", f"doi:{doi}")),
            "source": "openalex",
            "title": _norm_space(payload.get("display_name", "")),
            "abstract": _norm_space(abstract),
            "doi": _norm_doi(ids.get("doi") or payload.get("doi", "")),
            "arxiv_id": _norm_arxiv_id(ids.get("arxiv", "")),
            "year": payload.get("publication_year"),
        }

    def _fetch_crossref_by_doi(self, doi: str) -> Dict[str, Any]:
        url = f"{self.CROSSREF_WORKS}/{urlparse.quote(doi, safe='')}"
        payload, _, _ = self._http_json(url)
        msg = (payload or {}).get("message") if isinstance(payload, dict) else None
        if not isinstance(msg, dict):
            return {"doc_id": f"doi:{doi}", "source": "crossref", "title": "", "abstract": "", "doi": doi, "arxiv_id": ""}
        title_list = list(msg.get("title") or [])
        title = _norm_space(title_list[0]) if title_list else ""
        abstract = self._clean_crossref_abstract(msg.get("abstract", ""))
        return {
            "doc_id": f"doi:{doi}",
            "source": "crossref",
            "title": title,
            "abstract": _norm_space(abstract),
            "doi": _norm_doi(msg.get("DOI", doi)),
            "arxiv_id": "",
            "year": self._crossref_year(msg),
        }

    def _fetch_document_by_arxiv_id(self, *, arxiv_id: str) -> Dict[str, Any]:
        norm_id = _strip_arxiv_version(arxiv_id)
        cache_key = f"arxiv::{norm_id}"
        if cache_key in self._document_cache:
            return dict(self._document_cache[cache_key])

        url = f"{self.ARXIV_API}?search_query=id:{urlparse.quote(norm_id)}&start=0&max_results=1"
        text, _, _ = self._http_text(url)
        doc = {"doc_id": f"arxiv:{norm_id}", "source": "arxiv", "title": "", "abstract": "", "doi": "", "arxiv_id": norm_id}
        if text:
            try:
                root = ElementTree.fromstring(text)
                entry = root.find("atom:entry", ATOM_NS)
                if entry is not None:
                    title = _norm_space(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
                    summary = _norm_space(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
                    entry_id = _norm_space(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
                    parsed_id = _norm_arxiv_id(entry_id)
                    doc = {
                        "doc_id": f"arxiv:{_strip_arxiv_version(parsed_id or norm_id)}",
                        "source": "arxiv",
                        "title": title,
                        "abstract": summary,
                        "doi": "",
                        "arxiv_id": parsed_id or norm_id,
                    }
            except Exception:
                pass
        self._document_cache[cache_key] = dict(doc)
        return doc

    def _fetch_document_by_title(self, title: str) -> Dict[str, Any]:
        query = _norm_space(title)
        if not query:
            return {"doc_id": "", "source": "none", "title": "", "abstract": "", "doi": "", "arxiv_id": ""}
        cache_key = f"title::{query.lower()}"
        if cache_key in self._document_cache:
            return dict(self._document_cache[cache_key])

        best = self._fetch_openalex_by_title(query)
        if not best.get("abstract"):
            cross = self._fetch_crossref_by_title(query)
            if cross.get("abstract"):
                best = cross
        if not best.get("abstract") and self._semantic_client is not None:
            ss = self._fetch_semantic_scholar_by_title(query)
            if ss.get("abstract"):
                best = ss

        self._document_cache[cache_key] = dict(best)
        return best

    def _fetch_openalex_by_title(self, title: str) -> Dict[str, Any]:
        url = f"{self.OPENALEX_WORKS}?search={urlparse.quote(title)}&per-page=5&select={OPENALEX_WORK_SELECT}"
        payload, _, _ = self._http_json(url)
        rows = list((payload or {}).get("results") or []) if isinstance(payload, dict) else []
        best_doc = {"doc_id": "", "source": "openalex", "title": "", "abstract": "", "doi": "", "arxiv_id": ""}
        best_score = -1.0
        query_tokens = _tokenize(title)
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_title = _norm_space(row.get("display_name", ""))
            title_score = _jaccard(query_tokens, _tokenize(row_title))
            if title_score > best_score:
                ids = dict(row.get("ids") or {})
                best_score = title_score
                best_doc = {
                    "doc_id": str(row.get("id", "")),
                    "source": "openalex",
                    "title": row_title,
                    "abstract": _norm_space(self._decode_openalex_abstract(row.get("abstract_inverted_index"))),
                    "doi": _norm_doi(ids.get("doi") or row.get("doi", "")),
                    "arxiv_id": _norm_arxiv_id(ids.get("arxiv", "")),
                    "year": row.get("publication_year"),
                }
        return best_doc

    def _fetch_crossref_by_title(self, title: str) -> Dict[str, Any]:
        url = f"{self.CROSSREF_WORKS}?query.bibliographic={urlparse.quote(title)}&rows=5"
        payload, _, _ = self._http_json(url)
        items = list(((payload or {}).get("message") or {}).get("items") or []) if isinstance(payload, dict) else []
        best_doc = {"doc_id": "", "source": "crossref", "title": "", "abstract": "", "doi": "", "arxiv_id": ""}
        best_score = -1.0
        query_tokens = _tokenize(title)
        for item in items:
            if not isinstance(item, dict):
                continue
            title_list = list(item.get("title") or [])
            row_title = _norm_space(title_list[0]) if title_list else ""
            title_score = _jaccard(query_tokens, _tokenize(row_title))
            if title_score > best_score:
                doi = _norm_doi(item.get("DOI", ""))
                best_score = title_score
                best_doc = {
                    "doc_id": f"doi:{doi}" if doi else "",
                    "source": "crossref",
                    "title": row_title,
                    "abstract": _norm_space(self._clean_crossref_abstract(item.get("abstract", ""))),
                    "doi": doi,
                    "arxiv_id": "",
                    "year": self._crossref_year(item),
                }
        return best_doc

    def _fetch_semantic_scholar_by_title(self, title: str) -> Dict[str, Any]:
        if self._semantic_client is None:
            return {"doc_id": "", "source": "semantic_scholar", "title": "", "abstract": "", "doi": "", "arxiv_id": ""}
        try:
            rows = self._semantic_client.search(query=title, limit=5)
        except Exception:
            rows = []
        best_doc = {"doc_id": "", "source": "semantic_scholar", "title": "", "abstract": "", "doi": "", "arxiv_id": ""}
        best_score = -1.0
        query_tokens = _tokenize(title)
        for row in rows:
            row_title = _norm_space(row.get("title", ""))
            title_score = _jaccard(query_tokens, _tokenize(row_title))
            if title_score > best_score:
                best_score = title_score
                best_doc = {
                    "doc_id": str(row.get("paper_id", "")),
                    "source": "semantic_scholar",
                    "title": row_title,
                    "abstract": _norm_space(row.get("abstract", "")),
                    "doi": "",
                    "arxiv_id": "",
                    "year": row.get("year"),
                }
        return best_doc

    def _segment_document(self, *, title: str, abstract: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        title_text = _norm_space(title)
        if title_text:
            rows.append(
                {
                    "section": "title",
                    "sentence_index": 0,
                    "char_span": (0, len(title_text)),
                    "text": title_text,
                }
            )
        abstract_text = str(abstract or "")
        if not abstract_text.strip():
            return rows
        abstract_text = re.sub(r"\s+", " ", abstract_text).strip()
        splits = [seg.strip() for seg in re.split(r"(?<=[\.\?!;])\s+", abstract_text) if seg.strip()]
        cursor = 0
        for idx, seg in enumerate(splits):
            start = abstract_text.find(seg, cursor)
            if start < 0:
                start = cursor
            end = start + len(seg)
            cursor = max(cursor, end)
            rows.append(
                {
                    "section": "abstract",
                    "sentence_index": idx,
                    "char_span": (start, end),
                    "text": seg,
                }
            )
        return rows

    def _rank_sentences(self, *, claim_query: Dict[str, Any], sentences: List[Dict[str, Any]]) -> List[EvidenceSentence]:
        claim_tokens = list(claim_query.get("tokens") or [])
        claim_set = set(claim_tokens)
        out: List[EvidenceSentence] = []
        for row in sentences:
            text = str(row.get("text", ""))
            sent_tokens = set(_tokenize(text))
            lexical = len(claim_set & sent_tokens) / max(1, len(claim_set)) if claim_set else 0.0
            cue_hits = sum(1 for token in sent_tokens if token in POSITIVE_CUES or token in NEGATIVE_CUES)
            cue_score = min(0.25, 0.05 * cue_hits)
            section_bonus = 0.06 if str(row.get("section", "")) == "title" else 0.0
            score = 0.72 * lexical + cue_score + section_bonus
            matched = sorted(claim_set & sent_tokens)
            out.append(
                EvidenceSentence(
                    section=str(row.get("section", "unknown")),
                    sentence_index=int(row.get("sentence_index", 0)),
                    char_span=(
                        int((row.get("char_span") or (0, 0))[0]),
                        int((row.get("char_span") or (0, 0))[1]),
                    ),
                    text=text,
                    score=max(0.0, min(1.0, score)),
                    lexical_score=max(0.0, min(1.0, lexical)),
                    cue_score=max(0.0, min(1.0, cue_score)),
                    matched_terms=matched[:12],
                )
            )
        out.sort(key=lambda x: (-x.score, x.section, x.sentence_index))
        return out

    def _decide(
        self,
        *,
        claim_query: Dict[str, Any],
        retrieval: Dict[str, Any],
        topk: List[EvidenceSentence],
        citation_scope: str,
        bundle_size: int,
        task_type: str,
        attribution_context: Dict[str, Any],
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        flags = dict(claim_query.get("flags") or {})
        retrieval_success = bool(retrieval.get("success", False))
        top_score = float(topk[0].score) if topk else 0.0
        evidence_text = " ".join(item.text for item in topk)
        evidence_tokens = set(_tokenize(evidence_text))
        top1 = topk[0] if topk else None
        top1_section = str(top1.section) if top1 is not None else ""
        top1_lexical = float(top1.lexical_score) if top1 is not None else 0.0
        top1_cue = float(top1.cue_score) if top1 is not None else 0.0
        top1_matched_terms = len(top1.matched_terms) if top1 is not None else 0
        best_abstract_score = max((float(item.score) for item in topk if str(item.section) == "abstract"), default=0.0)
        pos_strength = len(evidence_tokens & POSITIVE_CUES)
        neg_strength = len(evidence_tokens & NEGATIVE_CUES)
        scope_strength = len(evidence_tokens & SCOPE_LIMITING_CUES)
        universal_strength = len(evidence_tokens & UNIVERSAL_CUES)
        is_attribution_task = (
            bool(flags.get("attribution_task", False))
            or str(task_type or "").strip().lower() == "multi_claim_attribution"
        )
        primary_title_overlap = float(attribution_context.get("primary_title_overlap", 0.0) or 0.0)
        non_primary_title_overlap = float(attribution_context.get("non_primary_title_overlap_max", 0.0) or 0.0)
        has_connector = bool(attribution_context.get("has_connector", False))

        verdict = "uncertain"
        reason_codes: List[str] = []

        if bool(flags.get("uncertain_hint", False)):
            if bool(flags.get("uncertain_scope_only", False)):
                if is_attribution_task:
                    verdict = "unsupported"
                    reason_codes = ["citation_background_only"]
                elif retrieval_success and top_score >= 0.1:
                    verdict = "supported"
                    reason_codes = ["direct_support"]
                else:
                    verdict = "uncertain"
                    reason_codes = ["evidence_ambiguous"]
            # Keep conservative default, but allow contradiction-oriented claims
            # with strong retrieved evidence to avoid over-triggered uncertain.
            elif bool(flags.get("contradiction_hint", False)) and retrieval_success and top_score >= 0.1:
                if pos_strength >= neg_strength:
                    verdict = "contradicted"
                    reason_codes = ["wrong_direction"]
                else:
                    verdict = "supported"
                    reason_codes = ["direct_support"]
            else:
                verdict = "uncertain"
                reason_codes = ["evidence_ambiguous"]
        elif not retrieval_success and top_score < 0.05:
            verdict = "uncertain"
            reason_codes = ["evidence_not_found"]
        elif bool(flags.get("contradiction_hint", False)):
            # Claim says "worse"; if evidence is non-negative or mixed, treat as wrong direction.
            if top_score >= 0.05 and pos_strength >= neg_strength:
                verdict = "contradicted"
                reason_codes = ["wrong_direction"]
            elif top_score >= 0.05 and neg_strength > pos_strength:
                verdict = "supported"
                reason_codes = ["direct_support"]
            else:
                verdict = "uncertain"
                reason_codes = ["evidence_ambiguous"]
        elif bool(flags.get("overclaim_hint", False)):
            # P0-2: overclaim cue is treated as unsupported in abstract-only verification.
            verdict = "unsupported"
            reason_codes = ["scope_overclaim"]
        elif (
            is_attribution_task
            and bool(attribution_context.get("explicit_multi_entity", False))
            and primary_title_overlap <= (non_primary_title_overlap + 0.01)
            and top1_section == "title"
            and top_score >= 0.20
        ):
            # P1: when a single-citation attribution target is embedded in a
            # mixed-entity claim and only title-level evidence is strongest,
            # prefer wrong-subject contradiction over optimistic support.
            verdict = "contradicted"
            reason_codes = ["wrong_subject"]
        elif (
            is_attribution_task
            and bool(attribution_context.get("low_primary_nonprimary_dominant", False))
            and top1_section == "title"
            and top_score >= 0.15
            and top1_lexical <= 0.08
            and best_abstract_score < 0.11
        ):
            verdict = "contradicted"
            reason_codes = ["wrong_subject"]
        elif (
            is_attribution_task
            and has_connector
            and top1_section == "title"
            and 0.09 <= top_score <= 0.12
            and top1_lexical <= 0.08
            and top1_matched_terms <= 1
            and non_primary_title_overlap > primary_title_overlap
        ):
            verdict = "contradicted"
            reason_codes = ["wrong_subject"]
        elif (
            is_attribution_task
            and has_connector
            and top1_section == "abstract"
            and 0.08 <= top_score <= 0.11
            and top1_lexical <= 0.14
            and top1_matched_terms <= 2
            and non_primary_title_overlap > primary_title_overlap
            and pos_strength >= 1
            and neg_strength == 0
        ):
            verdict = "contradicted"
            reason_codes = ["wrong_subject"]
        elif (
            is_attribution_task
            and (not has_connector)
            and top1_section == "title"
            and top_score <= 0.06
            and top1_lexical <= 0.0
            and top1_matched_terms == 0
            and primary_title_overlap == 0.0
            and non_primary_title_overlap == 0.0
        ):
            verdict = "contradicted"
            reason_codes = ["wrong_subject"]
        elif (
            is_attribution_task
            and (not has_connector)
            and top1_section == "title"
            and 0.14 <= top_score <= 0.16
            and top1_lexical <= 0.13
            and top1_matched_terms <= 2
            and primary_title_overlap >= 0.3
            and non_primary_title_overlap >= 0.2
            and best_abstract_score <= 0.14
        ):
            verdict = "contradicted"
            reason_codes = ["wrong_subject"]
        elif is_attribution_task and bool(flags.get("background_only_hint", False)):
            if (
                has_connector
                and top1_section == "title"
                and top_score <= 0.09
                and top1_lexical <= 0.02
                and best_abstract_score < 0.08
            ):
                verdict = "contradicted"
                reason_codes = ["wrong_subject"]
            elif top_score <= 0.18:
                # P0-4: avoid over-crediting citation-background mentions as direct support.
                verdict = "unsupported"
                reason_codes = ["citation_background_only"]
        else:
            if top_score >= 0.04:
                # P0-3: avoid weak abstract-only matches drifting to supported by default.
                weak_abstract_support = (
                    top1_section == "abstract"
                    and top_score < 0.1
                    and top1_cue < 0.01
                    and top1_lexical <= 0.13
                    and top1_matched_terms <= 2
                )
                # If abstract evidence is cue-only (no lexical overlap), treat as uncertain.
                cue_only_abstract = (
                    top1_section == "abstract"
                    and top1_lexical <= 0.0
                    and top1_matched_terms == 0
                )
                bundle_low_score_support_ok = (
                    (not is_attribution_task)
                    and str(citation_scope or "").strip().lower() == "bundle"
                    and top1_section == "abstract"
                    and top_score >= 0.06
                    and top1_lexical >= 0.08
                    and top1_matched_terms >= 1
                    and best_abstract_score >= 0.06
                )
                if (weak_abstract_support or cue_only_abstract) and not bundle_low_score_support_ok:
                    verdict = "uncertain"
                    reason_codes = ["evidence_ambiguous"]
                elif (
                    is_attribution_task
                    and top1_section == "abstract"
                    and top_score <= 0.11
                    and top1_lexical <= 0.09
                    and top1_matched_terms <= 1
                    and neg_strength >= pos_strength
                ):
                    verdict = "unsupported"
                    reason_codes = ["citation_background_only"]
                else:
                    verdict = "supported"
                    reason_codes = ["direct_support"]
                    if (
                        is_attribution_task
                        and str(citation_scope or "").strip().lower() == "bundle"
                        and int(bundle_size or 0) >= 2
                    ):
                        attribution_title_only_risk = (
                            top1_section == "title"
                            and top_score <= 0.18
                            and top1_lexical <= 0.13
                            and top1_matched_terms <= 2
                            and best_abstract_score < 0.12
                        )
                        if attribution_title_only_risk:
                            verdict = "uncertain"
                            reason_codes = ["evidence_ambiguous"]
            elif retrieval_success and topk:
                verdict = "uncertain"
                reason_codes = ["evidence_ambiguous"]
            else:
                verdict = "uncertain"
                reason_codes = ["evidence_not_found"]

        decision_trace = {
            "retrieval_success": retrieval_success,
            "top_score": round(top_score, 4),
            "topk_count": len(topk),
            "claim_flags": flags,
            "task_context": {
                "citation_scope": str(citation_scope or ""),
                "bundle_size": int(bundle_size or 0),
                "task_type": str(task_type or ""),
                "is_attribution_task": bool(is_attribution_task),
                "attribution_context": dict(attribution_context or {}),
            },
            "signal_counts": {
                "positive_cues": int(pos_strength),
                "negative_cues": int(neg_strength),
                "scope_cues": int(scope_strength),
                "universal_cues": int(universal_strength),
            },
            "top1_features": {
                "section": top1_section,
                "lexical_score": round(top1_lexical, 4),
                "cue_score": round(top1_cue, 4),
                "matched_terms_count": int(top1_matched_terms),
            },
            "decision_rule": f"flags={flags};top_score={top_score:.4f}",
            "verdict": verdict,
            "reason_codes": list(reason_codes),
        }
        return verdict, reason_codes, decision_trace

    @staticmethod
    def _decode_openalex_abstract(value: Any) -> str:
        if not isinstance(value, dict) or not value:
            return ""
        positions: Dict[int, str] = {}
        for token, pos_list in value.items():
            if not isinstance(pos_list, list):
                continue
            for pos in pos_list:
                try:
                    idx = int(pos)
                except Exception:
                    continue
                if idx >= 0:
                    positions[idx] = str(token)
        if not positions:
            return ""
        max_idx = max(positions.keys())
        words = [positions.get(i, "") for i in range(max_idx + 1)]
        return " ".join(word for word in words if word)

    @staticmethod
    def _crossref_year(payload: Dict[str, Any]) -> Optional[int]:
        for key in ("published-print", "published-online", "issued"):
            info = dict(payload.get(key) or {})
            parts = list(info.get("date-parts") or [])
            if parts and isinstance(parts[0], list) and parts[0]:
                try:
                    return int(parts[0][0])
                except Exception:
                    continue
        return None

    @staticmethod
    def _clean_crossref_abstract(value: Any) -> str:
        text = str(value or "")
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return _norm_space(text)

    @staticmethod
    def _extract_persistent_ids_from_bib_raw(local_bib_raw: str) -> Tuple[str, str]:
        raw = str(local_bib_raw or "")
        doi = ""
        arxiv_id = ""
        doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:a-z0-9]+)", raw, flags=re.IGNORECASE)
        if doi_match:
            doi = _norm_doi(doi_match.group(1))
        arxiv_match = re.search(r"arxiv\s*[:\s]\s*(\d{4}\.\d{4,5}(?:v\d+)?)", raw, flags=re.IGNORECASE)
        if arxiv_match:
            arxiv_id = _norm_arxiv_id(arxiv_match.group(1))
        return doi, arxiv_id

    def _http_json(self, url: str) -> Tuple[Optional[Dict[str, Any]], int, str]:
        if url in self._http_cache:
            payload, status, err = self._http_cache[url]
            return payload, status, err
        req = urlrequest.Request(
            url,
            headers={
                "User-Agent": "AcademicWritingCompanion/1.0 (citation-consistency)",
                "Accept": "application/json",
            },
        )
        payload: Optional[Dict[str, Any]] = None
        status = 0
        err_msg = ""
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
                status = int(getattr(resp, "status", 200))
                raw = resp.read().decode("utf-8", errors="ignore")
                payload = json.loads(raw)
        except urlerror.HTTPError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            err_msg = f"http_error:{status}"
        except urlerror.URLError as exc:
            err_msg = f"url_error:{str(exc.reason)[:160]}"
        except TimeoutError:
            err_msg = "timeout"
        except Exception as exc:  # pragma: no cover
            err_msg = f"unknown_error:{type(exc).__name__}"
        self._http_cache[url] = (payload, status, err_msg)
        return payload, status, err_msg

    def _http_text(self, url: str) -> Tuple[str, int, str]:
        if url in self._arxiv_text_cache:
            text, status, err = self._arxiv_text_cache[url]
            return text, status, err
        req = urlrequest.Request(
            url,
            headers={
                "User-Agent": "AcademicWritingCompanion/1.0 (citation-consistency)",
                "Accept": "application/atom+xml,text/xml,*/*",
            },
        )
        text = ""
        status = 0
        err_msg = ""
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
                status = int(getattr(resp, "status", 200))
                text = resp.read().decode("utf-8", errors="ignore")
        except urlerror.HTTPError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            err_msg = f"http_error:{status}"
        except urlerror.URLError as exc:
            err_msg = f"url_error:{str(exc.reason)[:160]}"
        except TimeoutError:
            err_msg = "timeout"
        except Exception as exc:  # pragma: no cover
            err_msg = f"unknown_error:{type(exc).__name__}"
        self._arxiv_text_cache[url] = (text, status, err_msg)
        return text, status, err_msg
