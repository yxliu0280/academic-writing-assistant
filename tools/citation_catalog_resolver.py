from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
from xml.etree import ElementTree

from tools.semantic_scholar_client import SemanticScholarClient


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
WORK_SELECT_FIELDS = "id,doi,display_name,publication_year,authorships,primary_location,ids"


def _norm_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _norm_text(value: Any) -> str:
    return _norm_space(value).lower()


def _normalize_latex_title(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    # Normalize common TeX markup noise in BibTeX titles:
    # "{G}en{AI}" -> "GenAI", outer braces removed, commands stripped.
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("~", " ").replace("_", " ")
    text = text.replace("\\&", "&")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", _norm_text(value))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


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


def _author_surnames(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = _norm_space(value)
        if not text:
            continue
        # Bib/TeX author strings are often "Family, Given"; prefer family part in that case.
        if "," in text:
            surname = _norm_text(text.split(",", 1)[0])
        else:
            parts = re.split(r"[\s,]+", text)
            if not parts:
                continue
            surname = _norm_text(parts[-1])
        if surname:
            out.append(surname)
    return out


@dataclass
class CatalogCandidate:
    source: str
    work_id: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "work_id": self.work_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "score": round(float(self.score), 4),
            "score_breakdown": {str(k): round(float(v), 4) for k, v in self.score_breakdown.items()},
        }


@dataclass
class Layer1ResolutionResult:
    verdict: str
    reason_codes: List[str]
    canonical_work_id: str
    matched_source: str
    match_status: str
    selected_score: float
    score_margin: float
    candidates: List[CatalogCandidate]
    query_trace: List[Dict[str, Any]]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        selected = self.candidates[0].to_dict() if self.candidates else {}
        return {
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
            "canonical_work_id": self.canonical_work_id,
            "matched_source": self.matched_source,
            "match_status": self.match_status,
            "selected_score": round(float(self.selected_score), 4),
            "score_margin": round(float(self.score_margin), 4),
            "selected_candidate": selected,
            "candidates": [item.to_dict() for item in self.candidates],
            "query_trace": list(self.query_trace),
            "notes": self.notes,
            "resolver_version": "l1_catalog_resolver_v1p4",
        }


class CitationCatalogResolver:
    """Resolve single-citation bibliographic records against external catalogs.

    Source roles:
    - OpenAlex: primary work graph for metadata harmonization.
    - Crossref: DOI authority and bibliographic fallback retrieval.
    - arXiv: arXiv-id authority for preprint records.
    - Semantic Scholar (optional): recall booster only.
    """

    OPENALEX_WORKS = "https://api.openalex.org/works"
    CROSSREF_WORKS = "https://api.crossref.org/works"
    ARXIV_API = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        timeout_sec: float = 12.0,
        use_semantic_scholar: bool = False,
        max_candidates: int = 8,
    ) -> None:
        self.timeout_sec = max(3.0, float(timeout_sec))
        self.use_semantic_scholar = bool(use_semantic_scholar)
        self.max_candidates = max(3, int(max_candidates))
        self._result_cache: Dict[str, Layer1ResolutionResult] = {}
        self._http_cache: Dict[str, Tuple[Optional[Dict[str, Any]], int, str]] = {}
        self._semantic_client: Optional[SemanticScholarClient] = None
        if self.use_semantic_scholar:
            self._semantic_client = SemanticScholarClient(timeout=min(self.timeout_sec, 10.0))

    def resolve_from_payload(self, payload: Dict[str, Any]) -> Layer1ResolutionResult:
        primary = dict(payload.get("primary_citation") or {})
        source = dict(payload.get("source") or {})
        citation_bundle = list(payload.get("citation_bundle") or [])
        local_title = str(primary.get("local_title", ""))
        local_authors = list(primary.get("local_authors") or [])
        local_year = primary.get("local_year")
        local_doi = str(primary.get("local_doi", ""))
        local_arxiv_id = str(primary.get("local_arxiv_id", ""))
        local_bib_raw = str(primary.get("local_bib_raw", ""))
        has_local_bib = bool(source.get("has_local_bib", False))
        if not has_local_bib:
            # Phase-2 bundle packets store this as count, not boolean.
            raw_count = source.get("has_local_bib_count")
            try:
                has_local_bib = int(raw_count) > 0
            except Exception:
                has_local_bib = False
        if not has_local_bib and citation_bundle:
            has_local_bib = any(bool(str(item.get("local_bib_raw", "")).strip()) for item in citation_bundle)
        cite_key = str(primary.get("cite_key", ""))
        result = self.resolve(
            cite_key=cite_key,
            local_title=local_title,
            local_authors=local_authors,
            local_year=local_year,
            local_doi=local_doi,
            local_arxiv_id=local_arxiv_id,
            local_bib_raw=local_bib_raw,
            has_local_bib=has_local_bib,
        )
        # Phase-2 bundle alias swap guard:
        # if primary local_title mismatches its own BibTeX title but strongly
        # matches another citation's BibTeX title in the same bundle, force ambiguous.
        if (
            citation_bundle
            and result.verdict in {"valid", "invalid"}
            and self._detect_bundle_alias_conflict(primary=primary, citation_bundle=citation_bundle)
        ):
            reason_codes = self._finalize_reason_codes(
                list(result.reason_codes) + ["version_alias_unresolved", "multiple_catalog_matches"]
            )
            result = Layer1ResolutionResult(
                verdict="ambiguous",
                reason_codes=reason_codes,
                canonical_work_id="",
                matched_source="none",
                match_status="ambiguous_bundle_alias_conflict",
                selected_score=result.selected_score,
                score_margin=result.score_margin,
                candidates=result.candidates,
                query_trace=result.query_trace,
                notes=f"{result.notes};bundle_alias_conflict=True",
            )
        return result

    def resolve(
        self,
        *,
        cite_key: str,
        local_title: str,
        local_authors: List[str],
        local_year: Any,
        local_doi: str,
        local_arxiv_id: str,
        local_bib_raw: str,
        has_local_bib: bool,
    ) -> Layer1ResolutionResult:
        normalized = self._normalize_local(
            cite_key=cite_key,
            local_title=local_title,
            local_authors=local_authors,
            local_year=local_year,
            local_doi=local_doi,
            local_arxiv_id=local_arxiv_id,
            local_bib_raw=local_bib_raw,
            has_local_bib=has_local_bib,
        )
        cache_key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]

        query_trace: List[Dict[str, Any]] = []
        candidates: List[CatalogCandidate] = []

        doi = str(normalized["doi"])
        arxiv_id = str(normalized["arxiv_id"])
        title = str(normalized["title"])

        # Persistent-id exact paths first.
        if doi:
            candidates.extend(self._query_openalex_by_doi(doi, query_trace))
            candidates.extend(self._query_crossref_by_doi(doi, query_trace))

        if arxiv_id:
            candidates.extend(self._query_openalex_by_arxiv(arxiv_id, query_trace))
            candidates.extend(self._query_arxiv_by_id(arxiv_id, query_trace))

        # Metadata fallback path.
        if title:
            candidates.extend(self._query_openalex_by_title(title, query_trace))
            candidates.extend(self._query_crossref_by_title(title, query_trace))
            if self._semantic_client is not None:
                candidates.extend(self._query_semantic_scholar_by_title(title, query_trace))

        scored = self._dedup_and_score(local=normalized, candidates=candidates)
        result = self._decide(local=normalized, candidates=scored, query_trace=query_trace)
        self._result_cache[cache_key] = result
        return result

    def _normalize_local(
        self,
        *,
        cite_key: str,
        local_title: str,
        local_authors: List[str],
        local_year: Any,
        local_doi: str,
        local_arxiv_id: str,
        local_bib_raw: str,
        has_local_bib: bool,
    ) -> Dict[str, Any]:
        year: Optional[int] = None
        try:
            if local_year is not None and str(local_year).strip():
                year = int(local_year)
        except Exception:
            year = None
        title = _norm_space(_normalize_latex_title(local_title))
        authors = [_norm_space(item) for item in local_authors if _norm_space(item)]
        doi = _norm_doi(local_doi)
        arxiv_id = _norm_arxiv_id(local_arxiv_id)
        raw_lower = str(local_bib_raw or "").lower()
        # Fallback only when structured IDs are absent; keep explicit fields as primary.
        raw_doi, raw_arxiv_id = self._extract_persistent_ids_from_bib_raw(local_bib_raw)
        raw_year = self._extract_year_from_bib_raw(local_bib_raw)
        if not doi and raw_doi:
            doi = raw_doi
        if not arxiv_id and raw_arxiv_id:
            arxiv_id = raw_arxiv_id
        return {
            "cite_key": _norm_space(cite_key),
            "title": title,
            "title_tokens": _tokenize(title),
            "authors": authors,
            "author_surnames": _author_surnames(authors),
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "raw_year": raw_year,
            "has_local_bib": bool(has_local_bib),
            "has_local_bib_raw": bool(_norm_space(local_bib_raw)),
            "raw_present": bool(_norm_space(local_bib_raw)),
            "has_url_hint": bool(
                ("http://" in raw_lower)
                or ("https://" in raw_lower)
                or ("\\url{" in raw_lower)
                or ("howpublished" in raw_lower)
                or ("urldate" in raw_lower)
            ),
            "title_token_count": len(_tokenize(title)),
        }

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

    @staticmethod
    def _extract_year_from_bib_raw(local_bib_raw: str) -> Optional[int]:
        raw = str(local_bib_raw or "")
        if not raw:
            return None
        year_match = re.search(r"year\s*=\s*[{\\\"]?(\d{4})", raw, flags=re.IGNORECASE)
        if not year_match:
            return None
        try:
            return int(year_match.group(1))
        except Exception:
            return None

    @staticmethod
    def _extract_title_from_bib_raw(local_bib_raw: str) -> str:
        raw = str(local_bib_raw or "")
        if not raw:
            return ""
        match = re.search(r"title\s*=\s*", raw, flags=re.IGNORECASE)
        if not match:
            return ""
        idx = int(match.end())
        n = len(raw)
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            return ""
        if raw[idx] == "{":
            idx += 1
            depth = 1
            out: List[str] = []
            while idx < n:
                ch = raw[idx]
                if ch == "{":
                    depth += 1
                    out.append(ch)
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                    out.append(ch)
                else:
                    out.append(ch)
                idx += 1
            return _norm_space(_normalize_latex_title("".join(out)))
        if raw[idx] == "\"":
            idx += 1
            out = []
            escaped = False
            while idx < n:
                ch = raw[idx]
                if ch == "\"" and not escaped:
                    break
                if ch == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
                out.append(ch)
                idx += 1
            return _norm_space(_normalize_latex_title("".join(out)))

        end = idx
        while end < n and raw[end] not in {",", "\n", "\r"}:
            end += 1
        return _norm_space(_normalize_latex_title(raw[idx:end]))

    def _detect_bundle_alias_conflict(
        self,
        *,
        primary: Dict[str, Any],
        citation_bundle: List[Dict[str, Any]],
    ) -> bool:
        primary_title = _norm_space(_normalize_latex_title(primary.get("local_title", "")))
        primary_tokens = _tokenize(primary_title)
        if not primary_tokens:
            return False

        primary_raw_title = self._extract_title_from_bib_raw(str(primary.get("local_bib_raw", "")))
        primary_raw_tokens = _tokenize(primary_raw_title)
        if not primary_raw_tokens:
            return False

        primary_raw_sim = _jaccard(primary_tokens, primary_raw_tokens)
        if primary_raw_sim > 0.45:
            return False

        max_other_raw_sim = 0.0
        for item in citation_bundle:
            if bool(item.get("is_primary_target", False)):
                continue
            other_raw_title = self._extract_title_from_bib_raw(str(item.get("local_bib_raw", "")))
            other_raw_tokens = _tokenize(other_raw_title)
            if not other_raw_tokens:
                continue
            sim = _jaccard(primary_tokens, other_raw_tokens)
            if sim > max_other_raw_sim:
                max_other_raw_sim = sim
        return max_other_raw_sim >= 0.9

    def _append_trace(
        self,
        trace: List[Dict[str, Any]],
        *,
        source: str,
        step: str,
        query: Dict[str, Any],
        status: str,
        hit_count: int,
        latency_ms: float,
        http_status: int = 0,
        error: str = "",
    ) -> None:
        trace.append(
            {
                "source": source,
                "step": step,
                "query": query,
                "status": status,
                "hit_count": int(hit_count),
                "http_status": int(http_status),
                "latency_ms": round(float(latency_ms), 2),
                "error": str(error or ""),
            }
        )

    def _http_json(self, url: str) -> Tuple[Optional[Dict[str, Any]], int, str, float]:
        if url in self._http_cache:
            payload, status, err = self._http_cache[url]
            return payload, status, err, 0.0
        started = time.perf_counter()
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
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._http_cache[url] = (payload, status, err_msg)
        return payload, status, err_msg, latency_ms

    def _http_text(self, url: str) -> Tuple[str, int, str, float]:
        if url in self._http_cache:
            payload, status, err = self._http_cache[url]
            text = ""
            if payload is not None:
                text = json.dumps(payload, ensure_ascii=False)
            return text, status, err, 0.0
        started = time.perf_counter()
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
        latency_ms = (time.perf_counter() - started) * 1000.0
        return text, status, err_msg, latency_ms

    def _query_openalex_by_doi(self, doi: str, trace: List[Dict[str, Any]]) -> List[CatalogCandidate]:
        url = f"{self.OPENALEX_WORKS}/https://doi.org/{urlparse.quote(doi, safe='')}?select={WORK_SELECT_FIELDS}"
        payload, status, err, latency = self._http_json(url)
        out: List[CatalogCandidate] = []
        if payload and isinstance(payload, dict) and payload.get("id"):
            out.append(self._openalex_to_candidate(payload))
            self._append_trace(
                trace,
                source="openalex",
                step="doi_exact",
                query={"doi": doi},
                status="ok",
                hit_count=1,
                latency_ms=latency,
                http_status=status,
            )
            return out
        self._append_trace(
            trace,
            source="openalex",
            step="doi_exact",
            query={"doi": doi},
            status="error" if err else "empty",
            hit_count=0,
            latency_ms=latency,
            http_status=status,
            error=err,
        )
        return out

    def _query_openalex_by_arxiv(self, arxiv_id: str, trace: List[Dict[str, Any]]) -> List[CatalogCandidate]:
        arxiv_norm = _strip_arxiv_version(arxiv_id)
        filt = urlparse.quote(f"ids.arxiv:{arxiv_norm}", safe=":,.")
        url = f"{self.OPENALEX_WORKS}?filter={filt}&per-page=5&select={WORK_SELECT_FIELDS}"
        payload, status, err, latency = self._http_json(url)
        out: List[CatalogCandidate] = []
        rows = list((payload or {}).get("results") or []) if isinstance(payload, dict) else []
        for row in rows:
            if isinstance(row, dict):
                out.append(self._openalex_to_candidate(row))
        self._append_trace(
            trace,
            source="openalex",
            step="arxiv_filter",
            query={"arxiv_id": arxiv_norm},
            status="ok" if rows else ("error" if err else "empty"),
            hit_count=len(rows),
            latency_ms=latency,
            http_status=status,
            error=err,
        )
        return out

    def _query_openalex_by_title(self, title: str, trace: List[Dict[str, Any]]) -> List[CatalogCandidate]:
        url = f"{self.OPENALEX_WORKS}?search={urlparse.quote(title)}&per-page=5&select={WORK_SELECT_FIELDS}"
        payload, status, err, latency = self._http_json(url)
        out: List[CatalogCandidate] = []
        rows = list((payload or {}).get("results") or []) if isinstance(payload, dict) else []
        for row in rows:
            if isinstance(row, dict):
                out.append(self._openalex_to_candidate(row))
        self._append_trace(
            trace,
            source="openalex",
            step="title_search",
            query={"title": title[:200]},
            status="ok" if rows else ("error" if err else "empty"),
            hit_count=len(rows),
            latency_ms=latency,
            http_status=status,
            error=err,
        )
        return out

    def _query_crossref_by_doi(self, doi: str, trace: List[Dict[str, Any]]) -> List[CatalogCandidate]:
        url = f"{self.CROSSREF_WORKS}/{urlparse.quote(doi, safe='')}"
        payload, status, err, latency = self._http_json(url)
        out: List[CatalogCandidate] = []
        msg = (payload or {}).get("message") if isinstance(payload, dict) else None
        if isinstance(msg, dict):
            out.append(self._crossref_to_candidate(msg))
            self._append_trace(
                trace,
                source="crossref",
                step="doi_exact",
                query={"doi": doi},
                status="ok",
                hit_count=1,
                latency_ms=latency,
                http_status=status,
            )
            return out
        self._append_trace(
            trace,
            source="crossref",
            step="doi_exact",
            query={"doi": doi},
            status="error" if err else "empty",
            hit_count=0,
            latency_ms=latency,
            http_status=status,
            error=err,
        )
        return out

    def _query_crossref_by_title(self, title: str, trace: List[Dict[str, Any]]) -> List[CatalogCandidate]:
        url = f"{self.CROSSREF_WORKS}?query.bibliographic={urlparse.quote(title)}&rows=5"
        payload, status, err, latency = self._http_json(url)
        out: List[CatalogCandidate] = []
        msg = (payload or {}).get("message") if isinstance(payload, dict) else {}
        items = list(msg.get("items") or []) if isinstance(msg, dict) else []
        for item in items:
            if isinstance(item, dict):
                out.append(self._crossref_to_candidate(item))
        self._append_trace(
            trace,
            source="crossref",
            step="title_search",
            query={"title": title[:200]},
            status="ok" if items else ("error" if err else "empty"),
            hit_count=len(items),
            latency_ms=latency,
            http_status=status,
            error=err,
        )
        return out

    def _query_arxiv_by_id(self, arxiv_id: str, trace: List[Dict[str, Any]]) -> List[CatalogCandidate]:
        query_id = _strip_arxiv_version(arxiv_id)
        url = f"{self.ARXIV_API}?search_query=id:{urlparse.quote(query_id)}&start=0&max_results=1"
        text, status, err, latency = self._http_text(url)
        out: List[CatalogCandidate] = []
        if text:
            try:
                root = ElementTree.fromstring(text)
                entry = root.find("atom:entry", ATOM_NS)
                if entry is not None:
                    out.append(self._arxiv_entry_to_candidate(entry))
            except Exception as exc:
                err = f"xml_parse_error:{type(exc).__name__}"
        self._append_trace(
            trace,
            source="arxiv",
            step="id_exact",
            query={"arxiv_id": query_id},
            status="ok" if out else ("error" if err else "empty"),
            hit_count=len(out),
            latency_ms=latency,
            http_status=status,
            error=err,
        )
        return out

    def _query_semantic_scholar_by_title(self, title: str, trace: List[Dict[str, Any]]) -> List[CatalogCandidate]:
        if self._semantic_client is None:
            return []
        started = time.perf_counter()
        out: List[CatalogCandidate] = []
        err = ""
        try:
            rows = self._semantic_client.search(query=title, limit=5)
            for row in rows:
                out.append(
                    CatalogCandidate(
                        source="semantic_scholar",
                        work_id=str(row.get("paper_id", "")),
                        title=str(row.get("title", "")),
                        authors=[],
                        year=self._to_int(row.get("year")),
                        venue=str(row.get("venue", "")),
                        doi="",
                        arxiv_id="",
                    )
                )
            status = "ok" if rows else "empty"
            hits = len(rows)
        except Exception as exc:  # pragma: no cover
            status = "error"
            hits = 0
            err = f"semantic_scholar_error:{type(exc).__name__}"
        latency = (time.perf_counter() - started) * 1000.0
        self._append_trace(
            trace,
            source="semantic_scholar",
            step="title_search",
            query={"title": title[:200]},
            status=status,
            hit_count=hits,
            latency_ms=latency,
            error=err,
        )
        return out

    def _openalex_to_candidate(self, row: Dict[str, Any]) -> CatalogCandidate:
        ids = dict(row.get("ids") or {})
        doi = _norm_doi(ids.get("doi") or row.get("doi", ""))
        arxiv_id = _norm_arxiv_id(ids.get("arxiv") or "")
        authorships = list(row.get("authorships") or [])
        authors: List[str] = []
        for item in authorships:
            author = dict((item or {}).get("author") or {})
            name = _norm_space(author.get("display_name", ""))
            if name:
                authors.append(name)
        primary_loc = dict(row.get("primary_location") or {})
        source = dict(primary_loc.get("source") or {})
        venue = _norm_space(source.get("display_name", ""))
        return CatalogCandidate(
            source="openalex",
            work_id=str(row.get("id", "")),
            title=_norm_space(row.get("display_name", "")),
            authors=authors,
            year=self._to_int(row.get("publication_year")),
            venue=venue,
            doi=doi,
            arxiv_id=arxiv_id,
        )

    def _crossref_to_candidate(self, row: Dict[str, Any]) -> CatalogCandidate:
        title_list = list(row.get("title") or [])
        title = _norm_space(title_list[0]) if title_list else ""
        author_rows = list(row.get("author") or [])
        authors: List[str] = []
        for item in author_rows:
            family = _norm_space((item or {}).get("family", ""))
            given = _norm_space((item or {}).get("given", ""))
            name = " ".join([part for part in [given, family] if part]).strip()
            if name:
                authors.append(name)
        year = None
        for key in ("published-print", "published-online", "issued"):
            info = dict(row.get(key) or {})
            parts = list(info.get("date-parts") or [])
            if parts and isinstance(parts[0], list) and parts[0]:
                year = self._to_int(parts[0][0])
                if year is not None:
                    break
        venue = ""
        container = list(row.get("container-title") or [])
        if container:
            venue = _norm_space(container[0])
        doi = _norm_doi(row.get("DOI", ""))
        return CatalogCandidate(
            source="crossref",
            work_id=f"doi:{doi}" if doi else "",
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            arxiv_id="",
        )

    def _arxiv_entry_to_candidate(self, entry: ElementTree.Element) -> CatalogCandidate:
        title = _norm_space(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        author_nodes = list(entry.findall("atom:author", ATOM_NS))
        authors: List[str] = []
        for node in author_nodes:
            name = _norm_space(node.findtext("atom:name", default="", namespaces=ATOM_NS))
            if name:
                authors.append(name)
        entry_id = _norm_space(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
        arxiv_id = _norm_arxiv_id(entry_id)
        published = _norm_space(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
        year = None
        if published:
            year = self._to_int(published[:4])
        return CatalogCandidate(
            source="arxiv",
            work_id=f"arxiv:{_strip_arxiv_version(arxiv_id)}" if arxiv_id else "",
            title=title,
            authors=authors,
            year=year,
            venue="arXiv",
            doi="",
            arxiv_id=arxiv_id,
        )

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            text = str(value).strip()
            if not text:
                return None
            return int(text)
        except Exception:
            return None

    def _dedup_and_score(self, *, local: Dict[str, Any], candidates: List[CatalogCandidate]) -> List[CatalogCandidate]:
        dedup: Dict[str, CatalogCandidate] = {}
        for item in candidates:
            key = self._candidate_dedup_key(item)
            if not key:
                continue
            if key not in dedup:
                dedup[key] = item
                continue
            prev = dedup[key]
            # Prefer richer metadata record.
            if self._candidate_richness(item) > self._candidate_richness(prev):
                dedup[key] = item

        scored: List[CatalogCandidate] = []
        for item in dedup.values():
            score, breakdown = self._score_candidate(local=local, candidate=item)
            item.score = score
            item.score_breakdown = breakdown
            scored.append(item)
        scored.sort(key=lambda x: (-x.score, x.source, x.work_id))
        return scored[: self.max_candidates]

    @staticmethod
    def _candidate_dedup_key(item: CatalogCandidate) -> str:
        if item.doi:
            return f"doi::{item.doi}"
        if item.arxiv_id:
            return f"arxiv::{_strip_arxiv_version(item.arxiv_id)}"
        if item.work_id:
            return f"id::{item.work_id}"
        title = _norm_text(item.title)
        year = str(item.year or "")
        if title:
            return f"title_year::{title}::{year}"
        return ""

    @staticmethod
    def _candidate_richness(item: CatalogCandidate) -> int:
        score = 0
        if item.title:
            score += 2
        if item.authors:
            score += 2
        if item.year is not None:
            score += 1
        if item.doi:
            score += 2
        if item.arxiv_id:
            score += 1
        if item.venue:
            score += 1
        return score

    def _score_candidate(self, *, local: Dict[str, Any], candidate: CatalogCandidate) -> Tuple[float, Dict[str, float]]:
        breakdown: Dict[str, float] = {}
        score = 0.0

        local_doi = str(local["doi"])
        cand_doi = _norm_doi(candidate.doi)
        if local_doi and cand_doi:
            if local_doi == cand_doi:
                score += 0.55
                breakdown["doi_exact"] = 0.55
            else:
                score -= 0.25
                breakdown["doi_mismatch"] = -0.25
        elif local_doi and not cand_doi:
            score -= 0.08
            breakdown["doi_missing_in_candidate"] = -0.08

        local_arxiv = _strip_arxiv_version(str(local["arxiv_id"]))
        cand_arxiv = _strip_arxiv_version(_norm_arxiv_id(candidate.arxiv_id))
        if local_arxiv and cand_arxiv:
            if local_arxiv == cand_arxiv:
                score += 0.45
                breakdown["arxiv_exact"] = 0.45
            else:
                score -= 0.2
                breakdown["arxiv_mismatch"] = -0.2
        elif local_arxiv and not cand_arxiv:
            score -= 0.05
            breakdown["arxiv_missing_in_candidate"] = -0.05

        title_sim = _jaccard(local["title_tokens"], _tokenize(candidate.title))
        title_contrib = 0.35 * title_sim
        score += title_contrib
        breakdown["title_jaccard"] = title_contrib

        local_year = local.get("year")
        cand_year = candidate.year
        if local_year is not None and cand_year is not None:
            gap = abs(int(local_year) - int(cand_year))
            if gap == 0:
                score += 0.12
                breakdown["year_exact"] = 0.12
            elif gap == 1:
                score += 0.05
                breakdown["year_close"] = 0.05
            else:
                score -= 0.06
                breakdown["year_far"] = -0.06

        local_surnames = set(local["author_surnames"])
        cand_surnames = set(_author_surnames(candidate.authors))
        author_overlap = 0.0
        if local_surnames and cand_surnames:
            author_overlap = len(local_surnames & cand_surnames) / max(1, len(local_surnames))
            author_contrib = 0.18 * author_overlap
            score += author_contrib
            breakdown["author_overlap"] = author_contrib

        if local["has_local_bib"] and local["has_local_bib_raw"]:
            score += 0.02
            breakdown["local_bib_present"] = 0.02

        score = max(0.0, min(1.0, score))
        breakdown["title_similarity_raw"] = round(float(title_sim), 6)
        breakdown["author_overlap_raw"] = round(float(author_overlap), 6)
        return score, breakdown

    def _decide(
        self,
        *,
        local: Dict[str, Any],
        candidates: List[CatalogCandidate],
        query_trace: List[Dict[str, Any]],
    ) -> Layer1ResolutionResult:
        reason_codes: List[str] = []
        if not local["has_local_bib"] or not local["has_local_bib_raw"]:
            verdict = "invalid"
            reason_codes.append("no_catalog_match")
            match_status = "invalid_missing_local_bib"
            reason_codes = self._finalize_reason_codes(reason_codes)
            return Layer1ResolutionResult(
                verdict=verdict,
                reason_codes=reason_codes,
                canonical_work_id="",
                matched_source="none",
                match_status=match_status,
                selected_score=0.0,
                score_margin=1.0,
                candidates=candidates,
                query_trace=query_trace,
                notes="resolver_decision=invalid_missing_local_bib",
            )

        missing_fields: List[str] = []
        if not local["title"]:
            missing_fields.append("missing_title")
        if not local["author_surnames"]:
            missing_fields.append("missing_author")
        if local["year"] is None:
            missing_fields.append("missing_year")

        # Track persistent-id resolution status independently from title fallback.
        doi_requested = bool(local["doi"])
        arxiv_requested = bool(local["arxiv_id"])
        doi_hits = sum(
            int(item.get("hit_count", 0))
            for item in query_trace
            if item.get("step") == "doi_exact" and item.get("source") in {"openalex", "crossref"}
        )
        arxiv_hits = sum(
            int(item.get("hit_count", 0))
            for item in query_trace
            if item.get("step") in {"arxiv_filter", "id_exact"} and item.get("source") in {"openalex", "arxiv"}
        )
        persistent_conflict = (doi_requested and doi_hits == 0) or (arxiv_requested and arxiv_hits == 0)

        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        top_score = float(top.score) if top else 0.0
        margin = (float(top.score) - float(second.score)) if (top and second) else 1.0
        top_title_sim = (
            float((top.score_breakdown or {}).get("title_similarity_raw", 0.0))
            if top is not None
            else 0.0
        )
        top_author_overlap = (
            float((top.score_breakdown or {}).get("author_overlap_raw", 0.0))
            if top is not None
            else 0.0
        )

        # If DOI-exact and arXiv-exact matches resolve to disjoint candidates,
        # treat the record as unresolved alias/version conflict.
        persistent_exact_disagreement = False
        if doi_requested and arxiv_requested and candidates:
            local_doi = str(local["doi"])
            local_arxiv = _strip_arxiv_version(str(local["arxiv_id"]))
            doi_exact_keys = {
                self._candidate_identity_key(item)
                for item in candidates
                if local_doi and _norm_doi(item.doi) == local_doi
            }
            arxiv_exact_keys = {
                self._candidate_identity_key(item)
                for item in candidates
                if local_arxiv and _strip_arxiv_version(_norm_arxiv_id(item.arxiv_id)) == local_arxiv
            }
            if doi_exact_keys and arxiv_exact_keys and doi_exact_keys.isdisjoint(arxiv_exact_keys):
                persistent_exact_disagreement = True

        if missing_fields:
            top_breakdown = dict((top.score_breakdown if top is not None else {}) or {})
            has_exact_persistent_id = bool(
                float(top_breakdown.get("doi_exact", 0.0)) > 0.0
                or float(top_breakdown.get("arxiv_exact", 0.0)) > 0.0
            )
            # P0-4: if catalog resolution is near-certain with persistent-id exact match,
            # do not downgrade solely for missing local year metadata.
            if (
                top is not None
                and top_score >= 0.95
                and set(missing_fields).issubset({"missing_year"})
                and has_exact_persistent_id
            ):
                verdict = "valid"
                match_status = "resolved_unique_valid_highconf_backfill"
            else:
                reason_codes.extend(missing_fields)
                verdict = "incomplete"
                match_status = "incomplete_metadata"
        elif (
            local.get("year") is not None
            and local.get("raw_year") is not None
            and abs(int(local["year"]) - int(local["raw_year"])) >= 3
            and not str(local.get("doi", "")).strip()
            and not str(local.get("arxiv_id", "")).strip()
        ):
            # Bundle alias/key-swap style cases often keep raw BibTeX entry year
            # but perturb structured local fields; keep this explicitly ambiguous.
            verdict = "ambiguous"
            reason_codes.extend(["version_alias_unresolved", "multiple_catalog_matches"])
            match_status = "ambiguous_local_raw_year_conflict"
        elif persistent_exact_disagreement:
            verdict = "ambiguous"
            reason_codes.extend(["version_alias_unresolved", "multiple_catalog_matches"])
            match_status = "ambiguous_persistent_id_disagreement"
        elif (
            persistent_conflict
            and doi_requested
            and arxiv_requested
        ):
            verdict = "ambiguous"
            reason_codes.extend(["version_alias_unresolved", "multiple_catalog_matches"])
            match_status = (
                "ambiguous_persistent_id_conflict_no_candidate"
                if top is None or top_score < 0.45
                else "ambiguous_persistent_id_conflict"
            )
        elif not candidates or top_score < 0.35:
            if self._should_accept_local_web_fallback(
                local=local,
                persistent_conflict=persistent_conflict,
                top_score=top_score,
                top_title_sim=top_title_sim,
                top_author_overlap=top_author_overlap,
            ):
                verdict = "valid"
                match_status = "valid_local_web_fallback"
            elif self._should_accept_author_year_backfill(
                local=local,
                persistent_conflict=persistent_conflict,
                top_score=top_score,
                top_title_sim=top_title_sim,
                top_author_overlap=top_author_overlap,
            ):
                verdict = "valid"
                match_status = "resolved_unique_valid_author_year_backfill"
            else:
                verdict = "invalid"
                if persistent_conflict:
                    reason_codes.append("persistent_id_invalid")
                reason_codes.append("no_catalog_match")
                match_status = "invalid_no_catalog_match"
        elif second is not None and top_score >= 0.66 and margin <= 0.05:
            # Strong metadata alignment can break near-tie ambiguity safely.
            if not persistent_conflict and top_title_sim >= 0.95 and top_author_overlap >= 0.8:
                verdict = "valid"
                match_status = "resolved_unique_valid_tie_break"
            else:
                verdict = "ambiguous"
                reason_codes.extend(["multiple_catalog_matches", "version_alias_unresolved"])
                match_status = "ambiguous_multi_match"
        elif persistent_conflict and top_score >= 0.75:
            verdict = "ambiguous"
            reason_codes.extend(["persistent_id_invalid", "version_alias_unresolved"])
            match_status = "ambiguous_persistent_id_conflict"
        else:
            mismatch_codes = self._mismatch_codes(local=local, selected=top)
            if mismatch_codes:
                verdict = "invalid"
                reason_codes.extend(mismatch_codes)
                match_status = "invalid_metadata_mismatch"
            else:
                verdict = "valid"
                match_status = "resolved_unique_valid"

        reason_codes = self._finalize_reason_codes(reason_codes)
        canonical_work_id = ""
        matched_source = "none"
        if top is not None and match_status.startswith("resolved_unique"):
            canonical_work_id = top.work_id
            matched_source = top.source
        elif top is not None and verdict == "incomplete":
            canonical_work_id = top.work_id
            matched_source = top.source

        notes = (
            f"resolver_decision={match_status};top_score={top_score:.4f};margin={margin:.4f};"
            f"persistent_conflict={bool(persistent_conflict)}"
        )
        return Layer1ResolutionResult(
            verdict=verdict,
            reason_codes=reason_codes,
            canonical_work_id=canonical_work_id,
            matched_source=matched_source,
            match_status=match_status,
            selected_score=top_score,
            score_margin=margin,
            candidates=candidates,
            query_trace=query_trace,
            notes=notes,
        )

    @staticmethod
    def _should_accept_local_web_fallback(
        *,
        local: Dict[str, Any],
        persistent_conflict: bool,
        top_score: float,
        top_title_sim: float,
        top_author_overlap: float,
    ) -> bool:
        # Conservative fallback for non-catalog web references in local BibTeX.
        if persistent_conflict:
            return False
        if str(local.get("doi", "")).strip() or str(local.get("arxiv_id", "")).strip():
            return False
        if not bool(local.get("has_url_hint", False)):
            return False
        if int(local.get("title_token_count", 0) or 0) < 3:
            return False
        if not local.get("title") or not local.get("author_surnames") or local.get("year") is None:
            return False
        if top_score >= 0.35:
            return False
        if top_author_overlap > 0.0:
            return False
        if top_title_sim > 0.35:
            return False
        return True

    @staticmethod
    def _should_accept_author_year_backfill(
        *,
        local: Dict[str, Any],
        persistent_conflict: bool,
        top_score: float,
        top_title_sim: float,
        top_author_overlap: float,
    ) -> bool:
        # Conservative backfill when title indexing fails but author/year anchor is strong.
        if persistent_conflict:
            return False
        if top_score < 0.30:
            return False
        if top_author_overlap < 0.8:
            return False
        if local.get("year") is None:
            return False
        if bool(local.get("has_url_hint", False)):
            return True
        return top_title_sim >= 0.12

    def _mismatch_codes(self, *, local: Dict[str, Any], selected: Optional[CatalogCandidate]) -> List[str]:
        if selected is None:
            return []
        local_doi = str(local["doi"])
        selected_doi = _norm_doi(selected.doi)
        local_arxiv = _strip_arxiv_version(str(local["arxiv_id"]))
        selected_arxiv = _strip_arxiv_version(_norm_arxiv_id(selected.arxiv_id))
        # Persistent-id exact match has highest authority in L1 existence/integrity.
        if (local_doi and selected_doi and local_doi == selected_doi) or (
            local_arxiv and selected_arxiv and local_arxiv == selected_arxiv
        ):
            return []

        out: List[str] = []
        title_sim = _jaccard(local["title_tokens"], _tokenize(selected.title))
        if local["title"] and title_sim < 0.45:
            out.append("title_mismatch")
        local_surnames = set(local["author_surnames"])
        cand_surnames = set(_author_surnames(selected.authors))
        overlap = 0.0
        if local_surnames and cand_surnames:
            overlap = len(local_surnames & cand_surnames) / max(1, len(local_surnames))
            # For exact title matches, tolerate sparse/format-shifted author lists.
            if overlap < 0.2 and title_sim < 0.95:
                out.append("author_mismatch")
        if local["year"] is not None and selected.year is not None:
            year_gap = abs(int(local["year"]) - int(selected.year))
            if year_gap > 1 and not (title_sim >= 0.95 and overlap >= 0.3):
                out.append("year_mismatch")
        if local_doi and selected.doi and local_doi != _norm_doi(selected.doi):
            out.append("persistent_id_invalid")
        return self._finalize_reason_codes(out)

    @staticmethod
    def _candidate_identity_key(item: CatalogCandidate) -> str:
        if item.doi:
            return f"doi::{_norm_doi(item.doi)}"
        if item.arxiv_id:
            return f"arxiv::{_strip_arxiv_version(_norm_arxiv_id(item.arxiv_id))}"
        return f"id::{item.work_id}"

    @staticmethod
    def _finalize_reason_codes(values: Iterable[str]) -> List[str]:
        allowed = {
            "missing_bib_entry",
            "malformed_bib_entry",
            "missing_title",
            "missing_author",
            "missing_year",
            "persistent_id_invalid",
            "no_catalog_match",
            "multiple_catalog_matches",
            "title_mismatch",
            "author_mismatch",
            "year_mismatch",
            "venue_mismatch",
            "version_alias_unresolved",
            "duplicate_local_entry",
        }
        out: List[str] = []
        seen = set()
        for item in values:
            code = _norm_text(item)
            if not code or code not in allowed or code in seen:
                continue
            seen.add(code)
            out.append(code)
        return out
