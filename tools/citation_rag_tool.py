from __future__ import annotations

import re
from typing import Any, Dict, List

from tools.semantic_scholar_client import SemanticScholarClient


class CitationRAGTool:
    """Inspired by Towards AI-assisted Academic Writing and AgentRxiv:
    Contextual citation grounding and hallucination-free RAG.

    Policy:
    - prefer uploaded BibTeX matches first
    - then enrich with real Semantic Scholar metadata
    - recommendations always include provenance
    - no synthetic DOI/title generation
    """

    def __init__(self, client: SemanticScholarClient | None = None) -> None:
        self.client = client or SemanticScholarClient()

    def recommend(
        self,
        query: str,
        uploaded_bib_content: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        top_k = max(1, top_k)

        local_hits = self._search_local_bib(query, uploaded_bib_content, top_k=top_k)
        results = list(local_hits)

        if len(results) < top_k:
            remote = self.client.search(query=query, limit=max(top_k, 8))
            existing_titles = {self._normalize_title(x.get("title", "")) for x in results}
            for item in remote:
                title_norm = self._normalize_title(item.get("title", ""))
                if not title_norm or title_norm in existing_titles:
                    continue
                results.append(
                    {
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "year": item.get("year"),
                        "venue": item.get("venue", ""),
                        "url": item.get("url", ""),
                        "key": None,
                        "doi": None,
                        "score": 0.0,
                        "provenance": {
                            "source": "semantic_scholar",
                            "query": query,
                            "paper_id": item.get("paper_id", ""),
                        },
                    }
                )
                existing_titles.add(title_norm)
                if len(results) >= top_k:
                    break

        return results[:top_k]

    def _search_local_bib(self, query: str, bib_content: str, top_k: int) -> List[Dict[str, Any]]:
        entries = self._parse_bib_entries(bib_content)
        if not entries:
            return []

        scored = []
        for entry in entries:
            score = self._overlap_score(query, entry)
            if score <= 0:
                continue
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for score, entry in scored[:top_k]:
            results.append(
                {
                    "title": entry.get("title", ""),
                    "abstract": entry.get("abstract", ""),
                    "year": entry.get("year"),
                    "venue": entry.get("venue", ""),
                    "url": entry.get("url", ""),
                    "key": entry.get("key"),
                    "doi": entry.get("doi"),
                    "score": round(score, 4),
                    "provenance": {
                        "source": "local_bib",
                        "query": query,
                        "key": entry.get("key"),
                    },
                }
            )
        return results

    def _parse_bib_entries(self, bib_content: str) -> List[Dict[str, Any]]:
        if not bib_content.strip():
            return []

        entries: List[Dict[str, Any]] = []
        # Lightweight parser for typical BibTeX structure.
        entry_pattern = re.compile(
            r"@(\w+)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\s*\}",
            re.IGNORECASE,
        )
        field_pattern = re.compile(r"(\w+)\s*=\s*(\{[^{}]*\}|\"[^\"]*\"|[^,\n]+)", re.IGNORECASE)

        for m in entry_pattern.finditer(bib_content):
            entry_type = m.group(1).strip().lower()
            key = m.group(2).strip()
            body = m.group(3)

            fields: Dict[str, Any] = {"entry_type": entry_type, "key": key}
            for fm in field_pattern.finditer(body):
                name = fm.group(1).strip().lower()
                value = fm.group(2).strip().strip(",")
                value = value.strip("{}\"")
                fields[name] = value

            entries.append(
                {
                    "key": key,
                    "title": fields.get("title", ""),
                    "abstract": fields.get("abstract", ""),
                    "year": fields.get("year"),
                    "venue": fields.get("journal", fields.get("booktitle", "")),
                    "doi": fields.get("doi"),
                    "url": fields.get("url", ""),
                }
            )

        return entries

    def _overlap_score(self, query: str, entry: Dict[str, Any]) -> float:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return 0.0

        corpus = " ".join(
            [
                str(entry.get("title", "")),
                str(entry.get("abstract", "")),
                str(entry.get("venue", "")),
            ]
        )
        c_tokens = self._tokenize(corpus)
        if not c_tokens:
            return 0.0

        overlap = q_tokens.intersection(c_tokens)
        base = len(overlap) / max(1, len(q_tokens))

        title = str(entry.get("title", "")).lower()
        if query.lower() in title and query.strip():
            base += 0.2

        return min(base, 1.0)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
        return set(tokens)

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", title.strip().lower())
