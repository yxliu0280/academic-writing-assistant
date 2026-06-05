from __future__ import annotations

import json
from typing import Any, Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SemanticScholarClient:
    """Inspired by Towards AI-assisted Academic Writing and AgentRxiv:
    Contextual citation grounding and hallucination-free RAG.

    This client fetches only factual metadata from Semantic Scholar Graph API.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        params = {
            "query": query,
            "limit": max(1, min(limit, 20)),
            "fields": "title,abstract,year,venue,url,paperId",
        }
        url = f"{self.BASE_URL}?{urlencode(params)}"

        try:
            payload = self._fetch_json(url)
        except Exception:
            return []

        results: List[Dict[str, Any]] = []
        for row in payload.get("data", []) or []:
            results.append(
                {
                    "title": row.get("title", "") or "",
                    "abstract": row.get("abstract", "") or "",
                    "year": row.get("year"),
                    "venue": row.get("venue", "") or "",
                    "url": row.get("url", "") or "",
                    "paper_id": row.get("paperId", "") or "",
                    "source": "semantic_scholar",
                }
            )
        return results

    def _fetch_json(self, url: str) -> Dict[str, Any]:
        request = Request(
            url,
            headers={
                "User-Agent": "AcademicWritingCompanion/0.1",
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=self.timeout) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        return json.loads(data)
