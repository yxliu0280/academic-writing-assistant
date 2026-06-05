from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from agents.base import BaseAgent
from core.schemas import AppState, Issue, Location
from tools.text_utils import find_all_spans, normalize_term


class TerminologyAgent(BaseAgent):
    name = "terminology"

    def run(self, state: AppState) -> List[Issue]:
        text = state.current_text
        variants = self._extract_variants(text)
        if not variants:
            return []

        clusters = self._cluster_variants(list(variants.keys()))
        issues: List[Issue] = []

        for idx, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue

            occurrences = []
            freq = {}
            for variant in cluster:
                spans = variants.get(variant, [])
                freq[variant] = len(spans)
                for start, end in spans:
                    occurrences.append({"variant": variant, "char_span": [start, end]})

            canonical = max(freq.items(), key=lambda kv: kv[1])[0]
            first_start = min(o["char_span"][0] for o in occurrences)
            first_end = max(o["char_span"][1] for o in occurrences if o["char_span"][0] == first_start)
            snippet = text[first_start:first_end]

            issues.append(
                Issue(
                    id=f"term_inconsistent_{idx}_{first_start}",
                    type="terminology_inconsistent",
                    severity="medium",
                    status="detected",
                    location=Location(
                        sentence_index=-1,
                        char_span=(first_start, first_end),
                        snippet=snippet,
                    ),
                    evidence={
                        "canonical": canonical,
                        "variants": sorted(cluster),
                        "occurrences": occurrences,
                    },
                    message=(
                        f"Terminology drift detected among {sorted(cluster)}. "
                        f"Prefer a single form: '{canonical}'."
                    ),
                    suggested_fix=f"Use '{canonical}' consistently for all occurrences.",
                    provenance={"source": "rule_based_terminology_clustering"},
                )
            )

        return issues

    def _extract_variants(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        variants: Dict[str, List[Tuple[int, int]]] = {}

        acronym_pattern = re.compile(r"\b[A-Z]{2,}\b")
        title_case_pattern = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

        for m in acronym_pattern.finditer(text):
            token = m.group(0)
            variants.setdefault(token, []).append((m.start(), m.end()))

        for m in title_case_pattern.finditer(text):
            token = m.group(0)
            variants.setdefault(token, []).append((m.start(), m.end()))

        # Extract acronym definitions: Large Language Model (LLM)
        definition_pattern = re.compile(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(([A-Z]{2,})\)"
        )
        for m in definition_pattern.finditer(text):
            long_form = m.group(1)
            short_form = m.group(2)
            variants.setdefault(long_form, []).append((m.start(1), m.end(1)))
            variants.setdefault(short_form, []).append((m.start(2), m.end(2)))

        return variants

    def _cluster_variants(self, terms: List[str]) -> List[set[str]]:
        clusters: List[set[str]] = []

        alias_pairs = {
            "llm": "large language model",
            "llms": "large language model",
            "rnn": "recurrent neural network",
            "cnn": "convolutional neural network",
        }

        for term in terms:
            norm = normalize_term(term)
            norm = alias_pairs.get(norm, norm)
            placed = False
            for cluster in clusters:
                representative = next(iter(cluster))
                rep_norm = normalize_term(representative)
                rep_norm = alias_pairs.get(rep_norm, rep_norm)
                similar = SequenceMatcher(a=norm, b=rep_norm).ratio() >= 0.88
                if norm == rep_norm or similar:
                    cluster.add(term)
                    placed = True
                    break
            if not placed:
                clusters.append({term})
        return clusters
