from __future__ import annotations

import re
from typing import Dict, List, Tuple


def split_sentences_with_offsets(text: str) -> List[Dict[str, object]]:
    # Keep decimal numbers like 0.873 inside the same sentence span.
    pattern = re.compile(r"(?:\d+\.\d+|[^.!?\n])+[.!?]?", re.MULTILINE)
    sentences: List[Dict[str, object]] = []
    for idx, match in enumerate(pattern.finditer(text)):
        snippet = match.group(0).strip()
        if not snippet:
            continue
        sentences.append(
            {
                "index": idx,
                "start": match.start(),
                "end": match.end(),
                "text": snippet,
            }
        )
    return sentences


def char_span_to_line_range(text: str, start: int, end: int) -> Tuple[int, int]:
    start_line = text.count("\n", 0, max(start, 0)) + 1
    end_line = text.count("\n", 0, max(end, 0)) + 1
    return start_line, end_line


def normalize_term(term: str) -> str:
    term = term.strip().lower()
    term = term.replace("-", " ")
    term = re.sub(r"\s+", " ", term)
    if term.endswith("s") and len(term) > 4:
        term = term[:-1]
    return term


def find_all_spans(text: str, needle: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if not needle:
        return spans
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    for m in pattern.finditer(text):
        spans.append((m.start(), m.end()))
    return spans
