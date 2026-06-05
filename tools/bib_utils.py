from __future__ import annotations

import re
from typing import Dict, List, Set

from tools.text_utils import split_sentences_with_offsets

BIB_KEY_PATTERN = re.compile(r"@\w+\s*\{\s*([^,\s]+)", re.IGNORECASE)
CITE_PATTERN = re.compile(r"\\cite\w*\{([^}]+)\}")


def parse_bibtex_keys(content: str) -> Set[str]:
    return {m.group(1).strip() for m in BIB_KEY_PATTERN.finditer(content or "")}


def extract_cite_keys(text: str) -> List[Dict[str, object]]:
    cites: List[Dict[str, object]] = []
    for m in CITE_PATTERN.finditer(text or ""):
        raw = m.group(1)
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        cites.append(
            {
                "keys": keys,
                "start": m.start(),
                "end": m.end(),
                "snippet": m.group(0),
            }
        )
    return cites


def detect_suspected_missing_citation_sentences(text: str) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    triggers = re.compile(
        r"(state[- ]of[- ]the[- ]art|previous work|prior work|according to|reported|stud(y|ies)|\d{4})",
        re.IGNORECASE,
    )
    citation_in_sentence = re.compile(r"\\cite\w*\{[^}]+\}")

    for sentence in split_sentences_with_offsets(text):
        stext = str(sentence["text"])
        if citation_in_sentence.search(stext):
            continue
        if triggers.search(stext):
            candidates.append(sentence)
    return candidates


def make_provenance_entry(
    key: str,
    title: str | None,
    authors: str | None,
    year: str | None,
    venue: str | None,
    doi: str | None,
    source: str,
) -> Dict[str, str]:
    return {
        "key": key,
        "title": title or "",
        "authors": authors or "",
        "year": year or "",
        "venue": venue or "",
        "doi": doi or "",
        "source": source,
    }
