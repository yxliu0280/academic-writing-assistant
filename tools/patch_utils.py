from __future__ import annotations

from dataclasses import dataclass, asdict
import difflib
from typing import Dict, List, Tuple


@dataclass
class ReplaceOperation:
    start: int
    end: int
    replacement: str
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def apply_replace_operations(text: str, operations: List[ReplaceOperation]) -> str:
    # Apply from back to front to keep offsets valid.
    sorted_ops = sorted(operations, key=lambda op: op.start, reverse=True)
    out = text
    for op in sorted_ops:
        out = out[: op.start] + op.replacement + out[op.end :]
    return out


def compute_edit_ratio(original: str, modified: str) -> float:
    if not original and not modified:
        return 0.0
    matcher = difflib.SequenceMatcher(a=original, b=modified)
    return 1.0 - matcher.ratio()


def unified_diff(original: str, modified: str) -> str:
    diff = difflib.unified_diff(
        original.splitlines(),
        modified.splitlines(),
        fromfile="before.tex",
        tofile="after.tex",
        lineterm="",
    )
    return "\n".join(diff)


def count_modified_sentences(original: str, modified: str) -> int:
    # Coarse but deterministic: count changed lines as proxy for sentence edits.
    orig_lines = original.splitlines()
    mod_lines = modified.splitlines()
    matcher = difflib.SequenceMatcher(a=orig_lines, b=mod_lines)
    changed = 0
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed


def patch_guardrail_ok(
    original: str,
    modified: str,
    max_edit_ratio: float,
    max_sentence_changes: int,
) -> Tuple[bool, Dict[str, float]]:
    edit_ratio = compute_edit_ratio(original, modified)
    sentence_changes = float(count_modified_sentences(original, modified))
    ok = edit_ratio <= max_edit_ratio and sentence_changes <= max_sentence_changes
    return ok, {
        "edit_ratio": edit_ratio,
        "sentence_changes": sentence_changes,
        "max_edit_ratio": max_edit_ratio,
        "max_sentence_changes": float(max_sentence_changes),
    }
