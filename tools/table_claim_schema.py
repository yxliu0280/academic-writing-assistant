from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class LLMTableClaim(BaseModel):
    sentence_index: int = Field(ge=0)
    char_span: Tuple[int, int]
    claim_type: Optional[
        Literal[
            "direct_value",
            "from_to",
            "relative_improvement",
            "relative_decrease",
            "pp_change",
            "error_reduction",
            "relative_error_reduction",
        ]
    ] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    from_value: Optional[float] = None
    to_value: Optional[float] = None
    unit: Literal["%", "pp", "raw"]
    scale: Optional[Literal["0_1", "0_100", "raw", "unknown"]] = None
    subject: Optional[str] = None
    comparator: Optional[str] = None
    direction: Optional[Literal["increase", "decrease", "improve"]] = None
    reference: Optional[str] = None

    @field_validator("char_span")
    @classmethod
    def validate_char_span(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if len(v) != 2:
            raise ValueError("char_span must have exactly two integers")
        start, end = int(v[0]), int(v[1])
        if start < 0 or end < 0 or end < start:
            raise ValueError("char_span must satisfy 0 <= start <= end")
        return start, end


class LLMTableClaimsResponse(BaseModel):
    claims: List[LLMTableClaim] = Field(default_factory=list)
