"""Pydantic request/response models for the predict API."""

import re

from pydantic import BaseModel, Field, field_validator

_LETTER_RE = re.compile(r"^[A-Z]$")


class PredictRequest(BaseModel):
    question: str = Field(min_length=1)
    options: dict[str, str]

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("options")
    @classmethod
    def _validate_options(cls, value: dict[str, str]) -> dict[str, str]:
        if not (2 <= len(value) <= 10):
            raise ValueError("options must have between 2 and 10 entries")
        for key, option in value.items():
            if not _LETTER_RE.match(key):
                raise ValueError(f"option key {key!r} must be a single A-Z letter")
            if not option.strip():
                raise ValueError(f"option {key} must not be blank")
        return value


class PredictResponse(BaseModel):
    answer: str | None
    evidence: list[str]
    backend: str
    model_version: str
    latency_ms: float
    trace_id: str
