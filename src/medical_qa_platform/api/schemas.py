"""Pydantic request/response models for the predict API."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class PredictResponse(BaseModel):
    answer: str | None
    raw_output: str
    evidence: list[str]
    backend: str
    model_version: str
    contract_version: str
    latency_ms: float
    trace_id: str
