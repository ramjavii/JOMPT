"""API and domain schemas (immutable)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=200_000)
    system: str | None = Field(default=None, max_length=50_000)
    force_model: str | None = Field(default=None, max_length=200, pattern=r"^[a-zA-Z0-9._\-]+$")
    bias: int | None = Field(default=None, ge=-30, le=30)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=128_000)


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    model_score: int
    complexity_score: int
    input_tokens: int
    output_tokens: int
    finish_reason: str
    routed_by: str
    reason: str
    fallback_from: list[str] = Field(default_factory=list)


class CatalogModelOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    score: int
    strengths: list[str]
    context: int
    reasoning: bool
    source: str


class RankingsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    models: list[CatalogModelOut]


class ModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    requests: int


class StatsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    models: dict[str, ModelUsage]


class ErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str
    extra: dict[str, Any] | None = None
