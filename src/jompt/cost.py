"""Immutable cost / usage tracking."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostRecord:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


@dataclass(frozen=True, slots=True)
class _ModelTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0


@dataclass(frozen=True, slots=True)
class CostTracker:
    _records: tuple[CostRecord, ...] = ()

    def add(self, record: CostRecord) -> CostTracker:
        return CostTracker(self._records + (record,))

    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self._records)

    def request_count(self) -> int:
        return len(self._records)

    def model_totals(self) -> dict[str, _ModelTotals]:
        out: dict[str, _ModelTotals] = {}
        for r in self._records:
            prev = out.get(r.model, _ModelTotals())
            out[r.model] = _ModelTotals(
                input_tokens=prev.input_tokens + r.input_tokens,
                output_tokens=prev.output_tokens + r.output_tokens,
                requests=prev.requests + 1,
            )
        return out

    def summary(self) -> dict:
        totals = self.model_totals()
        return {
            "request_count": self.request_count(),
            "total_input_tokens": sum(r.input_tokens for r in self._records),
            "total_output_tokens": sum(r.output_tokens for r in self._records),
            "models": {
                m: {
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "requests": t.requests,
                }
                for m, t in totals.items()
            },
        }
