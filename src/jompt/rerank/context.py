"""Build truncated per-model enrichment blobs for the AI judge.

Model descriptions are untrusted data: callers must delimit them in the
prompt and never treat them as instructions.
"""
from __future__ import annotations

from typing import Any

MAX_CHARS_PER_MODEL = 300


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text[:limit] if len(text) <= limit else text[:limit].rstrip() + "…"


def build_enrichment(
    models: list[dict[str, Any]], max_chars_per_model: int = MAX_CHARS_PER_MODEL
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in models:
        desc = _clip(m.get("description") or "", max_chars_per_model)
        strengths = sorted(set(m.get("strengths") or []))
        out.append(
            {
                "id": str(m.get("id")),
                "name": _clip(m.get("name") or m.get("id"), 120),
                "description": desc,
                "context": int(m.get("context") or 0),
                "reasoning": bool(m.get("reasoning")),
                "tool_call": bool(m.get("tool_call")),
                "strengths": strengths,
            }
        )
    return out
