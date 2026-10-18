"""Load free models from the opencode models.json cache."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_LIGHT = ("flash", "tiny", "mini", "lightning", "lite", "nano")
_HEAVY = ("ultra", "flagship", "largest", "maximum", "thinking")
_WORD = re.compile(r"[a-z0-9]+")


def _strengths(model: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    modalities = model.get("modalities") or {}
    inputs = set(modalities.get("input") or [])
    if "image" in inputs or "pdf" in inputs:
        out.add("vision")
    if model.get("tool_call"):
        out.add("tools")
    if model.get("reasoning"):
        out.add("reasoning")
    text = f"{model.get('id', '')} {model.get('name', '')} {model.get('description', '')}".lower()
    if any(k in text for k in ("code", "coding", "software", "programming", "debug")):
        out.add("code")
    words = set(_WORD.findall(text))
    if "flash" in words or "lightning" in words or "nano" in words:
        out.add("flash")
    if "tiny" in words or "mini" in words or "lite" in words:
        out.add("light")
    if any(k in words for k in _HEAVY):
        out.add("heavy")
    limit = model.get("limit") or {}
    if (limit.get("context") or 0) >= 500_000:
        out.add("long-context")
    return out


def _is_free(model: dict[str, Any]) -> bool:
    cost = model.get("cost")
    if cost is None:
        return False
    return cost.get("input", 0) == 0 and cost.get("output", 0) == 0


def load_free_models(models_json: Path) -> list[dict[str, Any]]:
    if not models_json.is_file():
        raise FileNotFoundError(f"models.json not found: {models_json}")
    raw = json.loads(models_json.read_text(encoding="utf-8"))
    provider = raw.get("opencode") or {}
    models: list[dict[str, Any]] = []
    for mid, m in (provider.get("models") or {}).items():
        if not _is_free(m):
            continue
        limit = m.get("limit") or {}
        models.append(
            {
                "id": mid,
                "name": m.get("name") or mid,
                "description": m.get("description") or "",
                "reasoning": bool(m.get("reasoning")),
                "tool_call": bool(m.get("tool_call")),
                "structured_output": bool(m.get("structured_output")),
                "attachment": bool(m.get("attachment")),
                "context": int(limit.get("context") or 0),
                "output_limit": int(limit.get("output") or 0),
                "release_date": m.get("release_date") or "",
                "modalities": m.get("modalities") or {},
                "strengths": sorted(_strengths(m)),
            }
        )
    models.sort(key=lambda x: x["id"])
    return models
