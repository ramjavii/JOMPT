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


def list_free_ids(models_json: Path) -> list[str]:
    """Sorted free model ids from models.json."""
    return [m["id"] for m in load_free_models(models_json)]


def get_models_by_ids(models_json: Path, ids: list[str]) -> list[dict[str, Any]]:
    """Return free models for exactly the requested ids (order preserved).

    Raises ValueError on empty input, duplicates, unknown ids, or paid models.
    """
    if not ids:
        raise ValueError("id list is empty")
    cleaned = [i.strip() for i in ids if isinstance(i, str) and i.strip()]
    if len(cleaned) != len(ids) or any(not i for i in cleaned):
        raise ValueError("id list contains blank ids")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("id list has duplicate ids")
    free = {m["id"]: m for m in load_free_models(models_json)}
    # Detect paid-but-known vs truly unknown via raw file.
    import json as _json

    raw = _json.loads(models_json.read_text(encoding="utf-8"))
    known_all = set(((raw.get("opencode") or {}).get("models") or {}).keys())
    missing = [i for i in cleaned if i not in free]
    if missing:
        paid = [i for i in missing if i in known_all]
        unknown = [i for i in missing if i not in known_all]
        if paid:
            raise ValueError(f"paid/non-free models rejected: {paid}")
        raise ValueError(f"unknown model ids: {unknown}")
    return [free[i] for i in cleaned]


def diff_new_ids(models_json: Path, known_ids: set[str] | list[str]) -> list[str]:
    """Free ids in models.json not present in known_ids (drift discovery)."""
    known = set(known_ids)
    return sorted(i for i in list_free_ids(models_json) if i not in known)
