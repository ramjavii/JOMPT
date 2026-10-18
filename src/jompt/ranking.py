"""Capability ranking: auto-score, curated overrides, build sorted catalog."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from jompt.registry import load_free_models


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    id: str
    score: int
    strengths: tuple[str, ...] | frozenset[str] | set[str]
    context: int
    reasoning: bool
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "strengths", frozenset(self.strengths))


def auto_score(model: dict[str, Any]) -> int:
    """Derive 0–100 capability score from metadata and name/description keywords."""
    score = 10.0

    if model.get("reasoning"):
        score += 25.0
    if model.get("tool_call"):
        score += 10.0
    if model.get("structured_output"):
        score += 5.0
    if model.get("attachment"):
        score += 3.0

    ctx = int(model.get("context") or 0)
    if ctx >= 1_000_000:
        score += 18.0
    elif ctx >= 500_000:
        score += 12.0
    elif ctx >= 200_000:
        score += 6.0

    strengths = set(model.get("strengths") or [])
    text = f"{model.get('id', '')} {model.get('name', '')} {model.get('description', '')}".lower()

    words = set(re.findall(r"[a-z0-9]+", text))
    if "flash" in strengths or bool(words & {"flash", "lightning", "nano"}):
        score -= 18.0
    if "light" in strengths:
        score -= 22.0
    if "mini" in words and "max" not in words:
        score -= 12.0
    # Single heavy-title bonus (avoid stacking flagship + ultra + next-gen)
    if words & {"ultra", "maximum", "largest", "flagship"}:
        score += 18.0
    if "next-generation" in text or "next generation" in text:
        score += 4.0
    if "preview" in words or "stealth" in words:
        score -= 6.0
    if "coding" in words or "software" in words or "programming" in words:
        score += 5.0
    if "vision" in strengths:
        score += 4.0

    return int(max(0, min(100, round(score))))


def apply_overrides(entries: list[dict[str, Any]], overrides: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        e = dict(e)
        ov = overrides.get(e["id"])
        if isinstance(ov, dict):
            if "score" in ov:
                e["score"] = int(ov["score"])
                e["source"] = "override"
            if "strengths" in ov and ov["strengths"] is not None:
                e["strengths"] = sorted(set(ov["strengths"]))
        out.append(e)
    return out


def _load_overrides(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"overrides file must be a mapping: {path}")
    return data


def build_catalog(models_json: Path, overrides_path: Path | None = None) -> list[CatalogEntry]:
    free = load_free_models(models_json)
    scored = [
        {
            **m,
            "score": auto_score(m),
            "source": "auto",
        }
        for m in free
    ]
    scored = apply_overrides(scored, _load_overrides(overrides_path))
    scored.sort(key=lambda x: (-x["score"], x["id"]))
    return [
        CatalogEntry(
            id=e["id"],
            score=int(e["score"]),
            strengths=e["strengths"],
            context=int(e["context"]),
            reasoning=bool(e["reasoning"]),
            source=str(e["source"]),
        )
        for e in scored
    ]


def write_catalog(catalog: list[CatalogEntry], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": e.id,
            "score": e.score,
            "strengths": sorted(e.strengths),
            "context": e.context,
            "reasoning": e.reasoning,
            "source": e.source,
        }
        for e in catalog
    ]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
