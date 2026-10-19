"""Merge AI scores into overrides (immutable) and rebuild artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from jompt.ranking import CatalogEntry, apply_overrides, auto_score
from jompt.rerank.ai_ranker import AIRank

AUTO_RERANK_MARKER = "# auto-rerank"


def merge_overrides(
    existing: dict[str, Any], ai_ranks: list[AIRank], fallback_scores: dict[str, int] | None = None
) -> dict[str, Any]:
    """Return a NEW overrides dict; never mutates `existing`."""
    merged: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in existing.items()}
    fb = fallback_scores or {}
    for r in ai_ranks:
        entry: dict[str, Any] = {"score": int(r.score)}
        if r.strengths:
            entry["strengths"] = sorted(r.strengths)
        merged[r.id] = entry
    for mid, score in fb.items():
        if mid not in {r.id for r in ai_ranks}:
            merged.setdefault(mid, {"score": int(score)})
    return dict(merged)


def load_overrides(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"overrides file must be a mapping: {path}")
    return data


def write_overrides(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Curated capability overrides for JOMPT ranking.\n" + AUTO_RERANK_MARKER + ": entries below may be machine-updated by rerank_cli\n"
    path.write_text(header + yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def scored_diff_table(
    models: list[dict[str, Any]], new_scores: dict[str, int]
) -> list[tuple[str, int, int]]:
    """(id, old_auto, new) sorted heavy→light by new score."""
    rows = [(m["id"], auto_score(m), int(new_scores.get(m["id"], auto_score(m)))) for m in models]
    rows.sort(key=lambda r: (-r[2], r[0]))
    return rows


def format_diff_table(rows: list[tuple[str, int, int]]) -> str:
    lines = ["id  auto→new"]
    for mid, old, new in rows:
        lines.append(f"{new:3d}  {mid}  ({old}→{new})")
    return "\n".join(lines)
