"""Rank list: ordered config of models the JEV selector may choose from.

Config file shape:
    {"default": "<id>", "models": [{"id": "...", "description": "..."}, ...]}
Models are ordered heaviest first; the order is preserved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_ID_LEN = 200


@dataclass(frozen=True, slots=True)
class RankEntry:
    id: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class RankList:
    default: str
    models: tuple[RankEntry, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.models)


def _parse_entry(raw: Any) -> RankEntry:
    if not isinstance(raw, dict):
        raise ValueError("rank list entries must be objects")
    mid = raw.get("id")
    if not isinstance(mid, str) or not mid.strip() or len(mid) > MAX_ID_LEN:
        raise ValueError("rank list entry has invalid id")
    desc = raw.get("description") or ""
    if not isinstance(desc, str):
        raise ValueError("rank list description must be a string")
    return RankEntry(id=mid.strip(), description=desc.strip())


def parse_rank_list(data: Any) -> RankList:
    if not isinstance(data, dict):
        raise ValueError("rank list must be a JSON object")
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("rank list is empty")
    entries = tuple(_parse_entry(m) for m in raw_models)
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("rank list has duplicate ids")
    default = data.get("default")
    if default not in ids:
        raise ValueError("rank list default must be one of the listed ids")
    return RankList(default=default, models=entries)


def load_rank_list(path: Path) -> RankList:
    if not path.is_file():
        raise FileNotFoundError(f"rank list not found: {path}")
    return parse_rank_list(json.loads(path.read_text(encoding="utf-8")))
