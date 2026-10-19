"""Generate config/rank_list.json from catalog.json (single source of truth).

catalog.json is authoritative. rank_list.json is a generated artifact for the
JEV choice-question selector path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jompt.jev.rank_list import parse_rank_list
from jompt.ranking import CatalogEntry


def catalog_to_rank_payload(
    catalog: list[CatalogEntry], default: str | None = None
) -> dict[str, Any]:
    if not catalog:
        raise ValueError("catalog is empty")
    ordered = sorted(catalog, key=lambda e: (-e.score, e.id))
    ids = [e.id for e in ordered]
    if default is not None and default not in ids:
        raise ValueError("default must be one of the catalog ids")
    chosen_default = default if default is not None else ids[len(ids) // 2]
    models = []
    for e in ordered:
        tags = ",".join(sorted(e.strengths)) if e.strengths else ""
        desc = f"score {e.score}; {tags}" if tags else f"score {e.score}"
        models.append({"id": e.id, "description": desc})
    payload = {"default": chosen_default, "models": models}
    parse_rank_list(payload)  # validate shape before returning
    return payload


def load_catalog_entries(path: Path) -> list[CatalogEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"catalog is empty: {path}")
    out: list[CatalogEntry] = []
    for item in raw:
        out.append(
            CatalogEntry(
                id=str(item["id"]),
                score=int(item["score"]),
                strengths=tuple(item.get("strengths") or []),
                context=int(item.get("context") or 0),
                reasoning=bool(item.get("reasoning")),
                source=str(item.get("source") or "auto"),
            )
        )
    return out


def write_rank_list(payload: dict[str, Any], out_path: Path) -> None:
    parse_rank_list(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
