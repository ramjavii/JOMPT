"""Phase 0 tests: catalog → rank_list generation."""
from __future__ import annotations

import json

import pytest

from jompt.jev.rank_list import parse_rank_list
from jompt.ranking import CatalogEntry
from jompt.ranklist import catalog_to_rank_payload, load_catalog_entries, write_rank_list


def _cat() -> list[CatalogEntry]:
    return [
        CatalogEntry(id="b-mid", score=50, strengths=("tools",), context=0, reasoning=False, source="auto"),
        CatalogEntry(id="a-heavy", score=90, strengths=("code", "reasoning"), context=0, reasoning=True, source="auto"),
        CatalogEntry(id="c-light", score=10, strengths=(), context=0, reasoning=False, source="auto"),
    ]


def test_orders_heavy_first_and_validates() -> None:
    payload = catalog_to_rank_payload(_cat(), default="b-mid")
    assert [m["id"] for m in payload["models"]] == ["a-heavy", "b-mid", "c-light"]
    assert payload["models"][0]["description"] == "score 90; code,reasoning"
    parse_rank_list(payload)  # must not raise


def test_default_median_when_unspecified() -> None:
    payload = catalog_to_rank_payload(_cat())
    assert payload["default"] == "b-mid"


def test_rejects_empty_and_bad_default() -> None:
    with pytest.raises(ValueError, match="empty"):
        catalog_to_rank_payload([])
    with pytest.raises(ValueError, match="default"):
        catalog_to_rank_payload(_cat(), default="ghost")


def test_round_trip_write_and_load(tmp_path) -> None:
    from jompt.jev.rank_list import load_rank_list

    payload = catalog_to_rank_payload(_cat(), default="a-heavy")
    out = tmp_path / "rank_list.json"
    write_rank_list(payload, out)
    rl = load_rank_list(out)
    assert rl.ids == ("a-heavy", "b-mid", "c-light")

    cat_path = tmp_path / "catalog.json"
    cat_path.write_text(
        json.dumps([{"id": e.id, "score": e.score, "strengths": sorted(e.strengths),
                     "context": e.context, "reasoning": e.reasoning, "source": e.source}
                    for e in _cat()]),
        encoding="utf-8",
    )
    assert len(load_catalog_entries(cat_path)) == 3


def test_idempotent() -> None:
    p1 = catalog_to_rank_payload(_cat(), default="b-mid")
    p2 = catalog_to_rank_payload(_cat(), default="b-mid")
    assert p1 == p2
