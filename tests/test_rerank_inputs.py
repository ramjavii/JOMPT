"""Phase 1 tests: id ingestion + registry filtering."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jompt.rerank.inputs import load_requested_ids
from jompt.registry import diff_new_ids, get_models_by_ids, list_free_ids


def _models_json(tmp_path: Path) -> Path:
    p = tmp_path / "models.json"
    p.write_text(
        json.dumps(
            {
                "opencode": {
                    "models": {
                        "a-free": {"cost": {"input": 0, "output": 0}, "limit": {"context": 1000}},
                        "b-free": {"cost": {"input": 0, "output": 0}, "limit": {"context": 1000}},
                        "c-paid": {"cost": {"input": 1, "output": 1}, "limit": {"context": 1000}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return p


def test_get_models_by_ids_preserves_order(tmp_path: Path) -> None:
    mj = _models_json(tmp_path)
    out = get_models_by_ids(mj, ["b-free", "a-free"])
    assert [m["id"] for m in out] == ["b-free", "a-free"]


def test_get_models_rejects_paid_unknown_dup_empty(tmp_path: Path) -> None:
    mj = _models_json(tmp_path)
    with pytest.raises(ValueError, match="paid"):
        get_models_by_ids(mj, ["c-paid"])
    with pytest.raises(ValueError, match="unknown"):
        get_models_by_ids(mj, ["ghost"])
    with pytest.raises(ValueError, match="duplicate"):
        get_models_by_ids(mj, ["a-free", "a-free"])
    with pytest.raises(ValueError, match="empty"):
        get_models_by_ids(mj, [])


def test_diff_new_ids(tmp_path: Path) -> None:
    mj = _models_json(tmp_path)
    assert diff_new_ids(mj, {"a-free"}) == ["b-free"]
    assert list_free_ids(mj) == ["a-free", "b-free"]


def test_load_requested_ids_sources(tmp_path: Path) -> None:
    mj = _models_json(tmp_path)
    assert load_requested_ids(ids="b-free, a-free", models_json=mj) == ["b-free", "a-free"]
    f = tmp_path / "ids.txt"
    f.write_text("a-free\n# comment\nb-free\n", encoding="utf-8")
    assert load_requested_ids(ids_file=f, models_json=mj) == ["a-free", "b-free"]
    jf = tmp_path / "ids.json"
    jf.write_text(json.dumps(["a-free"]), encoding="utf-8")
    assert load_requested_ids(ids_file=jf, models_json=mj) == ["a-free"]
    assert load_requested_ids(from_models_json=mj, models_json=mj) == ["a-free", "b-free"]
    assert load_requested_ids(include_new=True, models_json=mj, known_ids=["a-free"]) == ["b-free"]
    assert load_requested_ids(stdin_text="a-free\n", models_json=mj) == ["a-free"]
    with pytest.raises(ValueError, match="exactly one"):
        load_requested_ids(models_json=mj)
    with pytest.raises(ValueError, match="duplicate"):
        load_requested_ids(ids="a-free,a-free", models_json=mj)
