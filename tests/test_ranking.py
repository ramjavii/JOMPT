"""Ranking builder: auto-score from metadata + curated overrides."""
from pathlib import Path

from jompt.ranking import apply_overrides, auto_score, build_catalog


def test_auto_score_orders_heavy_over_light(models_json_path: Path) -> None:
    from jompt.registry import load_free_models

    models = {m["id"]: m for m in load_free_models(models_json_path)}
    ultra = auto_score(models["nemotron-3-ultra-free"])
    tiny = auto_score(models["ling-3.0-tiny-free"])
    flash = auto_score(models["deepseek-v4-flash-free"])
    assert ultra > flash > tiny
    assert 0 <= tiny <= 100
    assert 0 <= ultra <= 100


def test_catalog_sorted_heavy_to_light(models_json_path: Path, empty_overrides: Path) -> None:
    catalog = build_catalog(models_json_path, empty_overrides)
    scores = [entry.score for entry in catalog]
    assert scores == sorted(scores, reverse=True)
    assert catalog[0].id == "nemotron-3-ultra-free"
    # lightest cluster at the bottom
    bottom_ids = {e.id for e in catalog if e.score <= 5}
    assert bottom_ids == {"ling-3.0-tiny-free", "ling-3.1-flash-free"}
    assert catalog[-1].score <= 5


def test_overrides_replace_auto_score(models_json_path: Path, tmp_path: Path) -> None:
    ov = tmp_path / "ov.yaml"
    ov.write_text("ling-3.0-tiny-free:\n  score: 99\n", encoding="utf-8")
    catalog = build_catalog(models_json_path, ov)
    by_id = {e.id: e for e in catalog}
    assert by_id["ling-3.0-tiny-free"].score == 99
    assert by_id["ling-3.0-tiny-free"].source == "override"


def test_strength_tags(models_json_path: Path, empty_overrides: Path) -> None:
    catalog = build_catalog(models_json_path, empty_overrides)
    by_id = {e.id: e for e in catalog}
    assert "vision" in by_id["nemotron-3-ultra-free"].strengths
    assert "code" in by_id["step-5-preview-free"].strengths


def test_apply_overrides_only_known_ids() -> None:
    entries = [
        {"id": "a", "score": 10, "source": "auto"},
        {"id": "b", "score": 20, "source": "auto"},
    ]
    out = apply_overrides(entries, {"b": {"score": 5}, "ghost": {"score": 100}})
    by_id = {e["id"]: e for e in out}
    assert by_id["b"]["score"] == 5
    assert by_id["b"]["source"] == "override"
    assert by_id["a"]["score"] == 10
    assert "ghost" not in by_id
