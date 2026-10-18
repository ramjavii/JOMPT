"""Registry: load only free opencode models."""
from pathlib import Path

from jompt.registry import load_free_models


def test_filters_to_free_opencode_only(models_json_path: Path) -> None:
    models = load_free_models(models_json_path)
    ids = {m["id"] for m in models}
    assert "paid-gpt" not in ids
    assert "ling-3.0-tiny-free" in ids
    assert len(models) == 6


def test_normalized_fields(models_json_path: Path) -> None:
    models = load_free_models(models_json_path)
    tiny = next(m for m in models if m["id"] == "ling-3.0-tiny-free")
    assert tiny["context"] == 262144
    assert tiny["reasoning"] is False
    assert tiny["tool_call"] is False
    assert "flash" not in tiny["strengths"]  # tiny, not flash


def test_missing_file_raises(tmp_path: Path) -> None:
    try:
        load_free_models(tmp_path / "nope.json")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
