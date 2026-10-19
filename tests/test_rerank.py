"""Phase 2+3 tests: AI parser, blend, writer, HF fail-open."""
from __future__ import annotations

import httpx
import respx

from jompt.rerank.ai_ranker import (
    build_judge_messages,
    parse_jev_scores,
    parse_judge_payload,
)
from jompt.rerank.blend import blend_scores, hf_bonus
from jompt.rerank.context import build_enrichment
from jompt.rerank.hf_adapter import clear_cache, fetch_hf_signals
from jompt.rerank.writer import format_diff_table, merge_overrides, scored_diff_table


def test_enrichment_truncates() -> None:
    m = [{"id": "a", "description": "x" * 500, "strengths": ["code"], "context": 1,
          "reasoning": True, "tool_call": True, "name": "A"}]
    out = build_enrichment(m)
    assert len(out[0]["description"]) <= 301
    msgs = build_judge_messages(out)
    assert "UNTRUSTED" in msgs[0]["content"]
    assert "DATA id=a" in msgs[1]["content"]


def test_parse_judge_happy_and_drops_hallucination() -> None:
    payload = [
        {"id": "a", "score": 90, "strengths": ["code"], "description": "great"},
        {"id": "ghost", "score": 100, "strengths": [], "description": "fake"},
        {"id": "b", "score": 999, "strengths": [], "description": "clamped"},
        {"id": "a", "score": 10, "strengths": [], "description": "dupe dropped"},
    ]
    out = parse_judge_payload(payload, {"a", "b"})
    by_id = {r.id: r for r in out}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"].score == 90
    assert by_id["b"].score == 100


def test_parse_judge_invalid_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_judge_payload("not json{{", {"a"})
    with pytest.raises(ValueError):
        parse_judge_payload([], {"a"})
    with pytest.raises(ValueError):
        parse_judge_payload([{"id": "ghost", "score": 5}], {"a"})


def test_parse_jev_scores() -> None:
    resp = {"answers": {"score_a": {"score": 70}, "score_b": {"score": "bad"}}}
    out = parse_jev_scores(resp, {"a", "b"})
    assert [(r.id, r.score) for r in out] == [("a", 70)]


def test_blend_math_and_weights() -> None:
    assert blend_scores(50) == 50
    assert blend_scores(50, ai=80) == 80
    assert blend_scores(50, ai=80, w_ai=0.5) == 65
    assert blend_scores(50, hf_signal={"likes": 1e6, "downloads": 1e6}) > 50
    assert blend_scores(50, hf_signal={"likes": 1e9, "downloads": 1e9}) <= 56
    assert blend_scores(50, ai=90, hf_signal={"likes": 0, "downloads": 0}) == 90
    import pytest

    with pytest.raises(ValueError):
        blend_scores(50, w_ai=2.0)
    assert hf_bonus(None) == 0.0


def test_writer_merge_immutable_and_diff() -> None:
    existing = {"keep": {"score": 10}}
    from jompt.rerank.ai_ranker import AIRank

    merged = merge_overrides(existing, [AIRank(id="a", score=77, strengths=("code",), description="d")],
                             fallback_scores={"b": 42})
    assert existing == {"keep": {"score": 10}}
    assert merged["a"]["score"] == 77
    assert merged["b"]["score"] == 42
    assert merged["keep"]["score"] == 10
    models = [{"id": "a", "reasoning": True, "tool_call": False, "structured_output": False,
               "attachment": False, "context": 0, "strengths": [], "name": "a", "description": ""}]
    rows = scored_diff_table(models, {"a": 77})
    assert "a" in format_diff_table(rows)


@respx.mock
def test_hf_timeout_fail_open() -> None:
    clear_cache()
    respx.get("https://huggingface.co/api/models").mock(side_effect=httpx.ConnectTimeout("boom"))
    assert fetch_hf_signals(["some-model-free"]) == {}


@respx.mock
def test_hf_happy_path() -> None:
    clear_cache()
    respx.get("https://huggingface.co/api/models").respond(
        200, json=[{"likes": 100, "downloads": 1000}])
    out = fetch_hf_signals(["some-model-free"])
    assert out["some-model-free"]["likes"] == 100.0
