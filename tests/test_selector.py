"""JEV selector: rank list loading, strict parsing, retry/fallback, CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jompt.jev.rank_list import RankList, load_rank_list
from jompt.jev.selector import build_questions, parse_choice, select_model
from jompt.providers.opencode import ProviderError
from jompt.select_cli import main as cli_main

RANK = [
    {"id": "heavy-a", "description": "largest reasoning model"},
    {"id": "mid-b", "description": "general coding model"},
    {"id": "light-c", "description": "tiny fast model"},
]


class FakeJev:
    def __init__(self, replies: list[object]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    def decide(self, state: str, questions: dict) -> dict:
        self.calls.append((state, questions))
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def rank_file(tmp_path: Path) -> Path:
    p = tmp_path / "rank_list.json"
    p.write_text(json.dumps({"default": "mid-b", "models": RANK}), encoding="utf-8")
    return p


def _rank() -> RankList:
    return RankList(
        default="mid-b",
        models=tuple(
            __import__("jompt.jev.rank_list", fromlist=["RankEntry"]).RankEntry(
                id=r["id"], description=r["description"]
            )
            for r in RANK
        ),
    )


# --- rank list -------------------------------------------------------------


def test_load_rank_list_preserves_order(rank_file: Path) -> None:
    rl = load_rank_list(rank_file)
    assert isinstance(rl, RankList)
    assert [e.id for e in rl.models] == ["heavy-a", "mid-b", "light-c"]
    assert rl.default == "mid-b"


def test_load_rank_list_rejects_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"default": "a", "models": [{"id": "a"}, {"id": "a"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_rank_list(p)


def test_load_rank_list_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"default": "a", "models": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_rank_list(p)


def test_load_rank_list_rejects_default_not_in_list(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"default": "ghost", "models": [{"id": "a"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="default"):
        load_rank_list(p)


def test_load_rank_list_rejects_blank_id(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"default": "a", "models": [{"id": "  "}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        load_rank_list(p)


# --- build_questions -------------------------------------------------------


def test_build_questions_includes_every_id() -> None:
    q = build_questions(_rank())
    criteria = q["route"]["criteria"]
    assert set(criteria) == {"heavy-a", "mid-b", "light-c"}
    assert q["route"]["type"] == "choice"


# --- parse_choice ----------------------------------------------------------


def test_parse_choice_extracts_valid_id() -> None:
    resp = {"answers": {"route": {"type": "choice", "choice": "light-c"}}}
    assert parse_choice(resp, ("heavy-a", "mid-b", "light-c")) == "light-c"


def test_parse_choice_rejects_unknown_id() -> None:
    resp = {"answers": {"route": {"type": "choice", "choice": "unknown"}}}
    assert parse_choice(resp, ("heavy-a", "mid-b", "light-c")) is None


def test_parse_choice_rejects_missing_answers() -> None:
    ids = ("heavy-a", "mid-b", "light-c")
    assert parse_choice({}, ids) is None
    assert parse_choice({"answers": {}}, ids) is None
    assert parse_choice({"answers": {"route": {}}}, ids) is None


# --- select_model ----------------------------------------------------------


def test_select_returns_jev_choice() -> None:
    jev = FakeJev([{"answers": {"route": {"choice": "light-c"}}}])
    assert select_model("hi", _rank(), jev) == "light-c"
    assert len(jev.calls) == 1


def test_select_falls_back_to_default_on_invalid_choice() -> None:
    jev = FakeJev([{"answers": {"route": {"choice": "bogus"}}}])
    assert select_model("hi", _rank(), jev) == "mid-b"


def test_select_falls_back_to_default_on_provider_error() -> None:
    jev = FakeJev([ProviderError("upstream provider error 403", status_code=403)])
    assert select_model("hi", _rank(), jev) == "mid-b"


def test_select_falls_back_to_default_on_malformed_response() -> None:
    jev = FakeJev([{"unexpected": "shape"}])
    assert select_model("hi", _rank(), jev) == "mid-b"


def test_select_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        select_model("   ", _rank(), FakeJev([]))


def test_select_rejects_oversized_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        select_model("x" * 32_001, _rank(), FakeJev([]))


# --- CLI -------------------------------------------------------------------


def test_cli_prints_selected_id(
    rank_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import jompt.select_cli as cli

    class FakeJev:
        def decide(self, state, questions):
            return {"answers": {"route": {"choice": "light-c"}}}

    monkeypatch.setattr(cli, "_make_client", lambda settings: FakeJev())
    code = cli_main(["--rank-list", str(rank_file), "say hi"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "light-c"


def test_cli_requires_prompt(rank_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        cli_main(["--rank-list", str(rank_file)])
    assert ei.value.code == 2
