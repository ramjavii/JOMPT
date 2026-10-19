"""Tests for the System One JEV client and selector."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from jompt.jev.rank_list import RankEntry, RankList
from jompt.jev.selector import build_questions, parse_choice, select_model
from jompt.jev.systemone import SystemOneClient
from jompt.providers.opencode import ProviderError

BASE = "https://opencode.ai/zen/v1"
IDS = ("heavy-a", "mid-b", "light-c")


def _rank() -> RankList:
    return RankList(
        default="mid-b",
        models=(
            RankEntry(id="heavy-a", description="largest reasoning model"),
            RankEntry(id="mid-b", description="general coding model"),
            RankEntry(id="light-c", description="tiny fast model"),
        ),
    )


class FakeSystemOne:
    def __init__(self, replies: list[object]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    def decide(self, state: str, questions: dict) -> dict:
        self.calls.append((state, questions))
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# --- build_questions -------------------------------------------------------


def test_build_questions_includes_every_id() -> None:
    q = build_questions(_rank())
    criteria = q["route"]["criteria"]
    assert set(criteria) == set(IDS)
    assert q["route"]["type"] == "choice"


# --- parse_choice ----------------------------------------------------------


def test_parse_choice_extracts_valid_id() -> None:
    resp = {"answers": {"route": {"type": "choice", "choice": "light-c", "confidence": 0.9}}}
    assert parse_choice(resp, IDS) == "light-c"


def test_parse_choice_rejects_unknown_id() -> None:
    resp = {"answers": {"route": {"type": "choice", "choice": "unknown"}}}
    assert parse_choice(resp, IDS) is None


def test_parse_choice_rejects_missing_answers() -> None:
    assert parse_choice({}, IDS) is None
    assert parse_choice({"answers": {}}, IDS) is None
    assert parse_choice({"answers": {"route": {}}}, IDS) is None


# --- select_model ----------------------------------------------------------


def test_select_returns_jev_choice() -> None:
    jev = FakeSystemOne([{"answers": {"route": {"choice": "light-c"}}}])
    assert select_model("hi", _rank(), jev) == "light-c"
    assert len(jev.calls) == 1


def test_select_falls_back_to_default_on_invalid_choice() -> None:
    jev = FakeSystemOne([{"answers": {"route": {"choice": "bogus"}}}])
    assert select_model("hi", _rank(), jev) == "mid-b"


def test_select_falls_back_to_default_on_provider_error() -> None:
    jev = FakeSystemOne([ProviderError("upstream provider error 403", status_code=403)])
    assert select_model("hi", _rank(), jev) == "mid-b"


def test_select_falls_back_to_default_on_malformed_response() -> None:
    jev = FakeSystemOne([{"unexpected": "shape"}])
    assert select_model("hi", _rank(), jev) == "mid-b"


def test_select_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        select_model("   ", _rank(), FakeSystemOne([]))


def test_select_rejects_oversized_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        select_model("x" * 32_001, _rank(), FakeSystemOne([]))


# --- SystemOneClient HTTP --------------------------------------------------


@respx.mock
def test_systemone_decide_success() -> None:
    route = respx.post(f"{BASE}/systemone").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "jev-1.13-free",
                "answers": {"route": {"type": "choice", "choice": "light-c"}},
                "cost": "0",
            },
        )
    )
    client = SystemOneClient(base_url=BASE, model="jev-1.13-free")
    result = client.decide("say hi", {"route": {"type": "choice", "criteria": {}}})
    assert result["answers"]["route"]["choice"] == "light-c"
    sent = route.calls.last.request
    body = json.loads(sent.content)
    assert body["model"] == "jev-1.13-free"
    assert body["state"] == "say hi"
    assert "route" in body["questions"]


@respx.mock
def test_systemone_decide_sends_auth_header_when_key_present() -> None:
    route = respx.post(f"{BASE}/systemone").mock(
        return_value=httpx.Response(200, json={"answers": {}})
    )
    client = SystemOneClient(base_url=BASE, model="jev-1.13", api_key="secret-key")
    client.decide("hi", {})
    assert route.calls.last.request.headers["Authorization"] == "Bearer secret-key"


@respx.mock
def test_systemone_decide_no_auth_header_without_key() -> None:
    route = respx.post(f"{BASE}/systemone").mock(
        return_value=httpx.Response(200, json={"answers": {}})
    )
    client = SystemOneClient(base_url=BASE, model="jev-1.13-free")
    client.decide("hi", {})
    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_systemone_decide_http_error() -> None:
    respx.post(f"{BASE}/systemone").mock(return_value=httpx.Response(403, text="forbidden"))
    client = SystemOneClient(base_url=BASE, model="jev-1.13-free")
    with pytest.raises(ProviderError) as ei:
        client.decide("hi", {})
    assert ei.value.status_code == 403
    assert "forbidden" not in str(ei.value)


@respx.mock
def test_systemone_decide_invalid_json() -> None:
    respx.post(f"{BASE}/systemone").mock(return_value=httpx.Response(200, text="not json"))
    client = SystemOneClient(base_url=BASE, model="jev-1.13-free")
    with pytest.raises(ProviderError, match="invalid JSON"):
        client.decide("hi", {})
