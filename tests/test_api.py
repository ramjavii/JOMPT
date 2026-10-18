"""API: /v1/chat routes via JEV; /v1/rankings; /v1/stats."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jompt.config import Settings
from jompt.main import create_app
from jompt.providers.opencode import ChatResult


@pytest.fixture
def client(models_json_path: Path, empty_overrides: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    settings = Settings(
        models_json_path=models_json_path,
        overrides_path=empty_overrides,
        opencode_api_key="test-key",
    )
    app = create_app(settings)
    return TestClient(app)


def test_rankings_sorted(client: TestClient) -> None:
    r = client.get("/v1/rankings")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 6
    scores = [m["score"] for m in body["models"]]
    assert scores == sorted(scores, reverse=True)
    assert body["models"][0]["id"] == "nemotron-3-ultra-free"


def test_chat_selects_and_calls_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jompt.providers import opencode as op

    captured: dict = {}

    async def fake_chat(self, model: str, messages: list, **kw) -> ChatResult:
        captured["model"] = model
        return ChatResult(
            content="ok",
            model=model,
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
        )

    monkeypatch.setattr(op.OpenCodeProvider, "chat", fake_chat)

    r = client.post("/v1/chat", json={"prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == captured["model"]
    # "hi" is trivial → lightest free model
    assert body["model"] == "ling-3.0-tiny-free"
    assert body["content"] == "ok"
    assert body["complexity_score"] <= 15
    assert body["model_score"] < 40


def test_chat_force_model(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from jompt.providers import opencode as op

    async def fake_chat(self, model: str, messages: list, **kw) -> ChatResult:
        return ChatResult(content="ok", model=model, input_tokens=1, output_tokens=1, finish_reason="stop")

    monkeypatch.setattr(op.OpenCodeProvider, "chat", fake_chat)
    r = client.post("/v1/chat", json={"prompt": "hi", "force_model": "nemotron-3-ultra-free"})
    assert r.status_code == 200
    assert r.json()["model"] == "nemotron-3-ultra-free"


def test_chat_rejects_unknown_force_model(client: TestClient) -> None:
    r = client.post("/v1/chat", json={"prompt": "hi", "force_model": "nope"})
    assert r.status_code == 400


def test_chat_validation_empty_prompt(client: TestClient) -> None:
    r = client.post("/v1/chat", json={"prompt": ""})
    assert r.status_code == 422


def test_stats_after_chat(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from jompt.providers import opencode as op

    async def fake_chat(self, model: str, messages: list, **kw) -> ChatResult:
        return ChatResult(content="ok", model=model, input_tokens=10, output_tokens=5, finish_reason="stop")

    monkeypatch.setattr(op.OpenCodeProvider, "chat", fake_chat)
    client.post("/v1/chat", json={"prompt": "hello world"})
    r = client.get("/v1/stats")
    assert r.status_code == 200
    stats = r.json()
    assert stats["request_count"] == 1
    assert "ling-3.0-tiny-free" in stats["models"]


def test_chat_complex_prompt_picks_heavier_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jompt.providers import opencode as op

    captured: dict = {}

    async def fake_chat(self, model: str, messages: list, **kw) -> ChatResult:
        captured["model"] = model
        return ChatResult(content="ok", model=model, input_tokens=100, output_tokens=100, finish_reason="stop")

    monkeypatch.setattr(op.OpenCodeProvider, "chat", fake_chat)

    prompt = (
        "Design a multi-region active-active database with conflict resolution. "
        "Prove consistency under partition and write rollback migration plans. "
        + "Deeply optimize consensus failure modes. " * 40
    )
    r = client.post("/v1/chat", json={"prompt": prompt})
    assert r.status_code == 200
    body = r.json()
    assert body["complexity_score"] >= 50
    assert body["model_score"] >= body["complexity_score"] or body["model"] == "nemotron-3-ultra-free"
    # must not pick the lightest for a hard task
    assert body["model"] != "ling-3.0-tiny-free"
