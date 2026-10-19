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


# --- RED: surface real upstream errors and fail fast on systemic ones ---------

def test_chat_all_fail_surfaces_sanitized_hint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """502 detail must carry a sanitized classification hint (raw body stays server-side)."""
    from jompt.providers import opencode as op
    from jompt.providers.opencode import ProviderError

    calls: list[str] = []

    async def failing_chat(self, model: str, messages: list, **kw) -> ChatResult:
        calls.append(model)
        raise ProviderError(
            "upstream provider error 403: FreeTierError: "
            "OpenCode's free tier can only be used from within OpenCode. secret-tenant-123",
            status_code=403,
            kind="free_tier",
        )

    monkeypatch.setattr(op.OpenCodeProvider, "chat", failing_chat)
    r = client.post("/v1/chat", json={"prompt": "hi"})
    assert r.status_code == 502
    detail = r.json()["detail"]
    # sanitized hint present and actionable
    assert "free tier" in detail.lower()
    assert "OpenCode app" in detail
    # but the raw upstream body (and any secrets in it) must NOT leak
    assert "secret-tenant-123" not in detail


def test_chat_fails_fast_on_systemic_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A systemic error (free_tier/auth/funds) must not burn the whole fallback pool."""
    from jompt.providers import opencode as op
    from jompt.providers.opencode import ProviderError

    calls: list[str] = []

    async def failing_chat(self, model: str, messages: list, **kw) -> ChatResult:
        calls.append(model)
        raise ProviderError("upstream provider error 403: FreeTierError", status_code=403, kind="free_tier")

    monkeypatch.setattr(op.OpenCodeProvider, "chat", failing_chat)
    r = client.post("/v1/chat", json={"prompt": "hi"})
    assert r.status_code == 502
    assert len(calls) == 1  # stop after first systemic failure


def test_chat_retries_non_systemic_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transient per-model errors (unavailable) still fall through the pool."""
    from jompt.providers import opencode as op
    from jompt.providers.opencode import ProviderError

    calls: list[str] = []

    async def flaky_chat(self, model: str, messages: list, **kw) -> ChatResult:
        calls.append(model)
        if len(calls) == 1:
            raise ProviderError(
                "upstream provider error 400: Upstream request failed: Model is unavailable.",
                status_code=400,
                kind="unavailable",
            )
        return ChatResult(content="ok", model=model, input_tokens=1, output_tokens=1, finish_reason="stop")

    monkeypatch.setattr(op.OpenCodeProvider, "chat", flaky_chat)
    r = client.post("/v1/chat", json={"prompt": "hi"})
    assert r.status_code == 200
    assert len(calls) == 2
    assert r.json()["fallback_from"] == [calls[0]]
