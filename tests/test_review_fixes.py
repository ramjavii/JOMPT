"""Post-review regression tests: fallback tags, _is_free missing cost, auth."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jompt.config import Settings
from jompt.jev.router import filter_by_tags, pick_model
from jompt.jev.scorer import PromptProfile
from jompt.main import create_app
from jompt.providers.opencode import ChatResult, ProviderError
from jompt.ranking import CatalogEntry
from jompt.registry import load_free_models, _is_free


def test_is_free_rejects_missing_cost() -> None:
    assert _is_free({"cost": None}) is False
    assert _is_free({}) is False
    assert _is_free({"cost": {"input": 0, "output": 0}}) is True
    assert _is_free({"cost": {"input": 1, "output": 0}}) is False


def test_filter_by_tags() -> None:
    cat = [
        CatalogEntry(id="v", score=50, strengths=frozenset({"vision"}), context=1, reasoning=False, source="auto"),
        CatalogEntry(id="t", score=20, strengths=frozenset(), context=1, reasoning=False, source="auto"),
    ]
    assert [e.id for e in filter_by_tags(cat, {"vision"})] == ["v"]
    assert len(filter_by_tags(cat, set())) == 2


def test_fallback_records_failed_ids(
    models_json_path: Path, empty_overrides: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    settings = Settings(
        models_json_path=models_json_path,
        overrides_path=empty_overrides,
        opencode_api_key="test-key",
    )
    client = TestClient(create_app(settings))

    from jompt.providers import opencode as op

    calls: list[str] = []

    async def flaky(self, model: str, messages: list, **kw) -> ChatResult:
        calls.append(model)
        if len(calls) < 3:
            raise ProviderError("boom", status_code=500)
        return ChatResult(content="ok", model=model, input_tokens=1, output_tokens=1, finish_reason="stop")

    monkeypatch.setattr(op.OpenCodeProvider, "chat", flaky)
    r = client.post("/v1/chat", json={"prompt": "hi", "force_model": "ling-3.0-tiny-free"})
    assert r.status_code == 200
    body = r.json()
    assert body["fallback_from"] == ["ling-3.0-tiny-free", calls[1]]
    assert body["model"] == calls[-1]
    assert len(body["fallback_from"]) == 2


def test_api_key_required_when_configured(
    models_json_path: Path, empty_overrides: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    settings = Settings(
        models_json_path=models_json_path,
        overrides_path=empty_overrides,
        opencode_api_key="test-key",
        jompt_api_key="sekrit",
    )
    client = TestClient(create_app(settings))
    assert client.get("/v1/rankings").status_code == 401
    assert client.get("/v1/rankings", headers={"X-JOMPT-Key": "wrong"}).status_code == 401
    assert client.get("/v1/rankings", headers={"X-JOMPT-Key": "sekrit"}).status_code == 200


def test_unknown_force_model_does_not_echo_input(
    models_json_path: Path, empty_overrides: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    settings = Settings(
        models_json_path=models_json_path,
        overrides_path=empty_overrides,
        opencode_api_key="test-key",
    )
    client = TestClient(create_app(settings))
    r = client.post("/v1/chat", json={"prompt": "hi", "force_model": "totally-unknown-model"})
    assert r.status_code == 400
    assert "totally-unknown-model" not in r.json()["detail"]


def test_provider_error_body_not_leaked_to_client(
    models_json_path: Path, empty_overrides: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    settings = Settings(
        models_json_path=models_json_path,
        overrides_path=empty_overrides,
        opencode_api_key="test-key",
    )
    client = TestClient(create_app(settings))

    from jompt.providers import opencode as op

    async def always_fail(self, model: str, messages: list, **kw) -> ChatResult:
        raise ProviderError("upstream provider error 500: secret-account-xyz", status_code=500)

    monkeypatch.setattr(op.OpenCodeProvider, "chat", always_fail)
    r = client.post("/v1/chat", json={"prompt": "hi"})
    assert r.status_code == 502
    assert "secret-account-xyz" not in r.text
    assert r.json()["detail"] == "all model attempts failed"
