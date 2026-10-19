"""OpenCodeProvider client tests via respx."""
from __future__ import annotations

import httpx
import pytest
import respx

from jompt.providers.opencode import ChatResult, OpenCodeProvider, ProviderError

BASE = "https://opencode.ai/inference/openai/v1"


def _provider() -> OpenCodeProvider:
    return OpenCodeProvider("k", BASE, timeout=5.0)


@pytest.mark.asyncio
@respx.mock
async def test_chat_success() -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "m1",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )
    )
    result = await _provider().chat("m1", [{"role": "user", "content": "hi"}])
    assert isinstance(result, ChatResult)
    assert result.content == "hello"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.model == "m1"


@pytest.mark.asyncio
@respx.mock
async def test_chat_http_error() -> None:
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(ProviderError) as ei:
        await _provider().chat("m1", [{"role": "user", "content": "hi"}])
    assert ei.value.status_code == 500


@pytest.mark.asyncio
@respx.mock
async def test_chat_no_choices() -> None:
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json={"choices": []}))
    with pytest.raises(ProviderError, match="no choices"):
        await _provider().chat("m1", [{"role": "user", "content": "hi"}])


# --- RED: upstream error body must be parsed and classified -------------------

FREE_TIER_BODY = {
    "type": "error",
    "error": {
        "type": "error",
        "error": {
            "type": "FreeTierError",
            "message": "OpenCode's free tier can only be used from within OpenCode.",
        },
    },
}


@pytest.mark.asyncio
@respx.mock
async def test_chat_error_includes_upstream_message() -> None:
    """Doubly-nested opencode error body → message + kind free_tier."""
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(403, json=FREE_TIER_BODY)
    )
    with pytest.raises(ProviderError) as ei:
        await _provider().chat("m1", [{"role": "user", "content": "hi"}])
    assert ei.value.status_code == 403
    assert "FreeTierError" in str(ei.value)
    assert "within OpenCode" in str(ei.value)
    assert ei.value.kind == "free_tier"


@pytest.mark.asyncio
@respx.mock
async def test_chat_error_openai_shape_body() -> None:
    """OpenAI-style {'error': {'message': ...}} body is parsed."""
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"type": "server_error", "message": "Upstream request failed: Model is unavailable."}},
        )
    )
    with pytest.raises(ProviderError) as ei:
        await _provider().chat("m1", [{"role": "user", "content": "hi"}])
    assert "Model is unavailable" in str(ei.value)
    assert ei.value.kind == "unavailable"


@pytest.mark.asyncio
@respx.mock
async def test_chat_error_plain_text_body() -> None:
    """Non-JSON body falls back to raw text in the message."""
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(403, text="Legacy inference endpoint retired")
    )
    with pytest.raises(ProviderError) as ei:
        await _provider().chat("m1", [{"role": "user", "content": "hi"}])
    assert "Legacy inference endpoint retired" in str(ei.value)


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (401, {"error": {"message": "Invalid credential"}}, "auth"),
        (402, {"error": {"message": "Upstream request failed: Insufficient account funds"}}, "funds"),
        (410, {"error": {"error": {"type": "ModelDeprecated", "message": "Model deprecated."}}}, "deprecated"),
        (429, {"error": {"message": "rate limited"}}, "rate_limit"),
        (500, {"error": {"message": "something odd"}}, "unknown"),
    ],
)
def test_chat_error_kind_classification(status: int, body: dict, kind: str) -> None:
    """Status/message combinations map to stable kind labels."""
    import json

    from jompt.providers.opencode import classify_provider_error

    message, classified = classify_provider_error(status, json.dumps(body))
    assert classified == kind
    assert message  # never empty


def test_provider_error_is_systemic() -> None:
    """Systemic errors cannot be fixed by retrying a different free model."""
    systemic = [
        ProviderError("a", status_code=403, kind="free_tier"),
        ProviderError("b", status_code=401, kind="auth"),
        ProviderError("c", status_code=402, kind="funds"),
    ]
    for e in systemic:
        assert e.is_systemic is True
    assert ProviderError("d", status_code=500, kind="unknown").is_systemic is False
    assert ProviderError("e", status_code=400, kind="unavailable").is_systemic is False


def test_provider_error_user_hint_is_sanitized() -> None:
    """user_hint is a fixed safe string per kind; unknown kinds fall back."""
    e = ProviderError("raw upstream body with secrets", status_code=403, kind="free_tier")
    assert "secrets" not in e.user_hint
    assert "OpenCode app" in e.user_hint
    assert ProviderError("x", kind="weird-new-kind").user_hint == "upstream provider error"


@pytest.mark.asyncio
@respx.mock
async def test_chat_passes_temperature_and_max_tokens() -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "m1",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )
    )
    await _provider().chat(
        "m1", [{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=64
    )
    sent = route.calls[0].request
    import json as _json

    payload = _json.loads(sent.content)
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 64


@pytest.mark.asyncio
@respx.mock
async def test_chat_invalid_json_body() -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, text="not-json{{{")
    )
    with pytest.raises(ProviderError, match="invalid JSON"):
        await _provider().chat("m1", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
@respx.mock
async def test_chat_transport_error() -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with pytest.raises(ProviderError, match="upstream request failed"):
        await _provider().chat("m1", [{"role": "user", "content": "hi"}])


def test_classify_list_shaped_body() -> None:
    """Error bodies that are JSON lists still yield a message."""
    from jompt.providers.opencode import classify_provider_error

    message, kind = classify_provider_error(400, '[{"message": "batch item failed"}]')
    assert message == "batch item failed"
    assert kind == "unknown"


def test_classify_empty_body() -> None:
    """Empty body degrades to a status-coded message, never raises."""
    from jompt.providers.opencode import classify_provider_error

    message, kind = classify_provider_error(502, "")
    assert message == "HTTP 502"
    assert kind == "unknown"


@pytest.mark.asyncio
async def test_chat_missing_key() -> None:
    p = OpenCodeProvider("", BASE)
    with pytest.raises(ProviderError, match="OPENCODE_API_KEY"):
        await p.chat("m1", [{"role": "user", "content": "hi"}])
