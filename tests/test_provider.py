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


@pytest.mark.asyncio
async def test_chat_missing_key() -> None:
    p = OpenCodeProvider("", BASE)
    with pytest.raises(ProviderError, match="OPENCODE_API_KEY"):
        await p.chat("m1", [{"role": "user", "content": "hi"}])
