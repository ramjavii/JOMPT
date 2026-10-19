"""OpenAI-compatible chat client for the opencode inference endpoint."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


class ProviderError(RuntimeError):
    """Upstream failure with a stable machine-readable kind.

    kind values: free_tier | auth | funds | unavailable | deprecated |
    rate_limit | unknown
    """

    SYSTEMIC_KINDS = frozenset({"free_tier", "auth", "funds"})

    # Sanitized, client-safe hints per kind. Raw upstream bodies stay
    # server-side (logged); never put upstream text here.
    KIND_HINTS: dict[str, str] = {
        "free_tier": (
            "free tier models can only be used from within the OpenCode app; "
            "run JOMPT against a workspace with paid model access"
        ),
        "auth": "upstream authentication failed; check OPENCODE_API_KEY",
        "funds": "upstream account has insufficient funds",
        "unavailable": "the selected model is unavailable upstream",
        "deprecated": "the selected model is deprecated upstream",
        "rate_limit": "upstream rate limit exceeded; retry shortly",
        "unknown": "upstream provider error",
    }

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        kind: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind

    @property
    def is_systemic(self) -> bool:
        """True when retrying a different model cannot fix the error."""
        return self.kind in self.SYSTEMIC_KINDS

    @property
    def user_hint(self) -> str:
        """Client-safe explanation; never echoes the upstream body."""
        return self.KIND_HINTS.get(self.kind, self.KIND_HINTS["unknown"])


_KIND_MARKERS: tuple[tuple[str, str], ...] = (
    ("freetier", "free_tier"),
    ("free tier", "free_tier"),
    ("invalid credential", "auth"),
    ("unauthorized", "auth"),
    ("insufficient account funds", "funds"),
    ("model is unavailable", "unavailable"),
    ("model access is disabled", "unavailable"),
    ("deprecated", "deprecated"),
    ("rate limit", "rate_limit"),
)

_STATUS_KINDS: dict[int, str] = {
    401: "auth",
    402: "funds",
    429: "rate_limit",
}


def _extract_message(parsed: Any) -> str | None:
    """Walk nested OpenAI/opencode error envelopes for a human message."""
    if isinstance(parsed, dict):
        for key in ("message", "detail"):
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for nest_key in ("error", "data"):
            nested = parsed.get(nest_key)
            if isinstance(nested, (dict, list)):
                found = _extract_message(nested)
                if found:
                    return found
    if isinstance(parsed, list):
        for item in parsed:
            found = _extract_message(item)
            if found:
                return found
    return None


def _extract_type_tag(parsed: Any) -> str | None:
    """Find a structured error type tag (e.g. FreeTierError, ModelDeprecated)."""
    if isinstance(parsed, dict):
        tag = parsed.get("type")
        if isinstance(tag, str) and tag not in ("error", "server_error") and tag.strip():
            return tag.strip()
        for nest_key in ("error", "data"):
            nested = parsed.get(nest_key)
            if isinstance(nested, (dict, list)):
                found = _extract_type_tag(nested)
                if found:
                    return found
    if isinstance(parsed, list):
        for item in parsed:
            found = _extract_type_tag(item)
            if found:
                return found
    return None


def classify_provider_error(status_code: int, body: str) -> tuple[str, str]:
    """Parse an upstream error body → (message, kind). Never raises."""
    message = ""
    parsed: Any = None
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if parsed is not None:
        message = _extract_message(parsed) or ""
    if not message:
        message = body.strip()[:300] or f"HTTP {status_code}"
    lowered = message.lower()
    # Also look at structured error type fields (e.g. "type": "FreeTierError")
    type_tag = _extract_type_tag(parsed) if parsed is not None else None
    haystack = lowered + (" " + type_tag.lower() if type_tag else "")
    for marker, kind in _KIND_MARKERS:
        if marker in haystack:
            if type_tag and type_tag not in message:
                message = f"{type_tag}: {message}"
            return message, kind
    return message, _STATUS_KINDS.get(status_code, "unknown")


class OpenCodeProvider:
    def __init__(self, api_key: str, base_url: str, *, timeout: float = 120.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        if not self._api_key:
            raise ProviderError("OPENCODE_API_KEY is not configured", kind="auth")
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError("upstream request failed", kind="unknown") from exc

        if resp.status_code >= 400:
            message, kind = classify_provider_error(resp.status_code, resp.text)
            raise ProviderError(
                f"upstream provider error {resp.status_code}: {message}",
                status_code=resp.status_code,
                kind=kind,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError("provider returned invalid JSON") from exc
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("provider returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            content=message.get("content") or "",
            model=data.get("model") or model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )
