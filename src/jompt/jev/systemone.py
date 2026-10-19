"""System One JEV client: POST /systemone with state + typed questions."""
from __future__ import annotations

from typing import Any

import httpx

from jompt.providers.opencode import ProviderError


class SystemOneClient:
    """Calls JEV via the System One endpoint.

    JEV does not generate text. It evaluates a `state` against typed
    `questions` and returns structured decisions (choice/noul/score).
    """

    def __init__(
        self,
        base_url: str,
        model: str = "jev-1.13-free",
        api_key: str = "",
        *,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        payload = {"model": self._model, "state": state, "questions": questions}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = httpx.post(
                f"{self._base_url}/systemone",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError("upstream request failed") from exc
        if resp.status_code >= 400:
            raise ProviderError(
                f"upstream provider error {resp.status_code}", status_code=resp.status_code
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError("provider returned invalid JSON") from exc
