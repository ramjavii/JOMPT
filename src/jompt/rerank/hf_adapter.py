"""Hugging Face Hub signal adapter (optional, fail-open, httpx only)."""
from __future__ import annotations

import math
import time
from typing import Any

import httpx

HF_API = "https://huggingface.co/api/models"
_cache: dict[str, tuple[float, dict[str, dict[str, float]]]] = {}
TTL_S = 3600.0


def _norm_opencode_id(mid: str) -> str:
    return mid.replace("-free", "").replace("_", "-").lower()


def fetch_hf_signals(
    ids: list[str], *, timeout: float = 10.0, client: httpx.Client | None = None
) -> dict[str, dict[str, float]]:
    """Map opencode id → {likes, downloads, trending}. Empty on any failure."""
    key = ",".join(sorted(ids))
    now = time.monotonic()
    if key in _cache and now - _cache[key][0] < TTL_S:
        return {k: dict(v) for k, v in _cache[key][1].items()}
    # Note: cache stores per-call result dict below.
    try:
        own_client = client is None
        c = client or httpx.Client(timeout=timeout)
        try:
            out: dict[str, dict[str, float]] = {}
            for mid in ids:
                try:
                    q = _norm_opencode_id(mid)
                    resp = c.get(HF_API, params={"search": q, "limit": 1})
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    item = data[0] if isinstance(data, list) and data else {}
                    if not isinstance(item, dict):
                        continue
                    likes = float(item.get("likes") or 0)
                    downloads = float(item.get("downloads") or 0)
                    out[mid] = {"likes": likes, "downloads": downloads,
                                "trending": math.log1p(likes) + math.log1p(downloads)}
                except (httpx.HTTPError, ValueError, KeyError):
                    continue
        finally:
            if own_client:
                c.close()
    except Exception:
        return {}
    _cache[key] = (now, out)
    return dict(out)


def clear_cache() -> None:
    _cache.clear()
