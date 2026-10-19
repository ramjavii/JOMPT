"""Blend auto / AI / HF signals into final 0–100 scores (pure function)."""
from __future__ import annotations

import math


def _check_weight(name: str, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be 0.0–1.0")
    return v


def hf_bonus(signal: dict[str, float] | None, cap: float = 6.0) -> float:
    if not signal:
        return 0.0
    raw = math.log1p(signal.get("likes", 0.0)) + math.log1p(signal.get("downloads", 0.0))
    # log1p(1e6)≈13.8 per term; scale so world-class ≈ cap, median ≈ 1-2.
    return max(0.0, min(cap, raw * (cap / 12.0)))


def blend_scores(
    auto: int,
    ai: int | None = None,
    hf_signal: dict[str, float] | None = None,
    *,
    w_ai: float = 1.0,
    w_hf: float = 1.0,
) -> int:
    _check_weight("w_ai", w_ai)
    _check_weight("w_hf", w_hf)
    base = float(ai) if ai is not None else float(auto)
    if ai is not None and w_ai < 1.0:
        base = w_ai * float(ai) + (1.0 - w_ai) * float(auto)
    final = base + w_hf * hf_bonus(hf_signal)
    return max(0, min(100, int(round(final))))
