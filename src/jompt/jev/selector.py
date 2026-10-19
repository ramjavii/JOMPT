"""JEV selector using System One choice questions.

JEV evaluates the prompt against a choice question whose criteria are the
rank list entries. It returns the chosen model id with probabilities.
"""
from __future__ import annotations

from typing import Any

from jompt.jev.rank_list import RankList
from jompt.jev.systemone import SystemOneClient
from jompt.providers.opencode import ProviderError

MAX_PROMPT_CHARS = 32_000
QUESTION_ID = "route"


def top_k_by_score(rank: RankList, k: int) -> RankList:
    """Pre-filter to top-K by embedded score for large rank lists.

    Parses leading 'score N' from descriptions (fallback: keep order).
    Default rank is already heavy→light, so slicing preserves intent.
    """
    if k <= 0 or len(rank.models) <= k:
        return rank
    import re as _re

    def _score(desc: str) -> int:
        m = _re.match(r"\s*score\s+(\d+)", desc or "")
        return int(m.group(1)) if m else -1

    scored = sorted(rank.models, key=lambda e: (-_score(e.description), e.id))
    kept = scored[:k]
    # Preserve original heavy→light order among kept.
    order = {e.id: i for i, e in enumerate(rank.models)}
    kept.sort(key=lambda e: order.get(e.id, 0))
    default = rank.default if rank.default in {e.id for e in kept} else kept[len(kept) // 2].id
    return RankList(default=default, models=tuple(kept))


def build_questions(rank: RankList) -> dict[str, Any]:
    criteria = {e.id: e.description or e.id for e in rank.models}
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": (
                "Which model should handle this request? Choose the lightest model "
                "that can handle it well."
            ),
            "criteria": criteria,
        }
    }


def parse_choice(response: dict[str, Any], ids: tuple[str, ...]) -> str | None:
    answers = response.get("answers") or {}
    answer = answers.get(QUESTION_ID) or {}
    choice = answer.get("choice")
    if isinstance(choice, str) and choice in ids:
        return choice
    return None


def select_model(prompt: str, rank: RankList, client: SystemOneClient) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("prompt exceeds maximum length")

    questions = build_questions(rank)
    try:
        response = client.decide(prompt, questions)
    except ProviderError:
        return rank.default
    choice = parse_choice(response, rank.ids)
    return choice if choice is not None else rank.default
