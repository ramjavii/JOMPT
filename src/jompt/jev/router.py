"""JEV router: pick the cheapest catalog model that satisfies the prompt profile."""
from __future__ import annotations

from jompt.jev.scorer import PromptProfile
from jompt.ranking import CatalogEntry


def _satisfies(entry: CatalogEntry, tag: str) -> bool:
    strengths = set(entry.strengths)
    if tag == "vision":
        return "vision" in strengths
    if tag == "code":
        return bool(strengths & {"code", "tools", "reasoning"}) or entry.reasoning
    if tag == "reasoning":
        return "reasoning" in strengths or entry.reasoning
    return True


def filter_by_tags(catalog: list[CatalogEntry], tags: set[str] | frozenset[str]) -> list[CatalogEntry]:
    if not tags:
        return list(catalog)
    return [e for e in catalog if all(_satisfies(e, tag) for tag in tags)]


def pick_model(
    profile: PromptProfile,
    catalog: list[CatalogEntry],
    *,
    bias: int = 0,
    strict_tags: bool = True,
) -> CatalogEntry:
    """Lowest-capability model whose score >= complexity+bias; heaviest fallback."""
    if not catalog:
        raise ValueError("empty model catalog")

    candidates = list(catalog)
    required = set(profile.required_tags)

    if required:
        filtered = filter_by_tags(candidates, required)
        if filtered:
            candidates = filtered
        elif strict_tags:
            raise ValueError(f"no model satisfies required capabilities: {sorted(required)}")

    ascending = sorted(candidates, key=lambda e: (e.score, e.id))
    target = profile.complexity_score + bias
    for e in ascending:
        if e.score >= target:
            return e
    return max(candidates, key=lambda e: (e.score, e.id))
