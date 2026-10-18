"""JEV router: pick the specific cheapest model with score >= prompt complexity."""
from jompt.jev.router import pick_model
from jompt.jev.scorer import PromptProfile
from jompt.ranking import CatalogEntry


def _cat(*rows: tuple[str, int, set[str]]) -> list[CatalogEntry]:
    return [
        CatalogEntry(id=i, score=s, strengths=set(t), context=200000, reasoning=False, source="auto")
        for i, s, t in rows
    ]


def test_picks_lowest_adequate_model() -> None:
    catalog = _cat(("heavy", 90, set()), ("mid", 50, set()), ("light", 20, set()))
    profile = PromptProfile(complexity_score=45, required_tags=set(), text="")
    assert pick_model(profile, catalog).id == "mid"


def test_picks_heaviest_when_none_adequate() -> None:
    catalog = _cat(("heavy", 90, set()), ("light", 20, set()))
    profile = PromptProfile(complexity_score=95, required_tags=set(), text="")
    assert pick_model(profile, catalog).id == "heavy"


def test_capability_filter_vision() -> None:
    catalog = _cat(("vision-heavy", 80, {"vision"}), ("text-mid", 50, set()))
    profile = PromptProfile(complexity_score=10, required_tags={"vision"}, text="")
    assert pick_model(profile, catalog).id == "vision-heavy"


def test_bias_negative_leans_lighter() -> None:
    catalog = _cat(("heavy", 90, set()), ("mid", 50, set()), ("light", 20, set()))
    profile = PromptProfile(complexity_score=50, required_tags=set(), text="")
    assert pick_model(profile, catalog).id == "mid"
    # target = 50-30 = 20 → light (score 20) qualifies
    assert pick_model(profile, catalog, bias=-30).id == "light"


def test_bias_positive_leans_heavier() -> None:
    catalog = _cat(("heavy", 90, set()), ("mid", 50, set()), ("light", 20, set()))
    profile = PromptProfile(complexity_score=40, required_tags=set(), text="")
    assert pick_model(profile, catalog, bias=30).id == "heavy"


def test_empty_catalog_raises() -> None:
    profile = PromptProfile(complexity_score=10, required_tags=set(), text="")
    try:
        pick_model(profile, [])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_no_vision_candidate_falls_back_to_best_text() -> None:
    catalog = _cat(("text-only", 70, set()))
    profile = PromptProfile(complexity_score=10, required_tags={"vision"}, text="")
    # strict: no candidate has vision → ValueError expected, caller handles fallback
    try:
        pick_model(profile, catalog, strict_tags=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError for strict missing tags")


def test_non_strict_ignores_missing_tags() -> None:
    catalog = _cat(("text-only", 70, set()), ("light", 20, set()))
    profile = PromptProfile(complexity_score=10, required_tags={"vision"}, text="")
    assert pick_model(profile, catalog, strict_tags=False).id == "light"
