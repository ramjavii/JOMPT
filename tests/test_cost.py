"""Immutable cost / usage tracker."""
from jompt.cost import CostRecord, CostTracker


def test_add_returns_new_tracker() -> None:
    t0 = CostTracker()
    t1 = t0.add(CostRecord(model="a", input_tokens=10, output_tokens=5, latency_ms=100))
    assert t0.total_tokens() == 0
    assert t1.total_tokens() == 15
    assert t1 is not t0


def test_per_model_totals() -> None:
    t = (
        CostTracker()
        .add(CostRecord(model="a", input_tokens=10, output_tokens=0, latency_ms=10))
        .add(CostRecord(model="a", input_tokens=5, output_tokens=5, latency_ms=20))
        .add(CostRecord(model="b", input_tokens=1, output_tokens=1, latency_ms=5))
    )
    assert t.model_totals()["a"].input_tokens == 15
    assert t.model_totals()["b"].output_tokens == 1
    assert t.request_count() == 3


def test_records_immutable() -> None:
    r = CostRecord(model="a", input_tokens=1, output_tokens=1, latency_ms=1)
    try:
        r.input_tokens = 9  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("expected frozen dataclass")
