"""JEV prompt scorer: continuous complexity 0–100 + capability tags."""
from jompt.jev.scorer import score_prompt


def test_trivial_greeting_is_low() -> None:
    result = score_prompt("hi")
    assert result.complexity_score <= 15
    assert result.required_tags == set()


def test_extraction_task_is_low() -> None:
    result = score_prompt("Extract the names and emails from this text:\nAlice, alice@example.com")
    assert result.complexity_score <= 35


def test_long_reasoning_is_high() -> None:
    prompt = (
        "Design a multi-region active-active database architecture with conflict resolution. "
        "Compare CRDTs vs last-writer-wins, prove consistency under network partition, "
        "and write a detailed migration plan with rollback strategy for each region. "
        + ("Consider edge cases in consensus protocols and failure modes. " * 30)
    )
    result = score_prompt(prompt)
    assert result.complexity_score >= 65


def test_code_blocks_raise_score() -> None:
    plain = "What does this mean?"
    coded = "Debug this stack trace and fix the race condition:\n```python\nimport asyncio\nasync def f():\n    await asyncio.gather(a(), b())\n```\n" + "See traceback above. " * 20
    assert score_prompt(coded).complexity_score > score_prompt(plain).complexity_score


def test_vision_tag_when_image_mentioned() -> None:
    result = score_prompt("Look at this diagram and describe it: [image attached]")
    assert "vision" in result.required_tags


def test_code_tag_when_asking_to_implement() -> None:
    result = score_prompt("Implement a rate limiter in Go with token bucket algorithm and tests")
    assert "code" in result.required_tags
    assert result.complexity_score >= 40


def test_score_bounded() -> None:
    mega = "Prove " + "optimize the distributed system architecture deeply. " * 200
    assert 0 <= score_prompt(mega).complexity_score <= 100
