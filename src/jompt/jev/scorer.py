"""JEV prompt complexity scorer: continuous 0–100 + capability tags."""
from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_HIGH_HINTS = re.compile(
    r"\b(prove|proof|architect(ure)?|design (a|the|an)|distributed|consensus|"
    r"race condition|rollback|migration plan|security audit|threat model|"
    r"optimi[sz]e|refactor|debug|stack trace|multi-?region|active-?active|"
    r"conflict resolution|algorithm|complexity analysis|formal)\b",
    re.I,
)
_MED_HINTS = re.compile(
    r"\b(implement|write (a|the|an|tests?)|create|build|explain|compare|"
    r"analy[sz]e|summar(y|ize)|translate|refactor|test|api|endpoint)\b",
    re.I,
)
_LOW_HINTS = re.compile(
    r"\b(hi|hello|thanks|thank you|yes|no|ok|okay|extract|list|rename|"
    r"what is|who is|when did|convert|format)\b",
    re.I,
)
_VISION = re.compile(r"(\[image|image attached|screenshot|diagram|photo|png|jpg|vision)\b", re.I)
_CODE = re.compile(
    r"(```|def |function |class |import |package main|public static void|"
    r"implement .*(in|using) (python|go|rust|java|typescript|javascript|c\+\+|sql)|"
    r"stack trace|unit test)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class PromptProfile:
    text: str
    complexity_score: int
    required_tags: frozenset[str] | set[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_tags", frozenset(self.required_tags))


def _length_points(text: str) -> float:
    n = len(text)
    if n < 80:
        return 0.0
    if n < 400:
        return 12.0
    if n < 2_000:
        return 22.0
    if n < 10_000:
        return 32.0
    return 40.0


def score_prompt(text: str) -> PromptProfile:
    stripped = text.strip()
    raw = stripped or text

    score = 5.0
    score += _length_points(raw)

    code_blocks = _CODE_FENCE.findall(raw)
    if code_blocks:
        score += min(20.0, 8.0 + 4.0 * len(code_blocks))
    if _CODE.search(raw) and not code_blocks:
        score += 10.0

    high = len(_HIGH_HINTS.findall(raw))
    med = len(_MED_HINTS.findall(raw))
    low = len(_LOW_HINTS.findall(raw))

    score += min(35.0, high * 8.0)
    score += min(12.0, med * 3.0)
    if high == 0 and med == 0 and low > 0:
        score -= 8.0

    if raw.lower().startswith(("hi", "hello", "hey")) and len(raw) < 50:
        score = min(score, 8.0)

    required: set[str] = set()
    if _VISION.search(raw):
        required.add("vision")
        score += 5.0
    if _CODE.search(raw) or code_blocks:
        required.add("code")
        if re.search(r"\b(implement|write|build|create|debug|fix)\b", raw, re.I):
            score += 12.0
    if "algorithm" in raw.lower() or "rate limiter" in raw.lower():
        score += 10.0
    if high >= 2:
        required.add("reasoning")

    clamped = int(max(0, min(100, round(score))))
    return PromptProfile(text=raw, complexity_score=clamped, required_tags=required)
