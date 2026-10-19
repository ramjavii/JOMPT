"""HTTP routes for JOMPT."""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from jompt.config import Settings
from jompt.cost import CostRecord
from jompt.jev.router import filter_by_tags, pick_model
from jompt.jev.scorer import score_prompt
from jompt.providers.opencode import ProviderError
from jompt.schemas import (
    CatalogModelOut,
    ChatRequest,
    ChatResponse,
    ModelUsage,
    RankingsResponse,
    StatsResponse,
)

logger = logging.getLogger("jompt")

router = APIRouter(prefix="/v1")


def get_state(request: Request) -> Any:
    return request.app.state.jompt


async def require_api_key(request: Request) -> None:
    settings: Settings = request.app.state.jompt.settings
    if not settings.jompt_api_key:
        return
    provided = request.headers.get("X-JOMPT-Key", "")
    if not provided or not secrets.compare_digest(provided, settings.jompt_api_key):
        raise HTTPException(status_code=401, detail="invalid api key")


@router.get("/rankings", response_model=RankingsResponse, dependencies=[Depends(require_api_key)])
async def rankings(state=Depends(get_state)) -> RankingsResponse:
    models = [
        CatalogModelOut(
            id=e.id,
            score=e.score,
            strengths=sorted(e.strengths),
            context=e.context,
            reasoning=e.reasoning,
            source=e.source,
        )
        for e in state.catalog
    ]
    return RankingsResponse(count=len(models), models=models)


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(body: ChatRequest, state=Depends(get_state)) -> ChatResponse:
    catalog = state.catalog
    if not catalog:
        raise HTTPException(status_code=503, detail="catalog unavailable")
    by_id = {e.id: e for e in catalog}

    if body.force_model is not None:
        if body.force_model not in by_id:
            raise HTTPException(status_code=400, detail="unknown force_model")
        chosen = by_id[body.force_model]
        profile = score_prompt(body.prompt)
        routed_by = "force_model"
        reason = "forced by client"
    else:
        profile = score_prompt(body.prompt)
        bias = body.bias if body.bias is not None else state.settings.default_bias
        try:
            chosen = pick_model(profile, catalog, bias=bias)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail="no model satisfies required capabilities") from exc
        routed_by = "jev"
        reason = (
            f"complexity={profile.complexity_score} bias={bias} → "
            f"cheapest model with score>={profile.complexity_score + bias}"
        )

    messages: list[dict[str, str]] = []
    if body.system:
        messages.append({"role": "system", "content": body.system})
    messages.append({"role": "user", "content": body.prompt})

    # Fallbacks must respect capability tags (do not silently drop vision/code/reasoning)
    pool = filter_by_tags(catalog, set(profile.required_tags)) or catalog
    attempt_order = [chosen] + [
        e for e in sorted(pool, key=lambda x: (-x.score, x.id)) if e.id != chosen.id
    ][:3]

    failed: list[str] = []
    last_err: Exception | None = None
    result = None
    started = time.perf_counter()
    for entry in attempt_order:
        try:
            result = await state.provider.chat(
                entry.id,
                messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
            chosen = entry
            break
        except ProviderError as exc:
            logger.warning("provider failure model=%s err=%s", entry.id, exc)
            last_err = exc
            failed.append(entry.id)
            if exc.is_systemic:
                # free_tier / auth / funds: retrying another model cannot help
                logger.error("systemic provider error (kind=%s); not retrying", exc.kind)
                break

    if result is None:
        detail = "all model attempts failed"
        if isinstance(last_err, ProviderError):
            detail = f"{detail}: {last_err.user_hint}"
        elif last_err is not None:
            detail = f"{detail}: upstream provider error"
        logger.error("%s (last_err=%s)", detail, last_err)
        raise HTTPException(status_code=502, detail=detail)

    latency_ms = int((time.perf_counter() - started) * 1000)
    state.tracker = state.tracker.add(
        CostRecord(
            model=chosen.id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
        )
    )

    return ChatResponse(
        content=result.content,
        model=chosen.id,
        model_score=chosen.score,
        complexity_score=profile.complexity_score,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
        routed_by=routed_by,
        reason=reason if not failed else f"{reason}; fell back from {failed}",
        fallback_from=failed,
    )


@router.get("/stats", response_model=StatsResponse, dependencies=[Depends(require_api_key)])
async def stats(state=Depends(get_state)) -> StatsResponse:
    data = state.tracker.summary()
    models = {k: ModelUsage(**v) for k, v in data["models"].items()}
    return StatsResponse(
        request_count=data["request_count"],
        total_input_tokens=data["total_input_tokens"],
        total_output_tokens=data["total_output_tokens"],
        models=models,
    )
