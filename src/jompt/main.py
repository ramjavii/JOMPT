"""FastAPI application factory."""
from __future__ import annotations

import os
from types import SimpleNamespace

from fastapi import FastAPI

from jompt.api.routes import router
from jompt.config import Settings, default_settings
from jompt.cost import CostTracker
from jompt.main_helpers import build_state
from jompt.providers.opencode import OpenCodeProvider


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or default_settings()
    state = build_state(settings)

    app = FastAPI(
        title="JOMPT",
        version="0.1.0",
        description="JEV model router for free opencode models",
    )
    app.state.jompt = state
    app.include_router(router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "catalog_size": len(state.catalog)}

    return app


def _module_app() -> FastAPI | None:
    """Build module-level app only when OPENCODE_API_KEY is present.

    Avoids import-time crash on machines without the key / models.json
    (tests import create_app and never need this instance).
    """
    if not os.environ.get("OPENCODE_API_KEY"):
        return None
    try:
        return create_app()
    except Exception:
        return None


app = _module_app()
