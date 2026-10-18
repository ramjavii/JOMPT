"""Shared app-state construction (kept separate so importing main is side-effect light)."""
from __future__ import annotations

from types import SimpleNamespace

from jompt.config import Settings
from jompt.cost import CostTracker
from jompt.providers.opencode import OpenCodeProvider
from jompt.ranking import build_catalog


def build_state(settings: Settings) -> SimpleNamespace:
    catalog = build_catalog(settings.models_json_path, settings.overrides_path)
    if not catalog:
        raise RuntimeError(
            "JOMPT_STARTUP: free-model catalog is empty; "
            "check models.json path / overrides"
        )
    provider = OpenCodeProvider(settings.opencode_api_key, settings.opencode_base_url)
    return SimpleNamespace(
        settings=settings,
        catalog=catalog,
        provider=provider,
        tracker=CostTracker(),
    )
