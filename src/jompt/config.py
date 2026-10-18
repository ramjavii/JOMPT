"""Settings loaded from environment; safe defaults for local dev."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CACHE = Path.home() / ".cache" / "opencode" / "models.json"
_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OVERRIDES = _PKG_ROOT / "catalog" / "catalog_overrides.yaml"
_DEFAULT_CATALOG = _PKG_ROOT / "catalog" / "catalog.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JOMPT_", env_file=".env", extra="ignore")

    models_json_path: Path = _DEFAULT_CACHE
    overrides_path: Path = _DEFAULT_OVERRIDES
    catalog_out_path: Path = _DEFAULT_CATALOG
    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/inference/openai/v1"
    jompt_api_key: str = ""
    default_bias: int = 0


def default_settings() -> Settings:
    import os

    key = os.environ.get("OPENCODE_API_KEY", "")
    s = Settings()
    if key and not s.opencode_api_key:
        s = s.model_copy(update={"opencode_api_key": key})
    return s
