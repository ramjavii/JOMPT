"""Rebuild catalog.json from models.json + overrides.

Usage:
  python -m jompt.ranking_cli
  JOMPT_MODELS_JSON_PATH=... JOMPT_OVERRIDES_PATH=... python -m jompt.ranking_cli
"""
from __future__ import annotations

from jompt.config import default_settings
from jompt.ranking import build_catalog, write_catalog


def main() -> None:
    s = default_settings()
    catalog = build_catalog(s.models_json_path, s.overrides_path)
    write_catalog(catalog, s.catalog_out_path)
    print(f"wrote {len(catalog)} models → {s.catalog_out_path}")
    for e in catalog:
        print(f"  {e.score:3d}  {e.id}  ({e.source})")


if __name__ == "__main__":
    main()
