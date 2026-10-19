"""CLI: print the JEV-selected model id for a prompt.

Usage:
  python -m jompt.select_cli "your prompt"
  python -m jompt.select_cli --rank-list config/rank_list.json "your prompt"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jompt.config import Settings, default_settings
from jompt.jev.rank_list import load_rank_list
from jompt.jev.selector import select_model
from jompt.jev.systemone import SystemOneClient


def _make_client(settings: Settings) -> SystemOneClient:
    return SystemOneClient(
        base_url=settings.jev_base_url,
        model=settings.jev_model,
        api_key=settings.opencode_api_key,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jompt-select", description="Select a model with JEV")
    parser.add_argument("prompt", nargs="?", help="prompt text to route")
    parser.add_argument("--rank-list", type=Path, default=None, help="path to rank list JSON")
    args = parser.parse_args(argv)

    if not args.prompt:
        parser.error("a prompt is required")

    settings = default_settings()
    rank_path = args.rank_list or settings.rank_list_path
    rank = load_rank_list(rank_path)
    chosen = select_model(args.prompt, rank, _make_client(settings))
    print(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
