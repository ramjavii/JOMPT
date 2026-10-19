"""Regenerate config/rank_list.json from catalog.json.

Usage:
  python -m jompt.ranklist_cli [--catalog PATH] [--out PATH] [--default ID] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jompt.config import default_settings
from jompt.ranklist import catalog_to_rank_payload, load_catalog_entries, write_rank_list


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jompt-ranklist")
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--default", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    settings = default_settings()
    catalog_path = args.catalog or settings.catalog_out_path
    out_path = args.out or settings.rank_list_path

    catalog = load_catalog_entries(catalog_path)
    payload = catalog_to_rank_payload(catalog, default=args.default)
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0
    write_rank_list(payload, out_path)
    print(f"wrote {len(payload['models'])} models → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
