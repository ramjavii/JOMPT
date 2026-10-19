"""Arbitrary re-rank input ingestion: --ids / --ids-file / stdin / --from-models-json."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from jompt.registry import diff_new_ids, list_free_ids


def _read_ids_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, list):
        ids = [str(x).strip() for x in data if str(x).strip()]
    elif isinstance(data, dict) and isinstance(data.get("models"), list):
        ids = [str(m.get("id", "")).strip() for m in data["models"] if isinstance(m, dict)]
        ids = [i for i in ids if i]
    else:
        ids = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not ids:
        raise ValueError(f"ids file is empty: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"ids file has duplicate ids: {path}")
    return ids


def load_requested_ids(
    *,
    ids: str | None = None,
    ids_file: Path | None = None,
    from_models_json: Path | None = None,
    include_new: bool = False,
    models_json: Path,
    known_ids: list[str] | None = None,
    stdin_text: str | None = None,
) -> list[str]:
    """Resolve the ordered id list from exactly one source.

    Sources (exactly one required): comma-separated `ids`, `ids_file`
    (.txt/.json), `from_models_json` (all free ids), `include_new` (free ids
    minus known catalog ids), or piped `stdin_text`.
    """
    sources = sum(
        [
            ids is not None,
            ids_file is not None,
            from_models_json is not None,
            include_new,
            stdin_text is not None,
        ]
    )
    if sources != 1:
        raise ValueError("exactly one id source is required")
    if ids is not None:
        out = [p.strip() for p in ids.split(",") if p.strip()]
        if not out:
            raise ValueError("id list is empty")
        if len(set(out)) != len(out):
            raise ValueError("id list has duplicate ids")
        return out
    if ids_file is not None:
        return _read_ids_file(ids_file)
    if from_models_json is not None:
        out = list_free_ids(from_models_json)
        if not out:
            raise ValueError("no free models found")
        return out
    if include_new:
        out = diff_new_ids(models_json, set(known_ids or []))
        if not out:
            raise ValueError("no new model ids found")
        return out
    assert stdin_text is not None
    out = [ln.strip() for ln in stdin_text.splitlines() if ln.strip()]
    if not out:
        raise ValueError("stdin id list is empty")
    if len(set(out)) != len(out):
        raise ValueError("stdin id list has duplicate ids")
    return out


def read_stdin_if_piped() -> str | None:
    if sys.stdin.isatty():
        return None
    text = sys.stdin.read()
    return text if text.strip() else None
