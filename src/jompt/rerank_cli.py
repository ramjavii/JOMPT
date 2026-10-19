"""Re-rank an arbitrary model list and regenerate catalog + rank_list.

Examples:
  python -m jompt.rerank_cli --ids a-free,b-free --with-ai off --dry-run
  python -m jompt.rerank_cli --from-models-json --with-ai jev
  python -m jompt.rerank_cli --include-new --with-ai off
  python -m jompt.rerank_cli --ids-file ids.txt --with-ai openai --with-hf bonus
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from jompt.config import default_settings
from jompt.ranking import auto_score, build_catalog, write_catalog
from jompt.ranklist import catalog_to_rank_payload, write_rank_list
from jompt.registry import get_models_by_ids
from jompt.rerank.ai_ranker import (
    AIRank,
    build_jev_score_questions,
    build_judge_messages,
    parse_jev_scores,
    parse_judge_payload,
)
from jompt.rerank.blend import blend_scores
from jompt.rerank.context import build_enrichment
from jompt.rerank.hf_adapter import fetch_hf_signals
from jompt.rerank.inputs import load_requested_ids, read_stdin_if_piped
from jompt.rerank.writer import (
    format_diff_table,
    load_overrides,
    merge_overrides,
    scored_diff_table,
    write_overrides,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="jompt-rerank")
    p.add_argument("--ids", default=None)
    p.add_argument("--ids-file", type=Path, default=None)
    p.add_argument("--from-models-json", action="store_true")
    p.add_argument("--include-new", action="store_true")
    p.add_argument("--with-ai", choices=["off", "openai", "jev"], default="off")
    p.add_argument("--with-hf", choices=["off", "bonus"], default="off")
    p.add_argument("--w-ai", type=float, default=1.0)
    p.add_argument("--w-hf", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-catalog", type=Path, default=None)
    p.add_argument("--out-rank-list", type=Path, default=None)
    p.add_argument("--default", default=None)
    return p.parse_args(argv)


async def _judge_via_opencode(messages: list[dict[str, str]], settings) -> object:
    from jompt.providers.opencode import OpenCodeProvider

    provider = OpenCodeProvider(api_key=settings.opencode_api_key, base_url=settings.opencode_base_url)
    res = await provider.chat(settings.jev_model, messages, temperature=0.0)
    try:
        return json.loads(res.content)
    except ValueError:
        return res.content


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = default_settings()
    stdin_text = read_stdin_if_piped() if args.ids is None and args.ids_file is None \
        and not args.from_models_json and not args.include_new else None

    known_ids: list[str] = []
    if args.include_new:
        try:
            raw = json.loads(settings.catalog_out_path.read_text(encoding="utf-8"))
            known_ids = [e["id"] for e in raw if isinstance(e, dict) and "id" in e]
        except (OSError, ValueError):
            known_ids = []

    wanted = load_requested_ids(
        ids=args.ids,
        ids_file=args.ids_file,
        from_models_json=settings.models_json_path if args.from_models_json else None,
        include_new=args.include_new,
        models_json=settings.models_json_path,
        known_ids=known_ids,
        stdin_text=stdin_text,
    )
    models = get_models_by_ids(settings.models_json_path, wanted)
    enriched = build_enrichment(models)

    ai_map: dict[str, int] = {}
    ai_ranks: list[AIRank] = []
    if args.with_ai == "openai":
        messages = build_judge_messages(enriched)
        payload = asyncio.run(_judge_via_opencode(messages, settings))
        try:
            ai_ranks = parse_judge_payload(payload, set(wanted))
        except ValueError:
            ai_ranks = []
    elif args.with_ai == "jev":
        from jompt.jev.systemone import SystemOneClient

        client = SystemOneClient(base_url=settings.jev_base_url, model=settings.jev_model,
                                 api_key=settings.opencode_api_key)
        questions = build_jev_score_questions(enriched)
        try:
            resp = client.decide("rank these free models by capability", questions)
            ai_ranks = parse_jev_scores(resp, set(wanted))
        except Exception:
            ai_ranks = []
    ai_map = {r.id: r.score for r in ai_ranks}

    hf_signals: dict[str, dict[str, float]] = {}
    if args.with_hf == "bonus":
        hf_signals = fetch_hf_signals(wanted)

    final_scores = {
        m["id"]: blend_scores(auto_score(m), ai_map.get(m["id"]),
                              hf_signals.get(m["id"]), w_ai=args.w_ai, w_hf=args.w_hf)
        for m in models
    }
    rows = scored_diff_table(models, final_scores)
    print(format_diff_table(rows))
    if args.dry_run:
        return 0

    ai_for_merge = [
        AIRank(id=mid, score=sc, strengths=tuple(sorted(
            next((mm.get("strengths") or [] for mm in models if mm["id"] == mid), []))),
            description=f"score {sc}")
        for mid, sc in final_scores.items()
    ]
    existing = load_overrides(settings.overrides_path)
    merged = merge_overrides(existing, ai_for_merge)

    out_catalog = args.out_catalog or settings.catalog_out_path
    out_rank = args.out_rank_list or settings.rank_list_path
    if args.out_catalog or args.out_rank_list or True:
        # Write merged overrides next to out catalog unless custom out without override update:
        # default behavior: persist overrides file, then rebuild full catalog.
        if args.out_catalog is None and args.out_rank_list is None:
            write_overrides(merged, settings.overrides_path)
            catalog = build_catalog(settings.models_json_path, settings.overrides_path)
        else:
            # Custom outputs: build from in-memory merged overrides without touching repo files.
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
                import yaml

                tf.write(yaml.safe_dump(merged, sort_keys=True))
                tmp = Path(tf.name)
            try:
                catalog = build_catalog(settings.models_json_path, tmp)
            finally:
                tmp.unlink(missing_ok=True)
            # Restrict rank list to wanted ids when a subset was requested.
            wanted_set = set(wanted)
            catalog = [e for e in catalog if e.id in wanted_set] or catalog
    write_catalog(catalog, out_catalog)
    payload = catalog_to_rank_payload(
        [e for e in catalog if e.id in set(wanted)] or catalog, default=args.default
    )
    write_rank_list(payload, out_rank)
    print(f"wrote {len(catalog)} catalog entries → {out_catalog}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
