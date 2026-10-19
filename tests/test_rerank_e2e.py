"""E2E: rerank subset with heuristic only regenerates both artifacts sorted."""
from __future__ import annotations

import json
from pathlib import Path

from jompt.jev.rank_list import load_rank_list
from jompt.rerank_cli import main as rerank_main


def _write_models(p: Path) -> None:
    def m(ctx=1000, reasoning=False, tools=False):
        return {"cost": {"input": 0, "output": 0}, "limit": {"context": ctx},
                "reasoning": reasoning, "tool_call": tools, "description": "test coding model"}

    p.write_text(json.dumps({"opencode": {"models": {
        "z-heavy-free": m(ctx=1_000_000, reasoning=True, tools=True),
        "a-light-free": m(ctx=8000),
        "m-mid-free": m(ctx=200_000, reasoning=True),
    }}}), encoding="utf-8")


def test_rerank_e2e_heuristic_subset(tmp_path: Path, monkeypatch) -> None:
    import jompt.rerank_cli as cli

    mj = tmp_path / "models.json"
    _write_models(mj)
    ov = tmp_path / "overrides.yaml"
    ov.write_text("{}\n", encoding="utf-8")
    cat = tmp_path / "catalog.json"
    rank = tmp_path / "rank_list.json"

    import jompt.config as cfg

    settings = cfg.Settings(models_json_path=mj, overrides_path=ov,
                            catalog_out_path=cat, rank_list_path=rank,
                            opencode_api_key="k")
    monkeypatch.setattr(cfg, "default_settings", lambda: settings)
    monkeypatch.setattr(cli, "default_settings", lambda: settings)

    code = rerank_main(["--ids", "z-heavy-free,a-light-free,m-mid-free", "--with-ai", "off"])
    assert code == 0
    catalog = json.loads(cat.read_text(encoding="utf-8"))
    rl = load_rank_list(rank)
    assert {e["id"] for e in catalog} >= {"z-heavy-free", "a-light-free", "m-mid-free"}
    assert set(rl.ids) == {"z-heavy-free", "a-light-free", "m-mid-free"}
    scores = [next(e["score"] for e in catalog if e["id"] == i) for i in rl.ids]
    assert scores == sorted(scores, reverse=True)
    assert rl.ids[0] == "z-heavy-free"
