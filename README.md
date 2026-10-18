# JOMPT

**J**EV-routed **O**pen **M**odel **P**roxy **T**ool — scores your prompt (0–100), then picks the *specific* cheapest free opencode model whose capability score is still ≥ the prompt score.

## How it works

```
prompt ──► JEV scorer (complexity 0–100 + tags)
              │
              ▼
         ranked catalog (free models, heavy → light)
              │
              ▼
         cheapest model with score ≥ complexity  ──► OpenCode chat API
              │
              ▼
         response + routing rationale + usage stats
```

Ranking is built three ways (in order):

1. **Auto** — scores from models.json metadata (reasoning, context, tool use, name keywords)
2. **Overrides** — `catalog/catalog_overrides.yaml` fixes wrong auto-scores
3. Optional **LLM refine** — re-rank uncertain models with a free model when the catalog drifts

## Setup

```bash
cd ~/Projects/JOMPT
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export OPENCODE_API_KEY=...   # from your opencode auth
python -m jompt.ranking_cli   # rebuild catalog/catalog.json
uvicorn jompt.main:app --reload
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat` | Prompt → JEV route → model completion |
| GET | `/v1/rankings` | Full heavy→light catalog |
| GET | `/v1/stats` | Immutable per-model token usage |
| GET | `/health` | Liveness |

### Example

```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"prompt": "hi"}'
# → routed to a tiny/flash free model

curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"prompt": "Design a multi-region active-active database with conflict resolution and prove consistency under partition..."}'
# → routed to a heavy free model (nemotron-ultra class)
```

Optional body fields: `system`, `force_model` (must be in catalog), `bias` (−30..+30 lean lighter/heavier), `temperature`, `max_tokens`.

### Optional gateway auth

If `JOMPT_API_KEY` (or `JOMPT_JOMPT_API_KEY`) is set, all `/v1/*` routes require header `X-JOMPT-Key: <key>`. Leave unset for localhost-only use.

## Security notes

- `OPENCODE_API_KEY` is read from env only; never written to disk by JOMPT
- Upstream error bodies are not returned to clients (logged server-side only)
- Free models only (`cost.input == 0 and cost.output == 0`, missing cost excluded)
- Bind to `127.0.0.1` unless you add auth + rate limiting

## Tests

```bash
pytest                 # 43 tests, 90% coverage
```

## Conventions

See `AGENTS.md`. Free models only; API key from env only; immutable usage records.
