# JOMPT

**JEV-routed Open Model Proxy Tool** — sends a prompt to JEV (TypeSafe's System One decision model), which picks the best model from a ranked list. Output is one model id.

## How it works

```
prompt ──► JEV (POST /zen/v1/systemone)
              │
              ▼
         ranked list (44 free models, heavy → light)
              │
              ▼
         selected model id + probabilities
```

JEV is not a chat model. It evaluates the prompt against a `choice` question whose criteria are the rank list entries, and returns a structured decision with calibrated probabilities.

## Deploy

```bash
cd ~/Projects/JOMPT
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the selector (no API key needed — uses `jev-1.13-free`):

```bash
python -m jompt.select_cli "say hi"
# → ling-3.0-tiny-free
```

Run the API server (needs `OPENCODE_API_KEY` for `/v1/chat`):

```bash
export OPENCODE_API_KEY=...
uvicorn jompt.main:app --reload
```

## Usage

### CLI

```bash
python -m jompt.select_cli "your prompt"
```

Prints one model id from `config/rank_list.json`. Options:

| Flag | Description |
|------|-------------|
| `--rank-list PATH` | Use a custom rank list file |

### API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat` | Score-based routing → model completion |
| GET | `/v1/rankings` | Full heavy→light catalog |
| GET | `/v1/stats` | Per-model token usage |
| GET | `/health` | Liveness |

```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"prompt": "hi"}'
```

Body fields: `prompt`, `system`, `force_model`, `bias` (-30..+30), `temperature`, `max_tokens`.

Set `JOMPT_API_KEY` to require `X-JOMPT-Key` header on all `/v1/*` routes.

## Ranking method

Each model gets a **capability score 0–100** from metadata in `~/.cache/opencode/models.json`:

| Signal | Effect |
|--------|--------|
| `reasoning: true` | +25 |
| Context ≥ 1M tokens | +18 |
| `tool_call` | +10 |
| `structured_output` | +5 |
| Name keywords (`ultra`, `largest`, `flagship`) | +18 |
| Name keywords (`flash`, `tiny`, `lightning`) | -18 |
| `coding` in description | +5 |
| Vision support | +4 |

Scores are clamped to 0–100. Curated overrides go in `catalog/catalog_overrides.yaml`:

```yaml
some-model-free:
  score: 75
  strengths: [code, reasoning]
```

Rebuild the catalog after editing:

```bash
python -m jompt.ranking_cli
```

## JEV selector

The selector (`src/jompt/jev/selector.py`) sends the prompt as `state` and the rank list as a `choice` question:

```json
{
  "model": "jev-1.13-free",
  "state": "<<<your prompt>>>",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which model should handle this request?",
      "criteria": {
        "step-5-preview-free": "score 96; code, heavy, ...",
        "ling-2.6-flash-free": "score 8; flash, tools"
      }
    }
  }
}
```

Response:

```json
{
  "model": "jev-1.13-free",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "ling-3.0-tiny-free",
      "confidence": 0.53,
      "probabilities": {"ling-3.0-tiny-free": 0.54, "ling-2.6-flash-free": 0.39}
    }
  }
}
```

The selector parses `answers.route.choice`, validates it against the rank list, and falls back to the default (`minimax-m3-free`) on invalid responses or provider errors.

## Code layout

```
src/jompt/
├── config.py              # Settings from env
├── main.py                # FastAPI app factory
├── ranking.py             # Auto-score + overrides → catalog.json
├── registry.py            # Load free models from models.json
├── cost.py                # Immutable usage tracker
├── schemas.py             # Pydantic request/response
├── select_cli.py          # CLI entry point
├── jev/
│   ├── rank_list.py       # Rank list loader (config/rank_list.json)
│   ├── selector.py        # JEV selection logic
│   └── systemone.py       # System One HTTP client
├── providers/
│   └── opencode.py        # OpenAI-compatible chat client
└── api/
    └── routes.py          # /v1/chat, /v1/rankings, /v1/stats
```

## Tests

```bash
pytest
```

93 tests, 91% coverage. Tests cover rank list validation, JEV response parsing, fallback behavior, provider error handling, and API endpoints.

## Files

| File | Purpose |
|------|---------|
| `config/rank_list.json` | Ranked model list (44 entries, heavy → light) |
| `catalog/catalog.json` | Generated capability scores |
| `catalog/catalog_overrides.yaml` | Curated score overrides |
| `src/jompt/jev/systemone.py` | JEV System One client |
| `src/jompt/jev/selector.py` | Selection + parsing |
| `src/jompt/jev/rank_list.py` | Rank list loader |

## Security

- `OPENCODE_API_KEY` read from env only, never written to disk
- Upstream error bodies not returned to clients
- Free models only (`cost.input == 0 and cost.output == 0`)
- JEV selector uses `jev-1.13-free` (no key needed)
- Optional `JOMPT_API_KEY` gates `/v1/*` via `X-JOMPT-Key` header
