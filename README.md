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

## Re-ranking (open rank, any model list)

`catalog/catalog.json` is the source of truth; `config/rank_list.json` is a
generated artifact. Never hand-edit either — edit inputs, then regenerate.

### What to edit

| File / flag | Purpose |
|-------------|---------|
| `catalog/catalog_overrides.yaml` | Hand-pinned scores: `some-model-free: {score: 75, strengths: [code, reasoning]}`. Rerank merges AI scores in, hand entries are preserved. |
| `~/.cache/opencode/models.json` | Input of truth (refreshed by OpenCode, not by you). `--from-models-json` takes all free ids from here. |
| `--ids` / `--ids-file` | The model subset to rank. `--ids a,b,c` or a file: `.txt` (one id per line, `#` comments ok) or `.json` (`["a","b"]` or `{"models":[{"id":"a"}]}`). |
| `--with-ai off\|jev\|openai` | Scoring backend. `off` = local heuristic (default, reproducible). `jev` = keyless JEV `score` questions. `openai` = OpenCode chat judge (needs `OPENCODE_API_KEY`). AI output is validated (unknown ids dropped, scores clamped 0–100); total failure falls back to heuristic. |
| `--with-hf off\|bonus`, `--w-ai`, `--w-hf` | Optional Hugging Face bonus (capped +6, fail-open) and blend weights 0.0–1.0 (`final = w_ai*ai + (1-w_ai)*auto + w_hf*hf_bonus`). |
| `--default ID` | Rank-list default fallback. Omit to keep the median. |
| `--out-catalog` / `--out-rank-list` | Write to custom paths instead of the repo files (leaves repo files untouched). |
| `catalog/RERANK_LOG.md` | Append one row per run (date, input count, backend, weights) for audit. |

### How to run

```bash
# 0. Always preview first — prints a score diff table, writes nothing
python -m jompt.rerank_cli --ids a-free,b-free,c-free --with-ai off --dry-run

# 1. Rank an arbitrary new list (heuristic only, writes both artifacts)
python -m jompt.rerank_cli --ids a-free,b-free,c-free,d-free,e-free --with-ai off

# 2. Rank ids from a file
python -m jompt.rerank_cli --ids-file ids.txt --with-ai off --dry-run

# 3. Discover models added since the last catalog build, then rank them
python -m jompt.rerank_cli --include-new --with-ai off --dry-run

# 4. Full refresh from current models.json (heuristic)
python -m jompt.rerank_cli --from-models-json --with-ai off

# 5. AI re-score: keyless JEV judge + HF bonus signal
python -m jompt.rerank_cli --from-models-json --with-ai jev --with-hf bonus

# 6. AI re-score via OpenCode chat judge (needs OPENCODE_API_KEY)
python -m jompt.rerank_cli --ids-file ids.txt --with-ai openai --dry-run

# 7. Regenerate only the rank list from the current catalog (no rescoring)
python -m jompt.ranklist_cli --dry-run   # preview
python -m jompt.ranklist_cli            # write config/rank_list.json
```

Flow per run: resolve ids → validate free-only → score (heuristic/AI/blend)
→ print `auto→new` diff table → `--dry-run` stops here, otherwise merge into
`catalog_overrides.yaml`, rebuild `catalog.json`, regenerate `rank_list.json`.

Notes: `--with-ai jev` is keyless (`jev-1.13-free` score questions);
`openai` posts a strict-JSON prompt via `OpenCodeProvider`. HF signals are
bonus-only (capped +6) and fail open. Large lists: `select_cli --top-k 50`
(default 50) pre-filters before the JEV `choice` call. Each run is logged in
`catalog/RERANK_LOG.md`.

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
