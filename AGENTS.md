# JOMPT — project conventions

## What this is
FastAPI service: accepts a prompt, JEV scores complexity 0–100, routes to the
specific free opencode model with the lowest capability score that still ≥ prompt score.

## Layout
- `src/jompt/` — package (registry, ranking, jev/, providers/, api/)
- `tests/` — pytest, TDD first
- `catalog/` — generated catalog.json + catalog_overrides.yaml

## Rules
- Immutability: cost/usage records are frozen dataclasses or immutable pydantic models
- Never hardcode API keys; read `OPENCODE_API_KEY` from env only
- Files < 400 lines; small focused modules
- All user input validated at API boundary
- Free models only (cost == 0 on opencode provider)

## Commands
```bash
pip install -e ".[dev]"
pytest
uvicorn jompt.main:app --reload
python -m jompt.ranking_cli   # rebuild catalog
```
