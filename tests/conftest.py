"""Shared fixtures: synthetic free-model catalog + override file."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FREE_MODELS_JSON = {
    "opencode": {
        "id": "opencode",
        "models": {
            "nemotron-3-ultra-free": {
                "id": "nemotron-3-ultra-free",
                "name": "Nemotron 3 Ultra",
                "description": "Largest Nemotron 3 model for maximum open-weight reasoning and agent accuracy",
                "reasoning": True,
                "tool_call": True,
                "structured_output": True,
                "attachment": True,
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": 1000000, "output": 64000},
                "release_date": "2026-01-15",
                "cost": {"input": 0, "output": 0},
            },
            "step-5-preview-free": {
                "id": "step-5-preview-free",
                "name": "StepFun Step 5 Preview",
                "description": "StepFun's next-generation flagship base model for coding and professional knowledge",
                "reasoning": True,
                "tool_call": True,
                "structured_output": True,
                "attachment": False,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 1000000, "output": 32000},
                "release_date": "2026-03-01",
                "cost": {"input": 0, "output": 0},
            },
            "kimi-k2.5-free": {
                "id": "kimi-k2.5-free",
                "name": "Kimi K2.5",
                "description": "Strong general model with tool calling",
                "reasoning": True,
                "tool_call": True,
                "structured_output": False,
                "attachment": False,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 16000},
                "release_date": "2025-11-01",
                "cost": {"input": 0, "output": 0},
            },
            "deepseek-v4-flash-free": {
                "id": "deepseek-v4-flash-free",
                "name": "DeepSeek V4 Flash",
                "description": "Official DeepSeek V4 Flash release with enhanced agentic capabilities",
                "reasoning": False,
                "tool_call": True,
                "structured_output": False,
                "attachment": False,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 200000, "output": 16000},
                "release_date": "2026-02-01",
                "cost": {"input": 0, "output": 0},
            },
            "ling-3.1-flash-free": {
                "id": "ling-3.1-flash-free",
                "name": "Ling 3.1 Flash",
                "description": "Efficient model for low-latency assistance, extraction, and routine automation",
                "reasoning": False,
                "tool_call": False,
                "structured_output": False,
                "attachment": False,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 8000},
                "release_date": "2026-04-01",
                "cost": {"input": 0, "output": 0},
            },
            "ling-3.0-tiny-free": {
                "id": "ling-3.0-tiny-free",
                "name": "Ling 3.0 Tiny",
                "description": "Compact MoE model for responsive agents, instruction following",
                "reasoning": False,
                "tool_call": False,
                "structured_output": False,
                "attachment": False,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 262144, "output": 4000},
                "release_date": "2025-09-01",
                "cost": {"input": 0, "output": 0},
            },
            "paid-gpt": {
                "id": "paid-gpt",
                "name": "Paid GPT",
                "description": "Not free",
                "reasoning": True,
                "tool_call": True,
                "structured_output": True,
                "attachment": True,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 200000, "output": 16000},
                "release_date": "2026-01-01",
                "cost": {"input": 5, "output": 15},
            },
        },
    }
}


@pytest.fixture
def models_json_path(tmp_path: Path) -> Path:
    p = tmp_path / "models.json"
    p.write_text(json.dumps(FREE_MODELS_JSON), encoding="utf-8")
    return p


@pytest.fixture
def empty_overrides(tmp_path: Path) -> Path:
    p = tmp_path / "catalog_overrides.yaml"
    p.write_text("{}\n", encoding="utf-8")
    return p
