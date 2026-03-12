from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from jsonschema import Draft202012Validator


# Minimal schema for run configs; you can extend it.
RUN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["env", "baseline", "seed", "t_steps", "n_agents"],
    "properties": {
        "env": {"type": "string", "enum": ["resource_sharing", "public_goods"]},
        "baseline": {
            "type": "string",
            "enum": [
                "ppo_only",
                "static_guard",
                "constrained_ppo",
                "fair_ppo",
                "aaf_full",
                "aaf_detector_only",
                "aaf_shaping_only",
                "aaf_patch_only",
                "aaf_no_attrib",
            ],
        },
        "seed": {"type": "integer"},
        "t_steps": {"type": "integer", "minimum": 1},
        "n_agents": {"type": "integer", "minimum": 2},
        "device": {"type": "string"},
        "byzantine_frac": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "byzantine_start": {"type": "integer", "minimum": 0},
        "notes": {"type": "string"},
        "env_kwargs": {"type": "object"},
        "ppo_kwargs": {"type": "object"},
        "aaf_kwargs": {"type": "object"},
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: Dict[str, Any]


def load_config(path: str | Path, validate: bool = True) -> LoadedConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config must be a dict-like mapping at the top level.")

    if validate:
        Draft202012Validator(RUN_SCHEMA).validate(data)

    return LoadedConfig(path=p, data=data)
