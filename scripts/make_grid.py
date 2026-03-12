from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from aaf_q1.utils.io import ensure_dir


PRESETS: Dict[str, Dict[str, Any]] = {
    "paper_fast": {
        "envs": ["resource_sharing"],
        "baselines": ["ppo_only", "static_guard", "aaf_full"],
        "n_agents": [10, 50],
        "t_steps": 1000,
        "penalty_factor": [0.2],
        "dist_alpha": [1.0],
        "partial_obs": [False],
        "byzantine": [{"byzantine_frac": 0.0, "byzantine_start": 0}],
        "seeds": [0, 1, 2, 3, 4],
        "device": "auto",
    },
    "paper_full": {
        "envs": ["resource_sharing"],
        "baselines": ["ppo_only", "static_guard", "aaf_full"],
        "n_agents": [10, 50],
        "t_steps": 1000,
        "penalty_factor": [0.05, 0.2, 0.35],
        "dist_alpha": [0.0, 0.25, 1.0],
        "partial_obs": [False, True],
        "byzantine": [
            {"byzantine_frac": 0.0, "byzantine_start": 0},
            {"byzantine_frac": 0.05, "byzantine_start": 200},
        ],
        "seeds": [0, 1, 2, 3, 4],
        "device": "auto",
    },
    "q1_deep": {
        "envs": ["resource_sharing", "public_goods"],
        "baselines": [
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
        "n_agents": [10, 50, 100],
        "t_steps": 2000,
        "penalty_factor": [0.05, 0.2, 0.35],
        "dist_alpha": [0.0, 0.25, 1.0],
        "partial_obs": [False, True],
        "byzantine": [
            {"byzantine_frac": 0.0, "byzantine_start": 0},
            {"byzantine_frac": 0.05, "byzantine_start": 200},
            {"byzantine_frac": 0.10, "byzantine_start": 200},
        ],
        "seeds": list(range(10)),  # 10 seeds for Q1 significance
        # For large sweeps on macOS, CPU is typically faster/more stable than MPS
        # because the env loop is NumPy/CPU and some sampling ops can fall back.
        # Override at runtime with: scripts.run_grid --device cuda
        "device": "cpu",
    },

    # --- Suggested split for cluster runs ---
    # Main comparison (no ablations) across both domains
    "q1_main": {
        "envs": ["resource_sharing", "public_goods"],
        "baselines": [
            "ppo_only",
            "static_guard",
            "constrained_ppo",
            "fair_ppo",
            "aaf_full",
        ],
        "n_agents": [10, 50, 100],
        "t_steps": 2000,
        "penalty_factor": [0.05, 0.2, 0.35],
        "dist_alpha": [0.0, 0.25, 1.0],
        "partial_obs": [False, True],
        "byzantine": [
            {"byzantine_frac": 0.0, "byzantine_start": 0},
            {"byzantine_frac": 0.05, "byzantine_start": 200},
            {"byzantine_frac": 0.10, "byzantine_start": 200},
        ],
        "seeds": list(range(10)),
        "device": "cpu",
    },

    # AAF ablations on the canonical slice (smaller, but reviewer-friendly)
    "q1_ablation": {
        "envs": ["resource_sharing", "public_goods"],
        "baselines": [
            "aaf_full",
            "aaf_detector_only",
            "aaf_shaping_only",
            "aaf_patch_only",
            "aaf_no_attrib",
        ],
        "n_agents": [50],
        "t_steps": 2000,
        "penalty_factor": [0.2],
        "dist_alpha": [1.0],
        "partial_obs": [False, True],
        "byzantine": [
            {"byzantine_frac": 0.0, "byzantine_start": 0},
            {"byzantine_frac": 0.05, "byzantine_start": 200},
            {"byzantine_frac": 0.10, "byzantine_start": 200},
        ],
        "seeds": list(range(10)),
        "device": "cpu",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a JSONL grid file of experiment configs.")
    p.add_argument("--preset", type=str, default="paper_fast", choices=sorted(PRESETS.keys()))
    p.add_argument("--out", type=str, required=True, help="Output JSONL path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    rows: List[Dict[str, Any]] = []
    for env in preset["envs"]:
        for baseline in preset["baselines"]:
            for n_agents in preset["n_agents"]:
                for pf in preset["penalty_factor"]:
                    for alpha in preset["dist_alpha"]:
                        for po in preset["partial_obs"]:
                            for byz in preset["byzantine"]:
                                for seed in preset["seeds"]:
                                    cfg: Dict[str, Any] = {
                                        "env": env,
                                        "baseline": baseline,
                                        "n_agents": int(n_agents),
                                        "t_steps": int(preset["t_steps"]),
                                        "seed": int(seed),
                                        "device": preset.get("device", "auto"),
                                        "penalty_factor": float(pf),
                                        "dist_alpha": float(alpha),
                                        "partial_obs": bool(po),
                                        **byz,
                                        "env_kwargs": {
                                            "penalty_factor": float(pf),
                                            "dist_alpha": float(alpha),
                                            "partial_obs": bool(po),
                                        },
                                    }
                                    rows.append(cfg)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} configs to {out_path}")


if __name__ == "__main__":
    main()
