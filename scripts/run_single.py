from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from aaf_q1.runner import run_experiment
from aaf_q1.utils.io import ensure_dir, write_json


def _hash_config(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single AAF experiment configuration.")
    p.add_argument("--config", type=str, default=None, help="YAML/JSON config path (recommended).")
    p.add_argument("--out", type=str, required=True, help="Output directory (run folder).")

    # Optional CLI overrides if not using config file
    p.add_argument("--env", type=str, default="resource_sharing", choices=["resource_sharing", "public_goods"])
    p.add_argument("--baseline", type=str, default="ppo_only")
    p.add_argument("--n_agents", type=int, default=50)
    p.add_argument("--t_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--penalty_factor", type=float, default=0.2)
    p.add_argument("--dist_alpha", type=float, default=1.0)
    p.add_argument("--partial_obs", type=str, default="off", choices=["on", "off"])
    p.add_argument("--byzantine_frac", type=float, default=0.0)
    p.add_argument("--byzantine_start", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--log_mode",
        type=str,
        default="full",
        choices=["summary", "agents", "steps", "full"],
        help="Logging mode: summary (fast, no per-step logs), agents, steps, full.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)

    cfg: Dict[str, Any]
    if args.config is not None:
        p = Path(args.config)
        if not p.exists():
            raise FileNotFoundError(str(p))
        if p.suffix.lower() in (".yaml", ".yml"):
            cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("Config file must contain a dict.")
    else:
        cfg = {
            "env": args.env,
            "baseline": args.baseline,
            "n_agents": args.n_agents,
            "t_steps": args.t_steps,
            "seed": args.seed,
            "penalty_factor": args.penalty_factor,
            "dist_alpha": args.dist_alpha,
            "partial_obs": (args.partial_obs == "on"),
            "byzantine_frac": args.byzantine_frac,
            "byzantine_start": args.byzantine_start,
            "device": args.device,
            "log_mode": args.log_mode,
            "env_kwargs": {
                "penalty_factor": args.penalty_factor,
                "dist_alpha": args.dist_alpha,
                "partial_obs": (args.partial_obs == "on"),
            },
        }

    # CLI device override
    if args.device is not None:
        cfg["device"] = args.device
    cfg["log_mode"] = args.log_mode

    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{cfg.get('baseline','run')}_{_hash_config(cfg)}"
    run_dir = ensure_dir(out / run_id)

    result = run_experiment(cfg)

    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "summary.json", result["summary"])

    if args.log_mode in ("steps", "full") and result.get("step_logs"):
        pd.DataFrame(result["step_logs"]).to_csv(run_dir / "step_logs.csv", index=False)
    if args.log_mode in ("agents", "full") and result.get("agent_logs"):
        pd.DataFrame(result["agent_logs"]).to_csv(run_dir / "agent_logs.csv", index=False)

    print("\nSummary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")

    print(f"\nSaved run to: {run_dir}")


if __name__ == "__main__":
    main()
