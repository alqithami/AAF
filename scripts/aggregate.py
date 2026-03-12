from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from aaf_q1.utils.io import ensure_dir


KEYS = [
    "env",
    "baseline",
    "n_agents",
    "t_steps",
    "penalty_factor",
    "dist_alpha",
    "partial_obs",
    "byzantine_frac",
    "byzantine_start",
]

METRICS = [
    "compromise_ratio_attempted",
    "compromise_ratio_executed",
    "social_welfare",
    "gini_alloc_mean",
    "gini_alloc_final",
    "gini_reward_mean",
    "gini_reward_final",
    "alarms_count",
    "detection_delay",
    "attrib_top1_correct",
    "attrib_recall3",
    "attrib_recall5",
    "bandwidth_overhead_bytes",
    "runtime_s",
]


def _collect_summaries(root: Path) -> List[Dict[str, Any]]:
    """Collect all summary.json under root (recursive).

    This supports sharded output layouts such as:
      out/q1_deep/shard_00/<run_id>/summary.json
    """
    runs: List[Dict[str, Any]] = []
    for s in root.rglob("summary.json"):
        try:
            runs.append(json.loads(s.read_text(encoding="utf-8")))
        except Exception:
            # ignore corrupted partial writes
            continue
    return runs


def _ci95(mean: float, std: float, n: int) -> float:
    if n <= 1:
        return float("nan")
    return 1.96 * std / np.sqrt(n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate run summaries into grouped results.")
    p.add_argument("--root", type=str, required=True, help="Root directory containing run folders (output of run_grid).")
    p.add_argument(
        "--use_flat",
        action="store_true",
        help="Use analysis/all_runs_flat.csv if present (faster). Default is to rebuild from summary.json.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    analysis_dir = ensure_dir(root / "analysis")

    flat_path = analysis_dir / "all_runs_flat.csv"
    if args.use_flat and flat_path.exists():
        df = pd.read_csv(flat_path)
    else:
        runs = _collect_summaries(root)
        df = pd.DataFrame(runs)
        df.to_csv(flat_path, index=False)

    # Normalize types
    for k in ["n_agents", "t_steps", "byzantine_start", "seed", "alarms_count"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
    for k in ["penalty_factor", "dist_alpha", "byzantine_frac", "social_welfare"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    if "partial_obs" in df.columns:
        df["partial_obs"] = df["partial_obs"].astype(bool)

    # Metrics to numeric where possible
    for m in METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    # Keep only columns we care about (but preserve any extras)
    missing_keys = [k for k in KEYS if k not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing key columns in summaries: {missing_keys}")

    # Group + aggregate
    grouped = df.groupby(KEYS, dropna=False)

    rows: List[Dict[str, Any]] = []
    for keys, g in grouped:
        row: Dict[str, Any] = {k: v for k, v in zip(KEYS, keys)}

        n = int(g.shape[0])
        row["n_runs"] = n

        for m in METRICS:
            if m not in g.columns:
                continue
            vals = g[m].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_ci95"] = float("nan")
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95"] = _ci95(mean, std, int(vals.size))
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    out_path = analysis_dir / "final_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote aggregated summary: {out_path}")


if __name__ == "__main__":
    main()
