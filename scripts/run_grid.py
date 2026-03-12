from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from aaf_q1.runner import run_experiment
from aaf_q1.utils.io import ensure_dir, write_json


def _hash_config(cfg: Dict[str, Any]) -> str:
    # IMPORTANT: keep run_id stable across *logging-only* changes.
    # If we hash log_mode (or other non-experimental knobs), resume becomes
    # impossible when toggling output verbosity.
    cfg_norm = {k: v for k, v in cfg.items() if k not in {"log_mode"}}
    blob = json.dumps(cfg_norm, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def run_id(cfg: Dict[str, Any]) -> str:
    """Deterministic run id used for resume."""
    env = cfg.get("env", "env")
    baseline = cfg.get("baseline", "base")
    n = int(cfg.get("n_agents", 0))
    t = int(cfg.get("t_steps", 0))
    seed = int(cfg.get("seed", 0))
    pf = float(cfg.get("penalty_factor", cfg.get("env_kwargs", {}).get("penalty_factor", 0.0)))
    alpha = float(cfg.get("dist_alpha", cfg.get("env_kwargs", {}).get("dist_alpha", 0.0)))
    po = bool(cfg.get("partial_obs", cfg.get("env_kwargs", {}).get("partial_obs", False)))
    byz = float(cfg.get("byzantine_frac", 0.0))
    start = int(cfg.get("byzantine_start", 0))
    h = _hash_config(cfg)
    return f"{env}__{baseline}__N{n}__T{t}__pf{pf:.3f}__a{alpha:.3f}__po{int(po)}__byz{byz:.3f}__s{start}__seed{seed}__{h}"


def _iter_grid(
    path: Path,
    shard_id: int = 0,
    num_shards: int = 1,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_runs: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream a JSONL grid file.

    Selection rules (in order):
      1) slice by [start_idx, end_idx)
      2) shard by (idx % num_shards == shard_id)
      3) cap by max_runs (after filtering)

    idx is the 0-based line index among *non-empty* lines.
    """

    kept = 0
    idx = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if idx < start_idx:
                idx += 1
                continue
            if end_idx is not None and idx >= end_idx:
                break
            if num_shards > 1 and (idx % num_shards) != shard_id:
                idx += 1
                continue

            cfg = json.loads(line)
            if not isinstance(cfg, dict):
                raise ValueError("Each JSONL line must decode to a dict.")

            yield cfg
            kept += 1
            idx += 1
            if max_runs is not None and kept >= max_runs:
                break


def _count_grid(
    path: Path,
    shard_id: int = 0,
    num_shards: int = 1,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_runs: Optional[int] = None,
) -> int:
    """Count how many configs will be executed with the given filters."""
    count = 0
    idx = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if idx < start_idx:
                idx += 1
                continue
            if end_idx is not None and idx >= end_idx:
                break
            if num_shards > 1 and (idx % num_shards) != shard_id:
                idx += 1
                continue
            count += 1
            idx += 1
            if max_runs is not None and count >= max_runs:
                break
    return count


def _maybe_cleanup_torch() -> None:
    """Best-effort cleanup to reduce long-run memory growth."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # torch.mps exists only on macOS builds
        if hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass


def _set_torch_threads(n: Optional[int]) -> None:
    if n is None:
        return
    try:
        import torch

        torch.set_num_threads(int(n))
        torch.set_num_interop_threads(max(1, int(n)))
    except Exception:
        return


def _worker(
    cfg: Dict[str, Any],
    out_dir: str,
    log_mode: str,
    device_override: Optional[str],
    write_config: bool,
    torch_threads: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    """Run one config and persist outputs.

    Returns (status, summary).
      - status is "skipped" if summary exists, else "ran".
    """

    _set_torch_threads(torch_threads)

    # Apply overrides
    cfg = dict(cfg)  # do not mutate caller copy
    cfg["log_mode"] = log_mode
    if device_override is not None:
        cfg["device"] = device_override

    rid = run_id(cfg)
    run_dir = ensure_dir(Path(out_dir) / rid)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            return "skipped", summary
        except Exception:
            # Corrupted partial file; re-run.
            pass

    result = run_experiment(cfg)

    if write_config:
        write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "summary.json", result["summary"])

    # Optional heavy logs
    if log_mode in ("steps", "full") and result.get("step_logs"):
        pd.DataFrame(result["step_logs"]).to_csv(run_dir / "step_logs.csv", index=False)
    if log_mode in ("agents", "full") and result.get("agent_logs"):
        pd.DataFrame(result["agent_logs"]).to_csv(run_dir / "agent_logs.csv", index=False)

    _maybe_cleanup_torch()

    return "ran", result["summary"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a JSONL grid of experiments (scalable + resumable).")
    p.add_argument("--grid", type=str, required=True, help="Path to JSONL grid file.")
    p.add_argument("--out", type=str, required=True, help="Output directory for run folders.")
    p.add_argument("--jobs", type=int, default=1, help="Parallel worker processes (CPU grids). Use 1 for GPU.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Override device for all configs.",
    )
    p.add_argument(
        "--log_mode",
        type=str,
        default="summary",
        choices=["summary", "agents", "steps", "full"],
        help="Logging mode (grid default is summary to keep output small).",
    )
    p.add_argument("--write_config", action="store_true", help="Also save config.json per run (more files).")
    p.add_argument("--max_runs", type=int, default=None, help="Optional cap (after shard/slice filters).")

    # Sharding/slicing
    p.add_argument("--shard_id", type=int, default=0, help="Shard index (0..num_shards-1).")
    p.add_argument("--num_shards", type=int, default=1, help="Total shards (use >1 for distributed runs).")
    p.add_argument("--start_idx", type=int, default=0, help="Start line index (0-based, among non-empty lines).")
    p.add_argument("--end_idx", type=int, default=None, help="End line index (exclusive).")

    # Robustness knobs
    p.add_argument(
        "--max_tasks_per_child",
        type=int,
        default=None,
        help="Recycle worker processes after N tasks (helps mitigate memory growth on long CPU grids).",
    )
    p.add_argument(
        "--torch_threads",
        type=int,
        default=None,
        help="Set torch intra/inter-op thread count per worker (recommended: 1 when jobs>1).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.out)

    grid_path = Path(args.grid)
    if not grid_path.exists():
        raise FileNotFoundError(str(grid_path))

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    total = _count_grid(
        grid_path,
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
        start_idx=int(args.start_idx),
        end_idx=(int(args.end_idx) if args.end_idx is not None else None),
        max_runs=(int(args.max_runs) if args.max_runs is not None else None),
    )

    cfg_iter = _iter_grid(
        grid_path,
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
        start_idx=int(args.start_idx),
        end_idx=(int(args.end_idx) if args.end_idx is not None else None),
        max_runs=(int(args.max_runs) if args.max_runs is not None else None),
    )

    # Helpful warning: writing tens of thousands of files into cloud-synced folders is a common failure mode.
    out_str = str(out)
    cloud_markers = ["CloudStorage", "GoogleDrive", "Dropbox", "OneDrive", "iCloud"]
    if any(m in out_str for m in cloud_markers):
        print(
            "[WARN] Output directory looks like a cloud-synced path.\n"
            "       For large grids, prefer a local scratch directory to avoid sync overhead and OS kills.\n"
            f"       out={out_str}\n",
            file=sys.stderr,
        )

    ran = 0
    skipped = 0

    if int(args.jobs) <= 1:
        pbar = tqdm(total=total, desc="Runs")
        for cfg in cfg_iter:
            status, _ = _worker(
                cfg,
                out_dir=str(out),
                log_mode=str(args.log_mode),
                device_override=(str(args.device) if args.device is not None else None),
                write_config=bool(args.write_config),
                torch_threads=int(args.torch_threads) if args.torch_threads is not None else None,
            )
            if status == "skipped":
                skipped += 1
            else:
                ran += 1
            pbar.update(1)
        pbar.close()
    else:
        # multiprocessing: safe for CPU-heavy workloads.
        # For GPU training, prefer --jobs 1 (one process controls the GPU).
        ctx = mp.get_context("spawn")
        max_tasks = int(args.max_tasks_per_child) if args.max_tasks_per_child is not None else None
        with ProcessPoolExecutor(
            max_workers=int(args.jobs),
            mp_context=ctx,
            max_tasks_per_child=max_tasks,
        ) as ex:
            it = iter(cfg_iter)
            in_flight = set()
            # Keep a small multiple of jobs in flight to reduce overhead.
            target_in_flight = max(1, int(args.jobs) * 2)

            def submit_one() -> bool:
                try:
                    c = next(it)
                except StopIteration:
                    return False
                fut = ex.submit(
                    _worker,
                    c,
                    str(out),
                    str(args.log_mode),
                    (str(args.device) if args.device is not None else None),
                    bool(args.write_config),
                    (int(args.torch_threads) if args.torch_threads is not None else None),
                )
                in_flight.add(fut)
                return True

            for _ in range(target_in_flight):
                if not submit_one():
                    break

            pbar = tqdm(total=total, desc="Runs")
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    in_flight.remove(fut)
                    status, _ = fut.result()
                    if status == "skipped":
                        skipped += 1
                    else:
                        ran += 1
                    pbar.update(1)
                    # Refill
                    while len(in_flight) < target_in_flight:
                        if not submit_one():
                            break
            pbar.close()

    print(f"Done. ran={ran} skipped={skipped} total={total}")
    print("Next: python -m scripts.aggregate --root", out)


if __name__ == "__main__":
    main()
