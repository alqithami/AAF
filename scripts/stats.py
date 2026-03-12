from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from aaf_q1.utils.io import ensure_dir


KEYS = [
    "env",
    "n_agents",
    "t_steps",
    "penalty_factor",
    "dist_alpha",
    "partial_obs",
    "byzantine_frac",
    "byzantine_start",
]


def cohens_d_paired(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if diff.size < 2:
        return float("nan")
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute paired statistical tests from all_runs_flat.csv.")
    p.add_argument("--flat", type=str, required=False, default=None, help="Path to all_runs_flat.csv (recommended).")
    p.add_argument("--summary", type=str, required=False, default=None, help="Path to final_summary.csv (used to infer flat path).")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--method", type=str, default="aaf_full", help="Method to test (default aaf_full).")
    p.add_argument("--baseline", type=str, default="ppo_only", help="Baseline for paired test.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    flat_path: Path
    if args.flat is not None:
        flat_path = Path(args.flat)
    elif args.summary is not None:
        flat_path = Path(args.summary).parent / "all_runs_flat.csv"
    else:
        raise ValueError("Provide --flat or --summary to locate raw runs.")

    if not flat_path.exists():
        raise FileNotFoundError(str(flat_path))

    df = pd.read_csv(flat_path)

    # Ensure numeric
    for c in ["seed", "n_agents", "t_steps", "byzantine_start"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["penalty_factor", "dist_alpha", "byzantine_frac"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "partial_obs" in df.columns:
        df["partial_obs"] = df["partial_obs"].astype(bool)

    metrics = ["compromise_ratio_executed", "social_welfare", "gini_alloc_mean", "alarms_count", "detection_delay"]

    rows: List[Dict[str, float]] = []
    grouped = df.groupby(KEYS, dropna=False)
    for keys, g in grouped:
        g_m = g[g["baseline"] == args.method].copy()
        g_b = g[g["baseline"] == args.baseline].copy()
        if g_m.empty or g_b.empty:
            continue

        # Pair by seed
        gm = g_m.set_index("seed")
        gb = g_b.set_index("seed")
        common = sorted(set(gm.index.dropna().tolist()) & set(gb.index.dropna().tolist()))
        if len(common) < 2:
            continue

        out: Dict[str, float] = {k: v for k, v in zip(KEYS, keys)}  # type: ignore[arg-type]
        out["n_pairs"] = float(len(common))

        for m in metrics:
            if m not in gm.columns or m not in gb.columns:
                continue
            a = pd.to_numeric(gm.loc[common, m], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(gb.loc[common, m], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask]
            b = b[mask]
            if a.size < 2:
                continue
            diff = a - b
            mean = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            se = sd / np.sqrt(diff.size)
            ci95 = 1.96 * se
            tstat, pval = stats.ttest_rel(a, b, nan_policy="omit")
            out[f"delta_{m}_mean"] = mean
            out[f"delta_{m}_ci95"] = ci95
            out[f"delta_{m}_pval"] = float(pval)
            out[f"delta_{m}_d"] = cohens_d_paired(diff)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = outdir / "stats_paired_tests.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote paired tests to {out_path}")


if __name__ == "__main__":
    main()
