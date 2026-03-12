from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aaf_q1.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-ready figures from aggregated results.")
    p.add_argument("--summary", type=str, required=True, help="Path to final_summary.csv.")
    p.add_argument("--flat", type=str, default=None, help="Optional path to all_runs_flat.csv (for distributions).")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--env", type=str, default="resource_sharing")
    p.add_argument("--byzantine_frac", type=float, default=0.0)
    p.add_argument("--penalty_factor", type=float, default=0.2)
    p.add_argument("--dist_alpha", type=float, default=1.0)
    p.add_argument("--partial_obs", type=int, default=0)
    p.add_argument("--n_agents", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    sdf = pd.read_csv(args.summary)

    # canonical slice
    filt = (
        (sdf["env"] == args.env)
        & (np.isclose(sdf["penalty_factor"], args.penalty_factor))
        & (np.isclose(sdf["dist_alpha"], args.dist_alpha))
        & (sdf["partial_obs"].astype(int) == int(args.partial_obs))
        & (np.isclose(sdf["byzantine_frac"], args.byzantine_frac))
    )
    df = sdf[filt].copy()
    if args.n_agents is not None:
        df = df[df["n_agents"] == int(args.n_agents)]

    if df.empty:
        # fallback
        df = sdf[(sdf["env"] == args.env) & (np.isclose(sdf["byzantine_frac"], args.byzantine_frac))].copy()

    # Pick one N for single-panel plots
    n_list = sorted(df["n_agents"].unique().tolist())
    N = int(args.n_agents) if args.n_agents is not None else int(n_list[len(n_list) // 2])
    dfN = df[df["n_agents"] == N].copy()

    # --- 1) Compromise bar plot ---
    baselines = ["ppo_only", "static_guard", "constrained_ppo", "fair_ppo", "aaf_full", "aaf_no_attrib"]
    df_plot = dfN[dfN["baseline"].isin(baselines)].copy()
    df_plot = df_plot.sort_values("baseline")

    x = np.arange(len(df_plot))
    y = df_plot["compromise_ratio_executed_mean"].to_numpy(dtype=float)
    e = df_plot.get("compromise_ratio_executed_ci95", pd.Series([np.nan]*len(df_plot))).to_numpy(dtype=float)

    plt.figure()
    plt.bar(x, y, yerr=e)
    plt.xticks(x, df_plot["baseline"], rotation=30, ha="right")
    plt.ylabel("Compromise ratio (executed)")
    plt.title(f"{args.env} (N={N}, byz={args.byzantine_frac})")
    plt.tight_layout()
    plt.savefig(outdir / "fig_compromise_bar.png", dpi=200)
    plt.close()

    # --- 2) Pareto scatter: compromise vs gini ---
    plt.figure()
    x2 = dfN["compromise_ratio_executed_mean"].to_numpy(dtype=float)
    y2 = dfN["gini_alloc_mean_mean"].to_numpy(dtype=float)
    plt.scatter(x2, y2)
    for _, r in dfN.iterrows():
        plt.text(float(r["compromise_ratio_executed_mean"]), float(r["gini_alloc_mean_mean"]), str(r["baseline"]), fontsize=8)
    plt.xlabel("Compromise ratio (executed)")
    plt.ylabel("Mean Gini (alloc/returns)")
    plt.title(f"Pareto: compromise vs inequality ({args.env}, N={N})")
    plt.tight_layout()
    plt.savefig(outdir / "fig_pareto_compromise_gini.png", dpi=200)
    plt.close()

    # --- 3) Detection delay CDF (requires flat file) ---
    flat_path: Optional[Path] = Path(args.flat) if args.flat is not None else None
    if flat_path is None:
        # try to infer from summary path
        cand = Path(args.summary).parent / "all_runs_flat.csv"
        if cand.exists():
            flat_path = cand

    if flat_path is not None and flat_path.exists():
        flat = pd.read_csv(flat_path)
        flat["detection_delay"] = pd.to_numeric(flat["detection_delay"], errors="coerce")
        dd = flat[
            (flat["env"] == args.env)
            & (flat["baseline"] == "aaf_full")
            & (flat["byzantine_frac"] > 0.0)
            & (flat["detection_delay"].notna())
        ]["detection_delay"].to_numpy(dtype=float)

        if dd.size > 0:
            dd = np.sort(dd)
            cdf = np.arange(1, dd.size + 1) / dd.size
            plt.figure()
            plt.plot(dd, cdf)
            plt.xlabel("Detection delay (steps)")
            plt.ylabel("Empirical CDF")
            plt.title(f"Detection delay CDF (AAF, {args.env})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "fig_detection_delay_cdf.png", dpi=200)
            plt.close()

    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()
