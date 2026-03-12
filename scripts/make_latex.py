from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from aaf_q1.utils.io import ensure_dir


def fmt_pm(mean: float, ci: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "--"
    if np.isnan(ci) or ci == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def make_table(df: pd.DataFrame, baselines: List[str]) -> str:
    lines = []
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Baseline & Compromise$\\downarrow$ & Welfare$\\uparrow$ & Gini$\\downarrow$ & Alarms & Delay \\\\")
    lines.append("\\midrule")
    for b in baselines:
        r = df[df["baseline"] == b]
        if r.empty:
            continue
        row = r.iloc[0]
        comp = fmt_pm(row["compromise_ratio_executed_mean"], row.get("compromise_ratio_executed_ci95", np.nan))
        welfare = fmt_pm(row["social_welfare_mean"], row.get("social_welfare_ci95", np.nan))
        g = fmt_pm(row["gini_alloc_mean_mean"], row.get("gini_alloc_mean_ci95", np.nan))
        alarms = row.get("alarms_count_mean", np.nan)
        delay = row.get("detection_delay_mean", np.nan)
        alarms_s = "--" if np.isnan(alarms) else f"{alarms:.2f}"
        delay_s = "--" if np.isnan(delay) else f"{delay:.2f}"
        lines.append(f"{b} & {comp} & {welfare} & {g} & {alarms_s} & {delay_s} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert aggregated CSV into LaTeX tables.")
    p.add_argument("--summary", type=str, required=True, help="Path to final_summary.csv (output of scripts.aggregate).")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for LaTeX tables.")
    p.add_argument("--env", type=str, default="resource_sharing")
    p.add_argument("--n_agents", type=int, default=None)
    p.add_argument("--byzantine_frac", type=float, default=0.0)
    p.add_argument("--penalty_factor", type=float, default=0.2)
    p.add_argument("--dist_alpha", type=float, default=1.0)
    p.add_argument("--partial_obs", type=int, default=0, help="0/1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    df = pd.read_csv(args.summary)

    # Filter to a canonical slice (paper-friendly)
    sdf = df[
        (df["env"] == args.env)
        & (np.isclose(df["penalty_factor"], args.penalty_factor))
        & (np.isclose(df["dist_alpha"], args.dist_alpha))
        & (df["partial_obs"].astype(int) == int(args.partial_obs))
        & (np.isclose(df["byzantine_frac"], args.byzantine_frac))
    ].copy()

    if args.n_agents is not None:
        sdf = sdf[sdf["n_agents"] == int(args.n_agents)]

    if sdf.empty:
        # fallback: ignore pf/alpha slice
        sdf = df[(df["env"] == args.env) & (np.isclose(df["byzantine_frac"], args.byzantine_frac))].copy()

    # Build main table per n_agents
    baselines = ["ppo_only", "static_guard", "constrained_ppo", "fair_ppo", "aaf_full"]
    n_agents_list = sorted(sdf["n_agents"].unique().tolist())

    tables = []
    for n in n_agents_list:
        dfn = sdf[sdf["n_agents"] == n].sort_values("baseline")
        tables.append(f"% env={args.env}, N={n}, byz={args.byzantine_frac}\n" + make_table(dfn, baselines))

    main_tex = "\n\n".join(tables)
    (outdir / "table_main_results.tex").write_text(main_tex + "\n", encoding="utf-8")

    # Ablation table: compare AAF variants (if present)
    ab_baselines = ["ppo_only", "aaf_full", "aaf_detector_only", "aaf_shaping_only", "aaf_patch_only", "aaf_no_attrib"]
    ab_tables = []
    for n in n_agents_list:
        dfn = sdf[sdf["n_agents"] == n].sort_values("baseline")
        ab_tables.append(f"% env={args.env}, N={n}, byz={args.byzantine_frac}\n" + make_table(dfn, ab_baselines))
    (outdir / "table_ablation.tex").write_text("\n\n".join(ab_tables) + "\n", encoding="utf-8")

    print(f"Wrote LaTeX tables to {outdir}")


if __name__ == "__main__":
    main()
