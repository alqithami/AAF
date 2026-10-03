#!/usr/bin/env python3
"""Re-audit the archived AAF main-grid and scaling CSV files.

The script makes no assumption that repeated rows are independent.  It verifies
same-seed multiplicities, removes the unused public-goods dist_alpha factor,
constructs seed-matched AAF-vs-PPO regime comparisons, applies Holm correction,
and reports detection/attribution coverage without assigning finite delays to
missed detections.

Example
-------
python audit_existing_results.py \
    --main path/to/all_runs_flat.csv \
    --scaling path/to/scaling_all_runs_flat.csv \
    --out audit_output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

CONFIG_SEED_KEYS = [
    "env", "baseline", "n_agents", "t_steps", "penalty_factor", "dist_alpha",
    "partial_obs", "byzantine_frac", "byzantine_start", "seed",
]
REGIME_KEYS = [k for k in CONFIG_SEED_KEYS if k not in {"baseline", "seed"}]
METRICS = ["compromise_ratio_executed", "social_welfare", "gini_alloc_mean"]
SCIENTIFIC_EXCLUSIONS = {
    "runtime_s", "device_requested", "device_resolved", "log_mode",
    "byzantine_ids", "config_path", "run_dir", "timestamp",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["n_agents", "t_steps", "byzantine_start", "seed", "alarms_count", "first_alarm_t"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["penalty_factor", "dist_alpha", "byzantine_frac", *METRICS,
              "attrib_top1_correct", "attrib_recall3", "detection_delay"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "partial_obs" in out:
        out["partial_obs"] = out["partial_obs"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(out["partial_obs"].astype(bool))
    return out


def scientific_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in SCIENTIFIC_EXCLUSIONS]


def verify_same_seed_copies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_columns(df, CONFIG_SEED_KEYS, "main CSV")
    counts = df.groupby(CONFIG_SEED_KEYS, dropna=False).size()
    scientific = scientific_columns(df)
    varying: dict[str, int] = {}
    for c in scientific:
        if c in CONFIG_SEED_KEYS:
            continue
        nunique = df.groupby(CONFIG_SEED_KEYS, dropna=False)[c].nunique(dropna=False)
        n = int((nunique > 1).sum())
        if n:
            varying[c] = n
    # Keep one scientifically identical row per config/seed. Runtime is not used
    # in scientific inference; retaining the first row preserves provenance.
    effective = df.sort_values(CONFIG_SEED_KEYS).drop_duplicates(CONFIG_SEED_KEYS, keep="first")
    report = {
        "raw_rows": int(len(df)),
        "distinct_config_seed_rows": int(len(effective)),
        "multiplicity_distribution": {str(int(k)): int(v) for k, v in counts.value_counts().sort_index().items()},
        "within_key_variation": varying,
    }
    return effective, report


def collapse_unused_public_goods_factor(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pg = df[df["env"] == "public_goods"].copy()
    rs = df[df["env"] != "public_goods"].copy()
    pg_keys = [c for c in CONFIG_SEED_KEYS if c != "dist_alpha"]
    scientific = [c for c in scientific_columns(pg) if c not in CONFIG_SEED_KEYS]
    variation: dict[str, int] = {}
    for c in scientific:
        n = int((pg.groupby(pg_keys, dropna=False)[c].nunique(dropna=False) > 1).sum())
        if n:
            variation[c] = n
    # Use one canonical label because the environment code does not consume
    # dist_alpha. Prefer 1.0 when present to align with the paper tables.
    pg = pg.sort_values("dist_alpha", ascending=False).drop_duplicates(pg_keys, keep="first")
    effective = pd.concat([rs, pg], ignore_index=True).sort_values(CONFIG_SEED_KEYS).reset_index(drop=True)
    return effective, {
        "public_goods_factor_values": sorted(df.loc[df.env == "public_goods", "dist_alpha"].dropna().unique().tolist()),
        "public_goods_outcome_variation_across_ignored_factor": variation,
        "effective_rows": int(len(effective)),
    }


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    ci95 = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
    t_p = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue) if n > 1 else float("nan")
    try:
        w_p = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    except ValueError:
        w_p = 1.0
    return {"mean": mean, "ci95": ci95, "t_p": t_p, "w_p": w_p}


def regime_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(REGIME_KEYS, dropna=False):
        gm = g[g.baseline == "aaf_full"].set_index("seed")
        gb = g[g.baseline == "ppo_only"].set_index("seed")
        common = sorted(set(gm.index) & set(gb.index))
        if len(common) < 2:
            continue
        row = {k: v for k, v in zip(REGIME_KEYS, keys)}
        row["n_pairs"] = len(common)
        for metric in METRICS:
            a = pd.to_numeric(gm.loc[common, metric], errors="coerce").to_numpy(float)
            b = pd.to_numeric(gb.loc[common, metric], errors="coerce").to_numpy(float)
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            test = paired_test(a, b)
            row[f"aaf_{metric}_mean"] = float(np.mean(a))
            row[f"ppo_{metric}_mean"] = float(np.mean(b))
            row[f"delta_{metric}_mean"] = test["mean"]
            row[f"delta_{metric}_ci95"] = test["ci95"]
            row[f"delta_{metric}_pval"] = test["t_p"]
            row[f"delta_{metric}_wilcoxon_pval"] = test["w_p"]
        ppo_comp = row["ppo_compromise_ratio_executed_mean"]
        ppo_reward = row["ppo_social_welfare_mean"]
        row["relative_compromise_reduction"] = (
            (ppo_comp - row["aaf_compromise_ratio_executed_mean"]) / abs(ppo_comp)
            if abs(ppo_comp) > 1e-12 else np.nan
        )
        row["relative_welfare_change"] = (
            (row["aaf_social_welfare_mean"] - ppo_reward) / abs(ppo_reward)
            if abs(ppo_reward) > 1e-12 else np.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(REGIME_KEYS).reset_index(drop=True)
    for metric in METRICS:
        for suffix in ["pval", "wilcoxon_pval"]:
            col = f"delta_{metric}_{suffix}"
            valid = out[col].notna()
            adj = np.full(len(out), np.nan)
            reject = np.zeros(len(out), dtype=bool)
            if valid.any():
                r, p, _, _ = multipletests(out.loc[valid, col].to_numpy(float), alpha=0.05, method="holm")
                adj[valid] = p
                reject[valid] = r
            out[f"{col.rsplit('_pval',1)[0]}_holm_pval"] = adj
            out[f"{col.rsplit('_pval',1)[0]}_holm_reject"] = reject
    return out


def realized_attacker_count(n: float, frac: float) -> int:
    # Python round uses banker's rounding; this matches the repository runner.
    return max(0, min(int(n), int(round(float(frac) * int(n)))))


def detection_attribution(df: pd.DataFrame) -> dict[str, Any]:
    aaf = df[(df.baseline == "aaf_full") & (df.byzantine_frac > 0)].copy()
    aaf["n_byzantine"] = [realized_attacker_count(n, r) for n, r in zip(aaf.n_agents, aaf.byzantine_frac)]
    aaf = aaf[aaf.n_byzantine > 0].copy()
    # Supervisor time is one-based while injection index is zero-based.
    injection_clock = aaf["byzantine_start"] + 1
    qualifying = aaf["first_alarm_t"].notna() & (aaf["first_alarm_t"] >= injection_clock)
    pre = aaf["first_alarm_t"].notna() & (aaf["first_alarm_t"] < injection_clock)
    missing = aaf["first_alarm_t"].isna()
    delays = (aaf.loc[qualifying, "first_alarm_t"] - injection_clock[qualifying]).to_numpy(float)
    top1 = pd.to_numeric(aaf.loc[qualifying, "attrib_top1_correct"], errors="coerce")
    r3 = pd.to_numeric(aaf.loc[qualifying, "attrib_recall3"], errors="coerce")
    n = len(aaf)
    return {
        "actual_attacker_aaf_runs": int(n),
        "qualifying_post_injection_first_alarm_count": int(qualifying.sum()),
        "detection_coverage": float(qualifying.mean()) if n else np.nan,
        "pre_injection_first_alarm_count": int(pre.sum()),
        "pre_injection_first_alarm_fraction": float(pre.mean()) if n else np.nan,
        "no_first_alarm_count": int(missing.sum()),
        "no_first_alarm_fraction": float(missing.mean()) if n else np.nan,
        "detection_delay_conditional_median": float(np.median(delays)) if len(delays) else np.nan,
        "detection_delay_conditional_q1": float(np.quantile(delays, 0.25)) if len(delays) else np.nan,
        "detection_delay_conditional_q3": float(np.quantile(delays, 0.75)) if len(delays) else np.nan,
        "attribution_coverage": float(top1.notna().sum() / n) if n else np.nan,
        "post_injection_ranking_completeness": float(top1.notna().mean()) if len(top1) else np.nan,
        "attribution_top1_conditional_mean": float(top1.mean()),
        "attribution_recall3_conditional_mean": float(r3.mean()),
    }


def frac(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, type=Path, help="Archived main-grid flat CSV")
    ap.add_argument("--scaling", type=Path, default=None, help="Optional scaling flat CSV")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = normalize(pd.read_csv(args.main))
    distinct, copy_report = verify_same_seed_copies(raw)
    effective, factor_report = collapse_unused_public_goods_factor(distinct)
    comparisons = regime_comparisons(effective)

    report: dict[str, Any] = {
        "raw_main_sha256": sha256(args.main),
        "raw_main_rows": int(len(raw)),
        **copy_report,
        **factor_report,
        "operational_regimes": int(len(comparisons)),
        "compromise_lower_regime_count": int((comparisons.delta_compromise_ratio_executed_mean < 0).sum()),
        "compromise_lower_regime_fraction": frac(comparisons.delta_compromise_ratio_executed_mean < 0),
        "compromise_median_relative_reduction": float(comparisons.relative_compromise_reduction.median()),
        "compromise_uncorrected_significant_fraction": frac(comparisons.delta_compromise_ratio_executed_pval < 0.05),
        "compromise_holm_significant_fraction": frac(comparisons.delta_compromise_ratio_executed_holm_reject),
        "compromise_wilcoxon_uncorrected_significant_fraction": frac(comparisons.delta_compromise_ratio_executed_wilcoxon_pval < 0.05),
        "compromise_wilcoxon_holm_significant_fraction": frac(comparisons.delta_compromise_ratio_executed_wilcoxon_holm_reject),
        "welfare_higher_regime_count": int((comparisons.delta_social_welfare_mean > 0).sum()),
        "welfare_higher_regime_fraction": frac(comparisons.delta_social_welfare_mean > 0),
        "welfare_median_relative_change": float(comparisons.relative_welfare_change.median()),
        "welfare_uncorrected_significant_fraction": frac(comparisons.delta_social_welfare_pval < 0.05),
        "welfare_holm_significant_fraction": frac(comparisons.delta_social_welfare_holm_reject),
        "welfare_wilcoxon_uncorrected_significant_fraction": frac(comparisons.delta_social_welfare_wilcoxon_pval < 0.05),
        "welfare_wilcoxon_holm_significant_fraction": frac(comparisons.delta_social_welfare_wilcoxon_holm_reject),
        "allocation_gini_lower_regime_count": int((comparisons.delta_gini_alloc_mean_mean < 0).sum()),
        "allocation_gini_lower_regime_fraction": frac(comparisons.delta_gini_alloc_mean_mean < 0),
        **detection_attribution(effective),
    }
    if args.scaling is not None:
        scaling = normalize(pd.read_csv(args.scaling))
        report["raw_scaling_sha256"] = sha256(args.scaling)
        report["scaling_rows"] = int(len(scaling))

    effective.to_csv(args.out / "effective_main_rows.csv", index=False)
    comparisons.to_csv(args.out / "regime_comparisons.csv", index=False)
    (args.out / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = ["# AAF empirical audit", ""]
    for k, v in report.items():
        md.append(f"- **{k}**: {v}")
    (args.out / "audit_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
