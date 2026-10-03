#!/usr/bin/env python3
"""Mechanism-level adversarial stress tests for the AAF major revision.

These tests complement, rather than replace, the archived reinforcement-learning
factorial study. They are deliberately small, deterministic, and scoped to the
reviewers' threat-model objections: forged self-reports, selective omission,
alert flooding, policy depth/action-shield semantics, and predictive-screening
failure under hidden confounding.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class CUSUMConfig:
    alpha: float = 0.05
    delta: float = 0.01
    h0: float = 5.0
    eta_exp: float = 0.6
    h_min: float = 0.5
    h_max: float = 20.0
    warmup: int = 100


class AdaptiveCUSUM:
    """Projected version of the repository detector, retaining all alarm times."""

    def __init__(self, cfg: CUSUMConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.S = 0.0
        self.h = float(self.cfg.h0)
        self.mu0: Optional[float] = None
        self._warm: List[float] = []
        self.alarms: List[int] = []

    def update(self, z_t: float) -> int:
        self.t += 1
        if self.mu0 is None:
            self._warm.append(float(z_t))
            if len(self._warm) >= self.cfg.warmup:
                self.mu0 = float(np.mean(self._warm))
            return 0
        self.S = max(0.0, self.S + float(z_t) - float(self.mu0) - self.cfg.delta)
        alarm = int(self.S >= self.h)
        if alarm:
            self.S = 0.0
            self.alarms.append(self.t - 1)  # return to zero-based stream index
        eta = self.t ** (-self.cfg.eta_exp)
        self.h = float(np.clip(self.h + eta * (alarm - self.cfg.alpha), self.cfg.h_min, self.cfg.h_max))
        return alarm


def _first_at_or_after(times: Iterable[int], start: int) -> Optional[int]:
    for t in times:
        if t >= start:
            return int(t)
    return None


def _score_window(obs: np.ndarray, visible: np.ndarray, end_t: int, window: int = 50) -> np.ndarray:
    lo = max(0, end_t - window + 1)
    num = np.sum(obs[lo : end_t + 1] * visible[lo : end_t + 1], axis=0)
    den = np.sum(visible[lo : end_t + 1], axis=0)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _ranking_metrics(scores: np.ndarray, attackers: np.ndarray, k: int = 3) -> Tuple[float, float]:
    order = np.argsort(-scores)
    aset = set(int(x) for x in attackers)
    top1 = float(int(order[0]) in aset)
    recall = float(len(set(int(x) for x in order[:k]) & aset) / max(1, min(k, len(aset))))
    return top1, recall


def generate_violation_stream(seed: int, *, T: int = 2000, N: int = 50, change: int = 200, k: int = 5,
                              p_honest: float = 0.03, p_attack: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    attackers = np.sort(rng.choice(np.arange(N), size=k, replace=False))
    p = np.full((T, N), p_honest, dtype=float)
    p[change:, attackers] = p_attack
    true = (rng.random((T, N)) < p).astype(float)
    return true, attackers


def run_detector(obs: np.ndarray, visible: Optional[np.ndarray] = None, *, cfg: Optional[CUSUMConfig] = None,
                 extra_stat: Optional[np.ndarray] = None) -> AdaptiveCUSUM:
    if visible is None:
        visible = np.ones_like(obs, dtype=float)
    cfg = cfg or CUSUMConfig()
    det = AdaptiveCUSUM(cfg)
    for t in range(obs.shape[0]):
        den = float(np.sum(visible[t]))
        rate = float(np.sum(obs[t] * visible[t]) / den) if den > 0 else 0.0
        if extra_stat is not None:
            rate = max(rate, float(extra_stat[t]))
        det.update(rate)
    return det


def telemetry_and_network_tests(seeds: int) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    T, N, change, k = 2000, 50, 200, 5
    scenarios = [
        "E0_forged_self_report",
        "E1_gateway_witness",
        "E2_independent_witness_80pct",
        "random_drop_20pct",
        "selective_drop_naive",
        "selective_drop_visible_gaps",
        "permanent_record_and_checkpoint_suppression",
    ]
    collectors: Dict[str, List[Dict[str, float]]] = {s: [] for s in scenarios}

    for seed in range(seeds):
        true, attackers = generate_violation_stream(seed, T=T, N=N, change=change, k=k)
        rng = np.random.default_rng(100_000 + seed)

        # E0: compromised agents publish compliant claims after injection.
        obs_e0 = true.copy()
        obs_e0[change:, attackers] = 0.0
        vis_all = np.ones_like(true)

        # E1: controlled gateway sees accepted actions.
        obs_e1 = true.copy()

        # E2: an independent witness observes 80% of events. Missing entries are
        # excluded from the rate denominator rather than silently treated as safe.
        vis_e2 = (rng.random(true.shape) < 0.80).astype(float)
        obs_e2 = true.copy()

        # Random loss.
        vis_random = (rng.random(true.shape) >= 0.20).astype(float)

        # Selective omission: incriminating attacker records are dropped. The
        # naive profile silently ignores the missing records.
        vis_sel = np.ones_like(true)
        incrim = np.zeros_like(true, dtype=bool)
        incrim[change:, attackers] = true[change:, attackers] > 0
        vis_sel[incrim] = 0.0

        # Visible-gap profile: a later sequence/checkpoint exposes omitted
        # interior records. We approximate one-step disclosure because each
        # source emits a subsequent monotone sequence number.
        gap_rate = np.zeros(T, dtype=float)
        missing_by_t = np.mean(1.0 - vis_sel, axis=1)
        gap_rate[1:] = missing_by_t[:-1]

        # Permanent suppression of all attacker records and all later
        # checkpoints: no interior gap is exposed. This is outside the bounded
        # delivery profile and is included as a negative control.
        vis_perm = np.ones_like(true)
        vis_perm[change:, attackers] = 0.0

        scenario_data = {
            "E0_forged_self_report": (obs_e0, vis_all, None),
            "E1_gateway_witness": (obs_e1, vis_all, None),
            "E2_independent_witness_80pct": (obs_e2, vis_e2, None),
            "random_drop_20pct": (obs_e1, vis_random, None),
            "selective_drop_naive": (obs_e1, vis_sel, None),
            "selective_drop_visible_gaps": (obs_e1, vis_sel, gap_rate),
            "permanent_record_and_checkpoint_suppression": (obs_e1, vis_perm, None),
        }

        for name, (obs, vis, extra) in scenario_data.items():
            det = run_detector(obs, vis, extra_stat=extra)
            post = _first_at_or_after(det.alarms, change)
            pre = any(t < change for t in det.alarms)
            if post is not None:
                scores = _score_window(obs, vis, post, window=50)
                top1, recall3 = _ranking_metrics(scores, attackers, 3)
                delay = float(post - change)
            else:
                top1 = recall3 = delay = float("nan")
            collectors[name].append({
                "detected": float(post is not None),
                "pre_alarm": float(pre),
                "delay": delay,
                "top1": top1,
                "recall3": recall3,
                "event_coverage": float(np.mean(vis[change:])),
                "attack_event_coverage": float(np.mean(vis[change:, attackers])),
            })

    for name in scenarios:
        df = pd.DataFrame(collectors[name])
        detected = df[df["detected"] == 1.0]
        rows.append({
            "experiment": "telemetry_network",
            "scenario": name,
            "seeds": float(seeds),
            "detection_coverage": float(df["detected"].mean()),
            "pre_injection_alarm_rate": float(df["pre_alarm"].mean()),
            "median_delay": float(detected["delay"].median()) if not detected.empty else float("nan"),
            "delay_q25": float(detected["delay"].quantile(.25)) if not detected.empty else float("nan"),
            "delay_q75": float(detected["delay"].quantile(.75)) if not detected.empty else float("nan"),
            "conditional_top1": float(detected["top1"].mean()) if not detected.empty else float("nan"),
            "conditional_recall3": float(detected["recall3"].mean()) if not detected.empty else float("nan"),
            "event_coverage": float(df["event_coverage"].mean()),
            "attack_event_coverage": float(df["attack_event_coverage"].mean()),
        })
    return pd.DataFrame(rows)


def alert_flood_tests(seeds: int) -> pd.DataFrame:
    T, change = 2000, 200
    legacy_rows: List[Dict[str, float]] = []
    bounded_rows: List[Dict[str, float]] = []

    for seed in range(seeds):
        rng = np.random.default_rng(200_000 + seed)
        z = rng.binomial(50, 0.03, size=T) / 50.0
        # Low-cost periodic pulses with jitter. They are long enough to trip the
        # same CUSUM, but separated enough to emulate repeated alert regions.
        t = change + int(rng.integers(0, 20))
        while t < T:
            width = int(rng.integers(12, 21))
            z[t : min(T, t + width)] = rng.binomial(50, 0.28, size=min(T, t + width) - t) / 50.0
            t += int(rng.integers(70, 91))
        det = AdaptiveCUSUM(CUSUMConfig())
        for val in z:
            det.update(float(val))
        alarms = det.alarms

        # Collapse adjacent alarms into alert regions.
        regions: List[int] = []
        for a in alarms:
            if not regions or a - regions[-1] > 10:
                regions.append(a)

        # Legacy proxy: after three regions in 300 steps, global learning is
        # frozen. Repeated pulses prevent the favorable 300-step quiet-period
        # clear condition, so the state stays latched.
        legacy_state = np.zeros(T, dtype=int)
        trigger: Optional[int] = None
        for i, a in enumerate(regions):
            recent = [x for x in regions[: i + 1] if a - 300 < x <= a]
            if len(recent) >= 3:
                trigger = a
                break
        if trigger is not None:
            legacy_state[trigger:] = 1
        legacy_rows.append({
            "duty": float(np.mean(legacy_state)),
            "max_contiguous": float(T - trigger) if trigger is not None else 0.0,
            "regions": float(len(regions)),
        })

        # Revised bounded scheduler: targeted mode only, maximum 50 steps per
        # activation, at most two automatic activations in any 300-step window,
        # and a 75-step cooldown. It never applies a global learning freeze.
        bounded_state = np.zeros(T, dtype=int)
        starts: List[int] = []
        last_end = -10_000
        for a in regions:
            starts = [s for s in starts if a - 300 < s <= a]
            if a < last_end + 75:
                continue
            if len(starts) >= 2:
                continue
            end = min(T, a + 50)
            bounded_state[a:end] = 1
            starts.append(a)
            last_end = end
        # contiguous max
        max_run = run = 0
        for x in bounded_state:
            run = run + 1 if x else 0
            max_run = max(max_run, run)
        bounded_rows.append({
            "duty": float(np.mean(bounded_state)),
            "max_contiguous": float(max_run),
            "regions": float(len(regions)),
            "global_freeze_duty": 0.0,
        })

    legacy = pd.DataFrame(legacy_rows)
    bounded = pd.DataFrame(bounded_rows)
    return pd.DataFrame([
        {
            "experiment": "alert_flooding",
            "scenario": "legacy_three_alert_global_freeze_proxy",
            "seeds": float(seeds),
            "median_degraded_duty": float(legacy["duty"].median()),
            "q25_degraded_duty": float(legacy["duty"].quantile(.25)),
            "q75_degraded_duty": float(legacy["duty"].quantile(.75)),
            "median_max_contiguous_steps": float(legacy["max_contiguous"].median()),
            "median_alert_regions": float(legacy["regions"].median()),
            "global_freeze_duty": float(legacy["duty"].median()),
        },
        {
            "experiment": "alert_flooding",
            "scenario": "bounded_targeted_scheduler",
            "seeds": float(seeds),
            "median_degraded_duty": float(bounded["duty"].median()),
            "q25_degraded_duty": float(bounded["duty"].quantile(.25)),
            "q75_degraded_duty": float(bounded["duty"].quantile(.75)),
            "median_max_contiguous_steps": float(bounded["max_contiguous"].median()),
            "median_alert_regions": float(bounded["regions"].median()),
            "global_freeze_duty": 0.0,
        },
    ])


if torch is not None:
    class MLPPolicy(nn.Module):
        def __init__(self, obs_dim: int, action_dim: int, depth: int, width: int = 64):
            super().__init__()
            layers: List[nn.Module] = []
            d = obs_dim
            for _ in range(depth):
                layers += [nn.Linear(d, width), nn.ReLU()]
                d = width
            layers += [nn.Linear(d, action_dim), nn.Sigmoid()]
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def _complete_projection(a: "torch.Tensor", cap: float = 0.6, semantic_cap: float = 3.2) -> "torch.Tensor":
    first = torch.clamp(a[:, :4], max=cap - 1e-4)
    block = a[:, 4:8]
    total = block.sum(dim=1, keepdim=True)
    scale = torch.clamp(semantic_cap / (total + 1e-8), max=1.0)
    middle = block * scale
    return torch.cat([first, middle, a[:, 8:]], dim=1)


def shield_tests(seeds: int) -> pd.DataFrame:
    if torch is None:
        return pd.DataFrame()
    torch.set_num_threads(1)
    rows: List[Dict[str, float | str]] = []
    obs_dim, action_dim = 32, 16
    cap, semantic_cap = 0.6, 3.2
    for depth in (2, 4, 8):
        per: List[Dict[str, float]] = []
        for seed in range(seeds):
            torch.manual_seed(300_000 + seed + depth * 1000)
            np.random.seed(300_000 + seed + depth * 1000)
            model = MLPPolicy(obs_dim, action_dim, depth)
            opt = torch.optim.Adam(model.parameters(), lr=3e-3)
            for _ in range(120):
                x = torch.randn(256, obs_dim)
                proposed = model(x)
                # The attacker optimizes executed utility against the incomplete
                # shield. It can move reward into unconstrained dimensions.
                executed = torch.cat([torch.clamp(proposed[:, :4], max=cap - 1e-4), proposed[:, 4:]], dim=1)
                utility = executed.sum(dim=1).mean()
                opt.zero_grad()
                (-utility).backward()
                opt.step()
            with torch.no_grad():
                x = torch.randn(2048, obs_dim)
                proposed = model(x)
                incomplete = torch.cat([torch.clamp(proposed[:, :4], max=cap - 1e-4), proposed[:, 4:]], dim=1)
                complete = _complete_projection(proposed, cap, semantic_cap)

                def metrics(a: torch.Tensor) -> Tuple[float, float]:
                    encoded = (a[:, :4] >= cap).any(dim=1).float().mean().item()
                    semantic = (a[:, 4:8].sum(dim=1) > semantic_cap + 1e-6).float().mean().item()
                    return encoded, semantic

                p_enc, p_sem = metrics(proposed)
                i_enc, i_sem = metrics(incomplete)
                c_enc, c_sem = metrics(complete)
                per.append({
                    "proposed_encoded": p_enc,
                    "proposed_semantic": p_sem,
                    "incomplete_encoded": i_enc,
                    "incomplete_semantic": i_sem,
                    "complete_encoded": c_enc,
                    "complete_semantic": c_sem,
                })
        df = pd.DataFrame(per)
        for profile, e_col, s_col in [
            ("no_shield", "proposed_encoded", "proposed_semantic"),
            ("predicate_specific_external_shield", "incomplete_encoded", "incomplete_semantic"),
            ("expanded_external_safe_set", "complete_encoded", "complete_semantic"),
        ]:
            rows.append({
                "experiment": "adaptive_policy_shield",
                "scenario": profile,
                "depth": float(depth),
                "seeds": float(seeds),
                "encoded_violation_rate": float(df[e_col].mean()),
                "semantic_harm_rate": float(df[s_col].mean()),
            })
    return pd.DataFrame(rows)


def _ols_rss(X: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(res @ res), X.shape[1]


def granger_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    yt = y[1:]
    ylag = y[:-1]
    xlag = x[:-1]
    Xr = np.column_stack([np.ones_like(ylag), ylag])
    Xu = np.column_stack([np.ones_like(ylag), ylag, xlag])
    rss_r, kr = _ols_rss(Xr, yt)
    rss_u, ku = _ols_rss(Xu, yt)
    df1 = ku - kr
    df2 = len(yt) - ku
    num = max(0.0, (rss_r - rss_u) / max(1, df1))
    den = rss_u / max(1, df2)
    if den <= 0:
        return 1.0
    f = num / den
    return float(stats.f.sf(f, df1, df2))


def predictive_screening_tests(seeds: int) -> pd.DataFrame:
    records: List[Dict[str, float | str]] = []
    T, N = 1200, 10
    for confounded in (False, True):
        per: List[Dict[str, float]] = []
        for seed in range(seeds):
            rng = np.random.default_rng(400_000 + seed + 10000 * int(confounded))
            h = np.zeros(T)
            for t in range(1, T):
                h[t] = 0.85 * h[t - 1] + rng.normal(scale=.6)
            x = rng.normal(scale=.7, size=(T, N))
            if confounded:
                x += h[:, None] * rng.uniform(.5, 1.2, size=(1, N))
            # Only agent 0 has a direct lagged contribution. The hidden state
            # also drives the outcome in the confounded condition.
            y = np.zeros(T)
            for t in range(1, T):
                direct = 0.9 * np.tanh(x[t - 1, 0]) ** 2
                hidden = 1.0 * h[t - 1] if confounded else 0.0
                y[t] = .45 * y[t - 1] + direct + hidden + rng.normal(scale=.8)
            pvals = np.array([granger_pvalue(x[:, i], y) for i in range(N)])
            selected = set(np.where(pvals < .05)[0].tolist())
            tp = float(0 in selected)
            fp = float(len(selected - {0}))
            precision = tp / max(1.0, tp + fp)
            top1 = float(int(np.argmin(pvals)) == 0)
            per.append({
                "true_edge_selected": tp,
                "false_edges": fp,
                "precision": precision,
                "top1": top1,
                "any_edge": float(len(selected) > 0),
            })
        df = pd.DataFrame(per)
        records.append({
            "experiment": "predictive_screening_negative_control",
            "scenario": "hidden_common_cause" if confounded else "no_hidden_common_cause",
            "seeds": float(seeds),
            "true_edge_selection_rate": float(df["true_edge_selected"].mean()),
            "mean_false_edges": float(df["false_edges"].mean()),
            "mean_precision": float(df["precision"].mean()),
            "top1_rate": float(df["top1"].mean()),
            "any_edge_rate": float(df["any_edge"].mean()),
        })
    return pd.DataFrame(records)


def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--stream-seeds", type=int, default=500)
    p.add_argument("--flood-seeds", type=int, default=500)
    p.add_argument("--shield-seeds", type=int, default=12)
    p.add_argument("--screening-seeds", type=int, default=250)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    telemetry = telemetry_and_network_tests(args.stream_seeds)
    flood = alert_flood_tests(args.flood_seeds)
    shield = shield_tests(args.shield_seeds)
    screening = predictive_screening_tests(args.screening_seeds)

    telemetry.to_csv(out / "telemetry_network_results.csv", index=False)
    flood.to_csv(out / "alert_flood_results.csv", index=False)
    shield.to_csv(out / "shield_results.csv", index=False)
    screening.to_csv(out / "predictive_screening_results.csv", index=False)

    payload = {
        "design": {
            "stream_seeds": args.stream_seeds,
            "flood_seeds": args.flood_seeds,
            "shield_seeds_per_depth": args.shield_seeds,
            "screening_seeds": args.screening_seeds,
            "stream_T": 2000,
            "stream_N": 50,
            "stream_attackers": 5,
            "change_index": 200,
            "cusum": CUSUMConfig().__dict__,
        },
        "telemetry_network": telemetry.to_dict(orient="records"),
        "alert_flooding": flood.to_dict(orient="records"),
        "adaptive_policy_shield": shield.to_dict(orient="records"),
        "predictive_screening": screening.to_dict(orient="records"),
    }
    (out / "targeted_robustness_results.json").write_text(
        json.dumps(clean_json(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(clean_json(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
