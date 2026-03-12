from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .aaf.supervisor import AAFSupervisor, SupervisorConfig
from .agents.ppo_shared import PPOConfig, PPOSharedAgent
from .envs.public_goods import PublicGoodsConfig, PublicGoodsEnv
from .envs.resource_sharing import ResourceSharingConfig, ResourceSharingEnv
from .metrics import detection_delay, gini
from .utils.device import resolve_device
from .utils.seeding import seed_everything


def _select_byzantine(n_agents: int, frac: float, rng: np.random.Generator) -> List[int]:
    frac = float(frac)
    if frac <= 0.0:
        return []
    k = int(round(frac * n_agents))
    k = max(0, min(n_agents, k))
    if k == 0:
        return []
    return [int(i) for i in rng.choice(np.arange(n_agents), size=k, replace=False)]


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment.

    Returns
    -------
    dict with keys:
      - summary: Dict[str, Any] (scalar metrics)
      - step_logs: List[Dict[str, Any]]  (optional; may be empty)
      - agent_logs: List[Dict[str, Any]] (optional; may be empty)

    Notes
    -----
    For large grids, you should set cfg["log_mode"] = "summary" (default for
    scripts.run_grid in v4) to avoid heavy per-step logging overhead.
    """

    t0 = time.time()

    # --- Core config ---
    env_name = str(cfg.get("env", "resource_sharing"))
    baseline = str(cfg.get("baseline", "ppo_only"))
    n_agents = int(cfg.get("n_agents", 50))
    t_steps = int(cfg.get("t_steps", 1000))
    seed = int(cfg.get("seed", 0))

    byz_frac = float(cfg.get("byzantine_frac", 0.0))
    byz_start = int(cfg.get("byzantine_start", 0))

    # Logging mode controls runtime + output size.
    log_mode = str(cfg.get("log_mode", "full")).lower().strip()
    if log_mode not in {"summary", "agents", "steps", "full"}:
        raise ValueError(f"Invalid log_mode: {log_mode} (expected summary/agents/steps/full)")
    want_step_logs = log_mode in {"steps", "full"}
    want_agent_logs = log_mode in {"agents", "full"}

    device_req = str(cfg.get("device", "auto"))
    device_info = resolve_device(device_req)  # resolves to cuda/mps/cpu

    # Shared RNG for byzantine selection
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    # --- Environment ---
    env_kwargs = dict(cfg.get("env_kwargs", {}) or {})
    if env_name == "resource_sharing":
        env_cfg = ResourceSharingConfig(
            n_agents=n_agents,
            t_steps=t_steps,
            r_max=float(env_kwargs.get("r_max", 100.0)),
            r_in=float(env_kwargs.get("r_in", 0.0)),
            q_max=float(env_kwargs.get("q_max", 100.0)),
            dist_alpha=float(env_kwargs.get("dist_alpha", cfg.get("dist_alpha", 1.0))),
            greedy_threshold=float(env_kwargs.get("greedy_threshold", 0.6)),
            penalty_factor=float(env_kwargs.get("penalty_factor", cfg.get("penalty_factor", 0.2))),
            social_lambda=float(env_kwargs.get("social_lambda", 0.3)),
            partial_obs=bool(env_kwargs.get("partial_obs", cfg.get("partial_obs", False))),
            obs_noise=float(env_kwargs.get("obs_noise", 0.01)),
            graph_k=int(env_kwargs.get("graph_k", 4)),
            graph_p=float(env_kwargs.get("graph_p", 0.1)),
        )
        env = ResourceSharingEnv(env_cfg, seed=seed)

        action_scale = env_cfg.q_max  # action_fraction * q_max -> executed request
        threshold_exec = env_cfg.greedy_threshold * env_cfg.r_max

        # For patch/guard: cap fraction corresponding to greedy threshold
        cap_fraction = threshold_exec / max(env_cfg.q_max, 1e-9)
        patch_mode = "upper"

        def violation_from_exec(exec_val: np.ndarray) -> np.ndarray:
            return (exec_val >= threshold_exec).astype(float)

        byz_action_fraction = 1.0  # always request max

    elif env_name == "public_goods":
        env_cfg = PublicGoodsConfig(
            n_agents=n_agents,
            t_steps=t_steps,
            endowment=float(env_kwargs.get("endowment", 1.0)),
            multiplier=float(env_kwargs.get("multiplier", 1.6)),
            min_contrib=float(env_kwargs.get("min_contrib", 0.2)),
            penalty_factor=float(env_kwargs.get("penalty_factor", cfg.get("penalty_factor", 0.2))),
            social_lambda=float(env_kwargs.get("social_lambda", 0.3)),
            partial_obs=bool(env_kwargs.get("partial_obs", cfg.get("partial_obs", False))),
            obs_noise=float(env_kwargs.get("obs_noise", 0.01)),
        )
        env = PublicGoodsEnv(env_cfg, seed=seed)

        action_scale = env_cfg.endowment  # fraction * endowment -> contribution
        threshold_exec = env_cfg.min_contrib

        # Patch in the opposite direction: enforce minimum contribution
        cap_fraction = threshold_exec / max(env_cfg.endowment, 1e-9)
        patch_mode = "lower"

        def violation_from_exec(exec_val: np.ndarray) -> np.ndarray:
            return (exec_val < threshold_exec).astype(float)

        byz_action_fraction = 0.0  # always free-ride

    else:
        raise ValueError(f"Unknown env: {env_name}")

    obs = env.reset()
    obs_dim = int(obs.shape[-1])

    # --- Learning agent (shared PPO) ---
    ppo_kwargs = dict(cfg.get("ppo_kwargs", {}) or {})
    ppo_cfg = PPOConfig(
        obs_dim=obs_dim,
        hidden_sizes=tuple(ppo_kwargs.get("hidden_sizes", (128, 128))),
        lr=float(ppo_kwargs.get("lr", 3e-4)),
        gamma=float(ppo_kwargs.get("gamma", 0.99)),
        gae_lambda=float(ppo_kwargs.get("gae_lambda", 0.95)),
        clip_range=float(ppo_kwargs.get("clip_range", 0.2)),
        ent_coef=float(ppo_kwargs.get("ent_coef", 0.01)),
        vf_coef=float(ppo_kwargs.get("vf_coef", 0.5)),
        max_grad_norm=float(ppo_kwargs.get("max_grad_norm", 0.5)),
        rollout_len=int(ppo_kwargs.get("rollout_len", 128)),
        n_epochs=int(ppo_kwargs.get("n_epochs", 4)),
        batch_size=int(ppo_kwargs.get("batch_size", 1024)),
    )
    agent = PPOSharedAgent(n_agents=n_agents, cfg=ppo_cfg, device=device_info.torch_device)

    # --- AAF supervisor (if applicable) ---
    sup: Optional[AAFSupervisor] = None
    aaf_kwargs = dict(cfg.get("aaf_kwargs", {}) or {})
    if baseline.startswith("aaf_"):
        # configure ablations via baseline name
        enable_detection = baseline in (
            "aaf_full",
            "aaf_detector_only",
            "aaf_shaping_only",
            "aaf_patch_only",
            "aaf_no_attrib",
        )
        enable_shaping = baseline in ("aaf_full", "aaf_shaping_only")
        enable_patch = baseline in ("aaf_full", "aaf_patch_only")
        use_attrib = baseline not in ("aaf_no_attrib",)

        sup_cfg = SupervisorConfig(
            monitored_stat=str(
                aaf_kwargs.get(
                    "monitored_stat",
                    "greedy" if env_name == "resource_sharing" else "violation",
                )
            ),
            resp_window=int(aaf_kwargs.get("resp_window", 50)),
            record_size_bytes=int(aaf_kwargs.get("record_size_bytes", 40)),
            enable_detection=bool(aaf_kwargs.get("enable_detection", enable_detection)),
            enable_shaping=bool(aaf_kwargs.get("enable_shaping", enable_shaping)),
            enable_patch=bool(aaf_kwargs.get("enable_patch", enable_patch)),
            use_attribution=bool(aaf_kwargs.get("use_attribution", use_attrib)),
            patch_mode=str(aaf_kwargs.get("patch_mode", patch_mode)),
        )
        # CUSUM + intervention overrides
        sup_cfg.cusum.alpha = float(aaf_kwargs.get("cusum_alpha", sup_cfg.cusum.alpha))
        sup_cfg.cusum.delta = float(aaf_kwargs.get("cusum_delta", sup_cfg.cusum.delta))
        sup_cfg.cusum.h0 = float(aaf_kwargs.get("cusum_h0", sup_cfg.cusum.h0))
        sup_cfg.cusum.eta_exp = float(aaf_kwargs.get("cusum_eta_exp", sup_cfg.cusum.eta_exp))
        sup_cfg.cusum.h_min = float(aaf_kwargs.get("cusum_h_min", sup_cfg.cusum.h_min))
        sup_cfg.cusum.warmup = int(aaf_kwargs.get("cusum_warmup", sup_cfg.cusum.warmup))

        sup_cfg.intervention.horizon = int(aaf_kwargs.get("horizon", sup_cfg.intervention.horizon))
        sup_cfg.intervention.lambda_penalty = float(
            aaf_kwargs.get("lambda_penalty", sup_cfg.intervention.lambda_penalty)
        )
        sup_cfg.intervention.enable_patch = bool(
            aaf_kwargs.get("intervention_enable_patch", sup_cfg.intervention.enable_patch)
        )
        sup_cfg.intervention.patch_eps = float(aaf_kwargs.get("patch_eps", sup_cfg.intervention.patch_eps))
        sup_cfg.intervention.top_k = int(aaf_kwargs.get("top_k", sup_cfg.intervention.top_k))

        sup = AAFSupervisor(n_agents=n_agents, cfg=sup_cfg)

    # --- Baseline-specific knobs ---
    # Constrained PPO
    lag_lambda = float(cfg.get("lagrangian_lambda0", 0.0))
    lag_lr = float(cfg.get("lagrangian_lr", 0.05))
    lag_eps = float(cfg.get("lagrangian_eps", 0.05))  # allowed mean violation

    # Fair PPO
    fair_lambda = float(cfg.get("fair_lambda", 0.1))

    # Bandwidth model
    base_record_bytes = float(cfg.get("bandwidth_record_bytes_per_agent_step", 43.2))

    # Byzantine selection
    byz_ids = _select_byzantine(n_agents, byz_frac, rng)
    byz_mask = np.zeros(n_agents, dtype=bool)
    byz_mask[byz_ids] = True

    # --- Logging accumulators ---
    step_logs: List[Dict[str, Any]] = []
    agent_logs: List[Dict[str, Any]] = []

    total_attempted = 0.0
    total_executed = 0.0
    total_reward_sum = 0.0
    gini_alloc_sum = 0.0
    gini_reward_sum = 0.0

    # Per-agent aggregates (only if needed)
    if want_agent_logs:
        per_agent_attempted = np.zeros(n_agents, dtype=float)
        per_agent_executed = np.zeros(n_agents, dtype=float)
        per_agent_reward_sum = np.zeros(n_agents, dtype=float)
        per_agent_alloc_sum = np.zeros(n_agents, dtype=float)
    else:
        per_agent_attempted = per_agent_executed = per_agent_reward_sum = per_agent_alloc_sum = None  # type: ignore[assignment]

    alarms_count = 0

    last_alloc = np.zeros(n_agents, dtype=float)
    last_reward = np.zeros(n_agents, dtype=float)
    last_gini_alloc = float("nan")
    last_gini_reward = float("nan")

    # PPO debug (only used when saving step logs)
    ppo_debug: Dict[str, float] = {}

    for t in range(t_steps):
        # Policy actions for all agents
        act_frac, logp, value = agent.act(obs, deterministic=False)  # in [0,1]
        act_frac = act_frac.reshape(n_agents)
        logp = logp.reshape(n_agents)
        value = value.reshape(n_agents)

        # Override byzantine actions after change-point
        if byz_mask.any() and t >= byz_start:
            act_frac = act_frac.copy()
            act_frac[byz_mask] = float(byz_action_fraction)

        # Attempted executed values before any guards/patches
        exec_attempted = act_frac * action_scale

        # Baseline: static guard clamps actions (env-dependent)
        if baseline == "static_guard":
            eps = 1e-4
            if env_name == "resource_sharing":
                # clamp DOWN to avoid greedy
                act_frac = np.minimum(act_frac, max(0.0, cap_fraction - eps))
            else:
                # clamp UP to avoid free-riding
                act_frac = np.maximum(act_frac, min(1.0, cap_fraction + eps))

        # AAF patching
        if sup is not None:
            act_frac = sup.apply_patch(act_frac, bound=cap_fraction)

        # Executed values
        exec_val = act_frac * action_scale

        # Attempted/executed violation flags used for compromise rates
        if env_name == "resource_sharing":
            attempted_flag = (exec_attempted >= threshold_exec).astype(float)
            executed_flag = (exec_val >= threshold_exec).astype(float)
        else:
            attempted_flag = (exec_attempted < threshold_exec).astype(float)
            executed_flag = (exec_val < threshold_exec).astype(float)

        total_attempted += float(np.sum(attempted_flag))
        total_executed += float(np.sum(executed_flag))
        if want_agent_logs:
            per_agent_attempted += attempted_flag  # type: ignore[operator]
            per_agent_executed += executed_flag  # type: ignore[operator]

        # Env step
        next_obs, reward, info = env.step(exec_val)
        reward = np.asarray(reward, dtype=float).reshape(n_agents)

        # Allocation-like quantity for inequality stats
        if env_name == "resource_sharing":
            alloc = np.asarray(info["alloc"], dtype=float).reshape(n_agents)
        else:
            alloc = np.asarray(info["returns"], dtype=float).reshape(n_agents)

        # Violation vector used by AAF and constrained PPO
        viol_vec = violation_from_exec(exec_val)

        # Inequality + welfare stats (streaming)
        g_alloc = float(gini(alloc))
        g_rew = float(gini(reward))
        gini_alloc_sum += g_alloc
        gini_reward_sum += g_rew
        total_reward_sum += float(np.sum(reward))
        last_alloc = alloc
        last_reward = reward
        last_gini_alloc = g_alloc
        last_gini_reward = g_rew

        if want_agent_logs:
            per_agent_reward_sum += reward  # type: ignore[operator]
            per_agent_alloc_sum += alloc  # type: ignore[operator]

        # AAF monitoring + shaping
        alarm = 0
        shaping_pen = np.zeros(n_agents, dtype=float)
        if sup is not None:
            # monitored statistic z_t
            if sup.cfg.monitored_stat == "gini_alloc":
                z_t = g_alloc
            else:
                z_t = float(np.mean(viol_vec))
            alarm = int(sup.step_monitor(z_t=z_t, viol_vec=viol_vec))
            shaping_pen = sup.shaping_penalty()
        alarms_count += int(alarm)

        # Constrained / fairness / shaping reward modifications
        if baseline == "constrained_ppo":
            reward_in = reward - lag_lambda * viol_vec
        elif baseline == "fair_ppo":
            reward_in = reward - fair_lambda * g_alloc
        elif sup is not None:
            reward_in = reward - shaping_pen
        else:
            reward_in = reward

        # PPO observe
        done = (t == t_steps - 1)
        agent.observe(obs=obs, actions=act_frac, log_probs=logp, values=value, rewards=reward_in, done=done)

        # PPO update
        ppo_stats = agent.maybe_update(next_obs)
        if ppo_stats:
            ppo_debug.update(ppo_stats)
            # Lagrangian update for constrained PPO
            if baseline == "constrained_ppo":
                mean_viol = float(np.mean(viol_vec))
                lag_lambda = max(0.0, lag_lambda + lag_lr * (mean_viol - lag_eps))

        # Optional per-step logs
        if want_step_logs:
            step_logs.append(
                {
                    "t": t,
                    "env": env_name,
                    "baseline": baseline,
                    "mean_reward": float(np.mean(reward)),
                    "mean_alloc": float(np.mean(alloc)),
                    "gini_alloc": g_alloc,
                    "gini_reward": g_rew,
                    "viol_rate_exec": float(np.mean(viol_vec)),
                    "alarm": int(alarm),
                    "lag_lambda": float(lag_lambda),
                    **{k: float(v) for k, v in ppo_debug.items()},
                }
            )

        obs = next_obs

    # --- Summary metrics ---
    denom = float(max(1, t_steps * n_agents))
    compromise_attempted = float(total_attempted / denom)
    compromise_executed = float(total_executed / denom)

    social_welfare = float(total_reward_sum / denom)
    gini_mean_alloc = float(gini_alloc_sum / max(1, t_steps))
    gini_final_alloc = float(last_gini_alloc)
    gini_mean_reward = float(gini_reward_sum / max(1, t_steps))
    gini_final_reward = float(last_gini_reward)

    first_alarm_t = sup.first_alarm_t if sup is not None else None
    det_delay = detection_delay(first_alarm_t, byz_start) if byz_frac > 0 else None

    # Attribution metrics (only meaningful when byzantine exists and AAF used)
    attrib_top1 = None
    attrib_recall3 = None
    attrib_recall5 = None
    if sup is not None and byz_frac > 0 and sup.first_alarm_topk is not None:
        topk = sup.first_alarm_topk
        byz_set = set(byz_ids)
        attrib_top1 = 1.0 if len(topk) > 0 and topk[0] in byz_set else 0.0
        attrib_recall3 = float(len(set(topk[:3]) & byz_set) / max(1, min(3, len(byz_set))))
        attrib_recall5 = float(len(set(topk[:5]) & byz_set) / max(1, min(5, len(byz_set))))

    # Bandwidth overhead
    base_bytes = base_record_bytes * n_agents * t_steps
    extra_bytes = float(sup.overhead_bytes) if sup is not None else 0.0
    total_bytes = int(round(base_bytes + extra_bytes))

    runtime_s = float(time.time() - t0)

    summary: Dict[str, Any] = {
        "env": env_name,
        "baseline": baseline,
        "n_agents": n_agents,
        "t_steps": t_steps,
        "penalty_factor": float(cfg.get("penalty_factor", env_kwargs.get("penalty_factor", 0.0))),
        "dist_alpha": float(cfg.get("dist_alpha", env_kwargs.get("dist_alpha", float("nan")))),
        "partial_obs": bool(cfg.get("partial_obs", env_kwargs.get("partial_obs", False))),
        "seed": seed,
        "device_requested": device_req,
        "device_resolved": device_info.resolved,
        "byzantine_frac": byz_frac,
        "byzantine_start": byz_start,
        "byzantine_ids": byz_ids,
        "log_mode": log_mode,
        "compromise_ratio_attempted": compromise_attempted,
        "compromise_ratio_executed": compromise_executed,
        "social_welfare": social_welfare,
        "gini_alloc_final": gini_final_alloc,
        "gini_alloc_mean": gini_mean_alloc,
        "gini_reward_final": gini_final_reward,
        "gini_reward_mean": gini_mean_reward,
        "alarms_count": int(alarms_count),
        "first_alarm_t": first_alarm_t,
        "detection_delay": det_delay,
        "attrib_top1_correct": attrib_top1,
        "attrib_recall3": attrib_recall3,
        "attrib_recall5": attrib_recall5,
        "bandwidth_base_bytes": int(round(base_bytes)),
        "bandwidth_aaf_extra_bytes": int(round(extra_bytes)),
        "bandwidth_overhead_bytes": total_bytes,
        "runtime_s": runtime_s,
        # Constrained PPO final lambda for traceability
        "lagrangian_lambda_final": lag_lambda if baseline == "constrained_ppo" else None,
    }

    # Agent-level logs
    if want_agent_logs:
        per_agent_comp_exec = per_agent_executed / max(1, t_steps)  # type: ignore[operator]
        per_agent_comp_att = per_agent_attempted / max(1, t_steps)  # type: ignore[operator]
        per_agent_reward = per_agent_reward_sum / max(1, t_steps)  # type: ignore[operator]
        per_agent_alloc = per_agent_alloc_sum / max(1, t_steps)  # type: ignore[operator]

        for i in range(n_agents):
            agent_logs.append(
                {
                    "agent_id": i,
                    "is_byzantine": bool(byz_mask[i]),
                    "compromise_exec": float(per_agent_comp_exec[i]),
                    "compromise_attempted": float(per_agent_comp_att[i]),
                    "mean_reward": float(per_agent_reward[i]),
                    "mean_alloc": float(per_agent_alloc[i]),
                }
            )

    return {"summary": summary, "step_logs": step_logs, "agent_logs": agent_logs}
