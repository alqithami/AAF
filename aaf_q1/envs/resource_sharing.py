from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def watts_strogatz_graph(n: int, k: int, p: float, rng: np.random.Generator) -> List[List[int]]:
    """Generate an undirected Watts–Strogatz small-world graph adjacency list.

    Parameters
    ----------
    n : number of nodes
    k : each node is connected to k nearest neighbors in ring topology (k must be even)
    p : rewire probability
    rng : numpy random generator

    Returns
    -------
    adj : adjacency list (neighbors for each node, no self loops)
    """
    if k % 2 != 0:
        raise ValueError("k must be even for Watts–Strogatz.")
    if k >= n:
        raise ValueError("k must be < n.")
    # Start with ring lattice
    neighbors = [set() for _ in range(n)]
    half = k // 2
    for i in range(n):
        for j in range(1, half + 1):
            a = i
            b = (i + j) % n
            neighbors[a].add(b)
            neighbors[b].add(a)
    # Rewire edges (i -> i+j) with probability p (only forward edges to avoid duplicates)
    for i in range(n):
        for j in range(1, half + 1):
            if rng.random() < p:
                a = i
                b = (i + j) % n
                # Remove original edge
                neighbors[a].discard(b)
                neighbors[b].discard(a)
                # Pick new b' not equal to a and not already connected
                candidates = list(set(range(n)) - {a} - neighbors[a])
                if len(candidates) == 0:
                    # restore edge
                    neighbors[a].add(b)
                    neighbors[b].add(a)
                    continue
                b_new = int(rng.choice(candidates))
                neighbors[a].add(b_new)
                neighbors[b_new].add(a)
    return [sorted(list(s)) for s in neighbors]


@dataclass
class ResourceSharingConfig:
    n_agents: int
    t_steps: int
    r_max: float = 100.0
    r_in: float = 0.0  # optional replenishment, not needed for the simple pool
    q_max: float = 100.0
    dist_alpha: float = 1.0
    greedy_threshold: float = 0.6  # fraction of r_max
    penalty_factor: float = 0.2
    social_lambda: float = 0.3
    partial_obs: bool = False
    obs_noise: float = 0.01
    graph_k: int = 4
    graph_p: float = 0.1


class ResourceSharingEnv:
    """Resource-sharing game used in the paper-style evaluation.

    At each step, agents submit requests q_i in [0, q_max], then the environment allocates up to r_max.

    Allocation rule (paper §7.1):
      - If sum(q) <= r_max: a_i = q_i
      - Else: a_i = (q_i^alpha / sum_j q_j^alpha) * r_max

    Notes
    -----
    - For alpha=0, we interpret q^0 as 1 for q>0 and 0 for q=0, yielding equal share among positive requesters.
    """

    def __init__(self, cfg: ResourceSharingConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self.adj = watts_strogatz_graph(cfg.n_agents, min(cfg.graph_k, cfg.n_agents - 1 - ((cfg.n_agents - 1) % 2)), cfg.graph_p, self.rng)
        self.t = 0
        self.last_q = np.zeros(cfg.n_agents, dtype=float)
        self.last_alloc = np.zeros(cfg.n_agents, dtype=float)
        self.last_reward = np.zeros(cfg.n_agents, dtype=float)
        self.queue = 0.0

        # Observation dimension
        self.obs_dim = 5 if cfg.partial_obs else 4

    def reset(self) -> np.ndarray:
        self.t = 0
        self.last_q[:] = 0.0
        self.last_alloc[:] = 0.0
        self.last_reward[:] = 0.0
        self.queue = 0.0
        return self._obs()

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        n = cfg.n_agents
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)

        # Features normalized to r_max
        q_norm = self.last_q / cfg.r_max
        alloc_norm = self.last_alloc / cfg.r_max
        reward_norm = self.last_reward / max(cfg.r_max, 1e-9)  # scale
        pool_norm = np.full(n, cfg.r_max / cfg.r_max, dtype=float)

        for i in range(n):
            neigh = self.adj[i]
            neigh_mean_q = float(np.mean(q_norm[neigh])) if len(neigh) > 0 else 0.0
            obs[i, 0] = float(alloc_norm[i])
            obs[i, 1] = float(q_norm[i])
            obs[i, 2] = float(neigh_mean_q)
            obs[i, 3] = float(pool_norm[i])
            if cfg.partial_obs:
                noisy_q = self.queue / cfg.r_max + self.rng.normal(0.0, cfg.obs_noise)
                obs[i, 4] = float(noisy_q)

        if cfg.obs_noise > 0:
            obs = obs + self.rng.normal(0.0, cfg.obs_noise, size=obs.shape).astype(np.float32)

        return obs

    def step(self, q_exec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        cfg = self.cfg
        self.t += 1

        q_exec = np.asarray(q_exec, dtype=float).reshape(cfg.n_agents)
        q_exec = np.clip(q_exec, 0.0, cfg.q_max)

        total_q = float(np.sum(q_exec))
        if total_q <= cfg.r_max:
            alloc = q_exec.copy()
        else:
            alpha = float(cfg.dist_alpha)
            if abs(alpha) < 1e-12:
                weights = (q_exec > 0).astype(float)
            else:
                weights = np.power(q_exec, alpha)
            denom = float(np.sum(weights))
            if denom <= 0:
                alloc = np.zeros_like(q_exec)
            else:
                alloc = (weights / denom) * cfg.r_max

        # Queue is "excess demand"
        self.queue = max(0.0, total_q - cfg.r_max)

        greedy_flag = (q_exec >= cfg.greedy_threshold * cfg.r_max).astype(float)
        r_private = alloc - cfg.penalty_factor * greedy_flag
        r_social = float(np.mean(alloc))
        reward = r_private + cfg.social_lambda * r_social

        self.last_q = q_exec
        self.last_alloc = alloc
        self.last_reward = reward

        obs = self._obs()
        info = {
            "t": self.t,
            "q_exec": q_exec,
            "alloc": alloc,
            "reward": reward,
            "r_social": r_social,
            "greedy_exec": greedy_flag,
            "queue": self.queue,
        }
        return obs, reward, info
