from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class PublicGoodsConfig:
    n_agents: int
    t_steps: int
    endowment: float = 1.0
    multiplier: float = 1.6  # public goods multiplier
    min_contrib: float = 0.2  # norm threshold
    penalty_factor: float = 0.2
    social_lambda: float = 0.3
    partial_obs: bool = False
    obs_noise: float = 0.01


class PublicGoodsEnv:
    """Simple repeated public-goods game for robustness evaluation.

    Each step:
      - Agent i chooses contribution c_i in [0, endowment].
      - Total pot = sum(c).
      - Each agent receives return = multiplier * pot / n_agents.
      - Private payoff: (endowment - c_i) + return - penalty * 1{c_i < min_contrib}.
      - Total reward adds a social term: social_lambda * mean(return).

    Violation is \"free-riding\" below min_contrib.
    """

    def __init__(self, cfg: PublicGoodsConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.last_c = np.zeros(cfg.n_agents, dtype=float)
        self.last_reward = np.zeros(cfg.n_agents, dtype=float)
        self.last_return = np.zeros(cfg.n_agents, dtype=float)
        self.obs_dim = 4 if cfg.partial_obs else 3

    def reset(self) -> np.ndarray:
        self.t = 0
        self.last_c[:] = 0.0
        self.last_reward[:] = 0.0
        self.last_return[:] = 0.0
        return self._obs()

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        n = cfg.n_agents
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        c_norm = self.last_c / max(cfg.endowment, 1e-9)
        ret_norm = self.last_return / max(cfg.endowment, 1e-9)
        rew_norm = self.last_reward / max(cfg.endowment, 1e-9)

        mean_c = float(np.mean(c_norm))
        for i in range(n):
            obs[i, 0] = float(c_norm[i])
            obs[i, 1] = float(ret_norm[i])
            obs[i, 2] = float(mean_c)
            if cfg.partial_obs:
                obs[i, 3] = float(mean_c + self.rng.normal(0.0, cfg.obs_noise))

        if cfg.obs_noise > 0:
            obs = obs + self.rng.normal(0.0, cfg.obs_noise, size=obs.shape).astype(np.float32)

        return obs

    def step(self, c_exec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        cfg = self.cfg
        self.t += 1

        c_exec = np.asarray(c_exec, dtype=float).reshape(cfg.n_agents)
        c_exec = np.clip(c_exec, 0.0, cfg.endowment)

        pot = float(np.sum(c_exec))
        ret = cfg.multiplier * pot / cfg.n_agents
        returns = np.full(cfg.n_agents, ret, dtype=float)

        violation = (c_exec < cfg.min_contrib).astype(float)
        r_private = (cfg.endowment - c_exec) + returns - cfg.penalty_factor * violation
        r_social = float(np.mean(returns))
        reward = r_private + cfg.social_lambda * r_social

        self.last_c = c_exec
        self.last_return = returns
        self.last_reward = reward

        obs = self._obs()
        info = {
            "t": self.t,
            "c_exec": c_exec,
            "returns": returns,
            "reward": reward,
            "r_social": r_social,
            "violation_exec": violation,
        }
        return obs, reward, info
