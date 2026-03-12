from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta


def _sample_beta_via_gamma(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Sample Beta(alpha, beta) using Gamma draws.

    Motivation: On macOS/MPS builds, torch.distributions.Beta.sample() may
    internally call a Dirichlet sampler that is not supported on MPS and falls
    back to CPU (slow, noisy warnings). The Gamma-based construction avoids
    that Dirichlet path.
    """
    # Gamma(k, 1) samples (shape=k, scale=1)
    if hasattr(torch, "_standard_gamma"):
        x = torch._standard_gamma(alpha)
        y = torch._standard_gamma(beta)
    else:
        # Fallback for older torch builds
        x = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
        y = torch.distributions.Gamma(beta, torch.ones_like(beta)).sample()
    return x / (x + y)


@dataclass
class PPOConfig:
    obs_dim: int
    hidden_sizes: Tuple[int, ...] = (128, 128)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_len: int = 128
    n_epochs: int = 4
    batch_size: int = 1024
    # for stability on small runs
    min_std: float = 1e-3


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.body = nn.Sequential(*layers)

        self.alpha_head = nn.Linear(last, 1)
        self.beta_head = nn.Linear(last, 1)
        self.value_head = nn.Linear(last, 1)
        self.softplus = nn.Softplus()

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.body(obs)
        # Beta params must be > 0; softplus ensures positivity
        alpha = self.softplus(self.alpha_head(x)) + 1.0
        beta = self.softplus(self.beta_head(x)) + 1.0
        value = self.value_head(x).squeeze(-1)
        return alpha.squeeze(-1), beta.squeeze(-1), value


class RolloutBuffer:
    def __init__(self, rollout_len: int, n_agents: int, obs_dim: int, device: torch.device):
        self.rollout_len = int(rollout_len)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.device = device
        self.reset()

    def reset(self) -> None:
        T = self.rollout_len
        N = self.n_agents
        self.obs = torch.zeros((T, N, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        t = self.ptr
        if t >= self.rollout_len:
            raise RuntimeError("Rollout buffer overflow")
        self.obs[t] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[t] = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        self.log_probs[t] = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        self.values[t] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.rewards[t] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.dones[t] = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.rollout_len


class PPOSharedAgent:
    """Parameter-sharing PPO for homogeneous agents.

    We treat each agent-time transition as an individual sample while sharing one actor-critic network.
    """

    def __init__(self, n_agents: int, cfg: PPOConfig, device: torch.device):
        self.n_agents = int(n_agents)
        self.cfg = cfg
        self.device = device

        self.net = ActorCritic(cfg.obs_dim, cfg.hidden_sizes).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

        self.buffer = RolloutBuffer(cfg.rollout_len, n_agents, cfg.obs_dim, device)
        self._last_obs: Optional[np.ndarray] = None

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return actions in [0,1] for each agent."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        alpha, beta, value = self.net(obs_t)
        dist = Beta(alpha, beta)
        if deterministic:
            action = alpha / (alpha + beta)
        else:
            # On Apple MPS, Beta.sample() can fall back to CPU via Dirichlet.
            # Use Gamma construction to keep sampling on-device.
            if self.device.type == "mps":
                action = _sample_beta_via_gamma(alpha, beta)
            else:
                action = dist.sample()
        # avoid exact 0/1 for log_prob stability
        eps = 1e-6
        action_clamped = torch.clamp(action, eps, 1 - eps)
        log_prob = dist.log_prob(action_clamped)
        return (
            action_clamped.detach().cpu().numpy().astype(np.float32),
            log_prob.detach().cpu().numpy().astype(np.float32),
            value.detach().cpu().numpy().astype(np.float32),
        )

    def observe(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        done: bool,
    ) -> None:
        dones = np.full(self.n_agents, float(done), dtype=np.float32)
        self.buffer.add(obs, actions, log_probs, values, rewards, dones)
        self._last_obs = obs

    def maybe_update(self, next_obs: np.ndarray) -> Dict[str, float]:
        if not self.buffer.is_full():
            return {}
        if self._last_obs is None:
            raise RuntimeError("No last obs stored for PPO update bootstrap.")

        # Bootstrap values for next_obs
        with torch.no_grad():
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            _, _, next_values = self.net(obs_t)  # shape [N]

        T = self.buffer.rollout_len
        N = self.n_agents
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        advantages = torch.zeros((T, N), dtype=torch.float32, device=self.device)
        last_gae = torch.zeros((N,), dtype=torch.float32, device=self.device)

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - self.buffer.dones[t]
            next_val = next_values if t == T - 1 else self.buffer.values[t + 1]
            delta = self.buffer.rewards[t] + gamma * next_val * next_non_terminal - self.buffer.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.buffer.values

        # Flatten
        b_obs = self.buffer.obs.reshape(T * N, self.cfg.obs_dim)
        b_actions = self.buffer.actions.reshape(T * N)
        b_old_logp = self.buffer.log_probs.reshape(T * N)
        b_adv = advantages.reshape(T * N)
        b_ret = returns.reshape(T * N)
        b_val = self.buffer.values.reshape(T * N)

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # PPO optimization
        batch_size = min(self.cfg.batch_size, T * N)
        idxs = torch.randperm(T * N, device=self.device)
        clip = self.cfg.clip_range

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []

        for _ in range(self.cfg.n_epochs):
            for start in range(0, T * N, batch_size):
                mb_idx = idxs[start : start + batch_size]

                alpha, beta, value = self.net(b_obs[mb_idx])
                dist = Beta(alpha, beta)

                eps = 1e-6
                act = torch.clamp(b_actions[mb_idx], eps, 1 - eps)
                logp = dist.log_prob(act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_old_logp[mb_idx])
                unclipped = ratio * b_adv[mb_idx]
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_pred = value
                value_loss = ((b_ret[mb_idx] - value_pred) ** 2).mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (b_old_logp[mb_idx] - logp).mean()

                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                approx_kls.append(float(approx_kl.detach().cpu()))

        stats = {
            "ppo/policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "ppo/value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "ppo/entropy": float(np.mean(entropies)) if entropies else 0.0,
            "ppo/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

        self.buffer.reset()
        return stats
