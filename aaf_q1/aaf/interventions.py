from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class InterventionConfig:
    # If an alarm fires, intervene for this many steps
    horizon: int = 50
    # Shaping penalty amplitude (added to reward as negative penalty)
    lambda_penalty: float = 0.2
    # Patch clamp (action restriction); if True, clamp actions for responsible agents
    enable_patch: bool = True
    # When patching, cap action fraction at (threshold - eps)
    patch_eps: float = 1e-4
    # How many agents to target per alarm (top-k responsibility)
    top_k: int = 3


class InterventionState:
    def __init__(self, cfg: InterventionConfig):
        self.cfg = cfg
        self.active_steps = np.zeros(0, dtype=int)  # initialized when n_agents known

    def reset(self, n_agents: int) -> None:
        self.active_steps = np.zeros(n_agents, dtype=int)

    def start(self, target_agents: List[int]) -> None:
        for i in target_agents:
            self.active_steps[i] = max(self.active_steps[i], int(self.cfg.horizon))

    def tick(self) -> None:
        self.active_steps = np.maximum(0, self.active_steps - 1)

    def is_active(self, i: int) -> bool:
        return bool(self.active_steps[i] > 0)
