from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .detectors import AdaptiveCUSUM, CUSUMConfig
from .interventions import InterventionConfig, InterventionState


@dataclass
class SupervisorConfig:
    # What statistic is monitored. For the included environments:
    # - resource_sharing: "greedy" (high requests)
    # - public_goods: "violation" (low contribution)
    monitored_stat: str = "greedy"  # {"greedy", "violation", "gini_alloc"}

    # Detector config
    cusum: CUSUMConfig = field(default_factory=CUSUMConfig)

    # Intervention config
    intervention: InterventionConfig = field(default_factory=InterventionConfig)

    # Responsibility window length (steps)
    resp_window: int = 50

    # Ledger overhead model (bytes per agent-step record)
    record_size_bytes: int = 40

    # Variant switches / ablations
    enable_detection: bool = True
    enable_shaping: bool = True
    enable_patch: bool = True
    use_attribution: bool = True  # if False, interventions are uniform

    # Patching direction:
    # - "upper": clamp actions DOWN to an upper bound (greedy norm)
    # - "lower": clamp actions UP to a lower bound (free-riding norm)
    patch_mode: str = "upper"


class AAFSupervisor:
    """Pragmatic AAF supervisor for simulation evaluation.

    Responsibilities
    --------------
    1) Online detection of harmful norm emergence via adaptive CUSUM.
    2) Sliding-window responsibility scoring (by default, per-agent violation frequency).
    3) Intervention scheduling:
       - reward shaping penalties (soft)
       - policy patching (hard bounds on actions)

    This implementation is intentionally simple and *experiment-first*:
    it is meant to be robust and easy to reproduce for Q1 submissions,
    and it supports ablations (detector-only, shaping-only, patch-only, no-attribution).
    """

    def __init__(self, n_agents: int, cfg: SupervisorConfig):
        self.n_agents = int(n_agents)
        self.cfg = cfg
        self.detector = AdaptiveCUSUM(cfg.cusum)
        self.intervention_state = InterventionState(cfg.intervention)
        self.intervention_state.reset(n_agents)

        self.t = 0
        self.overhead_bytes = 0

        # Sliding window of per-agent violation indicators (T_window x N)
        self._resp_window = int(cfg.resp_window)
        self._viol_hist = np.zeros((self._resp_window, n_agents), dtype=float)
        self._hist_ptr = 0
        self._filled = 0

        self.alarms: List[int] = []

        # Attribution evaluation hooks
        self.first_alarm_t: Optional[int] = None
        self.first_alarm_topk: Optional[List[int]] = None

    def reset(self) -> None:
        self.detector.reset()
        self.intervention_state.reset(self.n_agents)
        self.t = 0
        self.overhead_bytes = 0
        self._viol_hist[:] = 0.0
        self._hist_ptr = 0
        self._filled = 0
        self.alarms = []
        self.first_alarm_t = None
        self.first_alarm_topk = None

    def _update_window(self, viol_vec: np.ndarray) -> None:
        self._viol_hist[self._hist_ptr] = viol_vec
        self._hist_ptr = (self._hist_ptr + 1) % self._resp_window
        self._filled = min(self._resp_window, self._filled + 1)

    def responsibility(self) -> np.ndarray:
        if self._filled == 0:
            return np.zeros(self.n_agents, dtype=float)
        hist = self._viol_hist[: self._filled]
        resp = np.mean(hist, axis=0)
        s = float(np.sum(resp))
        if s > 1e-12:
            resp = resp / s
        return resp

    def step_monitor(self, z_t: float, viol_vec: np.ndarray) -> int:
        """Update detector + intervention state for one time step."""
        self.t += 1

        # ledger overhead: per agent-step record
        self.overhead_bytes += int(self.cfg.record_size_bytes) * self.n_agents

        self._update_window(viol_vec)

        alarm = 0
        if self.cfg.enable_detection:
            alarm = int(self.detector.update(float(z_t)))
        self.alarms.append(alarm)

        if alarm == 1 and self.first_alarm_t is None:
            self.first_alarm_t = self.t

        if alarm == 1:
            targets = self.select_targets()
            if self.first_alarm_topk is None:
                self.first_alarm_topk = targets.copy()
            if self.cfg.enable_shaping or (self.cfg.enable_patch and self.cfg.intervention.enable_patch):
                self.intervention_state.start(targets)

        self.intervention_state.tick()
        return alarm

    def select_targets(self) -> List[int]:
        k = int(self.cfg.intervention.top_k)
        if k <= 0:
            return []
        if not self.cfg.use_attribution:
            # Uniform intervention: target everyone
            return list(range(self.n_agents))

        resp = self.responsibility()
        idx = np.argsort(-resp)[:k]
        return [int(i) for i in idx]

    def apply_patch(self, actions: np.ndarray, bound: float) -> np.ndarray:
        """Patch actions for active agents.

        Parameters
        ----------
        actions:
            action fractions in [0, 1]
        bound:
            - if patch_mode == "upper": upper cap
            - if patch_mode == "lower": lower floor
        """
        if not (self.cfg.enable_patch and self.cfg.intervention.enable_patch):
            return actions
        eps = float(self.cfg.intervention.patch_eps)
        patched = np.asarray(actions, dtype=float).copy()
        mode = str(self.cfg.patch_mode).lower().strip()
        if mode not in ("upper", "lower"):
            mode = "upper"

        for i in range(self.n_agents):
            if not self.intervention_state.is_active(i):
                continue
            if mode == "upper":
                patched[i] = min(patched[i], max(0.0, float(bound) - eps))
            else:
                patched[i] = max(patched[i], min(1.0, float(bound) + eps))

        return patched

    def shaping_penalty(self) -> np.ndarray:
        """Per-agent shaping penalty (non-negative) to subtract from reward."""
        if not self.cfg.enable_shaping:
            return np.zeros(self.n_agents, dtype=float)
        resp = self.responsibility()
        lam = float(self.cfg.intervention.lambda_penalty)
        pen = lam * resp
        for i in range(self.n_agents):
            if not self.intervention_state.is_active(i):
                pen[i] = 0.0
        return pen
