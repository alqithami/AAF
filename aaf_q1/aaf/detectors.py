from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CUSUMConfig:
    alpha: float = 0.05  # target false-alarm rate
    delta: float = 0.01  # slack
    h0: float = 5.0      # initial threshold
    eta_exp: float = 0.6 # Robbins–Monro gain exponent (eta_t = t^-eta_exp)
    h_min: float = 0.5   # floor to avoid negative thresholds
    warmup: int = 100    # steps for baseline estimation


class AdaptiveCUSUM:
    """Adaptive one-sided CUSUM with Robbins–Monro threshold update (paper Alg. 2 style)."""

    def __init__(self, cfg: CUSUMConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.S = 0.0
        self.h = float(self.cfg.h0)
        self.mu0: Optional[float] = None
        self._warm_values = []

    def update(self, z_t: float) -> int:
        """Update with new statistic value. Returns 1 if alarm fires."""
        self.t += 1

        # Warmup baseline mean
        if self.mu0 is None:
            self._warm_values.append(float(z_t))
            if len(self._warm_values) >= self.cfg.warmup:
                self.mu0 = float(np.mean(self._warm_values))
            return 0

        mu0 = float(self.mu0)
        self.S = max(0.0, self.S + float(z_t) - mu0 - float(self.cfg.delta))

        alarm = 0
        if self.S >= self.h:
            alarm = 1
            self.S = 0.0

        eta_t = self.t ** (-float(self.cfg.eta_exp))
        self.h = max(self.cfg.h_min, self.h + eta_t * (alarm - float(self.cfg.alpha)))
        return alarm
