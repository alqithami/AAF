from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def gini(x: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Gini coefficient for a 1D array (non-negative preferred)."""
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0.0):
        return 0.0
    # Shift if negative values exist (rare in allocations, possible in rewards)
    if np.min(x) < 0:
        x = x - np.min(x)
    x = x + eps
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    # Gini = (n+1 - 2 * sum_i (cumx_i / cumx_n)) / n
    g = (n + 1 - 2.0 * np.sum(cumx / cumx[-1])) / n
    return float(np.clip(g, 0.0, 1.0))


@dataclass
class RunStepMetrics:
    t: int
    greedy_rate_attempted: float
    greedy_rate_executed: float
    gini_alloc: float
    gini_reward: float
    mean_alloc: float
    mean_reward: float
    alarm: int


def compute_compromise_ratio(greedy_flags: np.ndarray) -> float:
    """Compute compromise ratio as mean of boolean matrix [T, N]."""
    greedy_flags = np.asarray(greedy_flags, dtype=float)
    if greedy_flags.size == 0:
        return 0.0
    return float(np.mean(greedy_flags))


def detection_delay(first_alarm_t: Optional[int], change_t: int) -> Optional[int]:
    if first_alarm_t is None:
        return None
    if first_alarm_t < change_t:
        return None
    return int(first_alarm_t - change_t)
