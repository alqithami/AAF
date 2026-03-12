from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class SeedBundle:
    seed: int
    numpy_seed: int
    python_seed: int
    torch_seed: Optional[int]


def seed_everything(seed: int) -> SeedBundle:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_seed: Optional[int] = None
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            torch_seed = seed
        except Exception:
            torch_seed = None

    return SeedBundle(seed=seed, numpy_seed=seed, python_seed=seed, torch_seed=torch_seed)
