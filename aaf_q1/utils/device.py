from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


DeviceStr = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class DeviceInfo:
    requested: DeviceStr
    resolved: str
    torch_device: torch.device


def resolve_device(requested: DeviceStr = "auto") -> DeviceInfo:
    """Resolve device selection across CUDA/MPS/CPU.

    Parameters
    ----------
    requested:
        - "auto": prefer CUDA, then MPS, else CPU
        - "cuda": require CUDA
        - "mps": require Apple Metal Performance Shaders backend
        - "cpu": CPU

    Returns
    -------
    DeviceInfo
    """
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"Invalid device: {requested}")

    if requested == "cpu":
        d = torch.device("cpu")
        return DeviceInfo(requested=requested, resolved="cpu", torch_device=d)

    if requested in ("cuda", "auto"):
        if torch.cuda.is_available():
            d = torch.device("cuda")
            return DeviceInfo(requested=requested, resolved="cuda", torch_device=d)
        if requested == "cuda":
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    if requested in ("mps", "auto"):
        # Note: torch.backends.mps.is_available exists on macOS builds
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            d = torch.device("mps")
            return DeviceInfo(requested=requested, resolved="mps", torch_device=d)
        if requested == "mps":
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")

    d = torch.device("cpu")
    return DeviceInfo(requested=requested, resolved="cpu", torch_device=d)
