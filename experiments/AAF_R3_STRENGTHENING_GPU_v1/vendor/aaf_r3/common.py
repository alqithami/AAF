from __future__ import annotations
import hashlib, json, os, platform, random, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch

BASE_COMMIT = "a7fb5c0a2a2a55ef823658d7786389fd5337b264"
PROTOCOL = "AAF-R3-v1"

def digest(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, allow_nan=False).encode()).hexdigest()

def seed_for(*parts: Any) -> int:
    return int(digest(parts)[:8], 16) % (2**31-1)

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def choose_device(request: str) -> torch.device:
    if request == "auto": request = "cuda" if torch.cuda.is_available() else "cpu"
    if request not in ("cpu", "cuda"): raise ValueError("Use cpu, cuda, or auto. MPS is deliberately not selected.")
    if request == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    return torch.device(request)

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def source_hash() -> str:
    root = Path(__file__).parent
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    return digest(files)

def provenance(device: str) -> dict:
    import importlib.metadata as im
    packages = {}
    for name in ("torch", "numpy", "scipy", "pandas", "matplotlib", "vmas", "pytest"):
        try: packages[name] = im.version(name)
        except im.PackageNotFoundError: packages[name] = None
    return {"protocol": PROTOCOL, "historical_base_commit": BASE_COMMIT,
            "source_sha256": source_hash(), "python": sys.version, "platform": platform.platform(),
            "device": device, "packages": packages,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if np.min(x) < 0: x = x - np.min(x)
    if x.sum() <= 1e-12: return 0.0
    x = np.sort(x); n = x.size
    return float(2*np.dot(np.arange(1,n+1), x)/(n*x.sum()) - (n+1)/n)

def finite_mean(x: list) -> float | None:
    a = np.asarray([v for v in x if v is not None], dtype=float)
    return float(a.mean()) if a.size else None
