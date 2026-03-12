from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    """Write JSON atomically (best-effort).

    Long experiment grids are frequently interrupted (OOM kills, preemption,
    laptop sleep, etc.). Atomic writes reduce the risk of partially-written
    JSON files breaking resume logic.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = json.dumps(obj, indent=2, sort_keys=True)

    # Write to a temp file in the same directory, then rename.
    fd, tmp_path = tempfile.mkstemp(prefix=p.name + ".tmp.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None
