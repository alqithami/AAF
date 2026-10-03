#!/usr/bin/env python3
"""Verify released source snapshots using only the Python standard library."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_ledger(directory, name):
    count = 0
    for line in (directory / name).read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise RuntimeError(f"Missing or unsafe source path: {relative}")
        if sha(path) != expected:
            raise RuntimeError(f"Source hash mismatch: {path}")
        count += 1
    return count

def main():
    counts = {}
    for folder, ledger in [
        ("AAF_R3_CORRECTED_v1", "SOURCE_SHA256SUMS.txt"),
        ("AAF_R3_STRENGTHENING_GPU_v1", "SHA256SUMS.txt"),
        ("AAF_R3_STRENGTHENING_CONFIRM_v1", "PACKAGE_SHA256SUMS.txt"),
    ]:
        counts[folder] = check_ledger(ROOT / "experiments" / folder, ledger)
    confirm = ROOT / "experiments/AAF_R3_STRENGTHENING_CONFIRM_v1"
    expected = json.loads((confirm / "BASELINE_HASHES.json").read_text())
    for name, value in expected.items():
        path = confirm / "base" / name
        if not path.is_file() or sha(path) != value:
            raise RuntimeError(f"Changed confirmation base: {name}")
    provenance = json.loads((ROOT / "docs/CODE_PROVENANCE.json").read_text())
    for name, value in provenance["standalone_files"].items():
        if sha(ROOT / name) != value:
            raise RuntimeError(f"Changed standalone code: {name}")
    print(json.dumps({"status": "PASS", "source_entries_checked": counts,
                      "confirmation_base_files": len(expected),
                      "scope": "Source integrity only; no simulation or GPU test launched."}, indent=2))

if __name__ == "__main__":
    main()
