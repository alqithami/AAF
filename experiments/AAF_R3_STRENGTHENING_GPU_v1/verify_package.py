#!/usr/bin/env python3
from pathlib import Path
import hashlib,sys
root=Path(__file__).resolve().parent
ledger=root/'SHA256SUMS.txt'
if not ledger.exists():raise SystemExit('Package checksum ledger missing; extract the complete ZIP')
n=0
for line in ledger.read_text().splitlines():
    value,name=line.split('  ',1);p=(root/name).resolve()
    if root not in p.parents:raise SystemExit('Invalid ledger path')
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=value:
        raise SystemExit('Package integrity check failed: '+name)
    n+=1
print(f'Package checksums verified: {n} files')
