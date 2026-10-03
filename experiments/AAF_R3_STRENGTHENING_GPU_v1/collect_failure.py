#!/usr/bin/env python3
"""Standard-library-only failure collection; no credentials/environment dump."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,zipfile
root=Path(__file__).resolve().parent
out=root/'AAF_R3_STRENGTHENING_FAILURE.zip'
paths=[]
for p in (root/'logs').rglob('*'):
    if p.is_file():paths.append(p)
for name in ('LAST_ERROR.json','RUN_MANIFEST.json','PREFLIGHT.json'):
    paths += list((root/'results').rglob(name))
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('STATUS.json',json.dumps({'status':'failed or interrupted diagnostics','created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'scope':'Logs only; no API keys or environment variables collected'},indent=2))
    for p in sorted(set(paths)):z.write(p,p.relative_to(root))
print('Return this failure bundle:',out)
