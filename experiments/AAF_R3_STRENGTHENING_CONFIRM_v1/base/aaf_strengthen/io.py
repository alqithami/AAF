from __future__ import annotations
import hashlib,json,os,platform,sys,subprocess,time,zipfile
from pathlib import Path
import numpy as np
import torch
from aaf_r3.common import atomic_json,digest
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def source_manifest():
    paths=list((ROOT/'aaf_strengthen').glob('*.py'))+list((ROOT/'vendor'/'aaf_r3').glob('*.py'))
    paths += [ROOT/'run_strengthening.py',ROOT/'requirements.txt']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def environment(device):
    import importlib.metadata as im
    versions={}
    for n in ('torch','numpy','gym','vmas','pytest','scipy','pandas','matplotlib'):
        try:versions[n]=im.version(n)
        except im.PackageNotFoundError:versions[n]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':versions,
            'device':str(device),'cuda_runtime':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
            'torch_threads':torch.get_num_threads()}

def verify_immutable_vendor():
    ledger=json.loads((ROOT/'vendor'/'SOURCE_SHA256.json').read_text())
    bad=[name for name,value in ledger.items() if not (ROOT/'vendor'/name).is_file() or sha(ROOT/'vendor'/name)!=value]
    if bad:raise RuntimeError('Inherited source changed: '+', '.join(bad))

def configure_device(request='cuda',max_gpu_gib=12.0):
    if max_gpu_gib<=0:raise ValueError('GPU memory ceiling must be positive')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if request not in ('cuda','cpu'):raise ValueError('Explicit cpu or cuda required')
    if request=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA was requested but is unavailable. No silent CPU fallback.')
        free,total=torch.cuda.mem_get_info()
        if free < 8*1024**3:raise RuntimeError('Less than 8 GiB CUDA memory is free; use the allocated GPU when available. No other process was changed.')
        torch.cuda.set_per_process_memory_fraction(min(max_gpu_gib*1024**3/total,.8),0)
    # Repeatability within a fixed software/device stack, not CPU/CUDA equality.
    torch.use_deterministic_algorithms(True,warn_only=False)
    return torch.device(request)

def atomic_npz(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def save_completed(folder,result,arrays):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    atomic_npz(folder/'trajectories.npz',arrays)
    result['trajectory_sha256']=sha(folder/'trajectories.npz')
    atomic_json(folder/'summary.json',result)
    atomic_json(folder/'COMPLETE.json',{'summary_sha256':sha(folder/'summary.json'),
            'trajectory_sha256':result['trajectory_sha256']})

def load_completed(folder,cfg):
    folder=Path(folder)
    if not (folder/'COMPLETE.json').exists():return None
    marker=json.loads((folder/'COMPLETE.json').read_text())
    for name,key in [('summary.json','summary_sha256'),('trajectories.npz','trajectory_sha256')]:
        if sha(folder/name)!=marker[key]:raise RuntimeError(f'Checksum mismatch in {folder/name}')
    r=json.loads((folder/'summary.json').read_text())
    if r['config']!=cfg:raise RuntimeError('Incompatible treatment result')
    return r

def pack(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if not (root/'RUN_COMPLETE.json').exists() or not (root/'analysis'/'RECORD_AUDIT.json').exists():
        raise RuntimeError('Complete run and record audit required; use collect_failure.py for partial results')
    if json.loads((root/'analysis'/'RECORD_AUDIT.json').read_text()).get('status')!='PASS':
        raise RuntimeError('Record audit did not pass')
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.lock') and p.name!='SHA256SUMS.txt']
    # Includes complete trajectories and binary policies from this package only.
    ledger=''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in sorted(files))
    (root/'SHA256SUMS.txt').write_text(ledger)
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files)+[root/'SHA256SUMS.txt']:z.write(p,Path('results')/p.relative_to(root))
        for base in ('aaf_strengthen','vendor','docs','tests','validation'):
            for p in sorted((ROOT/base).rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.tmp'):
                    z.write(p,Path('source')/p.relative_to(ROOT))
        for p in sorted(ROOT.iterdir()):
            if p.is_file() and p.suffix in ('.py','.sh','.txt','.md','.json'):
                z.write(p,Path('source')/p.name)
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('ZIP integrity failed: '+bad)
    os.replace(tmp,output);output.with_suffix(output.suffix+'.sha256').write_text(f'{sha(output)}  {output.name}\n')
    print(f'RETURN FILE: {output} ({output.stat().st_size/1024**2:.1f} MiB)',flush=True)
