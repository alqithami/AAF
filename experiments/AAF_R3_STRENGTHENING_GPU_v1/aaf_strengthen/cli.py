from __future__ import annotations
import argparse,json,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from aaf_r3.common import atomic_json,digest
from aaf_r3.cli import output_lock
from .io import configure_device,environment,source_manifest,verify_immutable_vendor,save_completed,load_completed,pack,sha
from .protocol import plan,game_plan,VERSION


def run(args):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    device=configure_device(args.device,args.max_gpu_gib);verify_immutable_vendor()
    np=plan(args.profile) if args.suite in ('all','physics') else []
    gp=game_plan(args.profile) if args.suite in ('all','games') else None
    identity={'version':VERSION,'profile':args.profile,'suite':args.suite,'source':source_manifest(),
              'environment':environment(device),'plan_sha256':digest({'navigation':np,'game':gp})}
    with output_lock(root):
        mf=root/'RUN_MANIFEST.json'
        if mf.exists():
            if json.loads(mf.read_text())['identity']!=identity:
                raise RuntimeError('Refusing mixed code/config/device/environment. Use a NEW output folder, keep old data.')
        else:atomic_json(mf,{'identity':identity,'navigation_plan':np,'game_plan':gp,
                            'created_utc':datetime.now(timezone.utc).isoformat(),'statistical_status':'development diagnostics, not main'})
        from .preflight import check
        atomic_json(root/'PREFLIGHT.json',check(device,bool(np)))
        print(f'PLAN: {len(np)} navigation evaluations; games replay={gp is not None}; DEVELOPMENT ONLY',flush=True)
        if gp:
            from .game_replay import run_game_diagnostics
            marker=root/'game_replay'/'COMPLETE.json'
            if marker.exists():
                m=json.loads(marker.read_text())
                if sha(marker.parent/'results.json')!=m['results_sha256'] or sha(marker.parent/'proposal_streams.npz')!=m['stream_sha256']:
                    raise RuntimeError('Game replay result checksum mismatch')
                print('REUSE complete game diagnostic',flush=True)
            else:run_game_diagnostics(gp,device,root)
        from .navigation import run_navigation
        complete=reused=0;begin=time.perf_counter()
        for index,cfg in enumerate(np):
            folder=root/'runs'/digest(cfg)[:20]
            if load_completed(folder,cfg) is not None:
                reused+=1;print(f'REUSE {index+1}/{len(np)}',flush=True);continue
            print(f'RUN {index+1}/{len(np)} seed={cfg["seed"]} {cfg["method"]} k={cfg["authority"]["top_k"]} {cfg["filter"]} {cfg["attack"]} {cfg["evidence"]}',flush=True)
            result,arrays=run_navigation(cfg,device,root);save_completed(folder,result,arrays);complete+=1
        atomic_json(root/'RUN_COMPLETE.json',{'status':'complete','navigation_completed_now':complete,'navigation_reused':reused,
                    'navigation_planned':len(np),'elapsed_s':time.perf_counter()-begin,
                    'scope':'All development configurations complete, not a scientific acceptance test.'})
    print('Finished development run. Analyze and return; do not launch a main study.',flush=True)


def main():
    p=argparse.ArgumentParser(description='AAF GPU strengthening diagnostics; additive and development-only')
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('plan');a.add_argument('--profile',choices=['smoke','diagnostic'],default='diagnostic')
    a=s.add_parser('preflight');a.add_argument('--device',choices=['cpu','cuda'],default='cuda');a.add_argument('--out',required=True)
    a.add_argument('--max-gpu-gib',type=float,default=12.)
    a=s.add_parser('run');a.add_argument('--profile',choices=['smoke','diagnostic'],default='diagnostic')
    a.add_argument('--suite',choices=['games','physics','all'],default='all');a.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a.add_argument('--out',required=True);a.add_argument('--max-gpu-gib',type=float,default=12.)
    a=s.add_parser('analyze');a.add_argument('--out',required=True)
    a=s.add_parser('pack');a.add_argument('--out',required=True);a.add_argument('--zip',required=True)
    args=p.parse_args()
    try:
        if args.command=='plan':
            nav=plan(args.profile);games=game_plan(args.profile)
            print(json.dumps({'profile':args.profile,'navigation_evaluations':len(nav),
                              'policy_seeds':sorted(set(c['seed'] for c in nav)),
                              'navigation_episode_records':sum(c['eval_worlds'] for c in nav),
                              'game_plan':games,'first_navigation_config':nav[0]},indent=2))
        elif args.command=='preflight':
            from .preflight import check
            root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
            result=check(configure_device(args.device,args.max_gpu_gib));atomic_json(root/'PREFLIGHT.json',result)
            print(json.dumps(result,indent=2))
        elif args.command=='run':run(args)
        elif args.command=='analyze':
            from .analysis import analyze
            analyze(Path(args.out))
        elif args.command=='pack':pack(Path(args.out),Path(args.zip))
    except Exception as exc:
        if getattr(args,'out',None):
            root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
            atomic_json(root/'LAST_ERROR.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
