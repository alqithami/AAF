from __future__ import annotations
import argparse, csv, gzip, json, multiprocessing as mp, os, sys, time, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import socket,contextlib
import numpy as np
import torch
from .common import atomic_json,digest,provenance,source_hash,choose_device,PROTOCOL
from .governor import METHODS,Authority
from .game_runner import run_game,ATTACKS
from .network import NETWORKS

PROFILES={
 "smoke":{"game_seeds":1,"steps":256,"burnin_steps":64,"calibration_steps":32,"attack_start":80,
          "physics_seeds":1,"train_updates":2,"rollout":16,"train_worlds":2,"eval_worlds":2,"horizon":128,"physics_calibration":32},
 "pilot":{"game_seeds":2,"steps":1000,"burnin_steps":512,"calibration_steps":256,"attack_start":300,
          "physics_seeds":2,"train_updates":8,"rollout":128,"train_worlds":8,"eval_worlds":8,"horizon":256,"physics_calibration":256},
 "main":{"game_seeds":20,"steps":5000,"burnin_steps":1024,"calibration_steps":512,"attack_start":1000,
          "physics_seeds":10,"train_updates":128,"rollout":128,"train_worlds":16,"eval_worlds":32,"horizon":256,"physics_calibration":512}}

def plan(profile:str,suite:str)->list[dict]:
    p=PROFILES[profile];configs=[]
    domains=("resource_sharing","public_goods")
    attacks=("none","persistent","pulse") if profile!="smoke" else ("persistent",)
    if suite in ("games","all"):
        for domain in domains:
            for seed in range(p["game_seeds"]):
                for attack in attacks:
                    for method in METHODS+("legacy__ppo_only","legacy__aaf_full"):
                        configs.append({"suite":"games","domain":domain,"seed":({"smoke":71000,"pilot":81000,"main":91000}[profile])+seed,"profile":profile,
                            "method":method,"n_agents":50,"attack":attack,
                            **{k:p[k] for k in ("steps","burnin_steps","calibration_steps","attack_start")},
                            "authority":asdict(Authority()),"update_semantics":"executed_action_legacy" if method.startswith("legacy__") else "proposal_consistent_masked"})
    if suite in ("physics","all"):
        from .physics import PHYSICS_METHODS
        methods=tuple(m for m in PHYSICS_METHODS if m!="aaf_full")
        nets=NETWORKS if profile!="smoke" else ("clean","stale_delay")
        for seed in range(p["physics_seeds"]):
            for net in nets:
                for attack in (("none","pursuit") if profile!="smoke" else ("pursuit",)):
                    for method in methods:
                        configs.append({"suite":"physics","domain":"vmas_navigation","seed":({"smoke":171000,"pilot":181000,"main":191000}[profile])+seed,
                            "profile":profile,"method":method,"network":net,"attack":attack,"attack_start":min(32,p["horizon"]//4),
                            "n_agents":4,**{k:p[k] for k in ("train_updates","rollout","train_worlds","eval_worlds","horizon")},
                            "calibration_steps":p["physics_calibration"],
                            "authority":asdict(Authority(top_k=1,horizon=10,window=60,tokens=4,cooldown=5,history=20)),
                            "update_semantics":"nominal_training_then_frozen_evaluation"})
    return configs

def work_group(configs:list[dict],device_name:str,root_name:str)->dict:
    torch.set_num_threads(1)
    device=choose_device(device_name);root=Path(root_name)
    done=0;skipped=0
    for cfg in configs:
        runid=digest(cfg)[:20];folder=root/"runs"/runid;summary_file=folder/"summary.json"
        if summary_file.exists():
            old=json.loads(summary_file.read_text())
            if old.get("config")!=cfg or old.get("provenance",{}).get("source_sha256")!=source_hash():
                raise RuntimeError(f"Refusing incompatible resume: {summary_file}")
            skipped+=1;continue
        begin=time.perf_counter()
        if cfg["suite"]=="games":result=run_game(cfg,device,root)
        else:
            from .physics import run_navigation
            result=run_navigation(cfg,device,root)
        folder.mkdir(parents=True,exist_ok=True)
        trace=result.pop("step_trace",None)
        if trace is not None:
            with gzip.open(folder/"steps.json.gz","wt") as f:json.dump(trace,f,allow_nan=False)
        result["run_id"]=runid;result["status"]="complete"
        atomic_json(summary_file,result);done+=1
        print(f"DONE {cfg['suite']} {cfg['domain']} seed={cfg['seed']} {cfg['method']} {cfg['attack']} {cfg.get('network','')} {time.perf_counter()-begin:.1f}s",flush=True)
    return {"completed":done,"resumed":skipped}

@contextlib.contextmanager
def output_lock(root):
    lock=root/"RUNNING.lock"
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        info=json.loads(lock.read_text())
        if info.get("host")==socket.gethostname():
            try:os.kill(info["pid"],0)
            except ProcessLookupError:
                lock.unlink();fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            else:raise RuntimeError(f"Another run owns {root}, PID {info['pid']}")
        else:raise RuntimeError("Output is locked on another host. Confirm it stopped before removing RUNNING.lock.")
    with os.fdopen(fd,"w") as f:json.dump({"pid":os.getpid(),"host":socket.gethostname()},f)
    try:yield
    finally:lock.unlink(missing_ok=True)

def run(args):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    with output_lock(root):_run_locked(args)

def _run_locked(args):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    device=choose_device(args.device)
    if device.type=="cuda" and args.jobs!=1:raise ValueError("Use --jobs 1 on a single CUDA device; CPU permits multiple jobs.")
    configs=plan(args.profile,args.suite)
    if args.suite in ("physics","all"):
        from .physics import preflight_navigation
        # Must pass actual VMAS API checks BEFORE starting any expensive run.
        check=preflight_navigation(device);atomic_json(root/"VMAS_PREFLIGHT.json",check)
    identity={"protocol":PROTOCOL,"source_sha256":source_hash(),"profile":args.profile,"suite":args.suite,
              "device":str(device),"plan_sha256":digest(configs),"packages":provenance(str(device))["packages"]}
    mf=root/"RUN_MANIFEST.json"
    if mf.exists():
        old=json.loads(mf.read_text())
        if old["identity"]!=identity:raise RuntimeError("Output folder belongs to another code/config/environment. Use a NEW --out directory; do not mix runs.")
    else:atomic_json(mf,{"identity":identity,"provenance":provenance(str(device)),"planned_runs":len(configs),"configs":configs})
    grouped={}
    for cfg in configs:grouped.setdefault((cfg["domain"],cfg["seed"]),[]).append(cfg)
    print(f"PLAN {len(configs)} runs in {len(grouped)} independent seed/domain groups. Profile={args.profile}. Output={root}",flush=True)
    t0=time.perf_counter()
    if args.jobs==1:
        totals=[work_group(c,str(device),str(root)) for c in grouped.values()]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=mp.get_context("spawn")) as pool:
            futures=[pool.submit(work_group,c,str(device),str(root)) for c in grouped.values()]
            totals=[f.result() for f in as_completed(futures)]
    atomic_json(root/"RUN_COMPLETE.json",{"status":"all planned runs complete","planned":len(configs),
        "elapsed_s":time.perf_counter()-t0,"groups":totals,"source_sha256":source_hash(),"profile":args.profile})
    print("All planned runs completed. Now run the analyze command.",flush=True)

def pack(root:Path,output:Path):
    if not (root/"RUN_MANIFEST.json").exists():raise FileNotFoundError("No run manifest")
    selected=[]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in (".json",".csv",".md",".tex",".pdf",".png",".txt"):
            if "initial_policies" not in p.parts:selected.append(p)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(selected):z.write(p,p.relative_to(root))
    print(f"Results bundle: {output} ({len(selected)} files; checkpoints/step traces stay on your machine)")

def main():
    ap=argparse.ArgumentParser(description="AAF Reviewer-3: real experiments, no generated substitute results")
    sub=ap.add_subparsers(dest="command",required=True)
    p=sub.add_parser("run");p.add_argument("--profile",choices=PROFILES,default="pilot")
    p.add_argument("--suite",choices=("games","physics","all"),default="all")
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="cpu")
    p.add_argument("--jobs",type=int,default=1);p.add_argument("--out",required=True)
    p=sub.add_parser("plan");p.add_argument("--profile",choices=PROFILES,default="main")
    p.add_argument("--suite",choices=("games","physics","all"),default="all")
    p=sub.add_parser("analyze");p.add_argument("--out",required=True);p.add_argument("--allow-partial",action="store_true")
    p=sub.add_parser("pack");p.add_argument("--out",required=True);p.add_argument("--zip",required=True)
    p=sub.add_parser("preflight");p.add_argument("--device",choices=("cpu","cuda","auto"),default="cpu")
    p.add_argument("--physics",action="store_true")
    args=ap.parse_args()
    if args.command=="run":
        if args.jobs<1:ap.error("--jobs must be positive")
        run(args)
    elif args.command=="plan":
        configs=plan(args.profile,args.suite)
        print(json.dumps({"profile":args.profile,"suite":args.suite,"planned_runs":len(configs),
             "by_suite":{s:sum(c["suite"]==s for c in configs) for s in ("games","physics")},
             "config_example":configs[0]},indent=2))
    elif args.command=="analyze":
        from .analysis import analyze
        analyze(Path(args.out).resolve(),args.allow_partial)
    elif args.command=="pack":pack(Path(args.out).resolve(),Path(args.zip).resolve())
    elif args.command=="preflight":
        device=choose_device(args.device);torch.set_num_threads(1)
        report=provenance(str(device))
        if args.physics:
            from .physics import preflight_navigation
            report["physics"]=preflight_navigation(device)
        print(json.dumps(report,indent=2))
if __name__=="__main__":main()
