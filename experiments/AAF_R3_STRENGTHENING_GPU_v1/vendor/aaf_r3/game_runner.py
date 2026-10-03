from __future__ import annotations
import copy, time
from pathlib import Path
import numpy as np
import torch
from .common import seed_all,seed_for,atomic_json,digest,provenance
from .games import Game
from .ppo import PPO
from .governor import Governor,Authority,calibrate

ATTACKS=("none","persistent","pulse")

def attack_mask(t:int,start:int,attack:str,ids:np.ndarray,n:int)->np.ndarray:
    m=np.zeros(n,bool)
    on=attack=="persistent" or (attack=="pulse" and (t-start)%200<50)
    if attack!="none" and t>=start and on:m[ids]=True
    return m

def prepare_game(cfg:dict,device:torch.device,root:Path)->tuple[PPO,dict]:
    key={k:cfg[k] for k in ("domain","n_agents","seed","burnin_steps","calibration_steps")}
    path=root/"initial_policies"/(digest(key)[:16]+".pt")
    seed_all(seed_for(cfg["seed"],cfg["domain"],"policy_init"))
    env=Game(cfg["domain"],cfg["n_agents"],seed_for(cfg["seed"],"burnin_env"))
    agent=PPO(env.obs_dim,1,device)
    if path.exists():
        ck=torch.load(path,map_location=device,weights_only=False)
        if ck["key"]!=key:raise RuntimeError("Checkpoint identity mismatch")
        agent.net.load_state_dict(ck["net"]);agent.opt.load_state_dict(ck["opt"])
        agent.updates=0
        return agent,ck["calibration"]
    obs=env.reset()
    for t in range(cfg["burnin_steps"]):
        a,lp,v=agent.act(obs);nobs,r,_=env.step(a[:,0])
        done=np.full(env.n,t==cfg["burnin_steps"]-1,dtype=float)
        agent.add(obs,a,lp,v,r,done,np.ones(env.n));agent.update(nobs,force=bool(done[0]));obs=nobs
    env=Game(cfg["domain"],cfg["n_agents"],seed_for(cfg["seed"],"calibration_env"))
    seed_all(seed_for(cfg["seed"],"calibration_policy"));obs=env.reset();zs=[]
    for _ in range(cfg["calibration_steps"]):
        a,_,_=agent.act(obs);obs,_,info=env.step(a[:,0]);zs.append(float(info["violation"].mean()))
    cal=calibrate(zs)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    torch.save({"key":key,"net":agent.net.state_dict(),"opt":agent.opt.state_dict(),"calibration":cal},temp)
    temp.replace(path);atomic_json(path.with_suffix(".json"),{"key":key,"calibration":cal})
    agent.updates=0
    return agent,cal

def run_game(cfg:dict,device:torch.device,root:Path)->dict:
    if not 0 <= cfg["attack_start"] < cfg["steps"]: raise ValueError("attack_start must precede the end of the run")
    preparation_start=time.perf_counter();agent,cal=prepare_game(cfg,device,root)
    preparation_s=time.perf_counter()-preparation_start;tick=time.perf_counter()
    n=cfg["n_agents"];method=cfg["method"].removeprefix("legacy__")
    legacy=cfg["update_semantics"]=="executed_action_legacy"
    seed_all(seed_for(cfg["seed"],cfg["domain"],"treatment_policy"))
    env=Game(cfg["domain"],n,seed_for(cfg["seed"],"treatment_env"))
    auth=Authority(**cfg["authority"])
    gov=Governor(method,1,n,seed_for(cfg["seed"],"target_rng"),cal,auth)
    ids=np.random.default_rng(seed_for(cfg["seed"],"attacker_ids")).choice(n,max(1,int(round(.1*n))),False)
    if cfg["attack"]=="none":ids=np.array([],int)
    obs=env.reset();rows=[];alarms=[];selections=[];losses=[]
    first_rank=None
    rank_rng=np.random.default_rng(seed_for(cfg["seed"],"ranking_evaluation"))
    for t in range(cfg["steps"]):
        proposal,lp,v=agent.act(obs);submitted=proposal[:,0].copy()
        overridden=attack_mask(t,cfg["attack_start"],cfg["attack"],ids,n)
        submitted[overridden]=1.0 if env.name=="resource_sharing" else 0.0
        active=gov.active(t)[0];executed=env.shield(submitted,active)
        attempt=env.violations(submitted)
        nobs,reward,info=env.step(executed)
        shaping=gov.penalties(t)[0]
        observation=gov.observe(t,info["violation"][None,:])
        if observation["trigger"][0]:
            alarms.append(t)
            if first_rank is None and t>=cfg["attack_start"] and ids.size and method not in ("periodic_rank","ppo_only","static_guard"):
                scores=observation["scores"][0];eligible=np.flatnonzero(observation["coverage"][0]>=auth.minimum_coverage)
                if eligible.size and scores[eligible].max()>0:
                    shuffled=rank_rng.permutation(eligible)
                    first_rank=shuffled[np.argsort(-scores[shuffled],kind="stable")[:auth.top_k]].tolist()
        selected=np.flatnonzero(observation["selected"][0]).tolist()
        if selected:selections.append({"t":t,"ids":selected,"attacker_overlap":len(set(selected)&set(ids.tolist()))})
        done=np.full(n,t==cfg["steps"]-1,dtype=float)
        # Both arrays refer to the sampled proposal, not submitted or executed.
        learn=(~overridden).astype(float)
        if legacy:
            # Deliberately reproduce the historical mismatch ONLY as a labeled
            # diagnostic. Never pool this with corrected primary experiments.
            agent.add(obs,executed[:,None],lp,v,reward-shaping,done,np.ones(n))
        else:
            agent.add(obs,proposal,lp,v,reward-shaping,done,learn)
        stats=agent.update(nobs,force=bool(done[0]))
        if stats:losses.append({"t":t,**stats})
        rows.append({"t":t,"attempted":float(attempt.mean()),"executed":float(info["violation"].mean()),
                     "reward":float(reward.mean()),"gini":float(info["gini"]),
                     "active_fraction":float(active.mean()),"modified_fraction":float((abs(executed-submitted)>1e-7).mean()),
                     "action_l1":float(abs(executed-submitted).mean()),"shaping":float(shaping.mean()),
                     "non_scripted_constrained_agent_time":float((active & ~np.isin(np.arange(n),ids)).mean()),
                     "attack_active":bool(overridden.any()),"alarm":bool(observation["trigger"][0])})
        obs=nobs
    post=[r for r in rows if r["t"]>=cfg["attack_start"]]
    first=next((t for t in alarms if t>=cfg["attack_start"]),None)
    out={"config":cfg,"calibration":cal,"runtime_s":time.perf_counter()-tick,"preparation_s":preparation_s,
         "attacker_ids":ids.tolist(),"alarms":alarms,"selections":selections,"training_updates":agent.updates,
         "new_results_not_archival_reanalysis":True,"provenance":provenance(str(device)),
         "summary":{k:float(np.mean([r[k] for r in rows])) for k in ("attempted","executed","reward","gini","active_fraction","modified_fraction","action_l1","shaping","non_scripted_constrained_agent_time")},
         "post_summary":{k:float(np.mean([r[k] for r in post])) for k in ("attempted","executed","reward","active_fraction")},
         "ranking_at_first_post_trigger":first_rank,
         "ranking_top1":float(first_rank[0] in ids) if first_rank else None,
         "ranking_recall_at_k":len(set(first_rank)&set(ids.tolist()))/len(ids) if first_rank and len(ids) else None,
         "trigger_count":len(alarms),"admissions":int(gov.n_admissions[0]),"denied":int(gov.n_denied[0]),
         "first_post_trigger_delay":first-cfg["attack_start"] if first is not None and ids.size and method not in ("periodic_rank","ppo_only","static_guard") else None,
         "step_trace":rows,"training_trace":losses}
    return out
