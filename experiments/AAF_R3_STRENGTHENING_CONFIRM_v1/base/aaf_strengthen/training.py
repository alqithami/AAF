"""Nominal PPO training preserved from the R3 implementation, with progress logs.

New development seeds only. No outcome-selected checkpoint or held-out tuning.
Checkpoint weights are accompanied by identity/hash metadata and exported.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from aaf_r3.physics import Navigation
from aaf_r3.ppo import PPO
from aaf_r3.common import seed_all,seed_for,digest,atomic_json
from aaf_r3.governor import calibrate
from .io import sha

def key_for(cfg):return {k:cfg[k] for k in ('seed','n_agents','train_worlds','train_updates','rollout','horizon','calibration_steps')}

def prepare(cfg,device,root):
    key=key_for(cfg);path=Path(root)/'policies'/('nominal_'+digest(key)[:16]+'.pt')
    seed_all(seed_for(cfg['seed'],'vmas_init'))
    nav=Navigation(cfg['train_worlds'],cfg['n_agents'],seed_for(cfg['seed'],'vmas_train_env'),device)
    try:
        agent=PPO(nav.obs_dim,2,device,rollout=cfg['rollout'])
        if path.exists() and path.with_suffix('.json').exists():
            meta=json.loads(path.with_suffix('.json').read_text())
            if sha(path)!=meta['sha256'] or meta['key']!=key:raise RuntimeError('Checkpoint verification failed')
            ck=torch.load(path,map_location=device,weights_only=True)
            agent.net.load_state_dict(ck['net'])
            return agent,meta
        print(f'TRAIN policy seed={cfg["seed"]}: {cfg["train_updates"]} updates on {device}',flush=True)
        begin=time.perf_counter();obs=nav.obs;alive=np.ones(nav.e,bool);trace=[];block_reward=[]
        steps=cfg['train_updates']*cfg['rollout']
        for t in range(steps):
            a,lp,v=agent.act(obs);nobs,r,d=nav.step((2*a-1).reshape(nav.e,nav.n,2))
            mask=np.repeat(alive,nav.n).astype(float)
            terminal=d|(t%cfg['horizon']==cfg['horizon']-1)|(t==steps-1)
            agent.add(obs,a,lp,v,r.reshape(-1),np.repeat(terminal,nav.n),mask)
            stats=agent.update(nobs,force=t==steps-1)
            block_reward.append(float(r[alive].mean()) if alive.any() else 0.)
            alive&=~d
            if stats:
                trace.append({'t':t,'mean_reward':float(np.mean(block_reward)),**stats});block_reward=[]
                if agent.updates%16==0 or t==steps-1:
                    print(f'  update {agent.updates}/{cfg["train_updates"]}; elapsed {time.perf_counter()-begin:.0f}s',flush=True)
            obs=nobs
            if (t+1)%cfg['horizon']==0:obs=nav.reset(seed_for(cfg['seed'],'vmas_train_reset',t));alive[:]=True
    finally:nav.close()
    nav=Navigation(cfg['train_worlds'],cfg['n_agents'],seed_for(cfg['seed'],'vmas_calibration_env'),device)
    try:
        seed_all(seed_for(cfg['seed'],'vmas_calibration_policy'));obs=nav.obs;zs=[]
        for t in range(cfg['calibration_steps']):
            a,_,_=agent.act(obs);obs,_,_=nav.step((2*a-1).reshape(nav.e,nav.n,2));zs.extend(nav.diagnostics()['risk'].mean(1).tolist())
            if (t+1)%cfg['horizon']==0:obs=nav.reset(seed_for(cfg['seed'],'vmas_cal_reset',t))
        cal=calibrate(zs)
    finally:nav.close()
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save({'net':{k:v.detach().cpu() for k,v in agent.net.state_dict().items()},'key':key},tmp);tmp.replace(path)
    meta={'key':key,'sha256':sha(path),'calibration':cal,'training_trace':trace,
          'training_device':str(device),'elapsed_s':time.perf_counter()-begin,
          'scope':'Fresh development policy, fixed final checkpoint; no selection by AAF benefit.'}
    atomic_json(path.with_suffix('.json'),meta)
    return agent,meta

import json
