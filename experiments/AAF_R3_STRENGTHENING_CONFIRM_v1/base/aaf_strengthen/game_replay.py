"""Capacity/selection diagnostics, explicitly NOT online-learning ablations.

Frozen-PPO proposal streams come from the inherited corrected learner. Scripted
streams are separately named positive/negative controls. All selectors receive
exactly the same proposal sequence and fixed intervention windows.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from aaf_r3.games import Game
from aaf_r3.game_runner import prepare_game
from aaf_r3.common import seed_all,seed_for,atomic_json
from .io import atomic_npz,sha

def fixed_windows(T,H=50,cooldown=25,W=300,B=4):
    period=max(H+cooldown,int(np.ceil(W/B)))
    return [(s,min(s+H,T)) for s in range(period,T,period)]

def replay(domain,actions,selector,k,seed):
    T,N=actions.shape;env=Game(domain,N,seed);env.reset()
    viol=np.asarray([env.violations(a) for a in actions],float)
    active=np.zeros((T,N),bool);selected=[];rng=np.random.default_rng(seed)
    for start,end in fixed_windows(T):
        if selector=='future_window_diagnostic':score=viol[start:end].sum(0)
        elif selector=='current_violation':score=viol[start-1]
        elif selector=='recent_score':score=viol[max(0,start-50):start].mean(0)
        elif selector=='random':score=np.ones(N)
        else:raise ValueError(selector)
        # Fixed schedule/all identities eligible; no outcome-dependent admissions.
        # Zero-score random ties intentionally still consume the SAME authority.
        tie=rng.permutation(N)
        ids=tie[:k] if selector=='random' else tie[np.argsort(-score[tie],kind='stable')[:k]]
        active[start:end,ids]=True;selected.append({'begin':start,'end':end,'ids':ids.tolist()})
    executed=actions.copy();rewards=[];post=[];ginis=[]
    for t in range(T):
        executed[t]=env.shield(actions[t],active[t]);_,r,info=env.step(executed[t])
        rewards.append(float(r.mean()));post.append(info['violation']);ginis.append(float(info['gini']))
    post=np.asarray(post,float);gain=viol-post
    if np.any(gain>active+1e-9) or np.any(gain< -1e-9):raise AssertionError('Direct-suppression accounting violation')
    expected_random=sum((k/N)*float(viol[start:end].sum()) for start,end in fixed_windows(T))/(N*T)
    fixed_hindsight=sum(float(np.sort(viol[start:end].sum(0))[-k:].sum()) for start,end in fixed_windows(T))/(N*T)
    return {'expected_uniform_random_reduction':expected_random,
            'fixed_schedule_selection_opportunity':fixed_hindsight-expected_random,
            'domain':domain,'selector':selector,'top_k':k,'seed':seed,
            'attempted':float(viol.mean()),'executed':float(post.mean()),
            'direct_reduction':float(gain.mean()),'active_fraction':float(active.mean()),
            'modified_fraction':float((np.abs(executed-actions)>1e-7).mean()),
            'reward':float(np.mean(rewards)),'gini':float(np.mean(ginis)),
            'long_run_authority_cap':k/N*min(1,50/75,4*50/300),
            'fixed_schedule_starts':selected},executed,active

def scripts(domain,kind,seed,T,N,start):
    rng=np.random.default_rng(seed);p=np.full((T,N),.03)
    if kind=='scripted_diffuse':p[:]=.985
    else:
        ordering=rng.permutation(N)
        for t in range(start,T):
            shift=0 if kind=='scripted_sparse' else ((t-start)//100)*5
            ids=ordering[np.arange(shift,shift+5)%N];p[t,ids]=.95
    v=rng.random((T,N))<p
    if domain=='resource_sharing':a=np.where(v,.9+.1*rng.random((T,N)),.15+.3*rng.random((T,N)))
    else:a=np.where(v,.1*rng.random((T,N)),.4+.3*rng.random((T,N)))
    return a.astype(np.float32)

def run_game_diagnostics(cfg,device,root):
    root=Path(root);folder=root/'game_replay';folder.mkdir(parents=True,exist_ok=True)
    result=[];streams={}
    for domain in cfg['domains']:
        for seed in cfg['seeds']:
            print(f'GAME REPLAY development: {domain}, seed={seed}',flush=True)
            local={'domain':domain,'n_agents':cfg['n_agents'],'seed':seed,
                   'burnin_steps':cfg['burnin_steps'],'calibration_steps':cfg['calibration_steps']}
            agent,cal=prepare_game(local,device,folder)
            env=Game(domain,cfg['n_agents'],seed_for(seed,'fixed_proposal_env'));obs=env.reset()
            seed_all(seed_for(seed,'fixed_proposal_policy'))
            ids=np.random.default_rng(seed_for(seed,'fixed_proposal_attacker')).choice(env.n,5,False)
            learned=[]
            for t in range(cfg['steps']):
                a,_,_=agent.act(obs);a=a[:,0].copy()
                if t>=cfg['attack_start']:a[ids]=1. if domain=='resource_sharing' else 0.
                obs,_,_=env.step(a);learned.append(a)
            for kind in cfg['streams']:
                actions=np.stack(learned) if kind=='frozen_ppo_replay' else scripts(domain,kind,seed_for(seed,kind),cfg['steps'],env.n,cfg['attack_start'])
                streamkey=f'{domain}_{seed}_{kind}';streams[streamkey]=actions
                future={}
                for k in cfg['budgets']:
                    for selector in cfg['selectors']:
                        r,executed,active=replay(domain,actions,selector,k,seed_for(seed,kind,'targets'))
                        r.update(seed=seed,stream=kind,scope='same-proposal fixed-schedule diagnostic; NOT feedback control performance')
                        if selector=='future_window_diagnostic':future[k]=r['direct_reduction']
                        result.append(r)
                # Exact hindsight bound is valid only for this fixed stream and
                # disjoint fixed schedule, not as a bound on learning dynamics.
                for r in result:
                    if r['domain']==domain and r['seed']==seed and r['stream']==kind:
                        r['fixed_stream_hindsight_max']=future[r['top_k']]
                        if r['direct_reduction']>future[r['top_k']]+1e-7:raise AssertionError('Replay oracle accounting failed')
    atomic_npz(folder/'proposal_streams.npz',streams)
    report={'config':cfg,'results':result,'stream_sha256':sha(folder/'proposal_streams.npz'),
            'interpretation':'Scripted streams are engineering controls. The learned-policy stream is frozen open-loop replay, not a new online MARL result. Hindsight is privileged and not deployable.'}
    atomic_json(folder/'results.json',report)
    for p in (folder/'initial_policies').glob('*.pt'):
        p.with_suffix('.pt.sha256').write_text(sha(p)+'  '+p.name+'\n')
    atomic_json(folder/'COMPLETE.json',{'results_sha256':sha(folder/'results.json'),'stream_sha256':report['stream_sha256']})
    return report
