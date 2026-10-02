"""Development-only design. This package intentionally has no main-study switch."""
from dataclasses import asdict
from aaf_r3.governor import Authority
VERSION='AAF-R3-strengthening-dev-v1'

def plan(profile='diagnostic'):
    if profile not in ('smoke','diagnostic'):raise ValueError('Only smoke and diagnostic are authorized')
    smoke=profile=='smoke'
    seeds=[271000] if smoke else [281000,281001]
    common={'profile':profile,'n_agents':4,'train_worlds':2 if smoke else 16,
            'train_updates':2 if smoke else 128,'rollout':16 if smoke else 128,
            'horizon':64 if smoke else 256,'calibration_steps':32 if smoke else 512,
            'eval_worlds':2 if smoke else 8,'attack_start':16 if smoke else 32,
            'prediction_horizon':8,'prediction_margin':.05}
    out=[]
    for seed in seeds:
        for attack in (['pursuit'] if smoke else ['none','pursuit']):
            for k in ([1] if smoke else [1,2,4]):
                for filt in ['brake','predictive']:
                    for method in (['adaptive_rank'] if smoke else ['adaptive_rank','adaptive_random','threshold_rank','periodic_rank']):
                        out.append(dict(common,seed=seed,attack=attack,method=method,filter=filt,
                                        evidence='gateway',block='authority',
                                        authority=asdict(Authority(top_k=k,horizon=10,window=60,tokens=4,cooldown=5,history=20))))
                if not smoke:
                    out.append(dict(common,seed=seed,attack=attack,method='adaptive_benefit_state',filter='predictive',
                                    evidence='gateway',block='state_model_diagnostic',
                                    authority=asdict(Authority(top_k=k,horizon=10,window=60,tokens=4,cooldown=5,history=20))))
            auth=asdict(Authority(top_k=1,horizon=10,window=60,tokens=4,cooldown=5,history=20))
            for method,filt in [('ppo_only','brake'),('static_guard','brake'),('static_guard','predictive'),('random_policy','brake')]:
                out.append(dict(common,seed=seed,attack=attack,method=method,filter=filt,evidence='gateway',block='reference',authority=auth))
            out.append(dict(common,seed=seed,attack=attack,method='immediate_state',filter='predictive',
                            evidence='gateway',block='zero_delay_state_model_diagnostic',authority=auth))
            profiles=['selective_gapaware'] if smoke else ['forged_self_report','selective_naive','selective_gapaware',
                         'permanent_suppression','gateway_iid','gateway_burst','stale_evidence_only','stale_commands_only','stale_both']
            for ep in profiles:
                out.append(dict(common,seed=seed,attack=attack,method='adaptive_rank',filter='predictive',
                                evidence=ep,block='evidence_and_timing',authority=auth))
    return out

def game_plan(profile='diagnostic'):
    return {'profile':profile,'seeds':[271100] if profile=='smoke' else [281100,281101],
            'domains':['resource_sharing','public_goods'],'n_agents':50,
            'burnin_steps':64 if profile=='smoke' else 6024,'calibration_steps':32 if profile=='smoke' else 512,
            'steps':128 if profile=='smoke' else 2000,'attack_start':32 if profile=='smoke' else 400,
            'budgets':[3] if profile=='smoke' else [3,10,25],
            'streams':['frozen_ppo_replay','scripted_sparse','scripted_rotating','scripted_diffuse'],
            'selectors':['random','recent_score','current_violation','future_window_diagnostic']}
