"""Actual VMAS diagnostic evaluations with complete per-step state/action records."""
from __future__ import annotations
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from aaf_r3.physics import Navigation,apply_brake_filter,braking_command
from aaf_r3.common import seed_all,seed_for
from aaf_r3.governor import Authority
from .predictive import MotionModel,Predictor
from .controller import BudgetController
from .evidence import Transport
from .training import prepare


def first_where(mask):
    ids=np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def summarize_trajectory(arrays,attack_start):
    live=arrays['live'].astype(bool);T,E,N=arrays['risk'].shape
    out=[]
    for e in range(E):
        valid=live[:,e];ts=np.flatnonzero(valid);length=len(ts)
        if not length:raise AssertionError('Empty episode')
        aftercontact=arrays['contact'][:,e].any(-1)
        initialcontact=bool(arrays['initial_contact'][e])
        modifications=(np.linalg.norm(arrays['executed'][:,e]-arrays['incoming'][:,e],axis=-1)>1e-7)
        # Contact times are ACTION indices: t denotes the transition s_t -> s_{t+1}.
        # Initial contacts have a separate flag, not a fabricated first-impact time.
        post=valid&(np.arange(T)>=attack_start)
        post_exposed=bool((arrays['attack_active'][:,e]&valid).any())
        mean=lambda a:float(np.asarray(a)[valid].mean())
        first_contact=first_where(aftercontact&valid)
        first_change=first_where(modifications.any(-1)&valid)
        first_admission=first_where(arrays['selected'][:,e].any(-1)&valid)
        first_alarm=first_where(arrays['alarm'][:,e]&valid)
        riskstart=first_where(arrays['pre_risk'][:,e].any(-1)&valid)
        predstart=first_where((arrays['predicted_clearance_before'][:,e]<.05)&valid)
        finals=arrays['goal_distance'][-1,e] if ts[-1]==T-1 else arrays['goal_distance'][ts[-1],e]
        event={'episode':e,'length':length,'success':bool((arrays['done'][:,e]&valid).any()),
               'initial_contact':initialcontact,'any_contact':initialcontact or bool((aftercontact&valid).any()),
               'post_attack_any_contact':bool((aftercontact&post).any()) if post_exposed else None,
               'pre_attack_any_contact':initialcontact or bool((aftercontact&valid&(np.arange(T)<attack_start)).any()),
               'attack_exposed':post_exposed,
               'contact_time':mean(arrays['contact'][:,e]),'modified_fraction':mean(modifications),
               'active_fraction':mean(arrays['active'][:,e]),
               'action_l1':mean(np.abs(arrays['executed'][:,e]-arrays['incoming'][:,e])),
               'goal_progress':float(arrays['initial_goal_distance'][e].mean()-finals.mean()),
               'reward':mean(arrays['reward'][:,e]),'overspeed':mean(arrays['speed'][:,e]>.3),
               'evidence_coverage':mean(arrays['evidence_valid'][:,e]),
               'fallback_fraction':mean(arrays['fallback'][:,e]),
               'visible_gap_fraction':mean(arrays['visible_gap'][:,e]),
               'omission_target_fraction':mean(arrays['omission_target'][:,e]),
               'admission_count':int(arrays['selected'][:,e].any(-1)[valid].sum()),
               'alarm_count':int(arrays['alarm'][:,e][valid].sum()),
               'first_risk_action_index':riskstart,'first_predicted_risk_action_index':predstart,
               'first_alarm_action_index':first_alarm,'first_admission_action_index':first_admission,
               'admission_phase':'before_action' if int(arrays['admission_delay_steps'])==0 else 'after_action',
               'first_scheduled_action_index':first_admission+int(arrays['admission_delay_steps']) if first_admission is not None else None,
               'first_post_attack_alarm_action_index':first_where(arrays['alarm'][:,e]&post) if post_exposed else None,
               'first_post_attack_modification_action_index':first_where(modifications.any(-1)&post) if post_exposed else None,
               'first_modified_action_index':first_change,'first_contact_action_index':first_contact,
               'first_post_attack_contact_action_index':first_where(aftercontact&post) if post_exposed else None,
               'modification_lead_to_first_contact_steps':first_contact-first_change if first_contact is not None and first_change is not None else None,
               'prediction_position_error_mean':mean(arrays['prediction_position_error'][:,e])}
        out.append(event)
    metrics=('success','any_contact','contact_time','modified_fraction','active_fraction','action_l1',
             'goal_progress','reward','overspeed','evidence_coverage','fallback_fraction','visible_gap_fraction',
             'omission_target_fraction','admission_count','alarm_count','attack_exposed','prediction_position_error_mean')
    return out,{k:float(np.mean([x[k] for x in out])) for k in metrics}


def run_navigation(cfg,device,root):
    agent,policy_meta=prepare(cfg,device,root)
    E,N,T=cfg['eval_worlds'],cfg['n_agents'],cfg['horizon'];seed=cfg['seed']
    nav=Navigation(E,N,seed_for(seed,'STRENGTHENING_dev_env'),device)
    try:
        model=MotionModel.from_navigation(nav)
        predictor=Predictor(model,device,cfg['prediction_horizon'],cfg['prediction_margin'])
        seed_all(seed_for(seed,'STRENGTHENING_dev_policy'))
        gov=BudgetController(cfg['method'],E,N,seed_for(seed,'STRENGTHENING_targets'),
                             policy_meta['calibration'],Authority(**cfg['authority']))
        channel=Transport(E,N,seed_for(seed,'STRENGTHENING_transport'),cfg['evidence'])
        ids=np.random.default_rng(seed_for(seed,'STRENGTHENING_attacker')).integers(0,N,size=E)
        random=np.random.default_rng(seed_for(seed,'STRENGTHENING_random'))
        obs=nav.obs;alive=np.ones(E,bool);initial=nav.diagnostics();traces={}
        def record(name,value):traces.setdefault(name,[]).append(np.array(value,copy=True))
        timings=[];elapsed=time.perf_counter()
        for t in range(T):
            before=nav.diagnostics();previous_force=nav.last_force.copy()
            proposal,_,_=agent.act(obs)
            if cfg['method']=='random_policy':proposal=random.uniform(0,1,(E*N,2)).astype(np.float32)
            proposal=(2*proposal-1).reshape(E,N,2);submitted=proposal.copy()
            attack_mask=np.zeros((E,N),bool)
            if cfg['attack']=='pursuit' and t>=cfg['attack_start']:
                for w,i in enumerate(ids):
                    delta=before['pos'][w]-before['pos'][w,i]
                    distance=np.linalg.norm(delta,axis=1);distance[i]=np.inf
                    direction=delta[distance.argmin()]
                    submitted[w,i]=direction/max(float(np.linalg.norm(direction)),1e-8)
                    attack_mask[w,i]=alive[w]
            incoming,expired=channel.commands(t,submitted,braking_command(before['vel']))
            # Diagnostics use a public-state model; no true future or attack labels.
            pre_cost,pre_gap=predictor.evaluate(before['pos'],before['vel'],previous_force,predictor.tensor(incoming)[:,None])
            event_pre=None
            if cfg['method']=='immediate_state':
                gain,cost=predictor.target_benefit(incoming,before,previous_force)
                event_pre=gov.immediate(t,gain,cost,alive)
            active=gov.active(t)&alive[:,None]
            if device.type=='cuda':torch.cuda.synchronize()
            tick=time.perf_counter()
            if cfg['filter']=='brake':
                executed=apply_brake_filter(incoming,active,before)
                filt={'predicted_cost_before':pre_cost[:,0].cpu().numpy(),
                      'predicted_clearance_before':pre_gap[:,0].cpu().numpy()}
                fc,fg=predictor.evaluate(before['pos'],before['vel'],previous_force,predictor.tensor(executed)[:,None])
                filt.update(predicted_cost_after=fc[:,0].cpu().numpy(),predicted_clearance_after=fg[:,0].cpu().numpy())
            else:executed,filt=predictor.filter(incoming,active,before,previous_force)
            if device.type=='cuda':torch.cuda.synchronize()
            timings.append(time.perf_counter()-tick)
            if np.any(np.linalg.norm(executed-incoming,axis=-1)[~active]>1e-7):raise AssertionError('Unauthorized command change')
            # Compare an a-priori one-step free-motion prediction to actual VMAS.
            pp,_,_=predictor.one_step(predictor.tensor(before['pos'])[:,None],
                    predictor.tensor(before['vel'])[:,None],predictor.tensor(previous_force)[:,None],predictor.tensor(executed)[:,None])
            nextobs,reward,done=nav.step(executed);after=nav.diagnostics()
            prediction_error=np.linalg.norm(pp[:,0].cpu().numpy()-after['pos'],axis=-1).max(-1)
            evidence=channel.evidence(t,after['risk'],attack_mask)
            benefit=None
            if cfg['method']=='adaptive_benefit_state':
                benefit,_=predictor.target_benefit(incoming,after,nav.last_force)
            if event_pre is None:
                event=gov.observe(t,evidence['controller_values'],alive,benefit=benefit,
                                  gaps=evidence['visible_gap'] if cfg['evidence']=='selective_gapaware' else None)
            else:event=event_pre
            for name,value in {'live':alive,'done':done,'position_before':before['pos'],'velocity_before':before['vel'],
                'position_after':after['pos'],'velocity_after':after['vel'],'goal_distance':after['goal_distance'],
                'goal_occupancy':after['on_goal'],'proposal':proposal,'submitted':submitted,'incoming':incoming,
                'executed':executed,'previous_force':previous_force,'applied_force':nav.last_force,
                'pre_risk':before['risk'],'risk':after['risk'],'speed':after['speed'],'contact':after['contact'],
                'reward':reward,'active':active,'selected':event['selected'],'alarm':event['trigger'],
                'denied':event['denied'],'abstain':event['abstain'],'scores':event['scores'],
                'score_coverage':event['coverage'],'cusum':event['cusum'],'threshold':event['threshold'],
                'evidence_received':evidence['received'],'evidence_controller':evidence['controller_values'],
                'evidence_valid':evidence['valid'],'evidence_generation':evidence['generation'],
                'command_generation':channel.command_generation,'visible_gap_generation':evidence['visible_gap_generation'],
                'visible_gap':evidence['visible_gap'],'suppressed':evidence['suppressed'],
                'omission_target':event['omission_target'],'fallback':expired,'attack_active':attack_mask.any(1),
                'prediction_position_error':prediction_error,**filt}.items():record(name,value)
            obs=nextobs;alive&=~done
        arrays={k:np.stack(v) for k,v in traces.items()}
        arrays.update(initial_goal_distance=initial['goal_distance'],goals=initial['goals'],
                      initial_contact=initial['contact'].any(1),attacker_ids=ids,radius=nav.radius,
                      filter_elapsed_s=np.asarray(timings),time_index=np.arange(T),
                      admission_delay_steps=np.array(0 if cfg['method']=='immediate_state' else 1))
        for name,a in arrays.items():
            if name not in ('evidence_received','evidence_controller') and not np.isfinite(a).all():
                raise FloatingPointError('Nonfinite trajectory: '+name)
        episodes,summary=summarize_trajectory(arrays,cfg['attack_start'])
        result={'config':cfg,'summary':summary,'episodes':episodes,'policy':policy_meta,
                'motion_model':asdict(model),'runtime_s':time.perf_counter()-elapsed,
                'filter_and_prediction_p50_ms':float(np.quantile(timings,.5)*1000),
                'filter_and_prediction_p95_ms':float(np.quantile(timings,.95)*1000),
                'timing_scope':'Synchronized filter plus prediction diagnostics. Not full-loop/hard-real-time performance.',
                'scope':'Development diagnostics, two independent fresh policies; no confirmatory p-values. Finite-candidate model filter is not certified.',
                'admission_delay_steps':0 if cfg['method']=='immediate_state' else 1}
        return result,arrays
    finally:nav.close()
