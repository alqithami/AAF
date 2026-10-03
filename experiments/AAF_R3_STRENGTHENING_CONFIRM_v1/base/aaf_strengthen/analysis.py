"""Complete-condition diagnostics; deliberately no significance tests on two seeds."""
from __future__ import annotations
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from aaf_r3.common import atomic_json,digest
from .io import load_completed,sha
from .navigation import summarize_trajectory


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def audit_arrays(a,cfg):
    live=a['live'].astype(bool);T,E=live.shape;N=cfg['n_agents'];auth=cfg['authority']
    expected=np.zeros((T,E,N),bool)
    method=cfg['method'];delay=0 if method=='immediate_state' else 1
    if method=='static_guard':expected[:]=live[:,:,None]
    elif method not in ('ppo_only','random_policy'):
        for e in range(E):
            starts=[]
            for t in range(T):
                ids=np.flatnonzero(a['selected'][t,e])
                if len(ids)>auth['top_k']:raise AssertionError('Too many selected targets')
                if not len(ids):continue
                s=t+delay
                if starts and s<starts[-1]+auth['horizon']+auth['cooldown']:raise AssertionError('Cooldown violation')
                if sum(x>s-auth['window'] for x in starts)>=auth['tokens']:raise AssertionError('Token limit violation')
                starts.append(s);expected[s:min(T,s+auth['horizon']),e,ids]=True
        expected&=live[:,:,None]
    if not np.array_equal(a['active'].astype(bool),expected):raise AssertionError('Active time does not reconstruct from admissions')
    modified=np.linalg.norm(a['executed']-a['incoming'],axis=-1)>1e-7
    if np.any(modified&~expected):raise AssertionError('Unauthorized control')
    # Authenticate the trace's geometric endpoint using the actual recorded states.
    positions=a['position_after'];r=a['radius']
    dist=np.linalg.norm(positions[:,:,:,None,:]-positions[:,:,None,:,:],axis=-1)
    contact=dist<=(r[:,None]+r[None,:]+.005)[None,None]
    contact[:,:,np.arange(N),np.arange(N)]=False
    if not np.array_equal(contact.any(-1),a['contact'].astype(bool)):raise AssertionError('Contact geometry mismatch')
    return {'episodes':E,'live_steps':int(live.sum()),'authority':'PASS','contact_geometry':'PASS'}


def analyze(root):
    root=Path(root);manifest=json.loads((root/'RUN_MANIFEST.json').read_text());folder=root/'analysis';folder.mkdir(exist_ok=True)
    rows=[];eps=[];checks=[]
    for cfg in manifest['navigation_plan']:
        path=root/'runs'/digest(cfg)[:20];record=load_completed(path,cfg)
        if record is None:raise RuntimeError('Analysis refuses incomplete navigation treatments')
        with np.load(path/'trajectories.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        checks.append(audit_arrays(a,cfg))
        episodes,summary=summarize_trajectory(a,cfg['attack_start'])
        for k,v in summary.items():
            if not np.isclose(record['summary'][k],v,rtol=1e-6,atol=1e-8):raise AssertionError('Summary does not match trajectory: '+k)
        descriptor={k:cfg[k] for k in ('seed','block','method','filter','evidence','attack')};descriptor['top_k']=cfg['authority']['top_k']
        rows.append({**descriptor,**summary})
        eps.extend({**descriptor,**e} for e in episodes)
    write_csv(folder/'navigation_seed_means.csv',rows);write_csv(folder/'navigation_episode_timings.csv',eps)
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in ('block','method','filter','evidence','attack','top_k'))].append(r)
    means=[]
    for key,group in sorted(grouped.items()):
        r=dict(zip(('block','method','filter','evidence','attack','top_k'),key));r['independent_policy_seeds']=len(group)
        for k in ('success','any_contact','contact_time','modified_fraction','active_fraction','goal_progress','evidence_coverage'):
            r[k]=float(np.mean([x[k] for x in group]));r[k+'_seed_min']=min(x[k] for x in group);r[k+'_seed_max']=max(x[k] for x in group)
        means.append(r)
    write_csv(folder/'navigation_descriptive_means.csv',means)
    timing=[]
    for key,group in grouped.items():
        selected=[e for e in eps if tuple(e[k] for k in ('block','method','filter','evidence','attack','top_k'))==key]
        contacts=[e for e in selected if e['first_contact_action_index'] is not None]
        before=sum(e['first_modified_action_index'] is None or e['first_contact_action_index']<e['first_modified_action_index'] for e in contacts)
        timing.append({**dict(zip(('block','method','filter','evidence','attack','top_k'),key)),
                       'episodes':len(selected),'episodes_contact':len(contacts),
                       'contact_before_first_modification_or_no_modification':before,
                       'episodes_without_alarm':sum(e['first_alarm_action_index'] is None for e in selected),
                       'episodes_without_modification':sum(e['first_modified_action_index'] is None for e in selected),
                       'attack_exposed_episodes':sum(e['attack_exposed'] for e in selected)})
    write_csv(folder/'timing_failure_counts.csv',timing)
    gpath=root/'game_replay'/'results.json';game=[]
    if manifest['game_plan'] is not None:
        marker=json.loads((root/'game_replay'/'COMPLETE.json').read_text())
        if sha(gpath)!=marker['results_sha256'] or sha(root/'game_replay'/'proposal_streams.npz')!=marker['stream_sha256']:
            raise RuntimeError('Game replay checksum mismatch')
        game=json.loads(gpath.read_text())['results']
        write_csv(folder/'game_replay_capacity.csv',[{k:v for k,v in r.items() if k!='fixed_schedule_starts'} for r in game])
    report=['# AAF strengthening development diagnostics','',
        '**Not confirmatory evidence. No p-values, favorable-seed selection, or automatic main-study launch.**','',
        f'Navigation configurations: {len(rows)}. Episode records: {len(eps)}.',
        f'Game replay configurations: {len(game)}. These are fixed-proposal diagnostics, not online-learning treatments.','',
        '## Nominal policy competence (all seeds retained)','',
        '| Seed | Clean PPO completion | Clean PPO any contact | Goal progress |','|---|---:|---:|---:|']
    for r in rows:
        if r['method']=='ppo_only' and r['attack']=='none':
            report.append(f'| {r["seed"]} | {r["success"]:.3f} | {r["any_contact"]:.3f} | {r["goal_progress"]:.4f} |')
    report += ['', '## Read these files together','',
        '`navigation_seed_means.csv`: all methods/conditions, each policy seed separately.',
        '`navigation_episode_timings.csv`: first risk/alarm/admission/modification/contact; null means absent, not zero delay.',
        '`timing_failure_counts.csv`: contact before first intervention and missing-alarm counts.',
        '`game_replay_capacity.csv`: direct suppression, actual authorized time, and fixed-stream hindsight maximum.',
        '', '## Decision questions before a confirmatory protocol','',
        '1. Does better state-informed targeting or immediate response create genuine opportunity under the same cap?',
        '2. Does a shared predictive filter improve actual contact outcomes and retain task progress, not merely its own model score?',
        '3. Does AAF offer a protection/burden advantage over threshold and periodic baselines across authority settings?',
        '4. Does evidence assurance improve downstream outcomes, while visible gaps remain omission reasons rather than blame?',
        '5. Where is the predictor inaccurate, and do contacts precede the first feasible response?',
        '', 'No gain is assumed. Stop for interpretation; do not scale until protocol and method choices are frozen.',
        'The state-model diagnostic is NOT an optimal simulator oracle. Replay hindsight is an exact maximum ONLY for its recorded proposals and fixed disjoint windows.',
        'Full physical traces and final nominal policy weights are included in the return package.']
    (folder/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(report)+'\n')
    atomic_json(folder/'RECORD_AUDIT.json',{'status':'PASS','navigation_configurations':len(rows),'episodes':len(eps),
        'game_replay_configurations':len(game),'checks':checks,'statistical_status':'development only'})
    print('Analysis complete: '+str(folder/'DIAGNOSTIC_REPORT.md'),flush=True)
