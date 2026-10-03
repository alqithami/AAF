#!/usr/bin/env python3
"""Frozen held-out confirmation of the reviewed strengthening-v1 mechanisms.

No installation, baseline edits, outcome-dependent exclusions, or remote API calls.
The main command is intentionally separate from the development-only baseline CLI.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys, time, traceback, zipfile
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'base'
PROTOCOL=json.loads((ROOT/'CONFIRM_PROTOCOL.json').read_text())


def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def json_write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_source():
    expected=json.loads((ROOT/'BASELINE_HASHES.json').read_text())
    for name,h in expected.items():
        p=BASE/name
        if not p.is_file() or file_sha(p)!=h:raise RuntimeError('Baseline source differs: '+name)
    ledger=ROOT/'PACKAGE_SHA256SUMS.txt'
    if ledger.exists():
        for ln in ledger.read_text().splitlines():
            h,name=ln.split(maxsplit=1)
            if file_sha(ROOT/name)!=h:raise RuntimeError('Confirmation package changed: '+name)
    return expected


def bind_base():
    for p in [BASE,BASE/'vendor']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))


def plans():
    bind_base()
    from aaf_strengthen.protocol import plan,game_plan
    template=[x for x in plan('diagnostic') if x['seed']==281000]
    nav=[]
    for seed in PROTOCOL['navigation_seeds']:
        for original in template:
            c=copy.deepcopy(original);c.update(seed=seed,profile='confirmation_v1',eval_worlds=PROTOCOL['navigation_eval_worlds']);nav.append(c)
    game=game_plan('diagnostic');game.update(profile='confirmation_v1',seeds=PROTOCOL['game_seeds'])
    if len(nav)!=PROTOCOL['navigation_evaluations']:raise RuntimeError('Plan count changed')
    if sum(c['eval_worlds'] for c in nav)!=PROTOCOL['navigation_episode_records']:raise RuntimeError('Episode count changed')
    return nav,game


def self_identity():
    files=[p for p in ROOT.rglob('*') if p.is_file() and
        ('base' in p.relative_to(ROOT).parts or p.suffix in ('.py','.json','.sh','.md')) and
        not any(t in p.relative_to(ROOT).parts for t in ('__pycache__','results','logs','.pytest_cache','runtime-storage'))]
    return {str(p.relative_to(ROOT)):file_sha(p) for p in sorted(files)}


def run(out):
    verify_source();bind_base()
    import importlib.metadata as im
    for name,version in PROTOCOL['dependencies'].items():
        if im.version(name)!=version:raise RuntimeError(f'{name}: expected {version}, found {im.version(name)}. Do not upgrade this study in place.')
    from aaf_strengthen.io import configure_device,environment,load_completed,save_completed
    from aaf_strengthen.preflight import check
    from aaf_r3.cli import output_lock
    from aaf_strengthen.navigation import run_navigation
    from aaf_strengthen.game_replay import run_game_diagnostics
    device=configure_device('cuda',12.0);nav,game=plans();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    identity={'version':PROTOCOL['version'],'protocol_sha256':file_sha(ROOT/'CONFIRM_PROTOCOL.json'),
      'source':self_identity(),'environment':environment(device),'plan_sha256':digest({'navigation':nav,'game':game})}
    with output_lock(out):
        mf=out/'RUN_MANIFEST.json'
        if mf.exists():
            if json.loads(mf.read_text())['identity']!=identity:raise RuntimeError('Refusing mixed source, protocol, or environment in existing output.')
        else:json_write(mf,{'identity':identity,'navigation_plan':nav,'game_plan':game,
             'created_utc':datetime.now(timezone.utc).isoformat(),'statistical_status':'Held-out confirmation; frozen after separate development data.'})
        start=time.perf_counter();json_write(out/'PREFLIGHT.json',check(device,True))
        marker=out/'game_replay/COMPLETE.json'
        if marker.exists():
            z=json.loads(marker.read_text())
            if file_sha(marker.parent/'results.json')!=z['results_sha256'] or file_sha(marker.parent/'proposal_streams.npz')!=z['stream_sha256']:
                raise RuntimeError('Game replay integrity failure; no overwrite or deletion.')
            if json.loads((marker.parent/'results.json').read_text())['config']!=game:raise RuntimeError('Game configuration differs')
            print('REUSE completed game replay study',flush=True)
        else:run_game_diagnostics(game,device,out)
        made=reused=0
        for index,cfg in enumerate(nav):
            folder=out/'runs'/digest(cfg)[:20]
            if load_completed(folder,cfg) is not None:
                reused+=1;print(f'REUSE {index+1}/{len(nav)}',flush=True);continue
            print(f'RUN {index+1}/{len(nav)} seed={cfg["seed"]} {cfg["method"]} k={cfg["authority"]["top_k"]} {cfg["filter"]} {cfg["attack"]} {cfg["evidence"]}',flush=True)
            result,arrays=run_navigation(cfg,device,out)
            # Replace ONLY the inherited, hard-coded two-seed scope sentence.
            result['policy']['scope']='Held-out fixed final nominal-policy checkpoint; no selection by intervention results.'
            json_write(out/'policies'/('nominal_'+digest(result['policy']['key'])[:16]+'.json'),result['policy'])
            result['scope']='Held-out confirmation: 20 independent nominal navigation policies, 32 episodes/configuration. Shared finite-candidate filter is not certified; state-informed arms are privileged diagnostics.'
            save_completed(folder,result,arrays);made+=1
        json_write(out/'RUN_COMPLETE.json',{'status':'complete','navigation_planned':len(nav),
            'navigation_completed_now':made,'navigation_reused':reused,'elapsed_this_invocation_s':time.perf_counter()-start,
            'scope':'Execution completion only; scientific conclusions require the full prespecified analysis.'})


def holm(pvalues):
    import numpy as np
    p=np.asarray(pvalues,float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid p-value')
    order=np.argsort(p);ans=np.empty(len(p));high=0.0
    for j,k in enumerate(order):
        high=max(high,min(1.,(len(p)-j)*p[k]));ans[k]=high
    return ans


def paired(left,right,name):
    import numpy as np
    from scipy.stats import binomtest
    a=np.asarray(left,float);b=np.asarray(right,float)
    if len(a)!=20 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Exactly 20 finite paired policy seeds are required')
    d=a-b;positive=int(np.sum(d>1e-12));negative=int(np.sum(d< -1e-12));nonzero=positive+negative
    p=float(binomtest(positive,nonzero,.5,alternative='two-sided').pvalue) if nonzero else 1.
    rng=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
    boots=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
    return {'independent_seed_pairs':len(d),'left_mean':float(a.mean()),'right_mean':float(b.mean()),
        'mean_paired_difference':float(d.mean()),'bootstrap95_low':float(np.quantile(boots,.025)),'bootstrap95_high':float(np.quantile(boots,.975)),
        'positive_pairs':positive,'negative_pairs':negative,'ties':int(len(d)-nonzero),'sign_test_p':p,
        'test_scope':'Exact paired sign test of equal sign probabilities, not specifically a population-mean test; intervals marginal.'}


def analyze(out):
    verify_source();bind_base();out=Path(out).resolve()
    import numpy as np,pandas as pd
    from aaf_strengthen.analysis import analyze as validate_all_records
    m=json.loads((out/'RUN_MANIFEST.json').read_text());nav,game=plans()
    if m['navigation_plan']!=nav or m['game_plan']!=game:raise RuntimeError('Output differs from frozen confirmation plan')
    # Inherited analysis reopens every full trajectory and checks all configuration
    # summaries, contact geometry, and intervention windows before inference.
    validate_all_records(out)
    folder=out/'analysis'
    (folder/'DIAGNOSTIC_REPORT.md').unlink() # New output only; replaces inappropriate inherited development heading.
    v=json.loads((folder/'RECORD_AUDIT.json').read_text());v['statistical_status']='Frozen held-out confirmation; see CONFIRM_PROTOCOL.json.';json_write(folder/'RECORD_AUDIT.json',v)
    df=pd.read_csv(folder/'navigation_seed_means.csv');gf=pd.read_csv(folder/'game_replay_capacity.csv');rows=[]
    if len(df)!=1640 or len(gf)!=1920:raise RuntimeError('Incomplete main-study table')
    def select(frame,criteria,seeds):
        z=frame
        for k,v in criteria.items():z=z[z[k].eq(v)]
        if len(z)!=len(seeds) or sorted(z.seed.tolist())!=seeds:raise RuntimeError('Missing or duplicated matched seeds: '+str(criteria))
        return z.sort_values('seed')
    for domain in PROTOCOL['primary_game']['domains']:
        for k in PROTOCOL['primary_game']['top_k']:
            crit={'domain':domain,'stream':'frozen_ppo_replay','top_k':k}
            left=select(gf,{**crit,'selector':'recent_score'},PROTOCOL['game_seeds'])
            right=select(gf,{**crit,'selector':'random'},PROTOCOL['game_seeds'])
            name=f'game:{domain}:k{k}:recent_minus_random'
            rows.append({'contrast':name,'suite':'game_replay','metric':'direct_reduction','better_direction':'positive',**paired(left.direct_reduction,right.direct_reduction,name)})
    for contrast in PROTOCOL['primary_navigation']['contrasts']:
        crit={'attack':'pursuit','top_k':1}
        left=select(df,{**crit,**contrast['left']},PROTOCOL['navigation_seeds'])
        right=select(df,{**crit,**contrast['right']},PROTOCOL['navigation_seeds'])
        for metric in PROTOCOL['primary_navigation']['metrics']:
            name=f'navigation:{contrast["name"]}:pursuit:k1:{metric}'
            rows.append({'contrast':name,'suite':'navigation','metric':metric,'better_direction':'positive' if metric=='goal_progress' else 'negative',**paired(left[metric],right[metric],name)})
    assert len(rows)==30
    adj=holm([r['sign_test_p'] for r in rows])
    for r,p in zip(rows,adj):r['holm_p_30_tests']=float(p)
    pd.DataFrame(rows).to_csv(folder/'PRIMARY_CONTRASTS.csv',index=False)
    json_write(folder/'PRIMARY_CONTRASTS.json',rows)
    eps=pd.read_csv(folder/'navigation_episode_timings.csv')
    nominal=df[(df.method=='ppo_only')&(df.attack=='none')]
    if len(eps)!=52480 or sorted(nominal.seed.tolist())!=PROTOCOL['navigation_seeds']:raise RuntimeError('Incomplete episodes or nominal policy check')
    json_write(folder/'CONFIRM_ANALYSIS_COMPLETE.json',{'status':'PASS','primary_tests':30,'navigation_evaluations':1640,'navigation_episode_records':52480,'game_replays':1920,
        'policy_seeds':PROTOCOL['navigation_seeds'],'all_seeds_retained':True,'source_sha256':identity_sha()})
    lines=['# Held-out AAF strengthening confirmation','',
      'Execution and record validation passed. Completion and statistical significance do not guarantee practical advantage.',
      '','20 independent navigation policy seeds; 32 evaluation worlds/configuration. Game domains each use 20 separately trained seeds.',
      '','## Nominal policy competence — all policies retained','',
      '| Seed | Clean PPO completion | Any contact | Goal progress |','|---|---:|---:|---:|']
    for r in nominal.sort_values('seed').to_dict('records'):lines.append(f'| {r["seed"]} | {r["success"]:.4f} | {r["any_contact"]:.4f} | {r["goal_progress"]:.4f} |')
    lines+=['','## Prespecified inference','',
      'PRIMARY_CONTRASTS.csv includes all 30 primary tests jointly Holm-adjusted. Exact paired sign tests concern directional consistency, not specifically equality of population means. Mean differences and paired-seed percentile bootstrap intervals are reported separately; intervals are marginal, not multiplicity-adjusted.',
      '','All other conditions remain in navigation_seed_means.csv, navigation_descriptive_means.csv, navigation_episode_timings.csv and game_replay_capacity.csv. Do not pool episodes as independent policy replications or replace contact incidence with a more favorable endpoint.',
      '','Any-contact incidence, contact duration, task progress and actual command modification are all primary navigation endpoints for this NEW study. The earlier R3 primary endpoint remains unchanged. No non-inferiority or optimal-frontier claim is authorized.',
      '','Full physical traces remain in results/confirmation/runs. The review archive contains every summary and all policy weights, plus a fixed world-index-0 trace from each configuration. No outcome-selected trajectory export.']
    (folder/'CONFIRMATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Confirmation analysis complete. Review the actual effect sizes and adverse outcomes.',flush=True)


def identity_sha():return digest(self_identity())


def slice_episode0(a):
    """Deterministic compact review export, all configurations, no outcome selection."""
    out={};T,E=a['live'].shape
    initial={'initial_goal_distance','goals','initial_contact','attacker_ids'}
    for k,x in a.items():
        if k in initial:out[k]=x[:1]
        elif x.ndim>=2 and x.shape[:2]==(T,E):out[k]=x[:,:1]
        else:out[k]=x
    out['review_original_world_index']=__import__('numpy').array(0)
    out['review_original_world_count']=__import__('numpy').array(E)
    return out


def pack(out,dest):
    verify_source();bind_base();out=Path(out).resolve();dest=Path(dest).resolve()
    import numpy as np
    from aaf_strengthen.io import atomic_npz
    if json.loads((out/'analysis/CONFIRM_ANALYSIS_COMPLETE.json').read_text()).get('status')!='PASS':raise RuntimeError('Completed analysis required')
    nav,_=plans();small=out/'review_traces';small.mkdir(exist_ok=True)
    for c in nav:
        key=digest(c)[:20]
        with np.load(out/'runs'/key/'trajectories.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        atomic_npz(small/(key+'.npz'),slice_episode0(a))
    allfiles=[p for p in out.rglob('*') if p.is_file() and p.name not in ('FULL_SHA256SUMS.txt','REVIEW_SHA256SUMS.txt') and p.suffix not in ('.tmp','.lock')]
    ledger=out/'FULL_SHA256SUMS.txt';ledger.write_text(''.join(f'{file_sha(p)}  {p.relative_to(out)}\n' for p in sorted(allfiles)))
    chosen=[p for p in allfiles if p.name!='trajectories.npz']+[ledger]
    code=[p for p in ROOT.rglob('*') if p.is_file() and not any(t in p.relative_to(ROOT).parts for t in ('results','logs','__pycache__','.pytest_cache','runtime-storage')) and p.suffix not in ('.zip','.sha256','.pyc','.tmp','.lock') and not p.name.startswith('RUN.')]
    entries=[(p,'results/'+str(p.relative_to(out))) for p in chosen]+[(p,'source/'+str(p.relative_to(ROOT))) for p in code]
    logroot=ROOT/'logs'
    if logroot.exists():
        entries += [(p,'execution_logs/'+p.name) for p in logroot.iterdir() if p.is_file() and p.suffix in ('.log','.json','.xml','.txt')]
    hashes=''.join(f'{file_sha(p)}  {name}\n' for p,name in sorted(entries,key=lambda t:t[1]))
    dest.parent.mkdir(exist_ok=True,parents=True);tmp=dest.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p,name in entries:z.write(p,name)
        z.writestr('SHA256SUMS.txt',hashes)
        z.writestr('REVIEW_ARCHIVE_SCOPE.txt','All summaries and policy weights are included. Each review_traces file contains only evaluation world index 0, chosen in advance for EVERY configuration. Full 32-world physical arrays remain on the IBM server; FULL_SHA256SUMS.txt identifies them. This is not the full raw trajectory archive.\n')
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Archive CRC failure: '+bad)
    os.replace(tmp,dest);dest.with_suffix(dest.suffix+'.sha256').write_text(f'{file_sha(dest)}  {dest.name}\n')
    print(f'RETURN {dest} ({dest.stat().st_size/1024**2:.1f} MiB). Keep all server data.',flush=True)


def failure(out):
    dest=ROOT/'AAF_R3_CONFIRMATION_FAILURE.zip';out=Path(out)
    paths=[]
    for folder in [ROOT/'logs',out]:
        if folder.exists():paths += [p for p in folder.iterdir() if p.is_file() and p.suffix in ('.log','.txt','.json','.xml')]
    paths += [ROOT/'CONFIRM_PROTOCOL.json',ROOT/'BASELINE_HASHES.json']
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(ROOT)))
    print('Return failure bundle:',dest)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['verify','plan','run','analyze','pack','failure']);ap.add_argument('--out',type=Path,default=ROOT/'results/confirmation');ap.add_argument('--zip',type=Path,default=ROOT/'AAF_R3_CONFIRMATION_REVIEW.zip');a=ap.parse_args()
    try:
        if a.action=='verify':print('Source verification PASS:',len(verify_source()),'unchanged baseline files')
        elif a.action=='plan':
            verify_source();n,g=plans();print(json.dumps({'version':PROTOCOL['version'],'navigation_evaluations':len(n),'episodes':sum(c['eval_worlds'] for c in n),'game_comparisons':len(g['seeds'])*2*4*3*4,'navigation_seeds':PROTOCOL['navigation_seeds'],'game_seeds':PROTOCOL['game_seeds'],'primary_tests':30},indent=2))
        elif a.action=='run':run(a.out)
        elif a.action=='analyze':analyze(a.out)
        elif a.action=='pack':pack(a.out,a.zip)
        else:failure(a.out)
    except Exception as exc:
        if a.action not in ('verify','plan'):json_write(a.out/'LAST_ERROR.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':main()
