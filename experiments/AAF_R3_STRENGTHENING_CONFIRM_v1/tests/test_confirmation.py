"""Configuration/statistics/export tests; no synthetic results are scientific evidence."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('confirm_under_test',ROOT/'confirm.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def test_source_hashes():assert len(c.verify_source())==50

def test_fixed_plan_counts_and_disjoint_seeds():
    nav,g=c.plans();assert len(nav)==1640;assert sum(x['eval_worlds'] for x in nav)==52480
    assert len({c.digest(x) for x in nav})==1640
    assert sorted(set(x['seed'] for x in nav))==list(range(291000,291020))
    assert g['seeds']==list(range(291100,291120))
    assert len(g['seeds'])*len(g['domains'])*len(g['streams'])*len(g['budgets'])*len(g['selectors'])==1920

def test_methods_unchanged_in_every_condition():
    c.bind_base();from aaf_strengthen.protocol import plan,game_plan
    original=[x for x in plan('diagnostic') if x['seed']==281000]
    nav,g=c.plans();ignore={'seed','eval_worlds','profile'}
    for j in range(20):
        for a,b in zip(nav[j*82:(j+1)*82],original):
            assert {k:v for k,v in a.items() if k not in ignore}=={k:v for k,v in b.items() if k not in ignore}
    og=game_plan('diagnostic')
    assert {k:v for k,v in g.items() if k not in ('seed','seeds','profile')}=={k:v for k,v in og.items() if k not in ('seed','seeds','profile')}

def test_sign_test_all_ties():
    x=c.paired(np.zeros(20),np.zeros(20),'ties');assert x['sign_test_p']==1 and x['ties']==20
    assert x['bootstrap95_low']==x['bootstrap95_high']==0

def test_sign_test_all_positive():
    x=c.paired(np.arange(20.)+1,np.zeros(20),'positive')
    assert x['positive_pairs']==20 and x['sign_test_p']==pytest.approx(2/2**20)
    assert x['mean_paired_difference']==10.5

def test_balanced_signs_and_seed_resampling_deterministic():
    a=np.r_[-np.arange(1,11),np.arange(1,11)]
    x=c.paired(a,np.zeros(20),'balanced');assert x['sign_test_p']==1
    assert x==c.paired(a,np.zeros(20),'balanced')

def test_incomplete_or_nonfinite_pairs_rejected():
    with pytest.raises(ValueError):c.paired(np.zeros(19),np.zeros(19),'bad')
    with pytest.raises(ValueError):c.paired(np.r_[np.zeros(19),np.nan],np.zeros(20),'bad')

def test_holm_includes_null_tests():
    assert np.allclose(c.holm([.01,.03,.2,1]),[.04,.09,.4,1])
    with pytest.raises(ValueError):c.holm([np.nan])

def test_primary_count_and_not_just_favorable_endpoints():
    p=c.PROTOCOL
    assert len(p['primary_navigation']['contrasts'])*len(p['primary_navigation']['metrics'])+2*3==30
    assert {'any_contact','contact_time','goal_progress','modified_fraction'}==set(p['primary_navigation']['metrics'])

def test_every_primary_cell_exists_once_per_seed():
    import pandas as pd
    nav,_=c.plans();df=pd.DataFrame([{**x,'top_k':x['authority']['top_k']} for x in nav])
    for cc in c.PROTOCOL['primary_navigation']['contrasts']:
        for side in ('left','right'):
            s=df[(df.attack=='pursuit')&(df.top_k==1)]
            for k,v in cc[side].items():s=s[s[k]==v]
            assert len(s)==20 and sorted(s.seed.tolist())==c.PROTOCOL['navigation_seeds']

def test_review_slice_is_fixed_and_preserves_every_time_step():
    a={'live':np.ones((5,32),bool),'position_after':np.arange(5*32*4*2).reshape(5,32,4,2),
       'goals':np.zeros((32,4,2)),'attacker_ids':np.arange(32),'initial_contact':np.zeros(32,bool),
       'radius':np.ones(4),'time_index':np.arange(5),'admission_delay_steps':np.array(1)}
    b=c.slice_episode0(a);assert b['position_after'].shape==(5,1,4,2)
    assert np.array_equal(b['position_after'][:,0],a['position_after'][:,0])
    assert b['goals'].shape==(1,4,2) and b['radius'].shape==(4,)
    assert int(b['review_original_world_count'])==32 and int(b['review_original_world_index'])==0

def test_complete_analysis_wiring_with_synthetic_fixture(tmp_path,monkeypatch):
    """Only an analysis plumbing fixture; never a simulator evaluation."""
    import pandas as pd
    c.bind_base();import aaf_strengthen.analysis as analyzer
    nav,g=c.plans();out=tmp_path/'analysis_fixture';folder=out/'analysis';folder.mkdir(parents=True)
    (out/'RUN_MANIFEST.json').write_text(json.dumps({'navigation_plan':nav,'game_plan':g}))
    def fake_validate(root):
        rows=[]
        for cfg in nav:
            rows.append({**{k:cfg[k] for k in ('seed','block','method','filter','evidence','attack')},'top_k':cfg['authority']['top_k'],
               'success':.5,'any_contact':.5,'contact_time':.25,'goal_progress':1.,'modified_fraction':.1})
        pd.DataFrame(rows).to_csv(folder/'navigation_seed_means.csv',index=False)
        gs=[]
        for seed in g['seeds']:
            for domain in g['domains']:
                for stream in g['streams']:
                    for k in g['budgets']:
                        for selector in g['selectors']:
                            gs.append(dict(seed=seed,domain=domain,stream=stream,top_k=k,selector=selector,direct_reduction=.01))
        pd.DataFrame(gs).to_csv(folder/'game_replay_capacity.csv',index=False)
        pd.DataFrame({'fixture_row':np.arange(52480)}).to_csv(folder/'navigation_episode_timings.csv',index=False)
        (folder/'DIAGNOSTIC_REPORT.md').write_text('synthetic wiring fixture')
        (folder/'RECORD_AUDIT.json').write_text(json.dumps({'status':'PASS'}))
    monkeypatch.setattr(analyzer,'analyze',fake_validate)
    c.analyze(out)
    contrast=pd.read_csv(folder/'PRIMARY_CONTRASTS.csv')
    assert len(contrast)==30 and (contrast.holm_p_30_tests==1).all()
    assert (contrast.ties==20).all()
    assert not (folder/'DIAGNOSTIC_REPORT.md').exists()
    assert json.loads((folder/'CONFIRM_ANALYSIS_COMPLETE.json').read_text())['status']=='PASS'

def test_review_export_completeness_and_hashes_with_fixture(tmp_path,monkeypatch):
    """Synthetic serialization fixture, not experimental data."""
    import zipfile,hashlib
    c.bind_base();nav,_=c.plans();nav=nav[:2]
    home=tmp_path/'package';out=home/'results/confirmation';(out/'analysis').mkdir(parents=True)
    (home/'CONFIRM_PROTOCOL.json').write_text(json.dumps(c.PROTOCOL))
    (out/'analysis/CONFIRM_ANALYSIS_COMPLETE.json').write_text(json.dumps({'status':'PASS'}))
    monkeypatch.setattr(c,'ROOT',home);monkeypatch.setattr(c,'plans',lambda:(nav,{}));monkeypatch.setattr(c,'verify_source',lambda:{})
    for cfg in nav:
        p=out/'runs'/c.digest(cfg)[:20];p.mkdir(parents=True)
        with (p/'trajectories.npz').open('wb') as f:
            np.savez_compressed(f,live=np.ones((4,32),bool),position_after=np.zeros((4,32,4,2)),goals=np.zeros((32,4,2)))
        (p/'summary.json').write_text(json.dumps({'scope':'synthetic fixture','episodes':list(range(32))}))
    dest=home/'review.zip';c.pack(out,dest)
    with zipfile.ZipFile(dest) as z:
        assert not any(n.endswith('/trajectories.npz') for n in z.namelist())
        assert len([n for n in z.namelist() if '/review_traces/' in n])==2
        assert len([n for n in z.namelist() if n.endswith('summary.json')])==2
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            h,n=line.split(maxsplit=1);assert hashlib.sha256(z.read(n)).hexdigest()==h
    assert dest.with_suffix('.zip.sha256').read_text().split()[0]==c.file_sha(dest)
