import numpy as np
import pytest
import torch
from dataclasses import asdict
from aaf_r3.governor import Governor,Authority,calibrate
from aaf_strengthen.controller import BudgetController
from aaf_strengthen.evidence import Transport
from aaf_strengthen.predictive import Predictor,MotionModel
from aaf_strengthen.game_replay import replay,scripts,fixed_windows
from aaf_strengthen.protocol import plan,game_plan
from aaf_strengthen.io import verify_immutable_vendor,save_completed,load_completed

@pytest.mark.parametrize('method',['adaptive_rank','adaptive_random','threshold_rank','periodic_rank'])
def test_original_controller_parity(method):
    cal=calibrate(np.linspace(0,.2,128).tolist());a=Authority(top_k=2,horizon=7,cooldown=3,window=45,tokens=3,history=10)
    old=Governor(method,3,6,152,cal,a);new=BudgetController(method,3,6,152,cal,a)
    rng=np.random.default_rng(162)
    for t in range(500):
        assert np.array_equal(old.active(t),new.active(t))
        x=(rng.random((3,6)) < (.15 if t<70 else .7)).astype(float)
        x[rng.random((3,6))<.1]=np.nan
        o=old.observe(t,x);p=new.observe(t,x)
        for k in ('trigger','selected','scores','coverage'):np.testing.assert_allclose(o[k],p[k])

def test_zero_score_abstention_and_next_step():
    a=Authority(top_k=1,horizon=2,cooldown=1,window=3,tokens=1,history=2)
    c=BudgetController('periodic_rank',1,4,4,calibrate([0.]*8),a)
    for t in range(3):c.observe(t,np.zeros((1,4)))
    assert not c.active(3).any()
    for t in range(3,6):event=c.observe(t,np.ones((1,4)))
    assert event['selected'].sum()==1
    assert not c.active(5).any() and c.active(6).sum()==1

def test_gap_is_delayed_not_payload():
    c=Transport(1,4,1,'selective_gapaware');x=np.ones((1,4));m=np.array([[1,0,0,0]],bool)
    a=c.evidence(0,x,m);assert np.isnan(a['received'][0,0]) and not a['visible_gap'].any()
    b=c.evidence(1,x,m);assert b['visible_gap'][0,0] and np.isnan(b['received'][0,0])

def test_naive_and_forgery_are_labeled_distinct():
    m=np.array([[1,0,0,0]],bool);x=np.ones((1,4))
    f=Transport(1,4,1,'forged_self_report').evidence(0,x,m)
    n=Transport(1,4,1,'selective_naive').evidence(0,x,m)
    assert f['received'][0,0]==0 and f['valid'][0,0]
    assert np.isnan(n['received'][0,0]) and not n['valid'][0,0] and n['controller_values'][0,0]==0

def test_permanent_suppression_has_no_invented_checkpoint():
    c=Transport(1,4,1,'permanent_suppression')
    for t in range(10):
        r=c.evidence(t,np.ones((1,4)),np.array([[1,0,0,0]],bool))
        assert not r['visible_gap'].any()

def test_omission_control_is_labeled_not_fake_blame():
    a=Authority(top_k=1,horizon=2,cooldown=1,window=5,tokens=1)
    c=BudgetController('adaptive_rank',1,4,4,calibrate([0.]*8),a)
    x=np.zeros((1,4));x[0,2]=np.nan;gap=np.array([[0,0,1,0]],bool)
    r=c.observe(0,x,gaps=gap)
    assert r['selected'][0,2] and r['omission_target'][0,2]
    assert np.isnan(c.hist[0,0,2])

@pytest.mark.parametrize('profile',['stale_evidence_only','stale_commands_only'])
def test_evidence_command_failure_separation(profile):
    c=Transport(2,4,51,profile);u=np.ones((2,4,2));brake=np.zeros_like(u)
    received,expired=c.commands(0,u,brake)
    e=c.evidence(0,np.ones((2,4)),np.zeros((2,4),bool))
    if profile=='stale_evidence_only':assert not expired.any() and np.array_equal(received,u)
    else:assert e['valid'].all()

def example_predictor():return Predictor(MotionModel(.1,2,(1.,1.,1.,1.),(.25,)*4,(.1,)*4),torch.device('cpu'))

def test_predictor_free_motion_equation():
    q=example_predictor();p=torch.zeros((1,1,4,2));v=torch.zeros_like(p);f=torch.zeros_like(p);u=torch.ones_like(p)
    pp,vv,ff=q.one_step(p,v,f,u)
    assert torch.allclose(ff,torch.full_like(f,.5))
    assert torch.allclose(vv,torch.full_like(f,.05))
    assert torch.allclose(pp,torch.full_like(f,.00375))

def test_predictor_authorization_and_nonincrease():
    p=np.array([[[-.3,0],[.3,0],[0,1],[0,-1]]],np.float32)
    v=np.array([[[.2,0],[-.2,0],[0,0],[0,0]]],np.float32);u=v*3
    q=example_predictor();active=np.array([[1,0,0,0]],bool)
    out,info=q.filter(u,active,{'pos':p,'vel':v},np.zeros_like(p))
    assert np.array_equal(out[~active],u[~active]) and np.max(abs(out))<=1
    assert np.all(info['predicted_cost_after']<=info['predicted_cost_before']+1e-7)

def test_no_authority_predictive_identity():
    q=example_predictor();p=np.array([[[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]]],np.float32)
    u=np.random.default_rng(1).uniform(-1,1,p.shape).astype(np.float32)
    out,_=q.filter(u,np.zeros((1,4),bool),{'pos':p,'vel':u*.1},u*.2)
    assert np.array_equal(out,u)

def test_benefit_nonnegative_no_labels():
    q=example_predictor();p=np.array([[[-.3,0],[.3,0],[0,1],[0,-1]]],np.float32);u=np.zeros_like(p)
    b,c=q.target_benefit(u,{'pos':p,'vel':u},u)
    assert b.shape==(1,4) and np.all(b>=0) and np.allclose(b,0)

@pytest.mark.parametrize('domain',['resource_sharing','public_goods'])
@pytest.mark.parametrize('kind',['scripted_sparse','scripted_rotating','scripted_diffuse'])
def test_replay_hindsight_and_capacity(domain,kind):
    a=scripts(domain,kind,182,256,50,40)
    oracle,_,_=replay(domain,a,'future_window_diagnostic',3,198)
    for m in ('random','recent_score','current_violation'):
        r,_,_=replay(domain,a,m,3,198)
        assert r['direct_reduction']<=r['active_fraction']+1e-9
        assert r['direct_reduction']<=oracle['direct_reduction']+1e-9
        assert r['active_fraction']==oracle['active_fraction']
        if domain=='resource_sharing':assert np.isclose(r['reward'],2.6-.2*r['executed'],atol=1e-6)

def test_fixed_window_has_rolling_limit():
    s=[x[0] for x in fixed_windows(5000)]
    for t in range(5000):assert sum(t-300<x<=t for x in s)<=4

def test_development_only_and_disjoint_seeds():
    a=plan();seeds={c['seed'] for c in a}
    assert seeds=={281000,281001}
    assert seeds.isdisjoint(range(191000,191010)) and seeds.isdisjoint({181000,181001})
    assert len({str(c) for c in a})==len(a)
    with pytest.raises(ValueError):plan('main')

def test_inherited_sources_unchanged():verify_immutable_vendor()

def test_complete_checksums_and_incompatible_resume(tmp_path):
    save_completed(tmp_path,{'config':{'x':1}},{'a':np.ones((2,2))})
    assert load_completed(tmp_path,{'x':1})
    with pytest.raises(RuntimeError):load_completed(tmp_path,{'x':2})
    (tmp_path/'trajectories.npz').write_bytes(b'broken')
    with pytest.raises(RuntimeError):load_completed(tmp_path,{'x':1})

def test_uniform_saturation_no_selection_opportunity():
    actions=np.ones((256,50),np.float32)
    r,_,_=replay('resource_sharing',actions,'future_window_diagnostic',3,18)
    assert abs(r['fixed_schedule_selection_opportunity'])<1e-12
    assert abs(r['direct_reduction']-r['expected_uniform_random_reduction'])<1e-12

def test_sparse_sources_create_hindsight_opportunity():
    actions=np.full((256,50),.2,np.float32);actions[:,:3]=1.
    r,_,_=replay('resource_sharing',actions,'future_window_diagnostic',3,18)
    assert r['fixed_schedule_selection_opportunity']>0
    assert np.isclose(r['direct_reduction'],r['active_fraction'])

def test_finite_episode_capacity_is_not_long_run_bound():
    # Direct gain is bounded by REALIZED authority; finite-window effects explicit.
    actions=np.ones((128,50),np.float32)
    r,_,_=replay('resource_sharing',actions,'random',3,18)
    assert r['direct_reduction']==r['active_fraction']
    assert np.isclose(r['long_run_authority_cap'],.04)

def test_state_diagnostic_enforces_same_limits():
    a=Authority(top_k=1,horizon=10,cooldown=5,window=60,tokens=4)
    c=BudgetController('immediate_state',1,4,4,calibrate([0.]*8),a)
    starts=[]
    for t in range(300):
        r=c.immediate(t,np.ones((1,4)),np.ones(1),np.ones(1,bool))
        if r['selected'].any():starts.append(t);assert c.active(t).sum()==1
        assert c.active(t).sum()<=1
        assert sum(t-60<s<=t for s in starts)<=4
    assert all(b-a>=15 for a,b in zip(starts,starts[1:]))
