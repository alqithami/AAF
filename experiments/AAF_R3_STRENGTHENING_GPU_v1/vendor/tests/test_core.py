import copy,json
from dataclasses import asdict
import numpy as np
import pytest
import torch
from aaf_r3.games import Game
from aaf_r3.governor import Governor,Authority,calibrate
from aaf_r3.ppo import PPO
from aaf_r3.network import EvidenceChannel,CommandChannel
from aaf_r3.cli import plan
from aaf_r3.analysis import holm

CAL={"mu0":0.,"slack":.01,"cusum_h0":.05,"instant_threshold":.01,"period":1}

def test_game_resource_equations():
    e=Game("resource_sharing",10,0,noise=0)
    _,r,info=e.step(np.ones(10))
    assert np.allclose(info["allocation"],10)
    assert np.allclose(r,12.8)
    assert info["violation"].sum()==10
    _,r,info=e.step(np.full(10,.05))
    assert np.allclose(info["allocation"],5)
    assert np.allclose(r,6.5)

def test_public_equations():
    e=Game("public_goods",10,0,noise=0)
    _,r,i=e.step(np.full(10,.5));assert np.allclose(r,1.54)
    _,r,i=e.step(np.zeros(10));assert np.allclose(r,.8)

def test_projection():
    for domain,a in [("resource_sharing",1.),("public_goods",0.)]:
        e=Game(domain,10,0);m=np.arange(10)<3
        out=e.shield(np.full(10,a),m)
        assert e.violations(out)[m].sum()==0
        assert np.all(out[~m]==a)

def test_random_target_is_k_not_all():
    a=Authority(top_k=3,horizon=5,window=30,tokens=4,cooldown=0)
    g=Governor("adaptive_random",1,50,0,CAL,a)
    r=g.observe(0,np.ones((1,50)))
    assert r["selected"].sum()==3 and g.active(1).sum()==3

def test_rank_selects_evidence():
    a=Authority(top_k=2,horizon=5)
    g=Governor("adaptive_rank",1,10,0,CAL,a)
    x=np.zeros((1,10));x[0,[3,7]]=1
    r=g.observe(0,x)
    assert set(np.flatnonzero(r["selected"]))=={3,7}

def test_missing_is_not_compliance():
    g=Governor("adaptive_rank",1,10,0,CAL,Authority())
    g.observe(0,np.full((1,10),np.nan))
    assert g.n_triggers.sum()==0
    _,c=g.scores();assert c.sum()==0

def test_horizon_budget_and_no_renewal():
    a=Authority(top_k=3,horizon=7,window=40,tokens=2,cooldown=3)
    g=Governor("threshold_rank",1,10,0,CAL,a)
    starts=[];masks=[]
    for t in range(200):
        masks.append(g.active(t).copy());r=g.observe(t,np.ones((1,10)))
        if r["selected"].any():starts.append(t+1)
    for t in range(200):assert sum(t-40<s<=t for s in starts)<=2
    for s in starts:
        if s+7<=200:
            assert all(masks[t].sum()==3 for t in range(s,s+7))
            if s+7<200:assert masks[s+7].sum()==0
    assert max(m.sum() for m in masks)<=3

def test_ppo_proposals_and_likelihoods():
    torch.set_num_threads(1);torch.manual_seed(19)
    p=PPO(3,1,torch.device("cpu"),rollout=4)
    obs=np.zeros((10,3),np.float32);a,lp,v=p.act(obs)
    with torch.no_grad():d,_=p.net(torch.as_tensor(obs));assert torch.allclose(d.log_prob(torch.as_tensor(a)).sum(-1),torch.as_tensor(lp))
    before=a.copy();executed=np.minimum(a,.01)
    p.add(obs,a,lp,v,np.ones(10),np.zeros(10),np.ones(10))
    a[:]=0
    assert np.allclose(p.buffer[0][1],before)
    assert not np.allclose(p.buffer[0][1],executed)
    for _ in range(3):
        a,lp,v=p.act(obs);p.add(obs,a,lp,v,np.ones(10),np.zeros(10),np.ones(10))
    s=p.update(obs);assert p.updates==1 and np.isfinite(s["loss"])

def test_all_scripted_samples_excluded():
    p=PPO(3,1,torch.device("cpu"),rollout=1);obs=np.zeros((10,3),np.float32)
    a,lp,v=p.act(obs);old=copy.deepcopy(p.net.state_dict())
    p.add(obs,a,lp,v,np.ones(10),np.ones(10),np.zeros(10));p.update(obs)
    assert all(torch.equal(v,p.net.state_dict()[k]) for k,v in old.items())

def test_channels_no_future_leakage():
    c=EvidenceChannel(2,4,0,"stale_delay",max_age=2)
    for t in range(20):
        out=c.send_receive(t,np.full((2,4),t))
        seen=out[np.isfinite(out)];assert (seen<=t).all();assert (seen>=t-2).all()
    assert c.expired>0

def test_common_burst_loss_and_rate():
    c=EvidenceChannel(1,8,191,"burst_loss")
    for t in range(5000):
        x=c.send_receive(t,np.ones((1,8)))
        assert np.isfinite(x).sum() in (0,8)
    assert .12<c.dropped/c.sent<.28

def test_command_expiry_brakes():
    c=CommandChannel(2,4,0,"stale_delay",max_age=0)
    count=0
    for t in range(30):
        u,m=c.step(t,np.ones((2,4,2)),np.zeros((2,4,2)))
        assert np.all(u[m]==0);count+=m.sum()
    assert count>0

def test_plan_counts_no_duplicates():
    from aaf_r3.common import digest
    ps=plan("main","all")
    assert len(ps)==1640
    assert sum(c["suite"]=="games" for c in ps)==1080
    assert len({digest(c) for c in ps})==len(ps)
    assert all(c["method"]!="aaf_full" for c in ps if c["suite"]=="physics")

def test_holm():assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])

def test_pilot_and_main_seeds_are_disjoint():
    a={c["seed"] for c in plan("pilot","all")};b={c["seed"] for c in plan("main","all")}
    assert not a & b

def test_primary_and_legacy_labels_separate():
    p=plan("main","games")
    assert sum(c["update_semantics"]=="proposal_consistent_masked" for c in p)==840
    assert sum(c["update_semantics"]=="executed_action_legacy" for c in p)==240

def test_no_interventions_for_ppo():
    g=Governor("ppo_only",1,10,0,CAL,Authority())
    for t in range(100):
        g.observe(t,np.ones((1,10)));assert not g.active(t).any()

def test_periodic_trigger_does_not_disable_intervention():
    g=Governor("periodic_rank",1,10,0,CAL,Authority())
    for t in range(75): g.observe(t,np.ones((1,10)))
    assert g.active(74).sum()==0
    assert g.active(75).sum()==3

def test_smoke_legacy_ppo_identical_without_modification(tmp_path):
    from aaf_r3.game_runner import run_game
    c=plan("smoke","games")[0];c["attack"]="none";c["steps"]=24;c["attack_start"]=8
    a=run_game(c,torch.device("cpu"),tmp_path)
    c=dict(c,method="legacy__ppo_only",update_semantics="executed_action_legacy")
    b=run_game(c,torch.device("cpu"),tmp_path)
    assert a["summary"]==b["summary"]

def test_analysis_refuses_incomplete(tmp_path):
    from aaf_r3.analysis import analyze
    from aaf_r3.common import atomic_json,source_hash
    c=plan("smoke","games")[0]
    atomic_json(tmp_path/"RUN_MANIFEST.json",{"configs":[c],"identity":{"profile":"smoke","source_sha256":source_hash()}})
    with pytest.raises(RuntimeError,match="Incomplete"):analyze(tmp_path)


def test_future_admission_not_active_at_trigger_step():
    g=Governor("threshold_rank",1,10,0,CAL,Authority())
    g.observe(0,np.ones((1,10)))
    assert not g.active(0).any()
    assert g.active(1).sum()==3


def test_authority_invalid_duration_rejected():
    with pytest.raises(ValueError):
        Governor("adaptive_rank",1,10,0,CAL,Authority(horizon=0))


def test_pulse_on_off_and_before_start():
    from aaf_r3.game_runner import attack_mask
    ids=np.array([1,3])
    assert not attack_mask(99,100,"pulse",ids,10).any()
    assert attack_mask(100,100,"pulse",ids,10).sum()==2
    assert attack_mask(149,100,"pulse",ids,10).sum()==2
    assert not attack_mask(150,100,"pulse",ids,10).any()
    assert attack_mask(300,100,"pulse",ids,10).sum()==2
    assert not attack_mask(300,100,"none",ids,10).any()
