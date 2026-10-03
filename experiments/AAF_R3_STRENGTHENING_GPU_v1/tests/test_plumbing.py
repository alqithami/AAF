"""Test doubles here check wiring/serialization, not VMAS fidelity or outcomes."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from aaf_strengthen.protocol import plan
from aaf_strengthen.navigation import run_navigation,summarize_trajectory
from aaf_strengthen.analysis import audit_arrays
from aaf_r3.governor import calibrate

class Holonomic:pass
class FakeNav:
    def __init__(self,e,n,seed,device):
        self.e,self.n,self.device=e,n,device;self.radius=np.full(n,.1)
        self.pos=np.tile(np.array([[-.4,-.4],[.4,-.4],[-.4,.4],[.4,.4]],np.float32),(e,1,1))
        self.vel=np.zeros_like(self.pos);self.goals=-self.pos.copy();self.last_force=np.zeros_like(self.pos)
        self.obs=np.zeros((e*n,4),np.float32);self.obs_dim=4
        agents=[SimpleNamespace(dynamics=Holonomic(),action=SimpleNamespace(u_multiplier=1.),mass=1.,drag=None,shape=SimpleNamespace(radius=.1)) for _ in range(n)]
        self.env=SimpleNamespace(agents=agents,world=SimpleNamespace(dt=.1,_substeps=2,_drag=.25,_x_semidim=None,_y_semidim=None))
    def diagnostics(self):
        dist=np.linalg.norm(self.pos[:,:,None]-self.pos[:,None],axis=-1)
        contact=dist<=.205;near=dist<.3
        contact[:,np.arange(self.n),np.arange(self.n)]=False;near[:,np.arange(self.n),np.arange(self.n)]=False
        speed=np.linalg.norm(self.vel,axis=-1);gd=np.linalg.norm(self.pos-self.goals,axis=-1)
        return dict(pos=self.pos.copy(),vel=self.vel.copy(),goals=self.goals.copy(),contact=contact.any(-1),near=near.any(-1),
                    speed=speed,risk=((speed>.3)|contact.any(-1)).astype(float),goal_distance=gd,on_goal=gd<.05)
    def step(self,u):
        self.last_force=.5*self.last_force+.5*np.clip(u,-1,1);self.vel*=.75
        for _ in range(2):self.vel+=self.last_force*.05;self.pos+=self.vel*.05
        done=np.zeros(self.e,bool)
        return self.obs,np.zeros((self.e,self.n),np.float32),done
    def close(self):pass
class FakePolicy:
    def act(self,obs):
        return np.full((len(obs),2),.9,np.float32),np.zeros(len(obs)),np.zeros(len(obs))

@pytest.mark.parametrize('method,filtername,evidence',[('adaptive_rank','brake','gateway'),('adaptive_random','predictive','gateway'),
    ('threshold_rank','predictive','forged_self_report'),('periodic_rank','predictive','stale_both'),
    ('adaptive_benefit_state','predictive','gateway'),('immediate_state','predictive','gateway'),
    ('adaptive_rank','predictive','selective_gapaware'),('static_guard','predictive','gateway'),('ppo_only','brake','gateway')])
def test_trajectory_pipeline(monkeypatch,tmp_path,method,filtername,evidence):
    import aaf_strengthen.navigation as mod
    monkeypatch.setattr(mod,'Navigation',FakeNav)
    monkeypatch.setattr(mod,'prepare',lambda *args:(FakePolicy(),{'calibration':calibrate([0.]*16),'training_trace':[]}))
    cfg=dict(plan('smoke')[0]);cfg.update(method=method,filter=filtername,evidence=evidence,horizon=36,attack_start=8)
    r,a=run_navigation(cfg,torch.device('cpu'),tmp_path)
    assert audit_arrays(a,cfg)['authority']=='PASS'
    episodes,summary=summarize_trajectory(a,cfg['attack_start'])
    assert r['summary']==summary and r['episodes']==episodes
    assert a['position_before'].shape==(36,2,4,2)
    assert all(e['attack_exposed'] for e in episodes)
