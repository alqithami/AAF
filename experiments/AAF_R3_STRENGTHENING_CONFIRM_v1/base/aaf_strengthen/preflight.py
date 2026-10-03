from __future__ import annotations
from dataclasses import asdict
import numpy as np
import torch
from .predictive import MotionModel,Predictor
from .io import environment,verify_immutable_vendor


def real_engine_check(device):
    from aaf_r3.physics import Navigation
    nav=Navigation(2,4,261000,device)
    try:
        model=MotionModel.from_navigation(nav);pred=Predictor(model,device)
        # Known separated states, not sampled until favorable. Test the predictor
        # outside contacts, where it purports to match the integration equations.
        positions=np.array([[-.7,-.7],[.7,-.7],[-.7,.7],[.7,.7]],np.float32)
        rng=np.random.default_rng(261001)
        for i,a in enumerate(nav.env.agents):
            a.set_pos(torch.tensor(np.tile(positions[i],(2,1)),device=device),batch_index=None)
            a.set_vel(torch.tensor(rng.uniform(-.02,.02,(2,2)),dtype=torch.float32,device=device),batch_index=None)
        nav.last_force[:]=rng.uniform(-.04,.04,nav.last_force.shape)
        errors=[]
        for _ in range(12):
            d=nav.diagnostics();command=rng.uniform(-.08,.08,(2,4,2)).astype(np.float32)
            p,v,f=pred.one_step(pred.tensor(d['pos'])[:,None],pred.tensor(d['vel'])[:,None],
                               pred.tensor(nav.last_force)[:,None],pred.tensor(command)[:,None])
            obs,r,done=nav.step(command);after=nav.diagnostics()
            pe=float(np.max(np.abs(p[:,0].cpu().numpy()-after['pos'])))
            ve=float(np.max(np.abs(v[:,0].cpu().numpy()-after['vel'])))
            errors.append({'position_max_abs':pe,'velocity_max_abs':ve})
            if pe>3e-5 or ve>3e-5:raise RuntimeError(f'Free-motion model does not match installed VMAS: p={pe}, v={ve}. Stop; return logs, do not tune the threshold.')
            if after['contact'].any():raise RuntimeError('Unexpected contact in separated-state integration test')
            if obs.shape[0]!=8 or r.shape!=(2,4) or done.shape!=(2,):raise RuntimeError('VMAS API shape mismatch')
        active=np.zeros((2,4),bool);active[:,0]=True
        x,_=pred.filter(command,active,after,nav.last_force)
        if not np.isfinite(x).all():raise RuntimeError('Nonfinite predictive filter')
        return {'status':'PASS','engine':'vmas==1.5.2','motion_model':asdict(model),
                'free_motion_errors':errors,'observation_shape':list(obs.shape),
                'limit':'The probe validates contact-free one-step integration only; it does not validate collision prediction or certify safety.'}
    finally:nav.close()

def check(device,require_physics=True):
    verify_immutable_vendor()
    # Exercise actual device arithmetic and an optimizer, not just is_available().
    x=torch.tensor([[1.,2.],[3.,4.]],device=device,requires_grad=True)
    optimizer=torch.optim.Adam([x],lr=.01);loss=x.square().sum();loss.backward();optimizer.step()
    if not bool(torch.isfinite(x).all()):raise RuntimeError('Device arithmetic failed')
    report={'environment':environment(device),'torch_optimizer':'PASS','inherited_source':'unchanged'}
    if require_physics:report['physics']=real_engine_check(device)
    return report
