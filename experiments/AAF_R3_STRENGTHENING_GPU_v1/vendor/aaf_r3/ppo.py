"""Shared Beta-PPO with proposal-consistent likelihoods and explicit learning masks.

The policy samples a *proposal*. Gateway/attacker changes are environment-side.
The buffer ALWAYS stores the sampled proposal and its matching log probability.
Scripted attackers are excluded from both actor and critic regression after takeover.
This fixes, rather than reproduces, the historical executed-action/old-logp mismatch.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.alpha=nn.Linear(128,action_dim); self.beta=nn.Linear(128,action_dim)
        self.value=nn.Linear(128,1)
    def forward(self,x):
        h=self.body(x)
        return Beta(nn.functional.softplus(self.alpha(h))+1,
                    nn.functional.softplus(self.beta(h))+1),self.value(h).squeeze(-1)

class PPO:
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device,
                 rollout: int=128, epochs: int=4, batch_size: int=1024):
        self.device=device; self.net=ActorCritic(obs_dim,action_dim).to(device)
        self.opt=torch.optim.Adam(self.net.parameters(),lr=3e-4)
        self.rollout=rollout;self.epochs=epochs;self.batch_size=batch_size
        self.buffer=[];self.updates=0
    @torch.no_grad()
    def act(self,obs,deterministic=False):
        x=torch.as_tensor(obs,dtype=torch.float32,device=self.device)
        dist,v=self.net(x)
        a=dist.mean if deterministic else dist.sample()
        a=a.clamp(1e-6,1-1e-6)
        return a.cpu().numpy(),dist.log_prob(a).sum(-1).cpu().numpy(),v.cpu().numpy()
    def add(self,obs,proposal,old_logp,value,reward,done,learn_mask):
        # Copy: callers may modify submitted/executed actions after proposing.
        row=[np.array(x,copy=True) for x in (obs,proposal,old_logp,value,reward,done,learn_mask)]
        if not all(np.isfinite(x).all() for x in row): raise FloatingPointError("Non-finite rollout")
        self.buffer.append(row)
    def update(self,next_obs,force=False):
        if not self.buffer or (len(self.buffer)<self.rollout and not force): return {}
        tensors=[torch.as_tensor(np.stack([r[i] for r in self.buffer]),dtype=torch.float32,device=self.device) for i in range(7)]
        obs,act,old,v,r,d,mask=tensors
        with torch.no_grad():
            _,nv=self.net(torch.as_tensor(next_obs,dtype=torch.float32,device=self.device))
            adv=torch.zeros_like(r); carry=torch.zeros_like(nv)
            for t in reversed(range(len(self.buffer))):
                vn=nv if t==len(self.buffer)-1 else v[t+1]
                delta=r[t]+.99*vn*(1-d[t])-v[t]
                carry=delta+.99*.95*(1-d[t])*carry
                adv[t]=carry
            ret=adv+v
        keep=mask.reshape(-1)>0.5
        if not keep.any(): self.buffer.clear();return {"updates":self.updates}
        x=obs.reshape(-1,obs.shape[-1])[keep]; a=act.reshape(-1,act.shape[-1])[keep]
        lp0=old.reshape(-1)[keep]; y=ret.reshape(-1)[keep]; gae=adv.reshape(-1)[keep]
        gae=(gae-gae.mean())/(gae.std(unbiased=False)+1e-8)
        losses=[]
        for _ in range(self.epochs):
            inds=torch.randperm(x.shape[0],device=self.device)
            for j in range(0,len(inds),self.batch_size):
                ix=inds[j:j+self.batch_size];dist,vn=self.net(x[ix])
                lp=dist.log_prob(a[ix]).sum(-1);ratio=(lp-lp0[ix]).exp()
                objective=torch.minimum(ratio*gae[ix],ratio.clamp(.8,1.2)*gae[ix])
                loss=-objective.mean()+.5*(vn-y[ix]).square().mean()-.01*dist.entropy().sum(-1).mean()
                if not torch.isfinite(loss): raise FloatingPointError("Non-finite PPO loss")
                self.opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),.5);self.opt.step()
                losses.append(float(loss.detach().cpu()))
        self.buffer.clear();self.updates+=1
        return {"loss":float(np.mean(losses)),"updates":self.updates}
