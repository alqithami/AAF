"""Synthetic transport faults, not VMAS-provided wireless or attestation models."""
from __future__ import annotations
import numpy as np

NETWORKS=("clean","iid_loss","burst_loss","stale_delay")
class EvidenceChannel:
    def __init__(self,worlds:int,n:int,seed:int,profile:str,max_age:int=2):
        if profile not in NETWORKS: raise ValueError(profile)
        self.e,self.n,self.profile,self.max_age=worlds,n,profile,max_age
        self.rng=np.random.default_rng(seed);self.bad=self.rng.random(worlds)<.2
        self.queue={};self.last=np.full((worlds,n),-1,int)
        self.sent=self.dropped=self.expired=self.delivered=0
    def send_receive(self,t:int,values:np.ndarray)->np.ndarray:
        values=np.asarray(values,float).reshape(self.e,self.n)
        self.sent+=self.e*self.n
        lost=np.zeros_like(values,dtype=bool)
        delay=np.zeros_like(values,dtype=int)
        if self.profile=="iid_loss": lost=self.rng.random(values.shape)<.2
        elif self.profile=="burst_loss":
            u=self.rng.random(self.e)
            self.bad=np.where(self.bad,u>=.10,u<.025)
            lost=np.broadcast_to(self.bad[:,None],values.shape)
        elif self.profile=="stale_delay":
            delay=self.rng.choice([0,1,2,4],size=values.shape,p=[.25,.25,.25,.25])
        self.dropped+=int(lost.sum())
        for d in np.unique(delay):
            mask=(delay==d)&~lost
            self.queue.setdefault(t+int(d),[]).append((t,values.copy(),mask.copy()))
        out=np.full_like(values,np.nan)
        for generation,v,mask in self.queue.pop(t,[]):
            if t-generation>self.max_age:
                self.expired+=int(mask.sum());continue
            accepted=mask & (generation>self.last)
            out[accepted]=v[accepted];self.last[accepted]=generation
        self.delivered+=int(np.isfinite(out).sum())
        return out

class CommandChannel:
    """Monotone per-agent command sequences; expired command -> local braking.

    Only stale_delay injects command delays. Loss profiles affect supervisory
    evidence only, making that boundary explicit instead of faulting everything.
    """
    def __init__(self,worlds:int,n:int,seed:int,profile:str,max_age:int=2):
        self.e,self.n,self.profile,self.max_age=worlds,n,profile,max_age
        self.rng=np.random.default_rng(seed);self.queue={}
        self.last_t=np.full((worlds,n),-1,int);self.last_u=np.zeros((worlds,n,2))
    def step(self,t:int,u:np.ndarray,brake:np.ndarray):
        if self.profile!="stale_delay":return u.copy(),np.zeros((self.e,self.n),bool)
        delay=self.rng.choice([0,1,2,4],size=(self.e,self.n))
        for d in np.unique(delay):self.queue.setdefault(t+int(d),[]).append((t,u.copy(),delay==d))
        for generation,v,mask in self.queue.pop(t,[]):
            ok=mask & (generation>self.last_t) & (t-generation<=self.max_age)
            self.last_t[ok]=generation;self.last_u[ok]=v[ok]
        expired=(self.last_t<0)|(t-self.last_t>self.max_age)
        command=self.last_u.copy();command[expired]=brake[expired]
        return command,expired
