from __future__ import annotations
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np

METHODS = ("ppo_only", "aaf_full", "adaptive_rank", "adaptive_random", "threshold_rank", "periodic_rank", "static_guard")

@dataclass(frozen=True)
class Authority:
    top_k: int = 3
    horizon: int = 50
    window: int = 300
    tokens: int = 4
    cooldown: int = 25
    history: int = 50
    minimum_coverage: float = 0.5
    shaping: float = 0.2

class Governor:
    """Same admission/authority limits for all five bounded treatment arms.

    A single active target set per world, no extension on repeated alarms.
    Every admission starts NEXT step and lasts exactly H steps. The cap is on
    admitted intervals; realized constrained agent-time is measured separately.
    No absent observation is converted to a zero violation.
    """
    def __init__(self, method: str, worlds: int, n: int, seed: int,
                 calibration: dict, authority: Authority):
        if method not in METHODS: raise ValueError(method)
        if not 0 < authority.top_k <= n: raise ValueError("Invalid top_k")
        if min(authority.horizon,authority.window,authority.tokens,authority.history)<=0 or authority.cooldown<0: raise ValueError("Invalid authority duration or budget")
        if not 0<authority.minimum_coverage<=1: raise ValueError("Invalid coverage threshold")
        self.method,self.e,self.n,self.a=method,worlds,n,authority
        self.rng=np.random.default_rng(seed);self.cal=dict(calibration)
        self.cal["period"]=max(authority.horizon+authority.cooldown,int(np.ceil(authority.window/authority.tokens)))
        self.hist=np.full((authority.history,worlds,n),np.nan)
        self.filled=0;self.ptr=0;self.t=0
        self.s=np.zeros(worlds);self.h=np.full(worlds,float(calibration["cusum_h0"]))
        self.begin=np.zeros((worlds,n),int);self.end=np.zeros((worlds,n),int);self.next_allowed=np.zeros(worlds,int)
        self.starts=[deque() for _ in range(worlds)]
        self.n_admissions=np.zeros(worlds,int);self.n_triggers=np.zeros(worlds,int)
        self.n_denied=np.zeros(worlds,int);self.n_abstain=np.zeros(worlds,int)
    def active(self,t:int)->np.ndarray:
        if self.method=="static_guard": return np.ones((self.e,self.n),bool)
        if self.method=="ppo_only": return np.zeros((self.e,self.n),bool)
        return (self.begin<=t)&(self.end>t)
    def scores(self):
        if self.filled==0: return np.zeros((self.e,self.n)),np.zeros((self.e,self.n))
        h=self.hist[:self.filled]
        count=np.isfinite(h).sum(0);score=np.nansum(h,axis=0)/np.maximum(count,1)
        return score,count/self.filled
    def penalties(self,t:int)->np.ndarray:
        if self.method!="aaf_full": return np.zeros((self.e,self.n))
        score,_=self.scores(); score=score/np.maximum(score.sum(1,keepdims=True),1e-12)
        return self.a.shaping*score*self.active(t)
    def observe(self,t:int, evidence:np.ndarray)->dict:
        self.t=t
        x=np.asarray(evidence,float).reshape(self.e,self.n)
        self.hist[self.ptr]=x;self.ptr=(self.ptr+1)%self.a.history
        self.filled=min(self.filled+1,self.a.history)
        score,coverage=self.scores()
        current_count=np.isfinite(x).sum(1)
        usable=current_count>=max(1,int(np.ceil(self.a.minimum_coverage*self.n)))
        z=np.nansum(x,axis=1)/np.maximum(current_count,1)
        trigger=np.zeros(self.e,bool)
        if self.method.startswith("adaptive") or self.method=="aaf_full":
            self.s[usable]=np.maximum(0,self.s[usable]+z[usable]-self.cal["mu0"]-self.cal["slack"])
            trigger=usable & (self.s>=self.h)
            self.s[trigger]=0
            self.h[usable]=np.maximum(.05,self.h[usable]+(t+1)**(-.6)*(trigger[usable]-.05))
        elif self.method=="threshold_rank":
            trigger=usable & (z>self.cal["instant_threshold"])
        elif self.method=="periodic_rank":
            trigger[:]=((t+1)%self.cal["period"]==0)
        self.n_triggers+=trigger
        selected=np.zeros((self.e,self.n),bool)
        for e in np.flatnonzero(trigger):
            # Tokens count intervention starts in every rolling W-step window.
            start=t+1
            while self.starts[e] and self.starts[e][0]<=start-self.a.window: self.starts[e].popleft()
            if (self.end[e]>t).any() or start<self.next_allowed[e] or len(self.starts[e])>=self.a.tokens:
                self.n_denied[e]+=1;continue
            eligible=np.flatnonzero(coverage[e]>=self.a.minimum_coverage)
            if len(eligible)==0 or score[e,eligible].max()<=0:
                self.n_abstain[e]+=1;continue
            k=min(self.a.top_k,len(eligible))
            if self.method=="adaptive_random": ids=self.rng.choice(eligible,k,replace=False)
            else:
                # Randomize ties without affecting policy/environment RNGs.
                shuffled=self.rng.permutation(eligible)
                ids=shuffled[np.argsort(-score[e,shuffled],kind="stable")[:k]]
            selected[e,ids]=True;self.begin[e,ids]=start;self.end[e,ids]=start+self.a.horizon
            self.next_allowed[e]=start+self.a.horizon+self.a.cooldown
            self.starts[e].append(start);self.n_admissions[e]+=1
        return {"trigger":trigger,"selected":selected,"scores":score,"coverage":coverage}

def calibrate(statistics:list[float])->dict:
    z=np.asarray(statistics,float)
    z=z[np.isfinite(z)]
    if len(z)<8: raise ValueError("Calibration needs at least eight observations")
    # Nominal-only calibration. No attack outcomes or held-out test data used.
    return {"mu0":float(z.mean()),"slack":.01,"cusum_h0":5.0,
            "instant_threshold":float(np.quantile(z,.95)),"period_rule":"max(horizon+cooldown,ceil(window/tokens))",
            "target_alarm_rate":.05,"n_calibration_observations":len(z),
            "note":"Instant threshold uses nominal 95th percentile; periodic interval follows the authority cap, not test outcomes. Equal authority caps, not equal realized use."}
