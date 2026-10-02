"""Versioned controller with explicit authority, timing, and diagnostic selectors.

Legacy four arms reproduce the supplied R3 governor's numerical rules, including
next-step admission. New full-state diagnostic selectors are NOT causal blame.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from aaf_r3.governor import Authority

class BudgetController:
    def __init__(self,method,worlds,n,seed,calibration,authority:Authority):
        if method not in ('adaptive_rank','adaptive_random','threshold_rank','periodic_rank','adaptive_benefit_state','immediate_state','ppo_only','random_policy','static_guard'):
            raise ValueError('Unknown controller method: '+str(method))
        self.method,self.e,self.n,self.a=method,worlds,n,authority
        if not 1<=authority.top_k<=n: raise ValueError('Invalid target count')
        if min(authority.horizon,authority.window,authority.tokens,authority.history)<=0 or authority.cooldown<0:
            raise ValueError('Invalid authority')
        if not 0<authority.minimum_coverage<=1: raise ValueError('Invalid coverage')
        self.rng=np.random.default_rng(seed);self.cal=dict(calibration)
        self.period=max(authority.horizon+authority.cooldown,int(np.ceil(authority.window/authority.tokens)))
        self.hist=np.full((authority.history,worlds,n),np.nan);self.ptr=self.filled=0
        self.s=np.zeros(worlds);self.h=np.full(worlds,float(calibration['cusum_h0']))
        self.begin=np.zeros((worlds,n),int);self.end=np.zeros((worlds,n),int)
        self.next_allowed=np.zeros(worlds,int);self.starts=[deque() for _ in range(worlds)]
        self.counts={k:np.zeros(worlds,int) for k in ('triggers','admissions','denied','abstentions')}
    def active(self,t):
        if self.method=='static_guard': return np.ones((self.e,self.n),bool)
        if self.method in ('ppo_only','random_policy'): return np.zeros((self.e,self.n),bool)
        return (self.begin<=t)&(self.end>t)
    def scores(self):
        if not self.filled: return np.zeros((self.e,self.n)),np.zeros((self.e,self.n))
        h=self.hist[:self.filled];c=np.isfinite(h).sum(0)
        return np.nansum(h,0)/np.maximum(c,1),c/self.filled
    def _admit(self,t,trigger,score,coverage,delay,omission,alive):
        selected=np.zeros((self.e,self.n),bool);denied=np.zeros(self.e,bool)
        abstain=np.zeros(self.e,bool);start=t+delay
        for e in np.flatnonzero(trigger & alive):
            while self.starts[e] and self.starts[e][0]<=start-self.a.window:self.starts[e].popleft()
            # No overlapping intervals, no extension, equal rolling starts budget.
            # Historical R3 rule also denies an admission on the last active step.
            if (self.end[e]>t).any() or start<self.next_allowed[e] or len(self.starts[e])>=self.a.tokens:
                denied[e]=True;continue
            eligible=np.flatnonzero(coverage[e]>=self.a.minimum_coverage)
            if not len(eligible) or score[e,eligible].max()<=0:
                abstain[e]=True;continue
            k=min(self.a.top_k,len(eligible))
            if self.method=='adaptive_random':ids=self.rng.choice(eligible,k,False)
            else:
                perm=self.rng.permutation(eligible)
                ids=perm[np.argsort(-score[e,perm],kind='stable')[:k]]
            selected[e,ids]=True;self.begin[e,ids]=start;self.end[e,ids]=start+self.a.horizon
            self.next_allowed[e]=start+self.a.horizon+self.a.cooldown;self.starts[e].append(start)
        self.counts['triggers']+=trigger&alive;self.counts['denied']+=denied
        self.counts['abstentions']+=abstain;self.counts['admissions']+=selected.any(1)
        return {'trigger':trigger&alive,'selected':selected,'denied':denied,'abstain':abstain,
                'scores':score,'coverage':coverage,'omission_target':selected&omission,
                'begin_step':start,'threshold':self.h.copy(),'cusum':self.s.copy()}
    def observe(self,t,evidence,alive=None,benefit=None,gaps=None):
        if alive is None:alive=np.ones(self.e,bool)
        x=np.asarray(evidence,float).reshape(self.e,self.n)
        if np.isinf(x).any():raise ValueError('Infinite evidence')
        self.hist[self.ptr]=x;self.ptr=(self.ptr+1)%self.a.history;self.filled=min(self.filled+1,self.a.history)
        score,coverage=self.scores()
        count=np.isfinite(x).sum(1);usable=count>=max(1,int(np.ceil(self.a.minimum_coverage*self.n)))
        z=np.nansum(x,1)/np.maximum(count,1);trigger=np.zeros(self.e,bool)
        if self.method.startswith('adaptive'):
            self.s[usable]=np.maximum(0,self.s[usable]+z[usable]-self.cal['mu0']-self.cal['slack'])
            trigger=usable&(self.s>=self.h);self.s[trigger]=0
            self.h[usable]=np.maximum(.05,self.h[usable]+(t+1)**(-.6)*(trigger[usable]-.05))
        elif self.method=='threshold_rank':trigger=usable&(z>self.cal['instant_threshold'])
        elif self.method=='periodic_rank':trigger[:]=(t+1)%self.period==0
        if self.method=='adaptive_benefit_state':
            if benefit is None or np.shape(benefit)!=(self.e,self.n):raise ValueError('Missing state-model benefit')
            score=np.asarray(benefit,float);coverage=np.ones_like(score)
        omission=np.zeros_like(x,bool)
        if gaps is not None:
            omission=np.asarray(gaps,bool);trigger|=omission.any(1)
            # An omission is an availability reason, not a fabricated violation.
            # Eligibility is for a reversible missing-evidence control ONLY.
            score=np.where(omission,2.+score,score);coverage=np.where(omission,1.,coverage)
        return self._admit(t,trigger,score,coverage,1,omission,alive)
    def immediate(self,t,benefit,predicted_cost,alive):
        if self.method!='immediate_state': raise ValueError('Not an immediate diagnostic')
        b=np.asarray(benefit,float)
        return self._admit(t,np.asarray(predicted_cost)>1e-8,b,np.ones_like(b),0,np.zeros_like(b,bool),alive)
