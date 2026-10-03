"""Explicit simulated evidence attacks. A received omission is never a violation.

The benchmark's attack injector knows the script identity. The controller does
not: it only receives payloads, finite-value coverage, and checkpoint gap flags.
"""
from __future__ import annotations
import numpy as np
from aaf_r3.network import EvidenceChannel,CommandChannel

class Transport:
    def __init__(self,e,n,seed,profile):
        self.e,self.n,self.profile=e,n,profile
        network={'gateway_iid':'iid_loss','gateway_burst':'burst_loss',
                 'stale_evidence_only':'stale_delay','stale_both':'stale_delay'}.get(profile,'clean')
        self.ev=EvidenceChannel(e,n,seed,network)
        self.cmd=CommandChannel(e,n,seed+1,'stale_delay' if profile in ('stale_commands_only','stale_both') else 'clean')
        self.previous_suppressed=np.zeros((e,n),bool)
    def commands(self,t,u,fallback):
        command,expired=self.cmd.step(t,u,fallback)
        self.command_generation=self.cmd.last_t.copy() if self.cmd.profile=='stale_delay' else np.full((self.e,self.n),t,int)
        return command,expired
    def evidence(self,t,true_risk,attack_mask):
        x=np.asarray(true_risk,float).copy();attack_mask=np.asarray(attack_mask,bool)
        suppressed=np.zeros_like(attack_mask);gaps=np.zeros_like(attack_mask)
        if self.profile=='forged_self_report':x[attack_mask]=0.
        if self.profile in ('selective_naive','selective_gapaware','permanent_suppression'):
            suppressed=attack_mask & (x>0)
            if self.profile=='permanent_suppression':suppressed=attack_mask.copy()
            x[suppressed]=np.nan
            if self.profile=='selective_gapaware':
                # A next-step trusted checkpoint exposes the PREVIOUS omitted
                # interior payload. It does not reveal the missing payload.
                gaps=self.previous_suppressed.copy()
            self.previous_suppressed=suppressed.copy()
        received=x.copy() if self.profile in ('selective_naive','selective_gapaware','permanent_suppression') else self.ev.send_receive(t,x)
        valid=np.isfinite(received)
        controller=received.copy()
        if self.profile=='selective_naive':
            # Deliberately naive comparator, explicitly labeled in outputs.
            controller=np.nan_to_num(controller,nan=0.)
        return {'received':received,'controller_values':controller,'valid':valid,
                'suppressed':suppressed,'visible_gap':gaps,
                'generation':np.where(valid,self.ev.last if self.ev.profile!='clean' else t,-1),
                'visible_gap_generation':np.where(gaps,t-1,-1)}
