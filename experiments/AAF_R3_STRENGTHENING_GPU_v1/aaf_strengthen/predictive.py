"""Finite-candidate, collision-free-model look-ahead. NOT a certified safety filter.

All methods in a filter block receive the same primitive and trusted gateway state.
The model omits collision forces and holds commands constant over the prediction
horizon. Real-engine preflight must check free-motion one-step agreement.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch

@dataclass(frozen=True)
class MotionModel:
    dt: float
    substeps: int
    mass: tuple[float, ...]
    drag: tuple[float, ...]
    radius: tuple[float, ...]
    x_bound: float | None = None
    y_bound: float | None = None
    lag: float = 0.5
    def __post_init__(self):
        if self.dt <= 0 or self.substeps < 1 or not 0 <= self.lag <= 1:
            raise ValueError('Invalid integration parameters')
        if len(self.mass)<2:raise ValueError('At least two agents required')
        if not (len(self.mass) == len(self.drag) == len(self.radius)):
            raise ValueError('Agent parameter lengths disagree')
        if any(x <= 0 for x in self.mass + self.radius) or any(not 0 <= d < 1 for d in self.drag):
            raise ValueError('Unsupported agent parameters')
    @classmethod
    def from_navigation(cls, nav):
        world = nav.env.world
        for a in nav.env.agents:
            if type(a.dynamics).__name__ != 'Holonomic':
                raise RuntimeError('Predictor requires the standard Holonomic dynamics')
            for name in ('max_speed','v_range','max_f','f_range'):
                if getattr(a, name, None) is not None:
                    raise RuntimeError(f'Unexpected {name}; extend and validate model before use')
            if not np.allclose(np.asarray(a.action.u_multiplier), 1):
                raise RuntimeError('Unsupported action multiplier')
        # Private fields are intentional, version-checked, and tested against
        # actual free-motion transitions. A changed engine fails preflight.
        return cls(float(world.dt), int(world._substeps),
                   tuple(float(a.mass) for a in nav.env.agents),
                   tuple(float(a.drag if a.drag is not None else world._drag) for a in nav.env.agents),
                   tuple(float(a.shape.radius) for a in nav.env.agents),
                   world._x_semidim, world._y_semidim)

class Predictor:
    def __init__(self, model: MotionModel, device: torch.device, horizon: int = 8, margin: float = .05):
        if horizon < 1 or margin < 0: raise ValueError('Invalid prediction horizon/margin')
        self.model,self.device,self.horizon,self.margin=model,device,horizon,margin
        self.mass=torch.tensor(model.mass,device=device,dtype=torch.float32)[None,None,:,None]
        self.drag=torch.tensor(model.drag,device=device,dtype=torch.float32)[None,None,:,None]
        self.radius=torch.tensor(model.radius,device=device,dtype=torch.float32)
        self.pairs=torch.triu_indices(len(model.mass),len(model.mass),1,device=device)
    def tensor(self, x): return torch.as_tensor(x,dtype=torch.float32,device=self.device)
    def one_step(self, p, v, f, command):
        f=self.model.lag*f+(1-self.model.lag)*command.clamp(-1,1)
        v=v*(1-self.drag)
        for _ in range(self.model.substeps):
            v=v+(f/self.mass)*(self.model.dt/self.model.substeps)
            p=p+v*(self.model.dt/self.model.substeps)
            if self.model.x_bound is not None:
                p=torch.stack((p[...,0].clamp(-self.model.x_bound,self.model.x_bound),p[...,1]),-1)
            if self.model.y_bound is not None:
                p=torch.stack((p[...,0],p[...,1].clamp(-self.model.y_bound,self.model.y_bound)),-1)
        return p,v,f
    @torch.inference_mode()
    def evaluate(self,p,v,f,commands):
        """Input states E,N,2; candidate commands E,C,N,2. Returns E,C."""
        p=self.tensor(p)[:,None].expand_as(commands).clone()
        v=self.tensor(v)[:,None].expand_as(commands).clone()
        f=self.tensor(f)[:,None].expand_as(commands).clone()
        cost=torch.zeros(commands.shape[:2],device=self.device)
        clearance=torch.full_like(cost,float('inf'))
        a,b=self.pairs
        for _ in range(self.horizon):
            p,v,f=self.one_step(p,v,f,commands)
            gap=(p[:,:,a]-p[:,:,b]).norm(dim=-1)-self.radius[a]-self.radius[b]
            clearance=torch.minimum(clearance,gap.min(-1).values)
            # Dimensionless penalties for both declared local risk predicates.
            # Zero cost requires predicted clearance >= margin AND speed <= .3.
            cost+=((self.margin-gap).clamp(min=0)/max(self.margin,.005)).square().sum(-1)
            cost+=((v.norm(dim=-1)-.3).clamp(min=0)/.3).square().sum(-1)
        return cost,clearance
    @torch.inference_mode()
    def candidate_set(self, command, velocity, agent_id):
        e,n,_=command.shape
        dirs=[[0.,0.]]+[[math.cos(k*math.pi/4),math.sin(k*math.pi/4)] for k in range(8)]
        choices=self.tensor(dirs)[None].expand(e,-1,2)
        choices=torch.cat((command[:,agent_id,None],(-2*velocity[:,agent_id]).clamp(-1,1)[:,None],choices),1)
        candidates=command[:,None].expand(e,choices.shape[1],n,2).clone()
        candidates[:,:,agent_id]=choices
        return candidates
    @staticmethod
    def choose(cost, candidates, original):
        # Minimize risk first, then perturbation among numerically tied minima.
        best=cost.min(1,keepdim=True).values
        tied=cost<=best+1e-8
        deviation=(candidates-original[:,None]).square().sum((-1,-2))
        index=torch.where(tied,deviation,torch.full_like(deviation,float('inf'))).argmin(1)
        return index
    @torch.inference_mode()
    def filter(self,command,active,diag,previous_force):
        original=self.tensor(command);work=original.clone();v=self.tensor(diag['vel'])
        p=diag['pos'];f=previous_force
        initialcost,initialgap=self.evaluate(p,v,f,original[:,None])
        # Two deterministic coordinate passes. Only authorized agents can change.
        # Coordinate descent is not exhaustive joint optimization for k>1.
        for _ in range(2):
            for i in range(command.shape[1]):
                allowed=self.tensor(active[:,i]).bool()
                if not bool(allowed.any()): continue
                candidates=self.candidate_set(work,v,i)
                costs,_=self.evaluate(p,v,f,candidates)
                idx=self.choose(costs,candidates,original)
                chosen=candidates[torch.arange(len(work),device=self.device),idx]
                work=torch.where(allowed[:,None,None],chosen,work)
        finalcost,finalgap=self.evaluate(p,v,f,work[:,None])
        # Numerical guard: never accept a predicted-cost increase.
        accept=finalcost[:,0]<=initialcost[:,0]+1e-7
        work=torch.where(accept[:,None,None],work,original)
        output=work.cpu().numpy()
        if not np.array_equal(output[~active],np.asarray(command,dtype=np.float32)[~active]):
            raise AssertionError('Prediction filter modified an unauthorized agent')
        return output,{'predicted_cost_before':initialcost[:,0].cpu().numpy(),
                       'predicted_cost_after':torch.where(accept,finalcost[:,0],initialcost[:,0]).cpu().numpy(),
                       'predicted_clearance_before':initialgap[:,0].cpu().numpy(),
                       'predicted_clearance_after':torch.where(accept,finalgap[:,0],initialgap[:,0]).cpu().numpy()}
    @torch.inference_mode()
    def target_benefit(self,command,diag,previous_force):
        """Privileged STATE-MODEL diagnostic, not a perfect/future-aware oracle.

        Scores estimate the reduction achievable by a SINGLE target under the
        finite candidate model. No attacker labels or future simulator states.
        """
        u=self.tensor(command);v=self.tensor(diag['vel'])
        base,_=self.evaluate(diag['pos'],diag['vel'],previous_force,u[:,None])
        gains=[]
        for i in range(command.shape[1]):
            cand=self.candidate_set(u,v,i)
            cost,_=self.evaluate(diag['pos'],diag['vel'],previous_force,cand)
            gains.append((base[:,0]-cost.min(1).values).clamp(min=0))
        return torch.stack(gains,1).cpu().numpy(),base[:,0].cpu().numpy()
