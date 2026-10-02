"""VMAS navigation adapter. Requires the actual vmas==1.5.2 dependency.

There is intentionally no toy-environment substitute on import failure. Local
actuation adds first-order lag; communication faults are explicit extensions.
The braking filter is a heuristic, not a certified collision-avoidance shield.
"""
from __future__ import annotations
import importlib.metadata as metadata
import time
from pathlib import Path
import numpy as np
import torch
from .common import seed_all,seed_for,digest,atomic_json,provenance,finite_mean
from .ppo import PPO
from .governor import Governor,Authority,calibrate
from .network import EvidenceChannel,CommandChannel

PHYSICS_METHODS=("ppo_only","aaf_full","adaptive_rank","adaptive_random","threshold_rank","periodic_rank","static_guard","random_policy")

class Navigation:
    def __init__(self, worlds:int, n:int, seed:int, device:torch.device):
        try:
            import vmas
        except ImportError as e:
            raise RuntimeError("VMAS is required: python -m pip install vmas==1.5.2 . No substitute environment will be used.") from e
        version=metadata.version("vmas")
        if version!="1.5.2": raise RuntimeError(f"Expected vmas==1.5.2, found {version}. Use the supplied requirements.")
        self.e,self.n,self.device=worlds,n,device
        self.env=vmas.make_env(scenario="navigation",num_envs=worlds,device=str(device),
            continuous_actions=True,wrapper=None,max_steps=None,seed=seed,dict_spaces=False,
            n_agents=n,collisions=True,shared_rew=False,observe_all_goals=False)
        self.last_force=np.zeros((worlds,n,2),np.float32)
        self.obs=self.reset(seed)
        self.obs_dim=self.obs.shape[-1]
        self.radius=np.asarray([a.shape.radius for a in self.env.agents])
        for a in self.env.agents:
            if not np.isclose(float(a.action.u_range),1): raise RuntimeError("Unexpected VMAS action range")
    def reset(self,seed:int):
        obs=self.env.reset(seed=seed)
        self.last_force[:]=0
        return self.pack_obs(obs)
    def pack_obs(self,obs):
        if not isinstance(obs,(tuple,list)) or len(obs)!=self.n: raise RuntimeError("Unexpected VMAS observation API")
        return torch.stack(list(obs),dim=1).detach().cpu().numpy().reshape(self.e*self.n,-1)
    def state(self):
        p=torch.stack([a.state.pos for a in self.env.agents],1).detach().cpu().numpy()
        v=torch.stack([a.state.vel for a in self.env.agents],1).detach().cpu().numpy()
        goals=torch.stack([a.goal.state.pos for a in self.env.agents],1).detach().cpu().numpy()
        return p,v,goals
    def diagnostics(self,speed_limit:float=.3):
        p,v,g=self.state();dist=np.linalg.norm(p[:,:,None,:]-p[:,None,:,:],axis=-1)
        contact_threshold=self.radius[:,None]+self.radius[None,:]+.005
        contact=dist<=contact_threshold[None,:,:]
        contact[:,np.arange(self.n),np.arange(self.n)]=False
        near=dist<(self.radius[:,None]+self.radius[None,:]+.1)[None,:,:]
        near[:,np.arange(self.n),np.arange(self.n)]=False
        speed=np.linalg.norm(v,axis=-1)
        goal_radius=np.asarray([a.goal.shape.radius for a in self.env.agents])
        goal_distance=np.linalg.norm(p-g,axis=-1)
        on_goal=goal_distance<goal_radius[None,:]
        # Risk involvement is operational evidence, not agent culpability.
        risk=(speed>speed_limit)|contact.any(-1)
        return {"pos":p,"vel":v,"goals":g,"contact":contact.any(-1),"near":near.any(-1),
                "speed":speed,"risk":risk.astype(float),"goal_distance":goal_distance,"on_goal":on_goal}
    def step(self,command):
        # Standard force range followed by a declared actuator lag, beta=0.5.
        self.last_force=.5*self.last_force+.5*np.clip(command,-1,1)
        actions=[torch.as_tensor(self.last_force[:,i,:],dtype=torch.float32,device=self.device) for i in range(self.n)]
        result=self.env.step(actions)
        if len(result)!=4: raise RuntimeError("Unexpected VMAS step API; expected obs, reward, done, info")
        obs,reward,done,info=result
        return self.pack_obs(obs),torch.stack(list(reward),1).detach().cpu().numpy(),done.detach().cpu().numpy()
    def close(self):
        if hasattr(self.env,"close"):self.env.close()

def braking_command(vel:np.ndarray)->np.ndarray:
    return np.clip(-2.0*vel,-1,1)

def apply_brake_filter(command:np.ndarray,active:np.ndarray,diag:dict)->np.ndarray:
    out=command.copy()
    # Braking is triggered by local speed or near-contact evidence; collisions
    # due to momentum/contact forces may still occur and are measured separately.
    need=active & ((diag["speed"]>.3)|diag["near"])
    brake=braking_command(diag["vel"]);out[need]=brake[need]
    return out

def prepare_navigation(cfg:dict,device:torch.device,root:Path):
    key={k:cfg[k] for k in ("seed","n_agents","train_worlds","train_updates","rollout","horizon","calibration_steps")}
    path=root/"initial_policies"/("vmas_"+digest(key)[:16]+".pt")
    seed_all(seed_for(cfg["seed"],"vmas_init"))
    nav=Navigation(cfg["train_worlds"],cfg["n_agents"],seed_for(cfg["seed"],"vmas_train_env"),device)
    agent=PPO(nav.obs_dim,2,device,rollout=cfg["rollout"])
    if path.exists():
        ck=torch.load(path,map_location=device,weights_only=False)
        if ck["key"]!=key:raise RuntimeError("VMAS checkpoint identity mismatch")
        agent.net.load_state_dict(ck["net"]);agent.opt.load_state_dict(ck["opt"]);nav.close()
        return agent,ck["calibration"],ck["training_trace"]
    obs=nav.obs;alive=np.ones(nav.e,bool);trace=[];block_reward=[]
    steps=cfg["train_updates"]*cfg["rollout"]
    for t in range(steps):
        proposal,lp,v=agent.act(obs);command=(2*proposal-1).reshape(nav.e,nav.n,2)
        nobs,reward,done=nav.step(command)
        mask=np.repeat(alive,nav.n).astype(float)
        terminal=done|(t%cfg["horizon"]==cfg["horizon"]-1)|(t==steps-1)
        agent.add(obs,proposal,lp,v,reward.reshape(-1),np.repeat(terminal,nav.n),mask)
        stats=agent.update(nobs,force=t==steps-1)
        block_reward.append(float(reward[alive].mean()) if alive.any() else 0.0)
        alive &= ~done
        if stats:trace.append({"t":t,"mean_reward":float(np.mean(block_reward)),**stats});block_reward=[]
        obs=nobs
        if (t+1)%cfg["horizon"]==0:
            obs=nav.reset(seed_for(cfg["seed"],"vmas_train_reset",t));alive[:]=True
    nav.close()
    nav=Navigation(cfg["train_worlds"],cfg["n_agents"],seed_for(cfg["seed"],"vmas_calibration_env"),device)
    seed_all(seed_for(cfg["seed"],"vmas_calibration_policy"));obs=nav.obs;zs=[]
    for t in range(cfg["calibration_steps"]):
        a,_,_=agent.act(obs);obs,_,_=nav.step((2*a-1).reshape(nav.e,nav.n,2));zs.extend(nav.diagnostics()["risk"].mean(1).tolist())
        if (t+1)%cfg["horizon"]==0:obs=nav.reset(seed_for(cfg["seed"],"vmas_cal_reset",t))
    cal=calibrate(zs);nav.close()
    # Test episodes are disjoint from training and calibration episodes.
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(".tmp")
    torch.save({"key":key,"net":agent.net.state_dict(),"opt":agent.opt.state_dict(),"calibration":cal,"training_trace":trace},temp);temp.replace(path)
    atomic_json(path.with_suffix(".json"),{"key":key,"calibration":cal,"training_trace":trace})
    return agent,cal,trace

def run_navigation(cfg:dict,device:torch.device,root:Path)->dict:
    preparation_start=time.perf_counter();agent,cal,training_trace=prepare_navigation(cfg,device,root)
    preparation_s=time.perf_counter()-preparation_start;tick=time.perf_counter()
    e,n=cfg["eval_worlds"],cfg["n_agents"]
    nav=Navigation(e,n,seed_for(cfg["seed"],"vmas_TEST_env"),device)
    seed_all(seed_for(cfg["seed"],"vmas_TEST_policy"))
    method=cfg["method"];gov=Governor("ppo_only" if method=="random_policy" else method,e,n,
        seed_for(cfg["seed"],"vmas_targets"),cal,Authority(**cfg["authority"]))
    evidence=EvidenceChannel(e,n,seed_for(cfg["seed"],"vmas_evidence"),cfg["network"])
    commands=CommandChannel(e,n,seed_for(cfg["seed"],"vmas_commands"),cfg["network"])
    rng=np.random.default_rng(seed_for(cfg["seed"],"vmas_random_policy"))
    ids=np.random.default_rng(seed_for(cfg["seed"],"vmas_attacker")).integers(0,n,size=e)
    obs=nav.obs;alive=np.ones(e,bool);success=np.zeros(e,bool);contacts=np.zeros(e,bool)
    initial=nav.diagnostics();start_distance=initial["goal_distance"].mean(1)
    final_distance=start_distance.copy();on_goal=np.zeros(e);first_goal=np.full(e,np.nan)
    sums={k:np.zeros(e) for k in ("risk","reward","contact_fraction","overspeed","active_fraction","modified_fraction","action_l1","deadline_fallback","evidence_coverage")}
    lengths=np.zeros(e,int);exposed=np.zeros(e,bool);events=[];timings=[]
    for t in range(cfg["horizon"]):
        diag=nav.diagnostics();proposal,_,_=agent.act(obs)
        if method=="random_policy":proposal=rng.uniform(0,1,(e*n,2)).astype(np.float32)
        submitted=(2*proposal-1).reshape(e,n,2)
        if cfg["attack"]=="pursuit" and t>=cfg["attack_start"]:
            exposed |= alive
            # Scripted pursuit, NOT an adaptive trained attacker. Identity never
            # enters the governor or detector; it is used only here and in logs.
            for j,i in enumerate(ids):
                delta=diag["pos"][j]-diag["pos"][j,i]
                distance=np.linalg.norm(delta,axis=1);distance[i]=np.inf
                target=int(distance.argmin());d=delta[target]
                submitted[j,i]=d/max(np.linalg.norm(d),1e-8)
        command,expired=commands.step(t,submitted,braking_command(diag["vel"]))
        if device.type=="cuda":torch.cuda.synchronize()
        t0=time.perf_counter();active=gov.active(t)
        executed=apply_brake_filter(command,active,diag)
        if device.type=="cuda":torch.cuda.synchronize()
        timings.append(time.perf_counter()-t0)
        nextobs,reward,done=nav.step(executed);after=nav.diagnostics()
        delivered=evidence.send_receive(t,after["risk"])
        event=gov.observe(t,delivered)
        # aaf_full uses the same frozen nominal policy here. No online reward
        # update takes place; it is expected to equal adaptive_rank in physics.
        metrics={"risk":after["risk"].mean(1),"reward":reward.mean(1),
            "contact_fraction":after["contact"].mean(1),"overspeed":(after["speed"]>.3).mean(1),
            "active_fraction":active.mean(1),"modified_fraction":(np.linalg.norm(executed-command,axis=-1)>1e-7).mean(1),
            "action_l1":abs(executed-command).mean((1,2)),"deadline_fallback":expired.mean(1),
            "evidence_coverage":np.isfinite(delivered).mean(1)}
        for k in sums:sums[k]+=np.where(alive,metrics[k],0)
        lengths+=alive;contacts |= alive & after["contact"].any(1)
        on_goal[alive]=after["on_goal"].mean(1)[alive];final_distance[alive]=after["goal_distance"].mean(1)[alive]
        just=alive & done
        success|=just;first_goal[just]=t+1;alive &= ~done
        if event["trigger"].any(): events.append({"t":t,"trigger_worlds":np.flatnonzero(event["trigger"]).tolist(),"selected":np.argwhere(event["selected"]).tolist()})
        obs=nextobs
    episodes=[]
    for j in range(e):
        episodes.append({"episode":j,"length":int(lengths[j]),"success":bool(success[j]),
            "collision_episode":bool(contacts[j]),"attack_exposed":bool(exposed[j]),"first_goal_step":int(first_goal[j]) if np.isfinite(first_goal[j]) else None,
            "goal_fraction":float(on_goal[j]),"initial_goal_distance":float(start_distance[j]),
            "final_goal_distance":float(final_distance[j]),"goal_progress":float(start_distance[j]-final_distance[j]),
            **{k:float(v[j]/max(1,lengths[j])) for k,v in sums.items()}})
    nav.close()
    summary={k:float(np.mean([r[k] for r in episodes])) for k in sums}
    summary.update({k:float(np.mean([r[k] for r in episodes])) for k in ("success","collision_episode","goal_fraction","goal_progress","final_goal_distance","attack_exposed")})
    return {"config":cfg,"calibration":cal,"provenance":provenance(str(device)),"summary":summary,
        "episodes":episodes,"alarms":events,"training_trace":training_trace,"runtime_s":time.perf_counter()-tick,"preparation_s":preparation_s,
        "gateway_runtime_p50_ms":float(np.quantile(timings,.5)*1000),"gateway_runtime_p95_ms":float(np.quantile(timings,.95)*1000),
        "gateway_runtime_max_ms":float(max(timings)*1000),
        "timing_scope":"Measured filter computation only; excludes policy, physics, transport and rendering. Deadline faults are simulated control-step ages.",
        "scope":"Frozen held-out runtime evaluation after nominal PPO training; aaf_full and adaptive_rank coincide by design. Scripted pursuit only. Braking is not certified safety.",
        "channel":{"sent":evidence.sent,"dropped":evidence.dropped,"expired":evidence.expired,"delivered":evidence.delivered}}

def preflight_navigation(device:torch.device):
    nav=Navigation(2,4,671002,device)
    try:
        obs=nav.obs
        for _ in range(5):
            obs,r,d=nav.step(np.zeros((2,4,2),np.float32))
            assert obs.shape[0]==8 and r.shape==(2,4) and d.shape==(2,)
            assert np.isfinite(obs).all()
        return {"status":"PASS","engine":"VMAS 1.5.2","observation_shape":list(obs.shape),"world_dt":float(nav.env.world.dt)}
    finally:nav.close()
