"""Equations/features adapted from alqithami/AAF at the recorded base commit.

This is a separate experiment implementation, not a bitwise reproduction of the
historical runner. Environment reward and allocation equations are unchanged.
"""
from __future__ import annotations
import numpy as np
from .common import gini

class Game:
    def __init__(self, name: str, n: int, seed: int, alpha: float = 1.0,
                 penalty: float = 0.2, noise: float = 0.01):
        if name not in ("resource_sharing", "public_goods"): raise ValueError(name)
        if n < 6: raise ValueError("Use at least six agents for the graph.")
        self.name, self.n, self.alpha, self.penalty, self.noise = name, n, alpha, penalty, noise
        self.rng = np.random.default_rng(seed)
        self.adj = np.zeros((n,n), float)
        if name == "resource_sharing":
            neighbors = [set() for _ in range(n)]
            for i in range(n):
                for j in (1,2):
                    b = (i+j)%n; neighbors[i].add(b); neighbors[b].add(i)
            for i in range(n):
                for j in (1,2):
                    if self.rng.random() < 0.1:
                        b = (i+j)%n; neighbors[i].discard(b); neighbors[b].discard(i)
                        candidates = sorted(set(range(n)) - {i} - neighbors[i])
                        b2 = int(self.rng.choice(candidates)) if candidates else b
                        neighbors[i].add(b2); neighbors[b2].add(i)
            for i, ns in enumerate(neighbors):
                if ns: self.adj[i, sorted(ns)] = 1/len(ns)
        self.obs_dim = 4 if name == "resource_sharing" else 3
        self.last_action = np.zeros(n); self.last_alloc = np.zeros(n)
        self.reset()

    def reset(self) -> np.ndarray:
        self.last_action[:] = 0; self.last_alloc[:] = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        if self.name == "resource_sharing":
            x = np.column_stack([self.last_alloc/100, self.last_action,
                                 self.adj@self.last_action, np.ones(self.n)])
        else:
            x = np.column_stack([self.last_action, self.last_alloc,
                                 np.full(self.n, self.last_action.mean())])
        return (x + self.rng.normal(0, self.noise, x.shape)).astype(np.float32)

    def violations(self, actions: np.ndarray) -> np.ndarray:
        a = np.asarray(actions).reshape(self.n)
        return (a >= 0.6 if self.name == "resource_sharing" else a < 0.2).astype(float)

    def shield(self, proposed: np.ndarray, mask: np.ndarray) -> np.ndarray:
        a = proposed.copy()
        if self.name == "resource_sharing": a[mask] = np.minimum(a[mask], 0.6-1e-4)
        else: a[mask] = np.maximum(a[mask], 0.2+1e-4)
        return a

    def step(self, actions: np.ndarray):
        a = np.clip(np.asarray(actions).reshape(self.n), 0, 1)
        violation = self.violations(a)
        if self.name == "resource_sharing":
            q = a*100
            w = (q>0).astype(float) if abs(self.alpha)<1e-12 else q**self.alpha
            alloc = q.copy() if q.sum()<=100 else 100*w/max(w.sum(),1e-12)
            reward = alloc - self.penalty*violation + 0.3*alloc.mean()
        else:
            alloc = np.full(self.n, 1.6*a.mean())
            reward = 1-a+alloc-self.penalty*violation+0.3*alloc.mean()
        self.last_action = a.copy(); self.last_alloc = alloc.copy()
        return self.observation(), reward.astype(np.float32), {
            "violation": violation, "gini": gini(alloc), "allocation": alloc}
