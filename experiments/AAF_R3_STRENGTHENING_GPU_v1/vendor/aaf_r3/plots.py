"""Data-only plot generation; never replaces the preserved manuscript figures."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t

STYLE={
 "aaf_full":("AAF + shaping","#006D77","o"),
 "adaptive_rank":("Adaptive + evidence ranking","#006D77","s"),
 "adaptive_random":("Adaptive + random-k","#5C677D","^"),
 "threshold_rank":("Instant threshold + ranking","#3A86A8","D"),
 "periodic_rank":("Periodic + ranking","#7A6F9B","v"),
 "ppo_only":("PPO only","#5C677D","x"),
 "static_guard":("Unrestricted guard","#9A8F7A","P"),
 "random_policy":("Random policy (quality check)","#9A8F7A","+")}

def interval(x):
    x=np.asarray(x,float);n=len(x)
    return float(t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)) if n>1 else 0.

def plot_results(df,out:Path,profile:str):
    out=out/"figures";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"ps.fonttype":42})
    for key,g in df.groupby(["suite","domain","attack","network"],dropna=False):
        suite,domain,attack,network=key
        x,y=("active_fraction","executed") if suite=="games" else ("goal_progress","collision_episode")
        fig,ax=plt.subplots(figsize=(8,4.8))
        for method,(label,color,marker) in STYLE.items():
            v=g[g.method==method]
            if not len(v):continue
            ax.errorbar(v[x].mean(),v[y].mean(),xerr=interval(v[x]),yerr=interval(v[y]),
                        fmt=marker,color=color,markersize=6,capsize=3,elinewidth=.8,label=label)
        ax.set_xlabel("Constrained agent-time fraction" if suite=="games" else "Mean progress toward goals (simulator units)")
        ax.set_ylabel("Executed violation fraction" if suite=="games" else "Fraction of episodes with contact")
        ax.set_title(f"{domain} | {attack}"+(f" | {network}" if network else "")+(f" [{profile.upper()}: diagnostic only]" if profile!="main" else ""))
        ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.16)
        ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False,fontsize=8)
        fig.tight_layout()
        name="__".join(str(k) for k in key if str(k))
        fig.savefig(out/f"{name}.pdf",bbox_inches="tight")
        fig.savefig(out/f"{name}.png",dpi=240,bbox_inches="tight");plt.close(fig)
