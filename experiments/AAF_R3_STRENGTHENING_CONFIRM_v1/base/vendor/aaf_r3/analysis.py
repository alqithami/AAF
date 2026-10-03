from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .common import atomic_json,digest,seed_for

CONTRASTS={"games":[("adaptive_rank","adaptive_random"),("adaptive_rank","threshold_rank"),
                     ("adaptive_rank","periodic_rank"),("aaf_full","adaptive_rank")],
           "physics":[("adaptive_rank","adaptive_random"),("adaptive_rank","threshold_rank"),("adaptive_rank","periodic_rank")]}
PRIMARY={"games":["executed","reward","active_fraction"],"physics":["collision_episode","goal_progress","modified_fraction"]}

def holm(ps):
    ps=np.asarray(ps,float);order=np.argsort(ps);out=np.ones(len(ps));running=0.
    for j,i in enumerate(order):running=max(running,(len(ps)-j)*ps[i]);out[i]=min(1,running)
    return out

def paired(x,y,key):
    d=np.asarray(x)-np.asarray(y);n=len(d)
    if n<2:return {"n":n,"difference":float(d.mean()),"ci_low":None,"ci_high":None,"p_t":None}
    rng=np.random.default_rng(seed_for("bootstrap",key))
    boots=d[rng.integers(0,n,size=(10000,n))].mean(1)
    sd=d.std(ddof=1)
    p=float(stats.ttest_1samp(d,0).pvalue) if sd>1e-15 else (1. if abs(d.mean())<1e-15 else 0.)
    return {"n":n,"difference":float(d.mean()),"ci_low":float(np.quantile(boots,.025)),
            "ci_high":float(np.quantile(boots,.975)),"p_t":p}

def analyze(root:Path,allow_partial=False):
    mf=json.loads((root/"RUN_MANIFEST.json").read_text());configs=mf["configs"]
    rows=[];missing=[];invalid=[]
    for cfg in configs:
        p=root/"runs"/digest(cfg)[:20]/"summary.json"
        if not p.exists():missing.append(digest(cfg)[:20]);continue
        x=json.loads(p.read_text())
        if x.get("status")!="complete" or x.get("config")!=cfg or x["provenance"]["source_sha256"]!=mf["identity"]["source_sha256"]:
            invalid.append(str(p));continue
        row={k:cfg.get(k,"") for k in ("suite","domain","seed","method","attack","network","profile")}
        row.update(x["summary"]);row["runtime_s"]=x["runtime_s"];row["preparation_s"]=x.get("preparation_s",0)
        if cfg["suite"]=="games":
            row.update({"post_"+k:v for k,v in x.get("post_summary",{}).items()})
            row.update({k:x.get(k) for k in ("trigger_count","admissions","denied","first_post_trigger_delay","ranking_top1","ranking_recall_at_k")})
        rows.append(row)
    out=root/"analysis";out.mkdir(exist_ok=True)
    status={"planned":len(configs),"found":len(rows),"missing":missing,"invalid":invalid,
            "profile":mf["identity"]["profile"],"research_results":mf["identity"]["profile"]=="main" and not missing and not invalid}
    atomic_json(out/"COMPLETENESS.json",status)
    if invalid or (missing and not allow_partial):raise RuntimeError(f"Incomplete/invalid results: {len(missing)} missing, {len(invalid)} invalid. See {out/'COMPLETENESS.json'}")
    if not rows:raise RuntimeError("No completed results")
    df=pd.DataFrame(rows);df.to_csv(out/"seed_results.csv",index=False)
    keys=["suite","domain","attack","network"]
    contrasts=[]
    for group,g in df.groupby(keys,dropna=False):
        suite=group[0]
        for a,b in CONTRASTS[suite]:
            A=g[g.method==a].set_index("seed");B=g[g.method==b].set_index("seed")
            common=sorted(set(A.index)&set(B.index))
            if not common:continue
            for metric in PRIMARY[suite]:
                result=paired(A.loc[common,metric].values,B.loc[common,metric].values,(group,a,b,metric))
                contrasts.append({**dict(zip(keys,group)),"a":a,"b":b,"metric":metric,**result})
    c=pd.DataFrame(contrasts)
    if not c.empty:
        c["p_holm"]=np.nan
        for suite in ("games","physics"):
            idx=c.index[(c.suite==suite)&c.p_t.notna()]
            if len(idx):c.loc[idx,"p_holm"]=holm(c.loc[idx,"p_t"].values)
        c.to_csv(out/"paired_comparisons.csv",index=False)
    metrics=sorted(set(k for r in rows for k in r if k not in keys+["seed","method","profile"]))
    descriptive=df.groupby(keys+["method"],dropna=False)[metrics].agg(["mean","std","count"])
    descriptive.to_csv(out/"group_summary.csv")
    sensitivity=[]
    for group,g in df[df.suite=="games"].groupby(keys,dropna=False):
        for method in ("ppo_only","aaf_full"):
            A=g[g.method==method].set_index("seed");B=g[g.method=="legacy__"+method].set_index("seed")
            ids=sorted(set(A.index)&set(B.index))
            if ids:
                for metric in PRIMARY["games"]:
                    sensitivity.append({**dict(zip(keys,group)),"corrected":method,"metric":metric,
                        **paired(A.loc[ids,metric].values,B.loc[ids,metric].values,(group,method,"legacy",metric))})
    if sensitivity:pd.DataFrame(sensitivity).to_csv(out/"legacy_update_sensitivity.csv",index=False)
    # Physics quality check uses seed-level nominal-policy comparisons to random.
    quality=[]
    g=df[(df.suite=="physics")&(df.attack=="none")&(df.network=="clean")]
    if len(g):
        A=g[g.method=="ppo_only"].set_index("seed");B=g[g.method=="random_policy"].set_index("seed")
        ids=sorted(set(A.index)&set(B.index))
        if ids:
            q=paired(A.loc[ids,"goal_progress"].values,B.loc[ids,"goal_progress"].values,"nominal_policy_quality")
            quality.append({"test":"nominal PPO goal progress minus random-policy goal progress",**q,
                            "passes_prespecified_quality_check":q["ci_low"] is not None and q["ci_low"]>0})
    atomic_json(out/"PHYSICS_POLICY_QUALITY.json",quality)
    eligibility={"complete_main_dataset":status["research_results"],
        "physics_policy_quality_passed":all(q["passes_prespecified_quality_check"] for q in quality) if quality else None,
        "human_scientific_review_required":True,"automatic_submission_ready":False}
    atomic_json(out/"SCIENTIFIC_REVIEW_GATE.json",eligibility)
    lines=["# Reviewer-3 experiment analysis",f"Profile: **{status['profile']}**. Complete: {len(rows)}/{len(configs)}.",
           "One independent seed is the inferential unit. VMAS episodes are averaged within trained-policy seed.",
           "Differences are A minus B. Bootstrap intervals are marginal paired-seed intervals, not simultaneous intervals.",
           "Holm correction groups all prespecified contrasts and primary outcomes within each suite.",
           "No significance or superiority conclusion should be based on smoke/pilot outputs."]
    if not status["research_results"]:lines.append("**NOT FOR MANUSCRIPT RESULTS: smoke, pilot, or incomplete dataset.**")
    if quality and not all(x["passes_prespecified_quality_check"] for x in quality):
        lines.append("**Physics policy-quality check not passed. Inspect learning curves before interpreting runtime comparisons. Do not select a winning seed.**")
    if not c.empty:lines.append(c.to_string(index=False))
    (out/"REPORT.md").write_text("\n\n".join(lines)+"\n")
    # Tables only from completed main runs; never fill a manuscript with smoke data.
    if status["research_results"] and (not quality or all(q["passes_prespecified_quality_check"] for q in quality)):
        table=df.groupby(keys+["method"],dropna=False)[metrics].mean().reset_index()
        (out/"seed_mean_table.tex").write_text(table.to_latex(index=False,float_format="%.4f",escape=True))
    from .plots import plot_results
    plot_results(df,out,status["profile"])
    print(f"Analysis written to {out}. Research-results status: {status['research_results']}")
