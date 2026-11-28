"""V7 declarative recipe generation and performance-per-compute search.
Only reviewed, finite training operators; no generated executable Python or web code.
"""
from __future__ import annotations
import math
import statistics
import hashlib
import json
import ralph_v6 as ctl
import ralph_v610 as focus
import ralph_v611 as oldloop
import ralph_v6_refine as refine

VERSION="v7-declarative-v1"
# Immutable catalog, each mutation changes one scientific hypothesis (unless noted).
CATALOG=(
    ("continue","continue",{}),
    ("lr_half","continue",{"lr_factor":.5}),
    ("lr_quarter","continue",{"lr_factor":.25}),
    ("phase_derivatives","phase",{"v610_phase_derivatives":True,"v7_phase_weight":.08}),
    ("phase_light","phase",{"v610_phase_derivatives":True,"v7_phase_weight":.04}),
    ("phase_strong","phase",{"v610_phase_derivatives":True,"v7_phase_weight":.16}),
    ("spectral_balance","spectral",{"v7_spectral_weight":.04}),
    ("spectral_strong","spectral",{"v7_spectral_weight":.08}),
    ("hard_noise_first","noise",{"v610_hard_first":True,"v6_hard":True}),
    ("full_length","context",{"v6_crop":48000}),
    ("critic_warm24","critic",{"v6_critic":True,"v6_crop":48000,"v7_critic_warm_count":24,"v7_critic_weight":.025}),
    ("critic_warm48","critic",{"v6_critic":True,"v6_crop":48000,"v7_critic_warm_count":48,"v7_critic_weight":.025}),
    ("critic_light","critic",{"v6_critic":True,"v6_crop":48000,"v7_critic_warm_count":24,"v7_critic_weight":.015}),
    ("residual","residual",{"new_residual":True}),
    ("residual_trust","residual",{"v7_residual_weight":.02}),
    ("residual_trust_light","residual",{"v7_residual_weight":.01}),
    ("korean128","noise",{"v6_korean":128}),
)
BY_NAME={name:(family,dict(delta)) for name,family,delta in CATALOG}
OLD_FAMILY={"continue":"continue","phase_derivatives":"phase","hard_noise_first":"noise","hard_noise":"noise","full_length":"context","critic":"critic","residual":"residual"}

def source():
    code=focus.source()
    extra=(ctl.ROOT/"src/v7_objectives.py").read_text(encoding="utf-8")
    rep=focus.replace_once
    code=rep(code,"def v6_loss(pred,clean,cfg):",extra+"\n\ndef v6_loss(pred,clean,cfg):")
    needle='    if cfg.get("v610_phase_derivatives"): loss=loss+.08*v610_phase_derivatives(pred.float(),clean.float())'
    code=rep(code,needle,'''    if cfg.get("v610_phase_derivatives"): loss=loss+float(cfg.get("v7_phase_weight",.08))*v610_phase_derivatives(pred.float(),clean.float())
    if cfg.get("v7_spectral_weight"): loss=loss+float(cfg["v7_spectral_weight"])*v7_spectral_balance(pred.float(),clean.float())''')
    needle='    loss=v6_loss(pred,s,cfg)'
    code=rep(code,needle,needle+'''
    if cfg.get("v7_residual_weight"):
        if not isinstance(model,V6Refiner): raise ValueError("Residual trust requires cascade")
        loss=loss+float(cfg["v7_residual_weight"])*v7_residual_trust(pred,stages[0],x)''')
    needle='    opt=None;co=None;critic=None;replay=None;lr=float(cfg["lr"])'
    code=rep(code,needle,needle+'\n    warmup_updates=0;warmup_calibration=None')
    needle='            critic=PESQCritic().to(device);co=torch.optim.AdamW(critic.parameters(),lr=3e-4);replay=CriticReplay(seed=cfg["seed"])'
    code=rep(code,needle,needle+'''
            if cfg.get("v7_critic_warm_count"):
                warmup_updates,warmup_calibration=v7_critic_warmup(model,critic,co,replay,train,cal_ids,cal,device,pesq_fn,cfg["seed"],int(cfg["v7_critic_warm_count"]))
                updates+=warmup_updates;calibration=warmup_calibration
                print("V7 CRITIC WARMUP "+json.dumps({"updates":warmup_updates,"calibration":calibration}),flush=True)''')
    needle='            losses=[];weight=adaptive_metric_weight(calibration,epoch,updates*2) if critic is not None else 0.'
    code=rep(code,needle,needle+'''
            if critic is not None and cfg.get("v7_critic_warm_count"):
                weight=v7_critic_gate(calibration,updates*2,float(cfg["v7_critic_weight"]))''')
    needle='    (WORK/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")'
    code=rep(code,needle,'''    result["v7"]={"protocol":"v7-declarative-v1","recipe":cfg["v7_recipe"],"warmup_updates":warmup_updates,
        "warmup_calibration":warmup_calibration,"mutation":cfg["v7_mutation"],"parent":cfg.get("v6_parent")}
'''+needle)
    compile(code,"v7_runner.py","exec")
    return code

def digest(code):return hashlib.sha256(code.encode()).hexdigest()

def settings(old):
    return {"v610_phase_derivatives":bool(old.get("v610_phase_derivatives")),"v610_hard_first":bool(old.get("v610_hard_first")),
        "v6_hard":bool(old.get("v6_hard")),"v6_critic":bool(old.get("v6_critic")),"v6_crop":int(old["v6_crop"]),
        "v6_korean":int(old["v6_korean"]),"v7_phase_weight":float(old.get("v7_phase_weight",.08 if old.get("v610_phase_derivatives") else 0.)),
        "v7_spectral_weight":float(old.get("v7_spectral_weight",0.)),"v7_residual_weight":float(old.get("v7_residual_weight",0.)),
        "v7_critic_warm_count":int(old.get("v7_critic_warm_count",0)),"v7_critic_weight":float(old.get("v7_critic_weight",.025))}

def validate_settings(c):
    if c["v7_phase_weight"] not in (0.,.04,.08,.16) or c["v7_spectral_weight"] not in (0.,.04,.08) or c["v7_residual_weight"] not in (0.,.01,.02):
        raise ValueError("Unapproved loss weights")
    if c["v7_critic_warm_count"] not in (0,24,48) or c["v7_critic_weight"] not in (.015,.025,.035):
        raise ValueError("Unapproved critic settings")
    if c["v6_crop"] not in (16000,48000) or c["v6_critic"] and c["v6_crop"]!=48000:
        raise ValueError("Invalid crop/critic")
    if c["v7_critic_warm_count"] and not c["v6_critic"]:raise ValueError("Warmup without critic")
    if c["v7_residual_weight"] and not c.get("v6_depth"):raise ValueError("Trust without cascade")
    if c["v610_hard_first"] and not c["v6_hard"]:raise ValueError("Hard init without hard sampling")

def available(old,name,enable_korean=False):
    if name=="korean128" and not enable_korean:return False
    _,delta=BY_NAME[name]
    if delta.get("new_residual"):return not old.get("v6_depth")
    if name.startswith("residual_trust") and not old.get("v6_depth"):return False
    if name=="full_length" and old.get("v6_critic"):return False
    if name=="korean128" and old.get("v6_korean",0)>=128:return False
    if "lr_factor" in delta or not delta:return True
    cur=settings(old)
    return any(cur.get(k)!=v for k,v in delta.items())

def build(parent,name,sha,cache,epochs,mix,pub,smoke=False):
    st,r,old,_=parent
    if name not in BY_NAME:raise ValueError("Unapproved research mutation")
    family,delta=BY_NAME[name]
    if not available(old,name,True):raise ValueError("No-op or inapplicable mutation")
    if (old.get("v6_base_family") or old["architecture"])!="mp_dns":raise ValueError("MP-DNS only")
    s=settings(old);s.update({k:v for k,v in delta.items() if k in s})
    new_residual=bool(delta.get("new_residual"));depth=2 if new_residual else old.get("v6_depth")
    arch="v6_cascade" if new_residual else old["architecture"]
    lr=float(old["lr"])
    if r["v6"]["attempted_epochs"]>0:lr=min(lr,float(r["v6"]["lr"]))
    if new_residual:lr=1e-4
    lr=max(1e-8,lr*float(delta.get("lr_factor",1.)))
    if s["v6_critic"]:s["v6_crop"]=48000
    if not smoke and s["v6_critic"] and not s["v7_critic_warm_count"] and epochs<3:
        raise ValueError("Unwarmed critic needs 3 epochs")
    c=ctl.config(arch,"cascade" if depth else "train",sha,{},cache=None if smoke else cache,
        epochs=1 if smoke else epochs,mix=16 if smoke else mix,public_mix=16 if smoke else pub,
        korean=0 if smoke else s["v6_korean"],critic=s["v6_critic"],hard=s["v6_hard"],
        parent=None if smoke else ctl.parent_info(st,r),base="mp_dns" if depth else None,depth=depth,
        coarse=new_residual and not smoke,smoke=smoke,lr=lr)
    c.update(s)
    if smoke:c["v6_korean"]=0
    c["v6_phase"]=old.get("v6_phase",True);c["v6_external_margin"]=old.get("v6_external_margin",.15)
    if old.get("v6_sisdr"):c["v6_sisdr"]=True
    c.update(v7_recipe=name,v7_mutation=delta,v610_recipe=name)
    validate_settings(c)
    return refine.resign(c,sha)

def verify_result(r,c):
    v=r.get("v7",{})
    if v.get("protocol")!=VERSION or v.get("recipe")!=c["v7_recipe"] or v.get("mutation")!=c["v7_mutation"] or v.get("parent")!=c["v6_parent"]:
        raise ValueError("V7 result provenance mismatch")
    n=int(c["v7_critic_warm_count"])
    if v.get("warmup_updates")!=n or bool(v.get("warmup_calibration"))!=bool(n):
        raise ValueError("V7 warmup evidence mismatch")
    if n and v["warmup_calibration"].get("count")!=32:raise ValueError("V7 critic calibration count")
    return r

def history(entries):
    """Observed gain / measured epoch-hours; excludes smoke and estimated billing."""
    out={};attempts={}
    for st,r,c,folder in entries.values():
        if st.get("status") not in ("COMPLETE","ERROR_TERMINAL"):continue
        if c is None:
            try:c=ctl.config_of(folder)
            except (OSError,ValueError,SyntaxError,KeyError):continue
        if c.get("smoke") or not c.get("v6_parent"):continue
        if (c.get("v6_base_family") or c.get("architecture"))!="mp_dns":continue
        name=c.get("v7_recipe") or c.get("v611_recipe") or c.get("v610_recipe") or c.get("v6_refine_hypothesis")
        if not name:continue
        if name=="continue" and c.get("v611_lr_step")==1:name="lr_half"
        family=BY_NAME[name][0] if name in BY_NAME else OLD_FAMILY.get(name)
        if not family:continue
        h=c["v6_parent"]["sha256"]
        attempts.setdefault(h,set()).add(name)
        if r is not None and c.get("v7_recipe"):verify_result(r,c)
        if r is None:
            out.setdefault(family,[]).append(0.)
            continue
        seconds=sum(float(v) for v in r.get("epoch_seconds",[]) if math.isfinite(float(v)) and float(v)>0)
        # Approximate measured compute proxy, not total job time or Kaggle billing.
        if seconds<=0:continue
        gain=max(0.,float(r["best_validation_pesq_wb"])-float(c["v6_parent"]["score"]))
        out.setdefault(family,[]).append(min(.5,gain/max(.1,seconds/3600)))
    return out,attempts

def rank(parent,diag,entries,enable_korean=False):
    _,r,c,_=parent
    outcomes,attempts=history(entries);done=attempts.get(r["v6"]["checkpoint_sha256"],set())
    phase=float(diag["phase_gain"]);mag=float(diag["magnitude_gain"])
    if not math.isfinite(phase) or not math.isfinite(mag):raise ValueError("Invalid diagnostic")
    n=sum(len(v) for v in outcomes.values())
    prior={"phase":.06 if phase>=mag-.1 else .025,"spectral":.06 if mag>phase+.1 else .025,
           "noise":.045 if mag>phase else .02,"context":.025,"critic":.02,"residual":.015,"continue":.035}
    if r["v6"].get("attempted_epochs",0)==0 or r["v6"].get("gain",0.)>=.002:prior["continue"]+=.04
    scores=[]
    for i,(name,family,delta) in enumerate(CATALOG):
        if name in done or not available(c,name,enable_korean):continue
        vals=outcomes.get(family,[])
        mean=(sum(vals)+2*prior[family])/(len(vals)+2)
        exploration=.025*math.sqrt(math.log(n+2)/(len(vals)+1))
        # Prefer canonical changes over their tuning variants when evidence is tied.
        priority=mean+exploration-.001*i
        scores.append((priority,i,name,{"family":family,"observations":len(vals),"posterior_gain_per_epoch_hour":mean,
            "exploration_bonus":exploration,"priority":priority}))
    return [(name,why) for _,_,name,why in sorted(scores,key=lambda x:(-x[0],x[1]))]

def cost(c,entries):
    t=oldloop.estimate(c,entries)
    if c.get("v7_critic_warm_count"):t+=.2+.006*c["v7_critic_warm_count"]
    if c.get("v7_spectral_weight"):t*=1.1
    if c.get("v6_korean"):t+=.5
    return t

def pick(parent,diag,entries,sha,cache,remaining,max_job,max_epochs,chunk,allow_reduce,enable_korean=False):
    old=parent[2];left=max_epochs-parent[1]["v6"]["attempted_epochs"]
    if left<=0:return None
    for name,why in rank(parent,diag,entries,enable_korean):
        # Keep legacy critic's minimum 3 epochs unless warmup is enabled.
        s=settings(old);s.update({k:v for k,v in BY_NAME[name][1].items() if k in s})
        minimum=3 if s["v6_critic"] and not s["v7_critic_warm_count"] else 1
        upper=min(left,max(minimum,chunk) if name in ("continue","lr_half","lr_quarter") else minimum)
        for epochs in range(upper,minimum-1,-1):
            for mix,pub in oldloop.profiles(old,allow_reduce):
                c=build(parent,name,sha,cache,epochs,mix,pub)
                sm=build(parent,name,sha,cache,epochs,mix,pub,True)
                if c["v56_key"] in entries:continue
                prior=entries.get(sm["v56_key"])
                if prior and not prior[1]:continue
                a=cost(c,entries);b=0 if prior else cost(sm,entries)
                if a+b<=remaining and max(a,b)<=max_job:
                    return {"config":c,"smoke":sm,"reuse_smoke":bool(prior),"hours":a+b,"reason":why,
                        "compound_data_change":mix!=old["v6_mix"] or pub!=old["v6_public_mix"]}
    return None
