"""V6 standalone training entry point, embedded after the unchanged V4 evaluator.
All targets used for training come from train or licensed external audio.
"""
import hashlib
import shutil

V6_NATIVE={"complex_residual":ComplexResidualUNet,"residual_relaxed":RelaxedPhaseUNet,"dual_axis":DualAxisEnhancer}
V6_PUBLIC={"cmgan_transfer":CMGANTransfer,"metricgan":MetricGANTransfer,"sepformer_dns_transfer":SepformerTransfer,"sepformer_wham_transfer":SepformerTransfer}

def v6_sha(p):
    with Path(p).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

class V6Refiner(nn.Module):
    """Frozen coarse model, identity-initialized trainable residual/phase refiners."""
    def __init__(self,coarse,depth=2):
        super().__init__()
        if depth not in (1,2,3): raise ValueError("Invalid cascade depth")
        self.coarse=coarse.requires_grad_(False).eval()
        self.wave=ConditionedWaveRefiner()
        self.phase=ConditionedComplexRefiner() if depth>=2 else None
        self.final=ConditionedWaveRefiner(width=16,dilations=(1,2,4,8),limit=.2) if depth>=3 else None
    def forward_stages(self,x):
        self.coarse.eval()
        with torch.no_grad(): a=self.coarse(x)
        b=self.wave(x,a);out=[a,b]
        if self.phase is not None: out.append(self.phase(x,out[-1]))
        if self.final is not None: out.append(self.final(x,out[-1]))
        return out
    def forward(self,x): return self.forward_stages(x)[-1]

def v6_make_model(cfg,mp_root=None,pretrained=True,coarse=None):
    family=cfg["architecture"]
    base=cfg.get("v6_base_family",family)
    if coarse is None:
        if base in V6_MP_FAMILIES:
            mp_root=Path(mp_root) if mp_root else WORK/"v6_mp_assets"
            # Only original verified public blobs; offline final load needs no original weights.
            coarse=V6MPNet(base,mp_root,allow_download=pretrained,pretrained=pretrained)
        elif base in V6_NATIVE: coarse=V6_NATIVE[base]()
        elif base in V6_PUBLIC:
            if not pretrained: raise ValueError("Offline public adapter requires constructed coarse module")
            coarse=V6_PUBLIC[base]()
        else: raise ValueError("Unsupported V6 architecture: "+base)
    return V6Refiner(coarse,int(cfg["v6_depth"])) if cfg.get("v6_depth") else coarse

def v6_parent_path(parent):
    slug=parent["ref"].rsplit("/",1)[-1]
    matches=[p for p in Path("/kaggle/input").rglob("best_model.pt") if slug in p.parts]
    if len(matches)!=1: raise ValueError("Expected a single private parent checkpoint")
    if v6_sha(matches[0])!=parent["sha256"]: raise ValueError("V6 checkpoint hash mismatch")
    return matches[0]

def v6_loaders(cfg,device):
    train=SpeechPairs(locate_data_root(),"train")
    val=SpeechPairs(locate_data_root(),"validation")
    if len(val)!=164: raise ValueError("Expected full validation")
    fit,cal,gate_fit,gate_select=v6_split(train)
    zero=cfg.get("v6_mode")=="zero"
    smoke=bool(cfg.get("smoke"))
    mix=None
    if smoke:
        fixture,ext=smoke_pairs(torch.utils.data.Subset(train,fit))
        data=torch.utils.data.ConcatDataset([torch.utils.data.Subset(train,fit[:8]),fixture])
    else:
        tr,dv,tn,dn,_,_=public_assets(evaluation_only=zero)
        ext=fixed_pairs(torch.utils.data.ConcatDataset([SyntheticPairs(dv,dn,32,2026,False),SyntheticPairs(dv,dn,32,3026,False,stress=True)]))
        data=torch.utils.data.Subset(train,fit)
        if not zero:
            limit=int(cfg.get("v6_korean",0))
            korean=[]
            if limit:
                subprocess.check_call([sys.executable,"-m","pip","install","-q","datasets==3.6.0","pyarrow==20.0.0","huggingface_hub<1"])
                korean=fleurs_speech("train",limit=limit)
            # Use both public speech and challenge residuals. Separate public-only mix keeps coverage.
            mix=V6CrossMix(train,fit,tr+korean,tn,int(cfg["v6_mix"]),int(cfg["seed"]),hard=cfg.get("v6_hard",False))
            public=SyntheticPairs(tr,tn,int(cfg["v6_public_mix"]),42,True)
            data=torch.utils.data.ConcatDataset([data,mix,public])
            print("V6_DATA "+json.dumps({"fit":len(fit),"calibration":len(cal),"gate_fit":len(gate_fit),"gate_select":len(gate_select),
                   "external_speech_files":len(tr),"korean_files":len(korean),"crossmix_per_epoch":len(mix),"public_per_epoch":len(public)}),flush=True)
    # Full-length validation always; optional short training crops only.
    loader=DataLoader(data,batch_size=int(cfg["batch_size"]),shuffle=True,generator=torch.Generator().manual_seed(cfg["seed"]),num_workers=0)
    return loader,DataLoader(val,batch_size=1,shuffle=False),DataLoader(ext,batch_size=1,shuffle=False),DataLoader(torch.utils.data.Subset(train,cal[:32]),batch_size=1,shuffle=False),mix,train,cal[:32]

def v6_eval(model,val,ext,device,pesq_fn):
    score,rows=evaluate(model,val,device,pesq_fn,save_rows=True)
    es,er=evaluate(model,ext,device,pesq_fn,save_rows=True)
    dom=[float(np.mean([r[1] for r in er[:32]])),float(np.mean([r[1] for r in er[32:]]))]
    return float(score),rows,float(es),er,dom

def v6_loss(pred,clean,cfg):
    loss=enhancement_loss(pred.float(),clean.float(),cfg["id"])
    if cfg.get("v6_phase",True): loss=loss+.15*phase_complex_loss(pred.float(),clean.float())
    if cfg.get("v6_sisdr"):
        a=pred-pred.mean(-1,keepdim=True);b=clean-clean.mean(-1,keepdim=True)
        proj=(a*b).sum(-1,keepdim=True)*b/(b.square().sum(-1,keepdim=True)+1e-8)
        loss=loss-.005*10*torch.log10((proj.square().sum(-1)+1e-8)/((a-proj).square().sum(-1)+1e-8)).clamp(-3,3).mean()
    return loss

def v6_step(model,opt,x,s,cfg,critic,co,replay,pesq_fn,weight,update):
    opt.zero_grad(set_to_none=True)
    stages=model.forward_stages(x) if isinstance(model,V6Refiner) else [model(x)]
    pred=stages[-1]
    loss=v6_loss(pred,s,cfg)
    if len(stages)>2: loss=loss+.05*sum(F.l1_loss(y,s) for y in stages[1:-1])/len(stages[1:-1])
    if critic is not None:
        if update: critic_replay_step(critic,co,replay,pred.detach(),x,s,pesq_fn)
        if weight>0:
            critic.eval().requires_grad_(False)
            loss=loss-weight*critic(pred.clamp(-1,1),s.clamp(-1,1)).mean()
    if not torch.isfinite(loss): raise ValueError("Nonfinite V6 loss")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    opt.step()
    return float(loss.detach()),float(norm)

def v6_run():
    cfg=EXPERIMENT_CONFIG
    if cfg.get("transductive") or cfg.get("self_supervised"): raise ValueError("No validation/test training")
    WORK.mkdir(parents=True,exist_ok=True)
    seed_everything(cfg["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Path("/kaggle").exists() and device.type!="cuda": raise RuntimeError("GPU required")
    mode=cfg["v6_mode"];smoke=bool(cfg.get("smoke"))
    if mode not in ("zero","train","cascade"): raise ValueError("Unknown V6 mode")
    pesq_fn=ensure_pesq()
    model=v6_make_model(cfg).to(device)
    parent=cfg.get("v6_parent")
    if bool(parent)!=bool(cfg.get("warm_ref")) or (parent and parent["ref"]!=cfg["warm_ref"]): raise ValueError("V6 parent reference mismatch")
    attempted=int(parent["attempted_epochs"]) if parent else 0
    offset=int(parent["checkpoint_epoch"]) if parent else 0
    if parent:
        target=model.coarse if cfg.get("v6_parent_coarse") else model
        target.load_state_dict(torch.load(v6_parent_path(parent),map_location=device,weights_only=True),strict=True)
    loader,val,ext,cal,mix,train,cal_ids=v6_loaders(cfg,device)
    # Probe full-size noisy-only forward; no clean target to construct predictions.
    model.eval()
    with torch.no_grad():
        probe=model(torch.from_numpy(train.noisy[cal_ids[0]])[None].to(device))
        if probe.shape!=(1,48000) or not torch.isfinite(probe).all(): raise ValueError("Full clip model contract failed")
    base,br,be,ber,bd=v6_eval(model,val,ext,device,pesq_fn)
    if parent and abs(base-float(parent["score"]))>.025: raise RuntimeError("Parent score failed reproduction")
    if parent and any(abs(a-b)>.025 for a,b in zip(bd,parent["external_domains"])): raise RuntimeError("Parent domain failed reproduction")
    root_domains=parent["root_external_domains"] if parent else bd
    write_csv(WORK/"zero_shot_validation.csv",["index","pesq_wb"],br)
    write_csv(WORK/"zero_shot_external.csv",["index","pesq_wb"],ber)
    torch.save(model.state_dict(),WORK/"best_model.pt")
    best=base;best_epoch=0;stale=0;history=[];epoch_times=[];calibration=None;updates=0;metric_steps=0
    opt=None;co=None;critic=None;replay=None;lr=float(cfg["lr"])
    if mode!="zero":
        trainable=[p for p in model.parameters() if p.requires_grad]
        if not trainable: raise ValueError("No trainable parameters")
        opt=torch.optim.AdamW(trainable,lr=lr,weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=2,factor=.5,threshold=.001,threshold_mode="abs",min_lr=1e-8)
        if cfg.get("v6_critic"):
            critic=PESQCritic().to(device);co=torch.optim.AdamW(critic.parameters(),lr=3e-4);replay=CriticReplay(seed=cfg["seed"])
        for epoch in range(1,int(cfg["epochs"])+1):
            start=time.monotonic();e=attempted+epoch
            seed_everything(cfg["seed"]+e);loader.generator.manual_seed(cfg["seed"]+e)
            if mix is not None: mix.epoch=e
            set_data_epoch(loader.dataset,e)
            model.train();losses=[];weight=adaptive_metric_weight(calibration,epoch,updates*2) if critic is not None else 0.
            for j,(x,s) in enumerate(loader,1):
                x,s=x.to(device),s.to(device)
                length=int(cfg.get("v6_crop",48000))
                if length<48000:
                    k=int(np.random.randint(0,48000-length+1));x=x[:,k:k+length];s=s[:,k:k+length]
                update=critic is not None and j%16==1
                # PESQ critic uses full clips, not short crop labels.
                if critic is not None and length!=48000: raise ValueError("Critic requires full clips")
                loss,gn=v6_step(model,opt,x,s,cfg,critic,co,replay,pesq_fn,weight,update)
                updates+=int(update);metric_steps+=int(weight>0);losses.append(loss)
                if j==1 or j%25==0 or j==len(loader):
                    print("V4_TELEMETRY "+json.dumps({"event":"batch","epoch":epoch,"epochs":cfg["epochs"],"batch":j,"batches":len(loader),"loss":loss,"elapsed_s":time.monotonic()-start}),flush=True)
            if critic is not None:
                calibration=critic_calibration(model,critic,cal,device,pesq_fn)
                print("V6_CRITIC "+json.dumps(calibration),flush=True)
            if mix is not None and mix.hard:
                _,rows=evaluate(model,cal,device,pesq_fn,save_rows=True)
                key=next((k for k in ("noise_type","noise_category","noise_label","noise_class") if k in train.manifest.columns),None)
                grouped={}
                for i,v in rows:
                    label=str(train.manifest.iloc[cal_ids[i]][key]) if key else "unknown"
                    grouped.setdefault(label,[]).append(v)
                mix.update_hard({k:float(np.mean(v)) for k,v in grouped.items()})
            score,_,_,_,dom=v6_eval(model,val,ext,device,pesq_fn)
            eligible=transfer_candidate_allowed(score,dom,root_domains,margin=float(cfg.get("v6_external_margin",.15)))
            schedule.step(score if eligible else -1e6)
            if eligible and score>best+.001:
                best=score;best_epoch=epoch;stale=0;torch.save(model.state_dict(),WORK/"best_model.pt")
            else: stale+=1
            dt=time.monotonic()-start;epoch_times.append(dt)
            history.append([epoch,float(np.mean(losses)),score,best,stale])
            write_csv(WORK/"history.csv",["epoch","train_loss","validation_pesq_wb","best_pesq_wb","stale_epochs"],history)
            print("V4_TELEMETRY "+json.dumps({"event":"epoch","epoch":epoch,"loss":history[-1][1],"challenge":score,"best":best,"elapsed_s":dt}),flush=True)
            if not smoke and epoch>=5 and stale>=int(cfg["patience"]): break
    model.load_state_dict(torch.load(WORK/"best_model.pt",map_location=device,weights_only=True),strict=True)
    score,rows,es,er,dom=v6_eval(model,val,ext,device,pesq_fn)
    if abs(score-best)>1e-5 or score<base-1e-5: raise RuntimeError("Best checkpoint failed replay")
    if parent and any(a<b-float(cfg.get("v6_external_margin",.15))-1e-5 for a,b in zip(dom,root_domains)): raise RuntimeError("Parent domain guard failed")
    if cfg.get("v6_depth"):
        # Compare final and coarse on identical full validation; no oracle.
        coarse_score,cr=evaluate(model.coarse,val,device,pesq_fn,save_rows=True)
        write_csv(WORK/"coarse_validation.csv",["index","pesq_wb"],cr)
    else: coarse_score=None
    write_csv(WORK/"validation_per_file.csv",["index","pesq_wb"],rows)
    write_csv(WORK/"external_validation.csv",["index","pesq_wb"],er)
    result={"experiment":"V6_MAX_PESQ","evaluation_protocol":"v4-smoke-fixture-v1" if smoke else "v4.2-fixed-dev-v2",
        "validation_files":164,"external_validation_files":64,"best_validation_pesq_wb":score,"external_validation_pesq_wb":es,
        "joint_score":(score+es)/2,"domain_scores":{"challenge":score,"external_normal":dom[0],"external_stress":dom[1]},
        "robust_score":.5*float(np.mean([score,*dom]))+.5*min([score,*dom]),"model_family":cfg["architecture"],
        "architecture":cfg["architecture"],"seed":cfg["seed"],"smoke":smoke,"epochs_completed":len(history),"best_epoch":best_epoch,
        "epoch_seconds":epoch_times,"zero_shot":{"challenge":base,"external_domains":bd},"transductive_noisy":False,"self_supervised":False,
        "selection_target":"challenge","v56_key":cfg["v56_key"],"v56_action":cfg["v56_action"],"v56_source_sha":cfg["v56_source_sha"],
        "v56_requested_epochs":cfg["epochs"],"sepformer_scale_policies":cfg.get("sepformer_scale_policies",{}),
        "v6":{"protocol":"v6-max-pesq-v1","parent":parent,"attempted_epochs":attempted+len(history),"checkpoint_epoch":offset+best_epoch,
            "checkpoint_sha256":v6_sha(WORK/"best_model.pt"),"root_external_domains":root_domains,"start_pesq":base,"gain":score-base,
            "lr":opt.param_groups[0]["lr"] if opt else lr,"optimizer_restored":False,"critic_updates":updates,"metric_steps":metric_steps,
            "critic_calibration":calibration,"coarse_score":coarse_score,"fit_count":len(v6_split(train)[0]),
            "training_pairs_per_epoch":len(loader.dataset),"mode":mode,"base_family":cfg.get("v6_base_family",cfg["architecture"])}}
    (WORK/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print("FINAL_RESULT "+json.dumps(result,allow_nan=False),flush=True)

if __name__=="__main__":
    try: v6_run()
    finally: close_pesq_pool()
