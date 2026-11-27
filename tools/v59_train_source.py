"""Generate an isolated long-training runner without changing V5.7.2 inference."""
from __future__ import annotations

def build(source):
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1: raise ValueError("V5.9 source contract changed: "+old[:100])
        source=source.replace(old,new,1)

    replace('    warm_ref = cfg.get("warm_ref")', '''    v59_enabled=bool(cfg.get("v59_enabled"))
    v59_parent=cfg.get("v59_parent")
    v59_offset=int(v59_parent["checkpoint_epoch"]) if v59_parent else 0
    v59_attempted=int(v59_parent["attempted_epochs"]) if v59_parent else 0
    if v59_enabled and (cfg.get("v54_stage") or cfg.get("pesq_aware") or cfg.get("self_supervised") or cfg.get("distill") or cfg.get("v53_anchor") or cfg.get("transfer_mode")=="lora"):
        raise ValueError("Unsupported V5.9 continuation state")
    v59_pretrained_hash=transfer_state_hash(model) if v59_enabled and cfg.get("v53_stage") else None
    warm_ref = cfg.get("warm_ref")
    if v59_enabled and bool(warm_ref)!=bool(v59_parent): raise ValueError("Missing V5.9 parent metadata")
    if v59_parent and warm_ref!=v59_parent["ref"]: raise ValueError("Wrong V5.9 parent reference")''')
    replace('        model.load_state_dict(torch.load(matches[0], map_location=device, weights_only=True), strict=True)', '''        if v59_parent:
            import hashlib
            with matches[0].open("rb") as stream: actual_hash=hashlib.file_digest(stream,"sha256").hexdigest()
            if actual_hash!=v59_parent["sha256"]: raise RuntimeError("V5.9 parent checkpoint hash mismatch")
        model.load_state_dict(torch.load(matches[0], map_location=device, weights_only=True), strict=True)''')
    replace('        initial_hash=transfer_state_hash(model)', '        initial_hash=v59_pretrained_hash or transfer_state_hash(model)')
    replace('    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\n        optimizer, T_max=max(int(cfg["epochs"]), 1), eta_min=0. if transfer else 0.000025\n    )', '''    if v59_enabled:
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=3,threshold=.001,threshold_mode="abs",min_lr=1e-8)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(cfg["epochs"]), 1), eta_min=0. if transfer else 0.000025
        )''')
    replace('    if transfer or cascade:\n        initial_score,initial_rows=', '    if transfer or cascade or v59_parent:\n        initial_score,initial_rows=')
    replace('        zero_shot={"challenge":initial_score,"external_domains":initial_domains}', '''        zero_shot={"challenge":initial_score,"external_domains":initial_domains}
        if v59_parent:
            if abs(initial_score-float(v59_parent["score"]))>.02 or any(abs(a-b)>.02 for a,b in zip(initial_domains,v59_parent["external_domains"])):
                raise RuntimeError("V5.9 parent re-evaluation differs; inspect evaluation protocol or data")
            if any(a<b-.15001 for a,b in zip(initial_domains,v59_parent["root_external_domains"])):
                raise RuntimeError("V5.9 parent violates original external preservation gate")''')
    replace('    for epoch in range(1, int(cfg["epochs"]) + 1):\n        model.train()', '''    for epoch in range(1, int(cfg["epochs"]) + 1):
        effective_epoch=epoch+v59_offset
        if v59_enabled:
            data_epoch=v59_attempted+epoch
            seed_everything(int(cfg["seed"])+data_epoch)
            if train_loader.generator is not None: train_loader.generator.manual_seed(int(cfg["seed"])+data_epoch)
        model.train()''')
    replace('            event=transfer.apply(epoch)', '''            event=transfer.apply(effective_epoch)
            if v59_enabled: event.update(epoch=epoch,effective_epoch=effective_epoch)''')
    replace('set_data_epoch(train_set,epoch)\n        losses = []','set_data_epoch(train_set,v59_attempted+epoch if v59_enabled else epoch)\n        losses = []')
    replace('        scheduler.step()\n        val_pesq, _ = evaluate','        if not v59_enabled: scheduler.step()\n        val_pesq, _ = evaluate')
    replace('        eligible=not (transfer or cascade) or transfer_candidate_allowed(val_pesq,domain_scores[1:],zero_shot["external_domains"])', '''        eligible=not (transfer or cascade or v59_parent) or transfer_candidate_allowed(val_pesq,domain_scores[1:],zero_shot["external_domains"])
        if v59_parent:
            eligible=eligible and transfer_candidate_allowed(val_pesq,domain_scores[1:],v59_parent["root_external_domains"])
        if v59_enabled:
            scheduler.step(selection_score if eligible else -1e6)
            print("V59_EPOCH "+json.dumps({"epoch":epoch,"effective_epoch":effective_epoch,"attempted_epoch":v59_attempted+epoch,"eligible":bool(eligible),"learning_rates":[g["lr"] for g in optimizer.param_groups],"validation_pesq":val_pesq}),flush=True)''')
    replace('    if (transfer or cascade) and best_score<zero_shot["challenge"]-1e-5:', '    if (transfer or cascade or v59_parent) and best_score<zero_shot["challenge"]-1e-5:')
    replace('    if (transfer or cascade) and not transfer_candidate_allowed(best_score,','    if (transfer or cascade or v59_parent) and not transfer_candidate_allowed(best_score,')
    replace('        "optimizer": "AdamW+CosineAnnealingLR (grouped)" if transfer else "AdamW+CosineAnnealingLR",','        "optimizer": "AdamW+ReduceLROnPlateau (weights-only continuation)" if v59_enabled else ("AdamW+CosineAnnealingLR (grouped)" if transfer else "AdamW+CosineAnnealingLR"),')
    replace('    (WORK / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")\n    print("FINAL_RESULT "', '''    if v59_enabled:
        import hashlib
        with (WORK/"best_model.pt").open("rb") as stream: checkpoint_hash=hashlib.file_digest(stream,"sha256").hexdigest()
        domains=[result["domain_scores"][k] for k in ("external_normal","external_stress")]
        root_domains=v59_parent["root_external_domains"] if v59_parent else (zero_shot["external_domains"] if zero_shot else domains)
        if any(a<b-.15001 for a,b in zip(domains,root_domains)): raise RuntimeError("V5.9 original external domain regression")
        result["v59"]={"protocol":"v59-weights-continuation-v1","parent":v59_parent,
            "attempted_epochs":v59_attempted+len(history),"checkpoint_epoch":v59_offset+best_epoch,
            "checkpoint_sha256":checkpoint_hash,"root_external_domains":root_domains,
            "start_pesq":zero_shot["challenge"] if zero_shot else None,
            "gain":best_score-zero_shot["challenge"] if zero_shot else None,
            "final_lrs":[g["lr"] for g in optimizer.param_groups],"optimizer_restored":False,
            "stale_epochs":stale}
    (WORK / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT "''')
    compile(source,"v59_generated_trainer.py","exec")
    return source
