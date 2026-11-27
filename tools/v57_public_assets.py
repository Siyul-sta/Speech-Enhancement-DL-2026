"""V5.7 public adapter reconstruction from local verified assets only.
Archived Python/YAML is trusted executable input, not an untrusted-file sandbox.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch

CMGAN_REV="39946005d9b66fa1e824edf4bb6bc9a06088e443"
CMGAN_BLOBS={"LICENSE":"6750370caf9bdc4710c5c1c0da12450a7d1bf336",
            "src/models/conformer.py":"4594f11803a988f2540a30830c28e51c88f25126",
            "src/models/generator.py":"d0dfacc36128d8b51386c3bb7c0fc8edfad82074"}
PUBLIC={"cmgan_transfer":"CMGANTransfer","cascade_cmgan_two":"CascadeCMGANTwo",
        "cascade_cmgan_three":"CascadeCMGANThree","cascade_cmgan":"CascadeCMGANFour",
        "metricgan":"MetricGANTransfer","sepformer_dns_transfer":"SepformerTransfer","sepformer_wham_transfer":"SepformerTransfer"}

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git_blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def cmgan_family(family):
    return family=="cmgan_transfer" or family.startswith("cascade_cmgan")

def checked_local(root,name,sha):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path)!=sha:
        raise ValueError("Missing or invalid offline asset: "+str(name))
    return path

def package_public(family,run,source,out,allow_download=False,cmgan_root=None):
    """Copy pinned source or the actual archived SpeechBrain YAML.
    No original pretrained weights needed: the selected full state dict supplies them.
    """
    if family not in PUBLIC: raise ValueError("Unsupported public architecture")
    out.mkdir(parents=True,exist_ok=False)
    if cmgan_family(family):
        if CMGAN_REV not in source or any(h not in source for h in CMGAN_BLOBS.values()):
            raise ValueError("Archived CMGAN source does not match supported pinned revision")
        files={}
        for name,blob in CMGAN_BLOBS.items():
            dest=out/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            local=cmgan_root/name if cmgan_root else None
            if local and local.is_file():
                data=local.read_bytes()
            elif allow_download:
                import urllib.request
                with urllib.request.urlopen(f"https://raw.githubusercontent.com/ruizhecao96/CMGAN/{CMGAN_REV}/{name}",timeout=60) as r:
                    data=r.read(2_000_001)
            else:
                raise ValueError("Pinned CMGAN sources unavailable locally; use --cmgan-root or explicitly allow public download")
            if len(data)>2_000_000 or git_blob(data)!=blob: raise ValueError("Pinned CMGAN blob mismatch")
            dest.write_bytes(data)
            files[name]=file_hash(dest)
        return {"kind":"cmgan","revision":CMGAN_REV,"files":files}
    info=json.loads((run/"offline_hyperparams.json").read_text(encoding="utf-8"))
    config=run/"offline_hyperparams.yaml"
    if info.get("family")!=family or file_hash(config)!=info.get("sha256"): raise ValueError("Invalid archived SpeechBrain configuration")
    if not all(isinstance(info.get(k),str) and info[k] for k in ("speechbrain","hyperpyyaml")): raise ValueError("Missing SpeechBrain version identity")
    dest=out/"hyperparams.yaml";dest.write_bytes(config.read_bytes())
    return {"kind":"speechbrain","files":{"hyperparams.yaml":file_hash(dest)},"speechbrain":info["speechbrain"],"hyperpyyaml":info["hyperpyyaml"],"source":info.get("source"),"revision":info.get("revision")}

def make_cmgan_module(root,meta):
    if meta.get("revision")!=CMGAN_REV or set(meta["files"])!=set(CMGAN_BLOBS): raise ValueError("Unsupported CMGAN asset manifest")
    for name,blob in CMGAN_BLOBS.items():
        p=checked_local(root,name,meta["files"][name])
        if git_blob(p.read_bytes())!=blob: raise ValueError("Pinned CMGAN content mismatch")
    # Use a unique in-memory package; original source files remain unchanged.
    import types
    name="_ralph_cmgan_"+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    pkg=types.ModuleType(name);pkg.__path__=[]
    sys.modules[name]=pkg
    for part in ("conformer","generator"):
        qualified=name+"."+part
        m=types.ModuleType(qualified);m.__package__=name
        path=root/"src/models"/(part+".py")
        m.__file__=str(path)
        sys.modules[qualified]=m
        text=path.read_text(encoding="utf-8")
        if part=="generator": text=text.replace("from models.conformer import","from .conformer import")
        exec(compile(text,str(path),"exec"),m.__dict__)
    return sys.modules[name+".generator"]

def construct(mod,family,root,meta):
    """No downloads, no original checkpoint load, strict final weights loaded by caller."""
    if family not in PUBLIC: raise ValueError("Unknown public adapter")
    if cmgan_family(family):
        if meta.get("kind")!="cmgan": raise ValueError("Wrong CMGAN asset type")
        generator=make_cmgan_module(root,meta)
        original=mod.CMGANTransfer
        class LocalCMGAN(original):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.net=generator.TSCNet(num_channel=64,num_features=201)
                self.register_buffer("window",torch.hamming_window(400))
        mod.CMGANTransfer=LocalCMGAN
        try: return getattr(mod,PUBLIC[family])()
        finally: mod.CMGANTransfer=original
    if meta.get("kind")!="speechbrain" or set(meta.get("files",{}))!={"hyperparams.yaml"}: raise ValueError("Wrong SpeechBrain asset type")
    from importlib.metadata import version
    for dep in ("speechbrain","hyperpyyaml"):
        if version(dep)!=meta[dep]: raise ValueError("Offline dependency version mismatch: "+dep)
    if hasattr(mod,"speechbrain_adapter_setup"): mod.speechbrain_adapter_setup()
    else:
        import torchaudio
        if not hasattr(torchaudio,"list_audio_backends"): torchaudio.list_audio_backends=lambda: []
    from hyperpyyaml import load_hyperpyyaml
    path=checked_local(root,"hyperparams.yaml",meta["files"]["hyperparams.yaml"])
    with path.open(encoding="utf-8") as f: h=load_hyperpyyaml(f)
    modules=h["modules"]
    original=getattr(mod,PUBLIC[family])
    class LocalSpeechBrain(original):
        def __init__(self):
            torch.nn.Module.__init__(self)
            if family=="metricgan":
                self.generator=modules["enhance_model"]
                self.features=h["compute_stft"]
                self.resynthesize=h["resynth"]
            else:
                if int(h["num_spks"])!=1: raise ValueError("Expected one enhanced speaker")
                self.encoder=modules["encoder"]
                self.masknet=modules["masknet"]
                self.decoder=modules["decoder"]
    return LocalSpeechBrain()
