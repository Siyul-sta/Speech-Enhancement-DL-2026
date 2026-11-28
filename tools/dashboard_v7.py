"""V7 local read-only research dashboard. Python standard library only.
Never changes controllers, job states, weights, data, quota or remote kernels.
"""
import argparse
import ast
import csv
import json
import math
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime,timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
import dashboard_v4 as old

ROOT=Path(__file__).resolve().parents[1]
REMOTE=ROOT/"runs/ralph_v2"
RESEARCH=ROOT/"runs/v6_refine"
TERMINAL={"COMPLETE","ERROR_TERMINAL"}
REF=re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+")
QUOTA_LOCK=threading.Lock()
QUOTA_CACHE=None
RESULT_CACHE={}
RESULT_LOCK=threading.Lock()

def read(path):
    return old.read_json(path)

def finite(v):
    return old.finite(v)

def mtime(p):
    try:return p.stat().st_mtime
    except OSError:return 0

def config(path):
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("EXPERIMENT_CONFIG = "):
                    v=ast.literal_eval(line.split("=",1)[1].strip())
                    return v if isinstance(v,dict) else {}
    except (OSError,ValueError,SyntaxError):pass
    return {}

def catalog(path=ROOT/"tools/v7_search.py"):
    # Read the reviewed registry without importing torch or executing source.
    tree=ast.parse(path.read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CATALOG" for t in n.targets):
            return [{"name":a,"family":b,"mutation":c} for a,b,c in ast.literal_eval(n.value)]
    raise ValueError("V7 CATALOG not found")

def parse_quota(raw):
    for line in raw.splitlines():
        cells=line.split()
        if len(cells)<4 or cells[0]!="GPU":continue
        try:
            u,r,t=[float(x.rstrip("h")) for x in cells[1:4]]
            if not all(math.isfinite(x) and x>=0 for x in (u,r,t)) or t<=0 or abs(u+r-t)>.15 or r>t+.05:continue
            m=re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",line)
            refresh=datetime.fromisoformat(m[0].replace("Z","+00:00")).astimezone(timezone.utc).isoformat() if m else None
            return {"used":u,"remaining":r,"total":t,"refresh":refresh}
        except ValueError:continue
    raise ValueError("GPU quota not available")

def live_quota():
    global QUOTA_CACHE
    with QUOTA_LOCK:
        now=time.monotonic()
        if QUOTA_CACHE and now-QUOTA_CACHE[0]<60:return QUOTA_CACHE[1]
        prev=QUOTA_CACHE[1] if QUOTA_CACHE else {}
        try:
            cli=shutil.which("kaggle")
            if not cli:raise RuntimeError("Kaggle CLI not found")
            p=subprocess.run([cli,"quota"],capture_output=True,encoding="utf-8",errors="replace",timeout=12,check=False)
            if p.returncode:raise RuntimeError("Quota request failed")
            v={**parse_quota(p.stdout),"fetched_at":time.time(),"stale":False,"error":None}
        except (OSError,subprocess.SubprocessError,RuntimeError,ValueError) as exc:
            v={**prev,"stale":True,"error":str(exc)}
        QUOTA_CACHE=(now,v)
        return v

def result_score(folder,r):
    """Lightweight display integrity: fixed protocol, 164 unique rows and matching mean.
    Not a substitute for controller checkpoint SHA and external-domain verification.
    """
    score=finite(r.get("best_validation_pesq_wb"))
    if score is None or r.get("smoke") or r.get("evaluation_protocol")!="v4.2-fixed-dev-v2" or r.get("transductive_noisy") or r.get("self_supervised"):
        return None
    p=folder/"validation_per_file.csv"
    try:
        stat=p.stat();key=(str(p),stat.st_mtime_ns,stat.st_size,score)
        with RESULT_LOCK:
            if key in RESULT_CACHE:return RESULT_CACHE[key]
        with p.open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.DictReader(f))
        vals=[float(row["pesq_wb"]) for row in rows]
        ids=[int(row["index"]) for row in rows]
        value=score if len(vals)==164 and set(ids)==set(range(164)) and all(math.isfinite(v) for v in vals) and abs(sum(vals)/164-score)<=1e-6 else None
        with RESULT_LOCK:
            if len(RESULT_CACHE)>1000:RESULT_CACHE.clear()
            RESULT_CACHE[key]=value
        return value
    except (OSError,ValueError,KeyError,TypeError):return None

def remote_rows(base):
    rows=[]
    for p in sorted(base.glob("run_v4*/round_*/*/remote_state.json")):
        st=read(p)
        if not st:continue
        c=config(p.parent/"source.py")
        r=read(p.parent/"result.json") if st.get("status")=="COMPLETE" else {}
        if not (st.get("v6") or c.get("v7_recipe")):continue
        m=r.get("v6") or {};parent=c.get("v6_parent") or {};v7=r.get("v7") or {}
        score=result_score(p.parent,r)
        raw=finite(r.get("best_validation_pesq_wb"))
        smoke=bool(c.get("smoke") or r.get("smoke") or st.get("kind")=="v4_smoke")
        rows.append({"ref":st.get("ref"),"run":p.parents[2].name,"round":p.parents[1].name,
            "status":st.get("status"),"kind":st.get("kind"),"model":r.get("model_family") or c.get("architecture") or st.get("architecture"),
            "version":"V7" if c.get("v7_recipe") else "V6.11" if c.get("v611_recipe") else "V6.10" if c.get("v610_recipe") else "V6.9" if c.get("v6_refine_hypothesis") else "V6",
            "recipe":c.get("v7_recipe") or c.get("v611_recipe") or c.get("v610_recipe") or c.get("v6_refine_hypothesis") or ("zero-shot" if c.get("v6_mode")=="zero" else "legacy training"),
            "smoke":smoke,"score":score,"smoke_score":raw if smoke else None,"external":finite(r.get("external_validation_pesq_wb")),
            "parent_score":finite(parent.get("score")),"checkpoint":m.get("checkpoint_sha256"),"gain":score-parent["score"] if score is not None and finite(parent.get("score")) is not None else None,
            "epochs":c.get("epochs"),"completed_epochs":r.get("epochs_completed"),"best_epoch":r.get("best_epoch"),
            "epoch_seconds":sum(v for v in r.get("epoch_seconds",[]) if finite(v) is not None and v>0),
            "recorded_at":mtime(p),"result_at":mtime(p.parent/"result.json"),"metric_steps":m.get("metric_steps"),
            "crop":c.get("v6_crop"),"mix":c.get("v6_mix"),"public_mix":c.get("v6_public_mix"),"lr":finite(c.get("lr")),
            "mutation":c.get("v7_mutation"),"critic_calibration":m.get("critic_calibration"),"warmup_calibration":v7.get("warmup_calibration")})
    return rows

def journals(root):
    out=[]
    for p in sorted((root/"v7").glob("run_*.json")):
        v=read(p)
        if v.get("protocol")=="v7-controller-v1":
            out.append({**v,"run":p.stem,"updated_at":mtime(p)})
    return out

def final_report(root):
    for p in sorted(root.glob("v7_final_*/report.json"),reverse=True):
        v=read(p)
        if v.get("protocol")=="v6-refine-v1" and v.get("count")==164:
            return {"name":p.parent.name,"winner":v.get("winner"),"pesq":finite(v.get("winner_pesq")),"updated_at":mtime(p)}
    return None

def strategy_summary(rows,recipes):
    out=[]
    for c in recipes:
        trials=[r for r in rows if r["version"]=="V7" and r["recipe"]==c["name"] and not r["smoke"]]
        complete=[r for r in trials if r["status"]=="COMPLETE" and r["gain"] is not None]
        gains=[r["gain"] for r in complete]
        out.append({**c,"attempts":len(trials),"completed":len(complete),"improved":sum(g>=.002 for g in gains),
            "failed":sum(r["status"]=="ERROR_TERMINAL" for r in trials),"pending":sum(r["status"] not in TERMINAL for r in trials),
            "unverified":sum(r["status"]=="COMPLETE" and r["gain"] is None for r in trials),
            "best_gain":max(gains) if gains else None,"mean_gain":sum(gains)/len(gains) if gains else None})
    return out

def snapshot(base=REMOTE,research=RESEARCH,poll_remote=False):
    rows=remote_rows(base)
    ranked=sorted([r for r in rows if r["score"] is not None],key=lambda r:(r["result_at"] or r["recorded_at"],r["ref"] or ""))
    best=max(ranked,key=lambda r:r["score"],default=None)
    history=[];peak=None
    for r in ranked:
        peak=r["score"] if peak is None else max(peak,r["score"])
        history.append({"ref":r["ref"],"score":r["score"],"best":peak,"recipe":r["recipe"],"version":r["version"],"time":r["result_at"]})
    pending=[r for r in rows if r["status"] not in TERMINAL]
    active=max(pending,key=lambda r:r["recorded_at"],default=None)
    remote=old.remote_status(active["ref"]) if active and poll_remote else None
    js=journals(research);latest=js[-1] if js else None
    events=[{**ev,"run":j["run"]} for j in js for ev in j.get("events",[]) if isinstance(ev,dict)]
    event=latest.get("events",[])[-1] if latest and latest.get("events") else None
    lock=base/"active.lock";lock_present=lock.exists()
    reason=latest.get("stop_reason") if latest else None
    cycle=latest.get("cycles",[])[-1] if latest and latest.get("cycles") else None
    if active:
        phase="remote_pending"
        if active["version"]!="V7":phase="legacy_pending"
    elif lock_present and reason in {"budget","budget_before_preflight","budget_after_preflight","budget_after_smoke","no_affordable_recipe"} and cycle and cycle.get("end_quota"):
        phase="possible_quota_wait"
    elif lock_present:phase="local_activity_unknown"
    elif latest:phase="journal_idle"
    else:phase="no_active_record"
    return {"updated_at":time.time(),"phase":phase,"lock_present":lock_present,"active":active,"pending":pending,"remote":remote,
        "best":best,"history":history,"experiments":list(reversed(rows[-300:])),"strategies":strategy_summary(rows,catalog()),
        "journal":{k:latest.get(k) for k in ("run","updated_at","rounds","jobs","cycle","stop_reason","wait_hours")} if latest else None,
        "cycle":cycle,"event":event,"events":events[-100:],"final":final_report(research),
        "counts":{"scored":len(ranked),"completed":sum(r["status"]=="COMPLETE" for r in rows),"failed":sum(r["status"]=="ERROR_TERMINAL" for r in rows),
                  "v7_trials":sum(r["version"]=="V7" and not r["smoke"] for r in rows),"pending":len(pending)}}

def log_payload(base=REMOTE):
    rows=remote_rows(base);pending=[r for r in rows if r["status"] not in TERMINAL]
    job=max(pending or rows,key=lambda r:r["recorded_at"],default=None)
    return old.remote_logs(job["ref"]) if job else {"ref":None,"lines":[],"telemetry":old.parse_logs(""),"error":None}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=urlsplit(self.path).path
        try:
            if path=="/":
                data=(ROOT/"tools/dashboard_v7.html").read_bytes();mime="text/html; charset=utf-8"
            elif path in ("/api","/quota","/logs"):
                v=snapshot(poll_remote=True) if path=="/api" else live_quota() if path=="/quota" else log_payload()
                data=json.dumps(v,ensure_ascii=False,allow_nan=False).encode("utf-8");mime="application/json; charset=utf-8"
            else:self.send_error(404);return
            self.send_response(200);self.send_header("Content-Type",mime);self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):self.close_connection=True
        except Exception as exc:
            self.send_error(500,"Dashboard data unavailable")
            print("V7 DASHBOARD ERROR",repr(exc),flush=True)
    def log_message(self,*args):pass

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--port",type=int,default=8767);a=p.parse_args()
    if not 1<=a.port<=65535:p.error("Invalid port")
    with ThreadingHTTPServer(("127.0.0.1",a.port),Handler) as server:
        print(f"V7 dashboard http://127.0.0.1:{a.port}",flush=True);server.serve_forever()
if __name__=="__main__":main()
