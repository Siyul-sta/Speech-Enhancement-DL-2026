"""V7.3 read-only two-lane dashboard. No training or Google credentials.
Colab files can be read from a Google Drive for desktop mount or manual snapshot.
An unchanged report is NOT evidence that the runtime remains alive.
"""
import argparse
import csv
import json
import math
import re
import time
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from http.server import ThreadingHTTPServer
import dashboard_v7 as d

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def history(folder,epochs):
    try:
        with (folder/"history.csv").open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
        out=[]
        for r in rows:
            e=int(r["epoch"]);loss=float(r["train_loss"]);score=float(r["validation_pesq_wb"]);best=float(r["best_pesq_wb"])
            if not 1<=e<=epochs or not all(math.isfinite(x) for x in (loss,score,best)):return []
            out.append({"epoch":e,"loss":loss,"challenge":score,"best":best})
        return out if [r["epoch"] for r in out]==list(range(1,len(out)+1)) else []
    except (OSError,ValueError,KeyError,TypeError):return []

def colab_snapshot(root):
    if root is None:return {"status":"not_configured","note":"Specify --colab-root for Drive folder or downloaded snapshot.","candidates":[],"scores":[]}
    plan=d.read(root/"v71_plan.json");report=d.read(root/"v71_report.json");env=d.read(root/"colab_environment.json")
    if not plan or not report:return {"status":"waiting_files","note":"Drive files not visible yet. This is not a GPU status.","candidates":[],"scores":[]}
    try:
        if plan["protocol"]!="v71-plan-v1" or report["protocol"]!="v71-session-v1" or report["plan_sha"]!=digest(plan):raise ValueError("Plan/report mismatch")
        if env and env.get("runtime_plan_sha")!=digest(plan):raise ValueError("Environment/plan mismatch")
        records={r["index"]:r for r in report["records"]}
        items=[];scores=[];fresh=max(d.mtime(root/n) for n in ("v71_plan.json","v71_report.json","colab_environment.json"))
        for i,item in enumerate(plan["candidates"]):
            c=item["config"];r=records.get(i,{})
            stages={}
            for name,cfg in (("smoke",item["smoke"]),("train",c)):
                folder=root/"v71_trials"/f"trial_{i:02d}"/name
                h=history(folder,int(cfg["epochs"]));result=d.read(folder/"result.json")
                exists=folder.is_dir();state=r.get(name,"not_started")
                fresh=max(fresh,d.mtime(folder/"history.csv"),d.mtime(folder/"result.json"),d.mtime(folder))
                score=None
                if state=="complete" and name=="train" and result.get("v56_key")==cfg["v56_key"] and result.get("v7",{}).get("recipe")==c["v7_recipe"] and (folder/"checkpoint.pt").is_file():
                    score=d.result_score(folder,result)
                    if score is not None and (d.finite(r.get("score")) is None or abs(score-r["score"])>1e-6):score=None
                    if score is not None:scores.append({"recipe":c["v7_recipe"],"score":score,"lane":"colab","index":i})
                stages[name]={"status":state,"folder_seen":exists,"epochs":cfg["epochs"],"history":h,"score":score}
            items.append({"index":i,"recipe":c["v7_recipe"],"lr":c.get("lr"),"stages":stages,"error":r.get("error")})
        reason=report.get("reason")
        return {"status":"saved_running" if reason=="running" else reason,"note":"Last saved stage evidence; no live runtime heartbeat or batch/GPU usage.",
                "device":env.get("device"),"hours":plan.get("hours"),"parent":plan.get("parent",{}).get("score"),
                "updated_at":fresh,"age_seconds":max(0,time.time()-fresh),"candidates":items,"scores":scores}
    except (KeyError,TypeError,ValueError,IndexError,AttributeError) as e:
        return {"status":"inconsistent","note":str(e),"candidates":[],"scores":[]}

def kaggle_plan(base,ref):
    if not ref:return {}
    for p in base.glob("run_v4*/round_*/*/remote_state.json"):
        st=d.read(p)
        if st.get("ref")!=ref:continue
        plan=d.read(p.parent/"v71_plan.json")
        return {"ref":ref,"hours":plan.get("hours"),"estimate_hours":plan.get("estimate_hours"),
                "parallel_sub":bool(plan.get("parallel_sub")),
                "recipes":[{"name":i["config"].get("v7_recipe"),"epochs":i["config"].get("epochs"),"reuse_smoke":i.get("reuse_smoke")} for i in plan.get("candidates",[])]}
    return {}

def scoped_logs(payload):
    # Epoch/batch IDs restart between trials; never mix smoke and train.
    lines=payload.get("lines",[]);stage=None;start=0
    for i,line in enumerate(lines):
        if "V71_STAGE " in line:
            try:
                val,_=json.JSONDecoder().raw_decode(line.split("V71_STAGE ",1)[1])
                if isinstance(val,dict):stage=val;start=i
            except ValueError:pass
    return {**payload,"stage":stage,"telemetry":d.old.parse_logs("\n".join(lines[start:])) if stage else payload.get("telemetry",{})}

def combined(root,base=d.REMOTE,research=d.RESEARCH,poll=False):
    k=d.snapshot(base,research,poll);c=colab_snapshot(root)
    # Colab score evidence is lightweight, not full checkpoint verification.
    scores=[{"lane":"kaggle","recipe":k["best"]["recipe"],"score":k["best"]["score"]}] if k["best"] else []
    scores.extend(c["scores"]);best=max(scores,key=lambda r:r["score"],default=None)
    return {"colab":c,"kaggle":k,"kaggle_plan":kaggle_plan(base,(k["active"] or {}).get("ref")),"best_recorded":best,"updated_at":time.time()}

def page():
    html=(d.ROOT/"tools/dashboard_v7.html").read_text(encoding="utf-8")
    addition=(d.ROOT/"tools/dashboard_v73_panel.html").read_text(encoding="utf-8")
    panel,script=addition.split("<!--SCRIPT-->",1)
    return html.replace("SPEECH ENHANCEMENT / V7","SPEECH ENHANCEMENT / V7.3").replace("<section class=\"panel hero\">",panel+"\n<section class=\"panel hero\">",1).replace("</script></body>",script+"\n</script></body>",1)

class Handler(d.Handler):
    colab_root=None
    def do_GET(self):
        path=urlsplit(self.path).path
        if path not in ("/","/parallel","/logs"):return super().do_GET()
        try:
            if path=="/":data=page().encode("utf-8");mime="text/html; charset=utf-8"
            else:
                value=combined(self.colab_root,poll=False) if path=="/parallel" else scoped_logs(d.log_payload())
                data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();mime="application/json; charset=utf-8"
            self.send_response(200);self.send_header("Content-Type",mime);self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers();self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):self.close_connection=True
        except Exception as exc:self.send_error(500,"Dashboard data unavailable");print("V73 DASHBOARD",repr(exc),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--colab-root",type=Path);p.add_argument("--port",type=int,default=8768);a=p.parse_args()
    if not 1<=a.port<=65535:p.error("Invalid port")
    Handler.colab_root=a.colab_root.resolve() if a.colab_root else None
    with ThreadingHTTPServer(("127.0.0.1",a.port),Handler) as server:
        print(f"V7.3 dashboard http://127.0.0.1:{a.port}",flush=True)
        print("Colab files: "+str(Handler.colab_root),flush=True);server.serve_forever()
if __name__=="__main__":main()
