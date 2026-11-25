"""Local read-only V4 research dashboard; no GPU control and no audio exposure."""
import argparse
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"runs"/"ralph_v2"

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):
        return {}

def snapshot(base=OUTPUT):
    runs=sorted(base.glob("run_v4*"))
    rows=[]
    for run in runs:
        for p in sorted(run.glob("round_*/*/remote_state.json")):
            state=read_json(p)
            result=read_json(p.parent/"result.json") if state.get("status")=="COMPLETE" else {}
            rows.append({"run":run.name,"round":p.parent.parent.name,"kind":state.get("kind"),
                         "status":state.get("status"),"model":result.get("model_family",result.get("winner")),
                         "challenge":result.get("best_validation_pesq_wb"),
                         "external":result.get("external_validation_pesq_wb"),
                         "robust":result.get("robust_score"),"domains":result.get("domain_scores"),
                         "metrics":result.get("metrics"),"audit":result.get("audit_pesq")})
    latest=runs[-1] if runs else None
    ui=read_json(latest/"ui_state.json") if latest else {}
    meta=read_json(latest/"public_research.json") if latest else {}
    ranked=[r for r in rows if r["kind"]=="v4_external" and isinstance(r["robust"],(int,float))]
    return {"state":{k:ui.get(k) for k in ("stage","remote_status","round","current_epoch","live_pesq","updated_at")},
            "best":max(ranked,key=lambda r:r["robust"]) if ranked else None,
            "research_counts":{k:len(meta.get(k,[])) for k in ("datasets","models","papers")},
            "experiments":rows[-200:]}

PAGE=r"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Ralph V4</title>
<style>body{font:15px system-ui;background:#f7f8fa;color:#16181c;margin:28px;max-width:1200px}h1{font-size:25px}h2{font-size:18px}section{background:white;padding:20px;border:1px solid #d8dde5;margin:16px 0;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}svg{width:100%;height:220px}.scroll{overflow:auto}</style>
<h1>Ralph V4 — Generalization Research</h1><section><h2>현재 상태</h2><pre id="state">Loading</pre><h2>최고 일반화 모델</h2><pre id="best">—</pre><h2>공개 연구 탐색</h2><pre id="research">—</pre></section>
<section><h2>검증 점수 추이</h2><svg id="chart" viewBox="0 0 900 220" role="img"></svg><p>Challenge: 파란색 · External: 초록색 · Robust: 주황색</p></section>
<section class="scroll"><h2>실험 기록</h2><table><thead><tr><th>Run / Round</th><th>종류</th><th>상태</th><th>모델</th><th>Challenge</th><th>External</th><th>Robust</th><th>Audit</th></tr></thead><tbody id="rows"></tbody></table></section>
<script>const fmt=v=>typeof v==='number'?v.toFixed(4):v??'—';const text=(id,x)=>document.getElementById(id).textContent=JSON.stringify(x,null,2);
async function load(){try{const r=await fetch('/api',{cache:'no-store'});const d=await r.json();text('state',d.state);text('best',d.best);text('research',d.research_counts);const body=document.getElementById('rows');body.replaceChildren();for(const x of d.experiments.slice().reverse()){const row=body.insertRow();for(const v of [x.run+' / '+x.round,x.kind,x.status,x.model,x.challenge,x.external,x.robust,x.audit])row.insertCell().textContent=fmt(v)}const svg=document.getElementById('chart');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';const es=d.experiments.filter(x=>x.status==='COMPLETE');for(const [key,color] of [['challenge','#2563eb'],['external','#15803d'],['robust','#d97706']]){const pts=es.map((x,i)=>typeof x[key]==='number'?[20+i*860/Math.max(1,es.length-1),210-(x[key]+.5)*38]:null).filter(Boolean);if(pts.length){const p=document.createElementNS(ns,'polyline');p.setAttribute('points',pts.map(v=>v.join(',')).join(' '));p.setAttribute('stroke',color);p.setAttribute('stroke-width','2');p.setAttribute('fill','none');svg.append(p);for(const [cx,cy] of pts){const c=document.createElementNS(ns,'circle');c.setAttribute('cx',cx);c.setAttribute('cy',cy);c.setAttribute('r',3);c.setAttribute('fill',color);svg.append(c)}}}}catch(e){document.getElementById('state').textContent=String(e)}}load();setInterval(load,5000);</script></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/":
            payload=PAGE.encode("utf-8"); mime="text/html; charset=utf-8"
        elif self.path=="/api":
            payload=json.dumps(snapshot(),ensure_ascii=False,allow_nan=False).encode("utf-8"); mime="application/json; charset=utf-8"
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Type",mime)
        self.send_header("Content-Length",str(len(payload))); self.send_header("Cache-Control","no-store")
        self.end_headers(); self.wfile.write(payload)
    def log_message(self,*args): pass

def main():
    p=argparse.ArgumentParser(); p.add_argument("--port",type=int,default=8766); a=p.parse_args()
    print(f"V4 dashboard http://127.0.0.1:{a.port}",flush=True)
    with ThreadingHTTPServer(("127.0.0.1",a.port),Handler) as server:
        server.serve_forever()

if __name__=="__main__":
    main()
