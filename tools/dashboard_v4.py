"""Local read-only V4 dashboard. Never submits, cancels or modifies a job."""
import argparse
import json
import math
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"runs"/"ralph_v2"
TERMINAL={"COMPLETE","ERROR_TERMINAL"}
POLL_LOCK=threading.Lock()
REMOTE_CACHE={}
LOG_CACHE={}
LOG_LOCK=threading.Lock()

def read_json(path):
    try:
        value=json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):
        return {}

def finite(value):
    return value if type(value) in (int,float) and math.isfinite(value) else None

def remote_status(ref):
    """Read-only CLI status, cached for 30 seconds. No state file writes."""
    if not isinstance(ref,str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+",ref):
        return {"status":"UNKNOWN","error":"Invalid kernel reference"}
    with POLL_LOCK:
        now=time.monotonic()
        cached=REMOTE_CACHE.get(ref)
        if cached and now-cached[0]<30: return cached[1]
        cli=shutil.which("kaggle")
        try:
            if not cli: raise RuntimeError("Kaggle CLI not found")
            p=subprocess.run([cli,"kernels","status",ref],capture_output=True,encoding="utf-8",errors="replace",timeout=12,check=False)
            m=re.search(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout+" "+p.stderr)
            if p.returncode or not m: raise RuntimeError("Kaggle status unavailable")
            value={"status":m.group(1),"error":None}
        except (OSError,subprocess.TimeoutExpired,RuntimeError) as exc:
            value={"status":"UNKNOWN","error":str(exc)}
        REMOTE_CACHE[ref]=(now,value)
        return value

def parse_logs(raw):
    """Only observed telemetry; older jobs may expose epoch summaries alone."""
    epochs={}
    batches={}
    download=None
    activity=None
    number=r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for line in raw.splitlines():
        if "V4_TELEMETRY " in line:
            try:
                event,_=json.JSONDecoder().raw_decode(line.split("V4_TELEMETRY ",1)[1])
                if not isinstance(event,dict): continue
                kind=event.get("event")
                if kind=="data":
                    download={k:event.get(k) for k in ("dataset","stage","bytes","total_bytes","elapsed_s")}
                    activity="data"
                elif kind=="batch" and type(event.get("epoch")) is int and type(event.get("batch")) is int:
                    batches[(event["epoch"],event["batch"])]=event
                    activity="batch"
                elif kind=="epoch" and type(event.get("epoch")) is int:
                    epochs[event["epoch"]]=event
                    activity="epoch"
            except (ValueError,TypeError): pass
        match=re.search(r"epoch=(\d+)\s+loss=("+number+r")\s+val_pesq=("+number+r")\s+best=("+number+r")",line)
        if match:
            e=int(match[1])
            if e not in epochs:
                epochs[e]={"epoch":e,"loss":finite(float(match[2])),"challenge":finite(float(match[3])),"best":finite(float(match[4]))}
                activity="epoch"
    recent=list(batches.values())[-100:]
    return {"epochs":[epochs[k] for k in sorted(epochs)][-100:],"batches":recent,"batch":recent[-1] if recent else None,"download":download,"activity":activity}

def remote_logs(ref):
    """Independent bounded, cached read-only log fetch; never blocks status API."""
    if not isinstance(ref,str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+",ref):
        return {"ref":None,"error":"No job reference","lines":[],"telemetry":parse_logs("")}
    with LOG_LOCK:
        now=time.monotonic()
        cached=LOG_CACHE.get(ref)
        if cached and now-cached[0]<30: return cached[1]
        cli=shutil.which("kaggle")
        try:
            if not cli: raise RuntimeError("Kaggle CLI not found")
            p=subprocess.run([cli,"kernels","logs",ref],capture_output=True,encoding="utf-8",errors="replace",timeout=20,check=False)
            if p.returncode: raise RuntimeError("Kaggle log retrieval unavailable")
            raw=(p.stdout or "")[-250000:]
            value={"ref":ref,"error":None,"lines":raw.splitlines()[-200:],"telemetry":parse_logs(raw),"fetched_at":time.time()}
        except (OSError,subprocess.TimeoutExpired,RuntimeError) as exc:
            value=dict(cached[1]) if cached else {"ref":ref,"lines":[],"telemetry":parse_logs("")}
            value["error"]=str(exc)
        LOG_CACHE.clear()
        LOG_CACHE[ref]=(now,value)
        return value

def snapshot(base=OUTPUT,poll_remote=False):
    runs=sorted(base.glob("run_v4*"))
    rows=[]
    for run in runs:
        for p in sorted(run.glob("round_*/*/remote_state.json")):
            state=read_json(p)
            if not state: continue
            result=read_json(p.parent/"result.json") if state.get("status")=="COMPLETE" else {}
            rows.append({"run":run.name,"round":p.parent.parent.name,"kind":state.get("kind"),
                         "ref":state.get("ref"),"status":state.get("status"),"recorded_at":p.stat().st_mtime,
                         "model":result.get("model_family",result.get("winner")),
                         "challenge":finite(result.get("best_validation_pesq_wb")),
                         "external":finite(result.get("external_validation_pesq_wb")),
                         "robust":finite(result.get("robust_score")),"domains":result.get("domain_scores"),
                         "metrics":result.get("metrics"),"audit":finite(result.get("audit_pesq")),
                         "transductive":bool(result.get("transductive_noisy")),
                         "protocol":result.get("evaluation_protocol")})
    latest=runs[-1] if runs else None
    ui=read_json(latest/"ui_state.json") if latest else {}
    meta=read_json(latest/"public_research.json") if latest else {}
    current=[r for r in rows if latest and r["run"]==latest.name]
    active=next((r for r in reversed(current) if r["status"] not in TERMINAL),None)
    remote=remote_status(active["ref"]) if active and poll_remote else None
    ranked=[r for r in rows if r["kind"] in ("v4_external","v4_public") and r["status"]=="COMPLETE" and not r["transductive"] and r["protocol"]=="v4.2-fixed-dev-v2" and r["robust"] is not None]
    return {"state":{k:ui.get(k) for k in ("stage","remote_status","round","current_epoch","live_pesq","updated_at","message","ref")},
            "active":active,"remote":remote,"run":latest.name if latest else None,
            "best":max(ranked,key=lambda r:r["robust"]) if ranked else None,
            "research_counts":{k:len(meta.get(k,[])) for k in ("datasets","models","papers")},
            "experiments":rows[-200:],"ranked":ranked[-200:],
            "counts":{"completed":sum(r["status"]=="COMPLETE" for r in rows),"failed":sum(r["status"]=="ERROR_TERMINAL" for r in rows),"ranked":len(ranked)}}

PAGE=r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ralph V4 | Research Console</title>
<style>
:root{color-scheme:dark;--bg:#0b1120;--panel:#121b2c;--line:#25334b;--text:#e6edf8;--muted:#92a4bd;--blue:#60a5fa;--green:#34d399;--orange:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,'Malgun Gothic',sans-serif}main{max-width:1440px;margin:auto;padding:30px 32px}header{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:25px}h1{font-size:25px;margin:0;letter-spacing:-.5px}h2{font-size:15px;margin:0 0 16px}p{margin:5px 0;color:var(--muted)}.eyebrow{font-size:11px;font-weight:700;letter-spacing:1.6px;color:var(--blue)}.pill{border:1px solid var(--line);background:#16243a;border-radius:20px;padding:6px 12px;font-size:12px;white-space:nowrap}.panel,.metric{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:20px}.hero{display:grid;grid-template-columns:1.2fr 1fr;gap:24px;align-items:center}.status-title{font-size:23px;font-weight:700;margin:7px 0}.status-title[data-tone=good]{color:var(--green)}.status-title[data-tone=warn]{color:var(--orange)}.status-title[data-tone=bad]{color:#f87171}.small{font-size:12px;color:var(--muted)}.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.step{padding:10px 8px;border:1px solid var(--line);border-radius:8px;color:var(--muted);font-size:12px;text-align:center}.step.done{border-color:#245f50;color:var(--green)}.step.active{border-color:var(--blue);color:var(--blue);background:#132746}.step b{display:block;margin-bottom:4px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.metric .value{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;margin:5px 0}.metric .label{color:var(--muted);font-size:12px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:16px}.empty{min-height:190px;display:grid;place-items:center;color:var(--muted);text-align:center;border:1px dashed var(--line);border-radius:8px;padding:15px}svg{width:100%;height:220px}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}.kv{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding:11px 0}.kv:last-child{border:0}.kv span{color:var(--muted)}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:12px}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}td{font-variant-numeric:tabular-nums}.tag{border-radius:6px;padding:4px 8px;background:#1e2d44;color:var(--muted)}.tag.good{color:var(--green);background:#14362e}.tag.warn{color:var(--orange);background:#382d17}.tag.bad{color:#f87171;background:#3d2330}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}.footer{margin-top:15px;color:var(--muted);font-size:11px}.err{color:#f87171} @media(max-width:850px){main{padding:18px}.hero,.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}header{align-items:flex-start}.steps{grid-template-columns:repeat(2,1fr)}} 
.telemetry{margin:16px 0}.livegrid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}.livecell{padding:12px;border:1px solid var(--line);border-radius:8px}.livecell b{display:block;font-size:17px;margin-top:4px;overflow-wrap:anywhere}.bar{height:8px;border-radius:9px;background:#25334b;overflow:hidden;margin:12px 0}.bar span{display:block;height:100%;background:var(--blue);width:0}.console{background:#080e19;border:1px solid var(--line);padding:12px;border-radius:8px;height:260px;overflow:auto;font:12px/1.6 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#c7d4e7}.chartgrid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.chartgrid svg{height:150px}.loghead{display:flex;justify-content:space-between;gap:10px;align-items:center}.loghead h2{margin:0}.loghead label{color:var(--muted);font-size:12px}@media(max-width:850px){.livegrid{grid-template-columns:repeat(2,1fr)}.chartgrid{grid-template-columns:1fr}}
</style></head><body><main>
<header><div><div class="eyebrow">SPEECH ENHANCEMENT / V4</div><h1>Ralph Research Console</h1><p>일반화 성능 중심 자동 연구 · 읽기 전용 모니터링</p></div><span class="pill" id="connection">연결 확인 중</span></header>
<section class="panel hero"><div><div class="eyebrow">CURRENT WORKFLOW</div><div class="status-title" id="status">상태 확인 중</div><p id="description">로컬 연구 기록을 불러오는 중입니다.</p><p class="small" id="detail"></p><a id="kernel" target="_blank" rel="noopener noreferrer" hidden>현재 Kaggle 작업 보기 ↗</a></div><div><div class="steps" id="steps"><div class="step"><b>01</b>공개 데이터</div><div class="step"><b>02</b>GPU 검증</div><div class="step"><b>03</b>자동 연구</div><div class="step"><b>04</b>독립 평가</div></div><p class="small" id="remoteNote">실제 원격 상태는 최대 30초 간격으로 조회합니다.</p></div></section>
<section class="panel telemetry"><div class="loghead"><h2>실시간 작업 모니터</h2><span class="small" id="logFresh">로그 조회 전</span></div><div class="livegrid"><div class="livecell"><span class="small">현재 작업</span><b id="liveActivity">—</b></div><div class="livecell"><span class="small">로컬 제출 후 경과</span><b id="elapsed">—</b></div><div class="livecell"><span class="small">Epoch / Batch</span><b id="batch">—</b></div><div class="livecell"><span class="small">GPU 메모리 (할당/전체)</span><b id="vram">—</b><span class="small" id="gpuUtil">사용률 미계측</span></div></div><p id="progressText">진행률 계측값 없음</p><div class="bar" id="progressBar" hidden><span id="progressFill"></span></div><p class="small" id="eta">예상 시간: 계측 전</p><div class="chartgrid"><div><h2>Epoch PESQ</h2><svg id="livePesq" viewBox="0 0 400 150" role="img" aria-label="Epoch PESQ 추이"></svg><p class="small">Challenge / External / Robust</p></div><div><h2>Batch Loss</h2><svg id="liveLoss" viewBox="0 0 400 150" role="img" aria-label="Batch loss 추이"></svg><p class="small" id="lossLabel">기록된 표본만 표시</p></div></div><div class="loghead"><h2>Kaggle 실행 로그</h2><label><input type="checkbox" id="follow" checked> 자동 스크롤</label></div><pre class="console" id="console">로그를 불러오는 중입니다.</pre></section>
<div class="metrics"><div class="metric"><div class="label">최고 Robust PESQ</div><div class="value" id="robust">—</div><div class="small">평균 + 최저 환경 성능</div></div><div class="metric"><div class="label">Challenge PESQ</div><div class="value" id="challenge">—</div><div class="small">최고 일반화 모델 기준</div></div><div class="metric"><div class="label">External PESQ</div><div class="value" id="external">—</div><div class="small">외부 일반·스트레스 환경</div></div><div class="metric"><div class="label">완료 연구 실험</div><div class="value" id="count">0</div><div class="small" id="counts">검증된 순위 대상만 집계</div></div></div>
<div class="grid"><section class="panel"><h2>일반화 점수 추이</h2><div class="empty" id="empty">아직 순위에 포함할 평가 결과가 없습니다.<br>캐시·Smoke Test는 모델 순위에서 제외됩니다.</div><svg id="chart" viewBox="0 0 850 220" role="img" aria-label="PESQ 점수 추이" style="display:none"></svg><div class="legend"><span><i class="dot" style="background:#60a5fa"></i>Challenge</span><span><i class="dot" style="background:#34d399"></i>External</span><span><i class="dot" style="background:#fbbf24"></i>Robust</span></div></section>
<section class="panel"><h2>최고 모델 · 공개 연구</h2><div class="kv"><span>모델</span><b id="model">선정 전</b></div><div class="kv"><span>외부 일반</span><b id="normal">—</b></div><div class="kv"><span>외부 스트레스</span><b id="stress">—</b></div><div class="kv"><span>데이터셋 후보</span><b id="datasets">0</b></div><div class="kv"><span>공개 모델 후보</span><b id="models">0</b></div><div class="kv"><span>논문 후보</span><b id="papers">0</b></div></section></div>
<section class="panel"><h2>실험 기록</h2><div class="scroll"><table><thead><tr><th>실험 / Round</th><th>유형</th><th>상태</th><th>모델</th><th>Challenge</th><th>External</th><th>Robust</th><th>Audit</th></tr></thead><tbody id="rows"></tbody></table></div><p id="noRows" class="small">실험 기록이 아직 없습니다.</p></section>
<div class="footer">로컬 기록 및 Kaggle 읽기 전용 상태 조회 · 실제 진행률은 확인 가능한 정보만 표시 · GPU 제어 기능 없음</div>
</main><script>
const $=id=>document.getElementById(id),put=(id,x)=>$(id).textContent=x;
const fmt=x=>typeof x==='number'?x.toFixed(4):'—';
const names={v4_cache:'데이터 캐시',v4_smoke:'GPU Smoke',v4_public:'공개 모델',v4_external:'학습 실험',v4_audit:'독립 평가'};
const statusNames={SUBMIT_PENDING:'제출 / 확인 중',RUNNING:'실행 중',QUEUED:'대기 중',PENDING:'대기 중',INITIALIZING:'초기화 중',COMPLETE:'완료',ERROR:'오류',ERROR_TERMINAL:'종료 오류',CANCELLED:'취소',CANCELED:'취소',CANCEL_ACKNOWLEDGED:'취소 확인',UNKNOWN:'확인 불가'};
const tone=s=>s==='COMPLETE'?'good':/ERROR|CANCEL/.test(s||'')?'bad':'warn';
function draw(data){const svg=$('chart');svg.replaceChildren();$('empty').style.display=data.length?'none':'grid';svg.style.display=data.length?'block':'none';if(!data.length)return;
const ns='http://www.w3.org/2000/svg',el=(tag,attrs)=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);svg.append(n);return n};
const y=v=>190-(v+.5)/5*175,x=i=>50+i*765/Math.max(1,data.length-1);
for(const v of [0,1,2,3,4]){el('line',{x1:50,x2:815,y1:y(v),y2:y(v),stroke:'#25334b'});const t=el('text',{x:18,y:y(v)+4,fill:'#92a4bd','font-size':11});t.textContent=v}
for(const [key,color] of [['challenge','#60a5fa'],['external','#34d399'],['robust','#fbbf24']]){const pts=data.map((r,i)=>typeof r[key]==='number'?[x(i),y(r[key])]:null).filter(Boolean);if(pts.length>1)el('polyline',{points:pts.map(p=>p.join(',')).join(' '),stroke:color,'stroke-width':2,fill:'none'});for(const [cx,cy] of pts)el('circle',{cx,cy,r:4,fill:color})}}
function render(d){const a=d.active,st=d.state||{},remote=d.remote,rows=d.experiments||[],b=d.best;const next=a||null;if(latestJob?.ref!==next?.ref){$('console').textContent='새 작업 로그 조회 중';lastLogs=null}latestJob=next;elapsed();const raw=a?(remote&&remote.status!=='UNKNOWN'?remote.status:(st.remote_status&&st.ref===a.ref?st.remote_status:a.status)):null;
let title='실행 기록 없음',desc='연구 제어기가 시작되면 이곳에 표시됩니다.',phase=-1;
if(a){phase=({v4_cache:0,v4_smoke:1,v4_public:2,v4_external:2,v4_audit:3})[a.kind]??-1;title=(names[a.kind]||a.kind)+' · '+(statusNames[raw]||raw||'확인 중');desc=a.kind==='v4_cache'?'공개 데이터 다운로드 및 캐시 준비 단계입니다. GPU 학습은 아직 시작되지 않았습니다.':a.kind==='v4_smoke'?'소규모 학습 및 고정 검증으로 실제 Kaggle 환경을 확인합니다.':a.kind==='v4_audit'?'고정된 최종 모델의 독립 평가를 수행합니다.':'연구 실험을 실행하고 결과를 검증합니다.';if(raw==='COMPLETE')desc='원격 작업 완료. 로컬 결과 확인 및 다음 단계 전환을 기다리는 상태입니다.';if(/ERROR|CANCEL/.test(raw||''))desc='원격 작업이 종료됐습니다. 제어기 로그를 확인하세요.'}
else if(rows.length){const last=rows[rows.length-1];title=last.status==='COMPLETE'?(names[last.kind]||last.kind)+' 완료':'최근 작업 '+(statusNames[last.status]||last.status);desc=st.message||'현재 실행 중인 기록이 없습니다.';phase=({v4_cache:0,v4_smoke:1,v4_public:2,v4_external:2,v4_audit:3})[last.kind]??-1}
put('status',title);$('status').dataset.tone=raw?tone(raw):rows.length&&rows[rows.length-1].status==='COMPLETE'?'good':'warn';put('description',desc);put('detail',[d.run,st.current_epoch!=null?'Epoch '+st.current_epoch:null,typeof st.live_pesq==='number'?'PESQ '+fmt(st.live_pesq):null].filter(Boolean).join(' · '));
const link=$('kernel');link.hidden=!(a&&a.ref);if(!link.hidden)link.href='https://www.kaggle.com/code/'+a.ref;
put('remoteNote',remote?(remote.error?'원격 조회 실패: '+remote.error+' · 로컬 기록 표시':'Kaggle 직접 조회: '+remote.status):'실제 원격 상태는 최대 30초 간격으로 조회합니다.');
const completed=new Set(rows.filter(r=>r.status==='COMPLETE').map(r=>r.kind));[...$('steps').children].forEach((e,i)=>{e.className='step'+(a&&i===phase?' active':(i===0&&completed.has('v4_cache')||i===1&&completed.has('v4_smoke')||i===2&&(completed.has('v4_public')||completed.has('v4_external'))||i===3&&completed.has('v4_audit'))?' done':'')});
for(const k of ['robust','challenge','external'])put(k,fmt(b&&b[k]));put('count',d.counts.ranked);put('counts','전체 완료 '+d.counts.completed+' · 종료 오류 '+d.counts.failed);put('model',b?.model||'선정 전');put('normal',fmt(b?.domains?.external_normal));put('stress',fmt(b?.domains?.external_stress));for(const k of ['datasets','models','papers'])put(k,d.research_counts[k]);
const body=$('rows');body.replaceChildren();$('noRows').hidden=rows.length>0;for(const r of rows.slice().reverse()){const row=body.insertRow(),values=[r.run.replace('run_','')+' / '+r.round.replace('round_',''),names[r.kind]||r.kind,(a&&r.ref===a.ref&&remote&&remote.status!=='UNKNOWN')?remote.status:r.status,r.model||'—',fmt(r.challenge),fmt(r.external),fmt(r.robust),fmt(r.audit)];values.forEach((v,i)=>{const c=row.insertCell();if(i===2){const t=document.createElement('span');t.className='tag '+tone(v);t.textContent=statusNames[v]||v||'—';c.append(t)}else c.textContent=v??'—'})}
draw(d.ranked||[])}

let latestJob=null,lastLogs=null;
const duration=s=>Number.isFinite(s)&&s>=0?[Math.floor(s/3600),Math.floor(s%3600/60),Math.floor(s%60)].map(v=>String(v).padStart(2,'0')).join(':'):'—';
function elapsed(){put('elapsed',latestJob&&Number.isFinite(latestJob.recorded_at)?duration(Math.max(0,Date.now()/1000-latestJob.recorded_at)):'—')}
setInterval(elapsed,1000);
function miniChart(id,data,series){const svg=$(id);svg.replaceChildren();const ns='http://www.w3.org/2000/svg';const add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!=null)e.textContent=text;svg.append(e);return e};const all=data.flatMap(r=>series.map(([key])=>r[key])).filter(Number.isFinite);if(!all.length){add('text',{x:200,y:75,'text-anchor':'middle',fill:'#92a4bd','font-size':12},'계측값 없음');return}let lo=Math.min(...all),hi=Math.max(...all);if(lo===hi){lo-=.05;hi+=.05}const y=v=>122-(v-lo)/(hi-lo)*100;for(let i=0;i<3;i++){const v=lo+(hi-lo)*i/2;add('line',{x1:48,x2:390,y1:y(v),y2:y(v),stroke:'#25334b'});add('text',{x:2,y:y(v)+4,fill:'#92a4bd','font-size':10},v.toFixed(3))}for(const [key,color] of series){const pts=data.map((r,i)=>Number.isFinite(r[key])?[48+i*342/Math.max(1,data.length-1),y(r[key])]:null).filter(Boolean);if(pts.length>1)add('polyline',{points:pts.map(p=>p.join(',')).join(' '),stroke:color,'stroke-width':2,fill:'none'});for(const [cx,cy] of pts)add('circle',{cx,cy,r:2,fill:color})}}
function renderLogs(d){if(latestJob&&d.ref!==latestJob.ref)return;lastLogs=d;const t=d.telemetry||{},b=t.batch,dl=t.download,e=(t.epochs||[]).at(-1);put('logFresh',(d.error?'로그 조회 실패 · 이전 기록 유지':d.fetched_at?'로그 조회 '+new Date(d.fetched_at*1000).toLocaleTimeString():'로그 없음')+(d.error?' · '+d.error:''));const c=$('console');const body=(d.lines||[]).join('\n')||'Kaggle에서 아직 반환된 로그가 없습니다. 실행 중에도 로그 제공이 지연될 수 있습니다.';if(c.textContent!==body){c.textContent=body;if($('follow').checked)c.scrollTop=c.scrollHeight}
put('liveActivity',t.activity==='data'&&dl?dl.dataset+' · '+dl.stage:t.activity==='batch'?'학습 Batch':t.activity==='epoch'?'Epoch 평가':'세부 로그 계측 전');
put('batch',b?'E'+b.epoch+'/'+b.epochs+' · '+b.batch+'/'+b.batches:e?'Epoch '+e.epoch+' 완료':'—');
put('vram',b&&Number.isFinite(b.gpu_alloc_mib)&&Number.isFinite(b.gpu_total_mib)?(b.gpu_alloc_mib/1024).toFixed(2)+' / '+(b.gpu_total_mib/1024).toFixed(2)+' GB':'—');put('gpuUtil',b&&Number.isFinite(b.gpu_util)?'GPU 사용률 '+b.gpu_util+'%':'사용률 미계측');
let p=null,eta=null,label='진행률 계측값 없음';
if(t.activity==='data'&&dl){label=dl.dataset+' · '+dl.stage;if(dl.stage==='download'&&Number.isFinite(dl.bytes)){label+=' · '+(dl.bytes/1048576).toFixed(1)+' MiB';if(dl.total_bytes>0){p=dl.bytes/dl.total_bytes;label+=' / '+(dl.total_bytes/1048576).toFixed(1)+' MiB';if(p>0&&p<=1&&dl.elapsed_s>0)eta=dl.elapsed_s*(1-p)/p}}}
else if(t.activity==='batch'&&b&&b.batches>0){p=b.batch/b.batches;label='현재 Epoch의 학습 Batch '+b.batch+'/'+b.batches;if(p>0&&p<=1&&b.elapsed_s>0)eta=b.elapsed_s*(1-p)/p}
else if(t.activity==='epoch'&&e){label='Epoch '+e.epoch+' 평가 완료 · PESQ '+fmt(e.challenge)}
put('progressText',label+(p!=null?' · '+(100*Math.max(0,Math.min(1,p))).toFixed(1)+'%':''));$('progressBar').hidden=p==null;$('progressFill').style.width=p==null?'0':100*Math.max(0,Math.min(1,p))+'%';put('eta',eta!=null?'현재 단계 남은 시간 추정 '+duration(eta)+' (전체 완료 시간 아님)':'예상 시간: 계측값 부족');
miniChart('livePesq',t.epochs||[],[['challenge','#60a5fa'],['external','#34d399'],['robust','#fbbf24']]);miniChart('liveLoss',t.batches?.length?t.batches:t.epochs||[],[['loss','#60a5fa']]);put('lossLabel',t.batches?.length?'표본 Batch Loss (최근 100개)':'Epoch 평균 Loss (Batch 계측 전)')}
async function loadLogs(){try{const r=await fetch('/logs',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);renderLogs(await r.json())}catch(e){put('logFresh','로그 연결 오류 · '+e.message)}finally{setTimeout(loadLogs,5000)}}loadLogs();

async function load(){try{const r=await fetch('/api',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);render(await r.json());put('connection','● 연결됨 · '+new Date().toLocaleTimeString())}catch(e){put('connection','연결 오류 · '+e.message)}finally{setTimeout(load,5000)}}load();
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=="/":
            payload=PAGE.encode("utf-8"); mime="text/html; charset=utf-8"
        elif self.path=="/api":
            payload=json.dumps(snapshot(poll_remote=True),ensure_ascii=False,allow_nan=False).encode("utf-8"); mime="application/json; charset=utf-8"
        elif self.path=="/logs":
            d=snapshot()
            job=d["active"] or (d["experiments"][-1] if d["experiments"] else None)
            value=remote_logs(job["ref"]) if job else {"ref":None,"lines":[],"telemetry":parse_logs(""),"error":None}
            payload=json.dumps(value,ensure_ascii=False,allow_nan=False).encode("utf-8"); mime="application/json; charset=utf-8"
        else:
            self.send_error(404); return
        try:
            self.send_response(200); self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(payload))); self.send_header("Cache-Control","no-store")
            self.end_headers(); self.wfile.write(payload)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # A browser refresh/tab close or local socket interruption can abort
            # a response while status is being fetched. Only this client is gone.
            self.close_connection=True
    def log_message(self,*args): pass

def main():
    p=argparse.ArgumentParser(); p.add_argument("--port",type=int,default=8766); a=p.parse_args()
    print(f"V4 dashboard http://127.0.0.1:{a.port}",flush=True)
    with ThreadingHTTPServer(("127.0.0.1",a.port),Handler) as server:
        server.serve_forever()

if __name__=="__main__":
    main()
