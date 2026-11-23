"""Read-only, localhost-only dashboard for Adaptive Ralph V2.

Run: py tools/ralph_dashboard.py
Open: http://127.0.0.1:8765
No third-party requirements; this process NEVER submits a Kaggle GPU job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path
import re
import webbrowser
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "ralph_v2"
FAMILIES = {
    "magnitude": "Magnitude U-Net",
    "residual_phase": "Residual Phase U-Net",
    "complex": "Complex Mask U-Net",
}


def read_json(path: Path):
    try:
        if not path.is_file() or path.stat().st_size > 1000000:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None



# Read-only Kaggle observations are cached to avoid making a remote API call
# on every browser refresh. The UI does not submit/stop any GPU work.
_STATUS_CACHE = {}
_LOG_CACHE = {}
_REMOTE_LOCK = threading.Lock()
_REMOTE_TTL = 35.0
_SAFE_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")


def _cached_kaggle_probe(ref, command, cache, extractor):
    if not isinstance(ref, str) or len(ref) > 180 or not _SAFE_REF.fullmatch(ref):
        return None
    with _REMOTE_LOCK:
        now = time.monotonic()
        previous = cache.get(ref)
        if previous is not None and now - previous[0] < _REMOTE_TTL:
            return previous[1]
        cli = shutil.which("kaggle")
        value = None
        if cli:
            try:
                proc = subprocess.run(
                    [cli, "kernels", command, ref],
                    capture_output=True, text=True, timeout=12, check=False,
                    cwd=ROOT,
                )
                if proc.returncode == 0:
                    value = extractor(proc.stdout or "")
            except (OSError, subprocess.TimeoutExpired, ValueError):
                value = None
        cache[ref] = (time.monotonic(), value)
        return value


def kaggle_worker_status(ref):
    def extract(output):
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
        return match.group(1) if match else None
    return _cached_kaggle_probe(ref, "status", _STATUS_CACHE, extract)


def kaggle_live_epoch(ref):
    def extract(output):
        matches = re.findall(
            r"epoch=(\d+)\s+loss=[0-9.eE+-]+\s+val_pesq=([0-9.]+)",
            output[-250000:],
        )
        if not matches:
            return None
        epoch, pesq = matches[-1]
        value = number(pesq)
        if 0 < int(epoch) <= 1000 and value is not None:
            return {"epoch": int(epoch), "pesq": value}
        return None
    return _cached_kaggle_probe(ref, "logs", _LOG_CACHE, extract)


def most_recent_job(run):
    if not run:
        return None
    options = []
    for path in run.glob("round_*/**/remote_state.json"):
        info = read_json(path)
        if not isinstance(info, dict):
            continue
        try:
            round_id = int(path.parent.parent.name.removeprefix("round_"))
        except ValueError:
            continue
        mode = path.parent.name
        priority = 1 if mode == "full" else 0
        options.append((round_id, priority, path, info))
    return max(options, key=lambda x: (x[0], x[1])) if options else None


def infer_family(run, round_id):
    if not run or round_id is None:
        return ""
    source_path = run / ("round_%04d" % round_id) / "candidate.py"
    try:
        code = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    if "class ResidualPhaseUNet(" in code:
        return "residual_phase"
    if "class ComplexMaskUNet(" in code:
        return "complex"
    if "class SpectrogramUNet(" in code:
        return "magnitude"
    return ""


def log_tail():
    logs = sorted(OUTPUT.glob("adaptive_*.log"), reverse=True)
    if not logs:
        return []
    try:
        with logs[0].open("rb") as handle:
            handle.seek(0, 2)
            length = min(handle.tell(), 32000)
            handle.seek(-length, 2)
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        # Do not leak credential lines in the localhost API.
        deny = re.compile(r"api[_-]?key|password|authorization|secret|token", re.I)
        return [line.strip()[:190] for line in lines[-22:]
                if line.strip() and not deny.search(line)][-12:]
    except OSError:
        return []


def collect():
    runs = sorted((p for p in OUTPUT.glob("run_*") if p.is_dir()), key=lambda p: p.name)
    latest = runs[-1] if runs else None
    family_scores = {family: None for family in FAMILIES}
    global_score = None
    global_origin = ""
    for file in (ROOT / "runs" / "ralph").glob("run_*/best.json"):
        data = read_json(file)
        v = number(data.get("score")) if isinstance(data, dict) else None
        if v is not None and (global_score is None or v > global_score):
            global_score, global_origin = v, "Ralph V1"
    points = []
    for run in runs:
        champ = read_json(run / "global_best.json")
        v = number(champ.get("score")) if isinstance(champ, dict) else None
        if v is not None and (global_score is None or v > global_score):
            global_score, global_origin = v, str(champ.get("origin", ""))[:100]
        for family in FAMILIES:
            data = read_json(run / "families" / family / "best.json")
            score = number(data.get("score")) if isinstance(data, dict) else None
            if score is not None and (family_scores[family] is None or score > family_scores[family]):
                family_scores[family] = score
        hist = run / "experiments.jsonl"
        if hist.is_file():
            try:
                lines = hist.read_text(encoding="utf-8").splitlines()[-150:]
            except OSError:
                lines = []
            for line in lines:
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                score = number(record.get("score"))
                if score is None:
                    continue
                points.append({
                    "run": run.name, "round": record.get("round"),
                    "score": score, "family": str(record.get("family", ""))[:40],
                    "status": str(record.get("status", ""))[:40],
                })
        # Recovered Kaggle jobs may have finished before local controller
        # recorded its history; show the validated result exactly once.
        for result_file in run.glob("round_*/full/result.json"):
            folder = result_file.parent
            meta = read_json(folder / "remote_state.json")
            result = read_json(result_file)
            if not isinstance(meta, dict) or meta.get("status") != "COMPLETE":
                continue
            if not isinstance(result, dict) or result.get("validation_files") != 164:
                continue
            try:
                round_n = int(folder.parent.name.removeprefix("round_"))
            except ValueError:
                continue
            if any(r["run"] == run.name and r["round"] == round_n for r in points):
                continue
            family = {
                "V3_RESIDUAL_PHASE": "residual_phase",
                "V2_CRM_PHASE": "complex", "E02_UNET": "magnitude",
                "E03_MRSTFT": "magnitude"
            }.get(result.get("experiment"), "unknown")
            score = number(result.get("best_validation_pesq_wb"))
            if score is not None:
                points.append({
                    "run": run.name, "round": round_n, "score": score,
                    "family": family, "status": "RECOVERED_COMPLETE",
                })
    points.sort(key=lambda p: (p["run"], int(p["round"] or 0)))
    state = read_json(latest / "ui_state.json") if latest else None
    if not isinstance(state, dict):
        state = {}
    updated = str(state.get("updated_at", ""))[:48]
    alive = False
    if updated:
        try:
            age = (datetime.now(timezone.utc) -
                   datetime.fromisoformat(updated.replace("Z", "+00:00"))).total_seconds()
            alive = -10 <= age < 155
        except (TypeError, ValueError, OverflowError):
            pass
    stage = str(state.get("stage", "offline"))[:30]
    if not alive and stage not in ("error", "complete", "stopped"):
        stage = "offline"
    ref = str(state.get("ref", ""))[:150]
    remote = str(state.get("remote_status", ""))[:30]
    family = str(state.get("family", ""))[:40]
    active_round = state.get("round")
    message = str(state.get("message", ""))[:200]
    remote_live = False
    remote_checked = False

    # Older controllers do not publish ui_state.json. Recover the current
    # experiment from the last persisted submission instead of claiming idle.
    job = most_recent_job(latest)
    if job:
        round_id, _, path, info = job
        ref = ref or str(info.get("ref", ""))[:150]
        active_round = active_round if active_round is not None else round_id
        family = family or infer_family(latest, round_id)
        local_status = str(info.get("status", ""))[:36]
        if not remote:
            remote = local_status
        if local_status == "SUBMIT_PENDING":
            actual = kaggle_worker_status(str(info.get("ref", "")))
            remote_checked = True
            if actual:
                remote = actual
                remote_live = actual in ("RUNNING", "QUEUED", "PENDING", "INITIALIZING")
                if not alive:
                    if actual in ("RUNNING", "QUEUED", "PENDING", "INITIALIZING"):
                        stage = "training" if path.parent.name == "full" else "smoke"
                        message = "Kaggle 작업 %s · 로컬 제어기 상태는 미확인" % actual
                    elif actual == "COMPLETE":
                        stage = "recover"
                        message = "Kaggle 학습 완료 · 로컬 결과 회수 필요"
                    elif actual in ("ERROR", "CANCELLED", "CANCELED"):
                        stage = "error"
                        message = "Kaggle 원격 작업 종료/실패 · 로그 확인 필요"
            elif not alive:
                stage = "checking"
                message = "원격 작업 조회 불가 · 실행/중단 여부 확인 필요"
        elif not alive and stage == "offline" and local_status == "COMPLETE":
            stage = "complete"
            message = "마지막 Kaggle 작업 완료 · 새 실행은 확인되지 않음"
    epoch = number(state.get("current_epoch"))
    total = state.get("epochs")
    live_pesq = number(state.get("live_pesq"))
    if remote_live and remote == "RUNNING" and epoch is None:
        progress = kaggle_live_epoch(ref)
        if progress:
            epoch = progress["epoch"]
            live_pesq = progress["pesq"]
            if total is None:
                total = info.get("epochs") if job else None
    if latest and active_round is not None:
        try:
            round_dir = latest / ("round_%04d" % int(active_round))
            # Never display a completed 2-epoch smoke result as the progress
            # of the current full 25-epoch GPU job.
            mode = "smoke" if stage == "smoke" else "full"
            if job and job[0] == int(active_round):
                mode = job[2].parent.name
            data = read_json(round_dir / mode / "result.json")
            if isinstance(data, dict):
                if epoch is None:
                    epoch = data.get("epochs_completed")
                total = total or data.get("epochs_completed")
        except (ValueError, TypeError, OverflowError):
            pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alive": alive, "remoteLive": remote_live, "remoteChecked": remote_checked,
        "stage": stage, "message": message, "family": family,
        "round": active_round, "ref": ref, "remote": remote,
        "run": latest.name if latest else "No runs yet",
        "updated": updated,
        "epoch": epoch, "total": total, "livePESQ": live_pesq,
        "best": global_score, "origin": global_origin,
        "goal": 3.0,
        "families": [{"id": k, "name": name, "score": family_scores[k]}
                     for k, name in FAMILIES.items()],
        "points": points[-90:],
        "logs": log_tail(),
    }


HTML = "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Ralph / Speech Lab</title>\n<style>\n:root{font-family:Inter,system-ui,-apple-system,\"Segoe UI\",sans-serif;background:#090f1d;color:#eaf2ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top left,#1f2d5470,transparent 45%),#090f1d;min-height:100vh}\n.app{max-width:1370px;margin:auto;padding:30px}.header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:26px}.brand{display:flex;align-items:center;gap:14px}.logo{width:47px;height:47px;display:grid;place-items:center;border-radius:15px;font-weight:900;font-size:23px;color:#0c2532;background:linear-gradient(135deg,#79f3c6,#71a6ff);box-shadow:0 0 30px #58d8bf24}.h1{font-weight:800;font-size:21px;letter-spacing:-.7px}.sub{font-size:12px;color:#8e9db8;margin-top:5px}.chip{border:1px solid #2c3b59;background:#192339;color:#b4c7dd;border-radius:30px;padding:10px 14px;font-size:12px;white-space:nowrap;font-weight:650}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#8492ab;margin-right:8px}.dot.live{background:#73e9bd;box-shadow:0 0 12px #73e9bdaa;animation:pulse 2s infinite}@keyframes pulse{50%{opacity:.5}}\n.cards{display:grid;grid-template-columns:1.2fr 1fr 1fr;gap:15px;margin-bottom:15px}.card{background:linear-gradient(145deg,#18233a,#10192a);border:1px solid #283650;border-radius:19px;padding:23px;min-width:0;box-shadow:0 10px 30px #0002}.hero{background:radial-gradient(ellipse at top right,#20656870,transparent 65%),linear-gradient(145deg,#1d2c48,#10192c)}.over{font-size:11px;font-weight:750;letter-spacing:1.7px;text-transform:uppercase;color:#9badcb}.major{font-size:44px;font-weight:800;letter-spacing:-2px;margin:17px 0 8px}.minor{font-size:30px;font-weight:800;margin:19px 0 9px;letter-spacing:-1px}.green{color:#7bedc6}.blue{color:#9ab9ff}.subtext{font-size:13px;color:#95a7c3;line-height:1.6;overflow-wrap:anywhere}.track{height:8px;background:#2a3954;border-radius:20px;overflow:hidden;margin-top:18px}.track div{height:100%;background:linear-gradient(90deg,#82aefe,#72f4bf);width:0;border-radius:20px;transition:width .5s}.below{font-size:11px;color:#8395b4;margin-top:11px}\n.body{display:grid;grid-template-columns:1.55fr 1fr;gap:15px}.stack{display:grid;gap:15px;align-content:start;min-width:0}.cap{font-size:15px;font-weight:750;letter-spacing:-.2px}.titlebar{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:22px}.tag{font-size:10px;font-weight:800;padding:7px 10px;background:#1a393c;color:#85efd0;border-radius:8px;letter-spacing:.4px}\n.steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}.step{border-radius:13px;padding:13px 7px;background:#142034;border:1px solid #2c3a54;text-align:center;color:#7284a2}.step.active{color:#a5ffdf;border-color:#4aacae;background:#1b383f}.step.done{color:#98baff}.num{display:grid;place-items:center;width:26px;height:26px;margin:0 auto 8px;border-radius:50%;background:#24344e;font-size:12px;font-weight:800}.step.active .num{background:#7bebc9;color:#0e2429}.step span{display:block;font-size:11px;font-weight:700}\n.current{display:flex;justify-content:space-between;align-items:center;margin:26px 0 10px;gap:10px}.current strong{font-size:14px}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.muted{color:#8699b7}.ref{font-size:11px;color:#8fa0b8;line-height:1.5;word-break:break-all;margin-bottom:15px;min-height:30px}.loadbar{height:9px;background:#293b55;border-radius:99px;overflow:hidden;position:relative}.loadbar .fill{height:100%;width:34%;border-radius:99px;background:linear-gradient(90deg,#81b3ff,#72edc0)}.loadbar.running .fill{position:absolute;animation:loading 2.2s ease-in-out infinite}@keyframes loading{0%{left:-36%}45%{left:45%}100%{left:104%}}.hint{font-size:11px;color:#8a9bbb;margin-top:12px;line-height:1.6}\n#chart{width:100%;min-height:205px}.chartSvg{width:100%;height:210px;display:block}.gridline{stroke:#2e3d57;stroke-dasharray:3 6}.axis{fill:#889bb8;font:11px ui-monospace,monospace}.empty{height:195px;display:grid;place-items:center;color:#8093af;font-size:13px}.fam{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:15px 0;border-bottom:1px solid #283650}.fam:last-child{border-bottom:0;padding-bottom:0}.famName{font-size:13px;font-weight:650}.famMeta{font-size:11px;color:#8799b5;margin-top:5px}.famScore{font-size:19px;font-weight:800}\n.logs{max-height:265px;overflow:auto;scrollbar-width:thin}.log{padding:8px 0;border-bottom:1px solid #26354b;font-size:11px;line-height:1.6;color:#a0b1c8;overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace}.log:last-child{border-bottom:0}\n.tablewrap{border:1px solid #283850;border-radius:12px;overflow:auto}table{width:100%;border-collapse:collapse;min-width:480px;text-align:left;font-size:12px}th{background:#1e2b43;color:#92a6c3;font-size:10px;letter-spacing:.7px;padding:13px}td{padding:13px 12px;border-top:1px solid #2b3850;color:#d2def1}.foot{display:flex;justify-content:space-between;gap:20px;color:#7586a2;margin-top:25px;font-size:11px;line-height:1.8}\n@media(max-width:980px){.body{grid-template-columns:1fr}.cards{grid-template-columns:1fr 1fr}.hero{grid-column:1/-1}}@media(max-width:570px){.app{padding:18px 12px}.cards{grid-template-columns:1fr}.hero{grid-column:auto}.card{padding:18px}.major{font-size:38px}.steps{grid-template-columns:repeat(2,1fr)}.header .chip{max-width:125px;overflow:hidden;text-overflow:ellipsis}}\n</style></head>\n<body><div class=\"app\"><header class=\"header\"><div class=\"brand\"><div class=\"logo\">∿</div><div><div class=\"h1\">Ralph <span class=\"muted\">/ Speech Lab</span></div><div class=\"sub\">Adaptive speech enhancement · live experiment monitor</div></div></div><span class=\"chip\"><span class=\"dot\" id=\"dot\"></span><span id=\"conn\">연결 중</span></span></header>\n<section class=\"cards\"><div class=\"card hero\"><div class=\"over\">GLOBAL CHAMPION · PESQ-WB</div><div class=\"major green mono\" id=\"best\">—</div><div class=\"subtext\" id=\"bestName\">검증된 최고 모델</div><div class=\"track\"><div id=\"goalBar\"></div></div><div class=\"below\" id=\"goal\">목표 PESQ 3.0</div></div><div class=\"card\"><div class=\"over\">CURRENT EXPERIMENT</div><div class=\"minor blue\" id=\"round\">대기 중</div><div class=\"subtext\" id=\"model\">모델 준비 중</div><div class=\"below mono\" id=\"run\">실험 기록 확인 중</div></div><div class=\"card\"><div class=\"over\">PROCESS STATUS</div><div class=\"minor\" id=\"stage\">—</div><div class=\"subtext\" id=\"remote\">Kaggle 작업 상태 확인 중</div><div class=\"below\" id=\"lastUpdate\">파일 기반 자동 갱신</div></div></section>\n<div class=\"body\"><div class=\"stack\"><div class=\"card\"><div class=\"titlebar\"><div class=\"cap\">실험 진행 현황</div><span class=\"tag\" id=\"flag\">STANDBY</span></div><div class=\"steps\" id=\"steps\"></div><div class=\"current\"><strong id=\"message\">실험 대기 중</strong><span class=\"mono muted\" id=\"epoch\">—</span></div><div class=\"ref mono\" id=\"ref\">Kaggle 실험이 시작되면 노트북 ID가 표시돼.</div><div class=\"loadbar\" id=\"bar\"><div class=\"fill\" id=\"fill\"></div></div><div class=\"hint\" id=\"hint\">GPU 학습 중 진행률을 임의로 추정하지 않아.</div></div><div class=\"card\"><div class=\"titlebar\"><div class=\"cap\">PESQ 실험 추이</div><span class=\"muted mono\" id=\"count\">0 runs</span></div><div id=\"chart\" class=\"empty\">데이터 불러오는 중...</div><div class=\"hint\">완료된 실험 PESQ · 초록색 점선 = 최고 모델 점수</div></div><div class=\"card\"><div class=\"titlebar\"><div class=\"cap\">최근 실험 기록</div><span class=\"muted\" style=\"font-size:11px\">164-clip Validation</span></div><div class=\"tablewrap\"><table><thead><tr><th>ROUND</th><th>MODEL</th><th>PESQ</th><th>STATUS</th></tr></thead><tbody id=\"rows\"></tbody></table></div></div></div><div class=\"stack\"><div class=\"card\"><div class=\"titlebar\"><div class=\"cap\">모델 계열별 최고 성능</div><span class=\"muted mono\" style=\"font-size:11px\">BEST / FAMILY</span></div><div id=\"families\"></div></div><div class=\"card\"><div class=\"titlebar\"><div class=\"cap\">실행 로그</div><span class=\"tag\">LOCAL LOG</span></div><div class=\"logs\" id=\"logs\"><div class=\"log\">로그 대기 중</div></div></div><div class=\"card\"><div class=\"cap\">실험 안전 규칙</div><p class=\"subtext\">최소 +0.005 PESQ 향상 시 모델을 교체하며, 성능이 정체되면 다른 모델 계열을 탐색해. 평가 데이터나 체크포인트는 변경하지 않아.</p><p class=\"subtext\">이 대시보드는 <b style=\"color:#d7eafa\">읽기 전용</b>이고 Kaggle GPU 작업을 새로 제출하지 않아.</p></div></div></div>\n<footer class=\"foot\"><span>LOCAL ONLY · 127.0.0.1 · READ-ONLY KAGGLE STATUS</span><span id=\"updated\" class=\"mono\">4s refresh</span></footer></div>\n<script>\nconst q=id=>document.getElementById(id), names={magnitude:\"Magnitude U-Net\",residual_phase:\"Residual Phase U-Net\",complex:\"Complex Mask U-Net\"};\nconst phases=[\"codex\",\"smoke\",\"training\",\"scoring\"],phaseNames=[\"가설 설계\",\"스모크 테스트\",\"Kaggle GPU 학습\",\"PESQ 검증\"];\nconst stageNames={starting:\"시작 중\",codex:\"모델 개선\",smoke:\"스모크\",training:\"GPU 학습\",scoring:\"결과 분석\",switch:\"모델 변경\",complete:\"완료\",stopped:\"안전 중단\",error:\"오류\",offline:\"대기 중\",checking:\"상태 확인 필요\",recover:\"결과 회수 필요\"};\nconst fmt=x=>typeof x===\"number\"&&Number.isFinite(x)?x.toFixed(4):\"—\";\nfunction e(tag,cls,text){const a=document.createElement(tag);a.className=cls||\"\";if(text!==undefined)a.textContent=text;return a}\nfunction chart(points,champ){\n const valid=points.filter(p=>typeof p.score===\"number\").slice(-26);\n q(\"count\").textContent=valid.length+\" experiments\";\n if(!valid.length){q(\"chart\").replaceChildren(e(\"div\",\"empty\",\"완료된 실험 결과가 없습니다.\"));return}\n const vals=valid.map(p=>p.score);if(typeof champ===\"number\")vals.push(champ);\n const min=Math.max(0,Math.min(...vals)-0.15),max=Math.max(...vals)+0.15;\n const ns=\"http://www.w3.org/2000/svg\",svg=document.createElementNS(ns,\"svg\");\n svg.setAttribute(\"class\",\"chartSvg\");svg.setAttribute(\"viewBox\",\"0 0 620 220\");\n function node(tag,attrs){const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));svg.append(n);return n}\n const x=i=>44+540*i/Math.max(1,valid.length-1),y=v=>176-144*(v-min)/(max-min);\n for(let i=0;i<=4;i++){const yy=32+i*36;node(\"line\",{x1:44,x2:590,y1:yy,y2:yy,class:\"gridline\"});node(\"text\",{x:1,y:yy+4,class:\"axis\"}).textContent=(max-i*(max-min)/4).toFixed(2)}\n if(typeof champ===\"number\")node(\"line\",{x1:44,x2:590,y1:y(champ),y2:y(champ),stroke:\"#7deec4\",\"stroke-width\":1.5,\"stroke-dasharray\":\"6 5\",opacity:.7});\n node(\"polyline\",{points:valid.map((p,i)=>String(x(i))+\",\"+String(y(p.score))).join(\" \"),fill:\"none\",stroke:\"#8cb4ff\",\"stroke-width\":3,\"stroke-linejoin\":\"round\"});\n valid.forEach((p,i)=>{const n=node(\"circle\",{cx:x(i),cy:y(p.score),r:4.5,fill:\"#83b4ff\",stroke:\"#14243a\",\"stroke-width\":2});const tip=document.createElementNS(ns,\"title\");tip.textContent=names[p.family]+\" / \"+fmt(p.score);n.append(tip)});\n q(\"chart\").replaceChildren(svg);\n}\nfunction show(d){\n q(\"dot\").className=\"dot\"+(d.alive||d.remoteLive?\" live\":\"\");\n q(\"conn\").textContent=d.alive?\"LIVE · CONTROLLER\":d.remoteLive?\"LIVE · KAGGLE\":d.stage===\"recover\"?\"KAGGLE · COMPLETE\":d.stage===\"checking\"?\"상태 확인 필요\":\"OFFLINE · HISTORY\";\n q(\"best\").textContent=fmt(d.best);q(\"bestName\").textContent=d.origin||\"검증된 최고 모델\";\n q(\"goalBar\").style.width=(typeof d.best===\"number\"?Math.min(100,100*d.best/d.goal):0)+\"%\";\n q(\"goal\").textContent=typeof d.best===\"number\"?\"PESQ 3.0까지 \"+Math.max(0,d.goal-d.best).toFixed(4)+\" 남음\":\"목표 PESQ 3.0\";\n q(\"round\").textContent=d.round!=null?\"ROUND \"+String(d.round).padStart(2,\"0\"):\"대기 중\";\n q(\"model\").textContent=names[d.family]||\"현재 선택된 모델 없음\";q(\"run\").textContent=d.run;\n q(\"stage\").textContent=stageNames[d.stage]||d.stage;\n q(\"remote\").textContent=\"Kaggle 상태: \"+(d.remote||\"확인 전\");\n q(\"lastUpdate\").textContent=d.alive?\"컨트롤러 하트비트 확인됨\":d.remoteLive?\"Kaggle 실제 상태 직접 조회 · 35초 갱신\":d.stage===\"checking\"?\"Kaggle 조회 실패 · 상태 직접 확인 필요\":\"로컬 실행 기록 확인 중\";\n const pos=phases.indexOf(d.stage);\n q(\"steps\").replaceChildren(...phases.map((p,i)=>{const box=e(\"div\",\"step\"+(i===pos?\" active\":pos>=0&&i<pos?\" done\":\"\"));box.append(e(\"div\",\"num\",pos>=0&&i<pos?\"✓\":String(i+1)),e(\"span\",\"\",phaseNames[i]));return box}));\n q(\"flag\").textContent=d.alive?\"ACTIVE · \"+d.stage.toUpperCase():d.remoteLive?\"REMOTE · \"+d.remote:d.stage===\"checking\"?\"CHECK STATUS\":d.stage===\"recover\"?\"RECOVER OUTPUT\":\"STANDBY\";\n q(\"message\").textContent=d.message||stageNames[d.stage]||\"대기 중\";\n q(\"ref\").textContent=d.ref||\"현재 Kaggle 노트북 없음\";\n q(\"epoch\").textContent=d.epoch!=null?String(d.epoch)+\" / \"+(d.total||\"?\")+\" Epoch\":d.remoteLive?\"Epoch 조회 중\":\"—\";\n const busy=(d.alive||d.remoteLive)&&phases.includes(d.stage),known=typeof d.epoch===\"number\"&&typeof d.total===\"number\"&&d.total>0;\n q(\"bar\").className=\"loadbar\"+(busy&&!known?\" running\":\"\");\n q(\"fill\").style.width=known?Math.min(100,100*d.epoch/d.total)+\"%\":busy?\"34%\":\"0%\";\n q(\"hint\").textContent=known?\"실제 Epoch \"+d.epoch+\"/\"+d.total+(typeof d.livePESQ===\"number\"?\" · 현재 Validation PESQ \"+fmt(d.livePESQ):\"\"):d.stage===\"checking\"?\"상태 조회 불가. 'kaggle kernels status '+(d.ref||'작업ID')+' 로 직접 확인해.\":d.stage===\"recover\"?\"원격 학습이 완료됐어. py tools/reconcile_ralph_v2.py --resolve 로 결과 회수 가능.\":busy?\"Kaggle 원격 작업 확인됨 · Epoch 로그가 없을 때는 진행 애니메이션만 표시해.\":\"학습 결과와 로그를 4초마다 자동 갱신해.\";\n q(\"families\").replaceChildren(...d.families.map(f=>{const box=e(\"div\",\"fam\"),left=e(\"div\");left.append(e(\"div\",\"famName\",f.name),e(\"div\",\"famMeta\",f.id));box.append(left,e(\"div\",\"famScore\"+(f.score===d.best?\" green\":\"\"),fmt(f.score)));return box}));\n chart(d.points,d.best);\n const recent=d.points.slice(-8).reverse();\n q(\"rows\").replaceChildren(...(recent.length?recent.map(r=>{const tr=e(\"tr\");[String(r.round||\"—\"),names[r.family]||r.family,fmt(r.score),r.status].forEach((v,i)=>tr.append(e(\"td\",i===2?\"mono\":\"\",v)));return tr}):[(()=>{const tr=e(\"tr\"),td=e(\"td\",\"\",\"완료된 기록이 없어.\");td.colSpan=4;tr.append(td);return tr})()]));\n q(\"logs\").replaceChildren(...(d.logs.length?d.logs.slice().reverse().map(v=>e(\"div\",\"log\",v)):[e(\"div\",\"log\",\"최근 로그가 없습니다.\")]));\n q(\"updated\").textContent=\"LAST SYNC \"+new Date().toLocaleTimeString(\"ko-KR\",{hour12:false})+\" · 4 SEC\";\n}\nasync function sync(){try{const response=await fetch(\"/api/status\",{cache:\"no-store\"});if(!response.ok)throw Error(\"HTTP \"+response.status);show(await response.json())}catch(error){q(\"conn\").textContent=\"서버 연결 실패\";q(\"dot\").className=\"dot\"}}\nsync();setInterval(sync,4000);\n</script></body></html>"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            mime = "text/html; charset=utf-8"
        elif path == "/api/status":
            body = json.dumps(collect(), ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
            mime = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Read-only local Ralph UI")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.port < 1024 or args.port > 65535:
        parser.error("Port must be 1024..65535")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("Ralph dashboard: http://127.0.0.1:%d" % args.port, flush=True)
    print("Read-only. Press Ctrl+C to close the dashboard server.", flush=True)
    try:
        webbrowser.open("http://127.0.0.1:%d" % args.port)
    except Exception:
        pass
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
