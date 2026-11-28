# V7 research dashboard

Read-only local web dashboard, independent of all training controllers.
No Python ML dependencies, no GPU use, no job submission/cancellation.
Uses existing V6/V7 remote-state records and V7 journals.

Open a new PowerShell (keep the controller window running):
```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
py -m unittest discover -s tests -p "test_dashboard_v7.py" -v
py -u tools/dashboard_v7.py --port 8767
```
Open http://127.0.0.1:8767. The older V4 dashboard can continue on 8766.

## Displays
- Challenge PESQ and cumulative best for completed V6/V7 fixed-development
  runs with 164 matching per-file records. Smoke scores never affect ranking.
  This lightweight dashboard check is not a replacement for the trainer's
  complete checkpoint SHA and external-domain verification.
- All unresolved V6/V7 jobs across runs, including older jobs being resumed.
- Live Kaggle status, Epoch/Batch, loss and PESQ telemetry when provided.
- All 17 V7 strategies and actual non-smoke attempt / improvement counts;
  gain >= .002 counts as improvement, not a predicted success probability.
- Most recent persisted strategy selection rationale and quota cycle journal.
- Live GPU quota and KST refresh time; query errors explicitly mark stale values.
- Final PCM16 report when produced, plus version/smoke experiment filters.

Status/API refresh every 5s after responses, log refresh every 10s,
Kaggle status/log results cached 30s, quota results 60s.
During CPU preflight or before the first V7 journal is written, precise
controller phase may be unknown. Lock existence is not process liveness;
quota wait is inferred from the available journal and explicitly labeled
possible. No ETA for whole research. The monitor never writes experiment
files or changes an executing training process. Closing the browser or
stopping this dashboard does not stop V7. Only localhost is bound.
