# V7.1 single-allocation research

V7.1 is an opt-in controller separate from V7. One private Kaggle GPU job
runs up to three candidates by default (max four), each with its own smoke
and full training unless a verified smoke is reusable. The session uses
the existing V7 evaluator and frozen session-start parent. Actual family
gain can reorder remaining candidates. The best verified candidate is
eligible to warm-start the next session; **no within-session sequential
warm start** or optimizer-state restoration. All completed candidate
evidence is preserved; only one best_model.pt is exposed in the mount.
The local verifier checks every completed stage, including 164 challenge
and 64 external PESQ rows, lineage, checkpoint and V7 metadata. A partial
or failed stage is not promoted. If a session fails before any verified
training, stop and inspect. A remote hard shutdown can lose outputs.

## Startup (PowerShell)
Do not start while a V7 controller is running. If an old job is queued,
either let it finish (resume the same ref) or cancel it manually, confirm
CANCEL_ACKNOWLEDGED, and reconcile it. Do not delete checkpoints or lock files.
```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
$Py = ".\.venv-v571\Scripts\python.exe"
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
& $Py -m unittest discover -s tests -p "test_v71.py" -v
& $Py -u tools/ralph_v71.py --data-root "$DataRoot" --execute --resume --allow-public-assets --max-candidates 3 --session-hours 4.5 --max-sessions 0 --reserve-hours 2 --allow-reduce-mix --enable-korean --wait-for-quota --max-quota-cycles 2 --max-wait-hours 168 --top 3
```
Dry-run is default without --execute. First GPU allocation can still queue.
The runtime budget is measured after allocation, includes all stages, and
uses a child-process timeout; Kaggle polling timeout includes queue time.
No duplicate pushes after uncertain network status. A pending V7.1 job
must be resumed using V7.1, not the legacy reconcile tool (except confirmed
terminal cancellation). The original V7 source hash and final offline
package flow remain unchanged. Actual Kaggle integration must be tested.
V7 dashboard shows the active V7.1 job and imported winner; individual
candidate evidence is in the session report and v71_trials, not V7
single-job strategy counts. Live logs include V71_STAGE and V71_SCORE.
