# V7 autonomous bounded research

V7 consolidates the V6.12–V6.15 roadmap into one independent controller.
It preserves all existing source-based evaluation contracts, checkpoints
and remote experiment history. No V6/V6.9/V6.11 process is stopped by this code.

## Implemented stages

- V6.12: observed MP-DNS improvement per measured epoch-hour, shrinkage
  priors based on exact-parent Train phase/magnitude diagnosis, exploration
  bonus and terminal-attempt memory. Epoch-hours are a compute proxy, not
  billed GPU-hours. Ranking is heuristic, not a guarantee of improvement.
- V6.13: energy-weighted band-balanced compressed spectral loss, tunable
  circular phase loss, bounded residual correction penalty and critic warmup
  on disjoint Train-fit examples. Critic calibration is on reserved Train
  speakers; Pearson >= .6 and MAE <= .5 gate metric gradients. Real metric
  steps are recorded; no guarantee the critic passes calibration.
- V6.14: optionally wait for actual verified Kaggle GPU quota recovery,
  bounded by quota cycles and total waiting hours. Never submit during
  the wait, never infer quota from the timestamp alone, retain shared
  exclusive lock and check unresolved jobs.
- V6.15: 17 reviewed declarative candidate mutations plus automatically
  generated epoch/augmentation profiles. Successful flags are inherited.
  Candidates vary LR, phase weight, spectral weight, hard-noise mining,
  full length, critic warmup/weight, residual cascade/trust and optional
  Korean speech. No arbitrary code generation, external paper download,
  or unreviewed architecture execution.
- V7: unified closed loop: diagnose, rank, budget, actual-Train CPU
  preflight, smoke, private Kaggle GPU training, unchanged 164+64 score
  verification, best-checkpoint retention, restart/reconcile and final
  PCM16 offline comparison. Local run journals persist. Optimizer and
  critic replay state are not restored; weights continue.

## Safe handoff

If an old local controller is still running, do not start V7 concurrently.
If you choose to migrate before its natural stop, stop waiting older
controllers first, then interrupt the active local controller normally.
Do not cancel the existing Kaggle job. Confirm old local processes
have exited; --resume polls the same remote job before any new submission.
Never manually remove a live lock or uncertain remote state.

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
$Py = ".\.venv-v571\Scripts\python.exe"
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
& $Py -m unittest discover -s tests -p "test_v7.py" -v
& $Py -u tools/ralph_v7.py --data-root "$DataRoot"
```

Dry-run uses the verified existing ledger; it does not submit or query
Kaggle quota. Optional real upstream CPU test locally:
```powershell
$env:V7_TEST_MP = "1"
& $Py -m unittest discover -s tests -p "test_v7.py" -v
Remove-Item Env:V7_TEST_MP
```

## Continuous operation within explicit bounds

```powershell
& $Py -u tools/ralph_v7.py --data-root "$DataRoot" --execute --resume --allow-public-assets --max-rounds 0 --max-stale 0 --max-epochs 200 --chunk-epochs 3 --reserve-hours 2 --allow-reduce-mix --wait-for-quota --max-quota-cycles 2 --max-wait-hours 168 --top 3
```

Keep this local PowerShell and computer running. No ChatGPT background
execution. Add --enable-korean only if you want the approved public
Korean corpus experiment considered; it can add download/compute costs.
Optional --budget-hours 3 limits each quota cycle further.
Two cycles means the initial cycle and at most one verified renewal;
168h limits cumulative waiting time, not training duration. No score-based
stop, but stop for exhausted/unaffordable candidates, budget, 200 attempted
epochs on selected lineage, three consecutive terminal failures,
unexpected overall champion/source, independent audit seal or unresolved job.
Quota and time estimates are not guaranteed hard billing caps.

Outputs:
- runs/v6_refine/v7/run_*.json: events, candidate rationale, score/gain,
  actual metric steps, quota cycles and stop reason.
- runs/v6_refine/v7/preflight_*/report.json: actual Train CPU acceptance.
- runs/v6_refine/unified/diagnostics/: reused exact-parent Train diagnoses.
- runs/v6_refine/v7_final_*/report.json: verified development PCM16 results.
- runs/v6_refine/v7_final_*/package/: frozen offline bundle.

Local private outputs are ignored by Git. Validation Clean is for
evaluation/selection only; final inference takes only noisy input.
No private test score or 4.0 breakthrough claimed until actually measured.
