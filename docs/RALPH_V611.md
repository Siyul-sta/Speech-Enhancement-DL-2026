# V6.11 unified Ralph loop

One controller replaces the separate V6.9/V6.10 continuation commands.
It reuses the exact V6.10 generated training source (no evaluator change)
and all verified V6/V6.9/V6.10 completed checkpoints. The original V6
preflight source and existing controllers remain unchanged.

## Policy
1. Select the best verified 164-clip non-smoke checkpoint across V6.
   Exact ties prefer fewer attempted epochs (epoch-zero fallback is not
   treated as a new improvement).
2. Reuse an exact-parent Train diagnostic from an existing analysis, or
   infer on the reserved 32 Train-select clips and compute the target-
   informed phase/magnitude diagnostic. No Validation targets train models.
3. Continue a productive checkpoint; if stagnant, choose untried phase
   derivatives, first-epoch hard noise, full-length, PESQ critic or
   frozen residual cascade. Only one new hypothesis at a time, except
   explicit mix-reduction profiles and critic's required full length.
4. Estimate both smoke and actual training, verify CPU parent/gradient,
   smoke then train only if live quota and wall budget allow both.
   Critic trials require at least 3 epochs; its calibration may still
   reject metric gradients, so report actual metric_steps.
5. Re-evaluate 164 challenge and 64 external examples using unchanged
   V6 evaluator and root-domain guard. Never overwrite prior checkpoints.
   Keep the best, continue or switch recipe. Failed terminal trials are
   recorded and not repeated on the same parent/hypothesis; lower LR
   continuation has at most one retry per parent.
6. At termination, CPU compare top unique checkpoints (same family allowed),
   Train-only gate fit/select, 164 PCM16 comparison and verified offline
   bundle. Development results only, not private test performance.

No score-based hard stop. --max-rounds 0 and --max-stale 0 disable those
limits; default lineage cap is 200 attempted epochs. Stop for initial
wall budget/live GPU reserve, no affordable untried recipe, three
consecutive terminal failures, audit seal, unresolved remote state or
non-MP-DNS champion. No automatic wait for quota refresh. This is not
unbounded GPU usage. Cost estimates can differ from billed time.
Optimizer state and critic replay are not restored; weights are continued.

## Run

Do not run while V6 or V6.9 local processes are active or waiting.
The shared lock and a 90-second stable idle check prevent simultaneous
submissions but cannot identify an older controller waiting outside the lock.
If migrating mid-job, do not cancel Kaggle: stop the waiting V6.9 local
process first, then stop the old V6 local controller normally (Ctrl+C),
verify no old local controller remains, and use --resume. Already submitted
jobs are reconciled by exact ref before any new submission. Never delete
an active lock or unresolved state by hand.

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
$Py = ".\.venv-v571\Scripts\python.exe"
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
& $Py -m unittest discover -s tests -p "test_v610_focus.py" -v
& $Py -m unittest discover -s tests -p "test_v611_loop.py" -v
& $Py -u tools/ralph_v611.py --data-root "$DataRoot"
```

Dry run only lists the verified champion and configured recipes.

```powershell
& $Py -u tools/ralph_v611.py --data-root "$DataRoot" --execute --resume --allow-public-assets --max-rounds 0 --max-stale 0 --max-epochs 200 --chunk-epochs 3 --reserve-hours 2 --allow-reduce-mix --top 3
```

Default initial budget = live GPU quota minus reserve. Optional
--budget-hours 3 caps the run further. --allow-reduce-mix explicitly
allows smaller augmentation budgets (1024/256, 512/128, 256/64 caps);
the event log marks these as compound changes, not isolated causal gains.
Use --skip-final only if final CPU packaging is not desired.

Logs: V611 PLAN, V610 PREFLIGHT PASS, V611 SCORE, V611 STOP, V611 FINAL.
Private local run logs: runs/v6_refine/unified/run_*.json.
Final bundle/report: runs/v6_refine/unified_final_*/.
These paths are ignored by Git. No private audio uploads to GitHub.
