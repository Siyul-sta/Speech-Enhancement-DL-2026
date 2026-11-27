# V5.9 — Progressive long training

## Design

V5.9 is a new optional controller, not a replacement for V5.6.
The V5.7.2 base training/inference file and existing CPU preflight
identity files are unchanged. A strict, source-checked generator
modifies only training control flow. The previous completed public
benchmark is reused if its verified source and amplitude policies match.

- Start with up to three distinct eligible pilot strategies (default:
  SepFormer DNS progressive, CMGAN progressive, complex-residual head).
- Use compatible verified V5.6 pilots if available. Do not assume the
  historical 2.4800 model is compatible.
- Interleave exploration with extensions near the strongest observed
  challenge PESQ (within 0.20); favor recent PESQ improvement.
- Attempt milestones 25, 50, 100 and up to 200 epochs per lineage.
  Blocks are budgeted, default maximum 20 additional epochs per job,
  minimum 12 for transfer and 5 for native. The minimum may cause a
  milestone to be skipped or the final maximum to be approached rather
  than reached exactly. The cap is attempted epochs, not the age of
  the selected best checkpoint.
- Continue from a verified best_model.pt with exact SHA-256, strict
  tensor loading and parent re-evaluation. Preserve cumulative best
  checkpoint age separately from total attempted epochs. Reset AdamW
  moments and LR scheduler each block: this is **weights-only warm
  continuation**, not a bit-exact optimizer/RNG resume.
- Reduce block starting LR (at most half the previous start, no higher
  than previous final LR). ReduceLROnPlateau within a block.
- Progressive transfer uses effective checkpoint epoch for unfreezing;
  augmentation and shuffle seeds use attempted epoch to avoid replay.
- Preserve the parent's best checkpoint if training fails to improve.
  Do not extend a lineage with gain below 0.005 or an unchanged initial
  checkpoint. Native training retains existing within-block early
  stopping. Transfer blocks run to their requested length (legacy
  verified contract), then apply the plateau rule.
- Re-evaluate parent score and both external domains before training.
  Keep the original external preservation floor across the lineage;
  fail closed on checkpoint corruption, incompatible data or metadata.
- Read-only existing V4 dashboard works; V59 lineage details are
  recorded in result.json, v59_best.json and v59_efficiency.jsonl.
- CPU ensembles are optional with --ensemble; not run between every
  training job by default.

Scope: 14 non-critic Native / conventional transfer strategies from the
V5 catalog. LoRA, anchored teacher, PESQ critic and cascades remain in
V5.6; they require additional continuation-state and stage-schedule
validation and are intentionally not extended by this version.
No new model architecture, dataset upload or change in the evaluation
metric. This is not a claim that 200 epochs or PESQ 4.5 is achievable.

## PowerShell

If a controller is already running, let its current job finish and
recover/verify any unresolved remote state before switching. Never run
V5.6 and V5.9 simultaneously. External assets must be permitted.

```powershell
git pull --ff-only
$Py = ".\.venv-v571\Scripts\python.exe"
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
& $Py -m compileall -q src tools tests
& $Py -m unittest discover -s tests -p "test_v59_progressive.py" -v
& $Py -u tools/ralph_v59.py --data-root "$DataRoot"
```

Existing PASS reports remain valid if they were generated from the
unchanged V5.7.2 files and environment. V5.9 still requires them.
Do not bypass a stale-report error.

```powershell
py tools/reconcile_ralph_v2.py --resolve
kaggle quota
& $Py -u tools/ralph_v59.py --execute --allow-public-assets --data-root "$DataRoot" --max-epochs 200 --max-rounds 1 --reserve-hours 2
```

The first new-model job is an isolated GPU smoke. Review the actual
output before repeated execution.

```powershell
& $Py -u tools/ralph_v59.py --execute --allow-public-assets --data-root "$DataRoot" --max-epochs 200 --max-rounds 0 --reserve-hours 2 --ensemble
```

0 means no fixed job-count limit, not an unlimited budget. Default
local per-job wait limit 360 minutes; timeouts preserve unresolved
state (not cancellation). Existing manual network recovery still
applies. GPU quotas and wall-time estimates are checked before jobs.
An affordable block can still exceed its estimate. No remote job is
launched by the dry run. Main branch unchanged. Actual live long
training/continuation and score gain still need verification.
