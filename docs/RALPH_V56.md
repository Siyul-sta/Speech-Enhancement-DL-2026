# V5.6 — unified Ralph scheduling

Run entrypoint: tools/ralph_v56.py. V5.5 local tests passed. This release
integrates the existing controlled stages, public benchmark, verified
checkpoint history and optional CPU ensemble step. No new GPU job was
submitted by development.

## Scheduling

1. Once per source/cache identity, try the V4 public zero-shot benchmark.
2. Before a full trial of a new architecture, require a successful two-epoch
   isolated GPU smoke of that architecture. Public adapters may download
   their pretrained assets; the smoke does not download external corpora.
   Smoke outputs are never ranked. A failed smoke blocks that architecture
   for this source version until reviewed.
3. The catalog contains all 26 V5.2/3/4 strategies (7 + 11 + 8).
   Pilot budgets are 5, 12 and 8 epochs respectively. Longer transfer
   pilots ensure that the existing progressive-unfreezing schedule actually
   runs. Existing V5.2 early stopping remains active.
4. After four successful pilots, prioritize promotion of the upper half by
   challenge development PESQ, alternating with exploration. Promotion
   ordering also uses a heuristic gain/estimated-hours bonus. This is a
   successive-halving-inspired schedule, NOT strict synchronous Hyperband.
   A candidate may be promoted as the pilot pool grows. Maximum promoted
   budget defaults to 25 epochs; higher tiers can be skipped if unaffordable.
5. Only native V5.2 no-critic promotions use an exact matching pilot
   checkpoint, half learning rate and fresh optimizer. Other promotions
   restart from the original initialization to preserve their protocol,
   including V5.3 zero-shot baselines. A promoted warm start receives the
   requested additional epochs, not a continuation of optimizer state.
6. Expand deterministic seeds if no affordable unseen candidate remains
   at earlier seeds. No target-PESQ stop. Stop if there is no eligible
   affordable trial, quota/budget guard, unresolved remote status or
   three consecutive verified terminal failures. Completed/terminal keys
   persist across restarts. Source, plan, seed, epochs, cache and parent
   identity enter the key. This is bounded strategy exploration, not
   arbitrary generated Python or Codex self-modification.
7. If --data-root is supplied, after eligible checkpoints accumulate,
   automatically run V5.5 on CPU for a new top-three native expert shortlist.
   Skip unchanged checkpoint combinations; record failures. Its PCM16
   development score is separate from float-waveform GPU history; never
   silently promote it as an equivalent GPU score. No actual public expert
   offline packaging yet.

## GPU safety and accounting

Exclusive active.lock shared with Ralph V2/V4, authenticated quota checks
before every submission, >=2-hour quota reserve, source/experiment identity
checked by normal submit and reconcile. Existing remote_state.json remains
unresolved if a status/download/timeout fails. Never resubmit in that case.
Terminal failures are recorded, not retried under the same experiment key.
GPU is private, single T4 job at a time; no paid GPU provisioning.

Cost estimates use median observed epoch times for the architecture when
available, otherwise explicit rough priors, plus overhead. These are
planning estimates, not billed GPU time or guaranteed runtimes. Each session
records estimated hours, observed wall time, actual quota change (can be
affected by other usage/reset) and development gain per wall hour. The
budget is a local wall-time admission budget, not a hard remote GPU kill
limit; a submitted job can overrun the estimate. The max-job-minutes option
limits local polling, not remote execution. If timeout occurs, check Kaggle
and reconcile before continuing. This controller does not automatically
cancel an uncertain remote job or override a quota guard.

FP32 remains the default and only V5.6 training mode. The earlier V4
three-seed mixed precision performance gate still exists, restricted to
its exact approved temporal profile. No unvalidated AMP or parallel jobs
are automatically enabled. Actual acceleration and memory use need GPU
experiments. No PESQ improvement is claimed from tests.

## CPU validation and preview

```powershell
git pull --ff-only
py -m compileall -q src tools tests
py -m unittest discover -s tests -p "test_v56_scheduler.py" -v
py -m unittest discover -s tests -p "test_ralph_v4.py" -v
py -u tools/ralph_v56.py
```

The last command is a dry run. It will not submit a GPU job.
All V5.6 tests use mock submission/quota/metrics, not live Kaggle.

## Future execution (after V5.7 and approval)
Only use public assets after instructor permission and license review.
The flag acknowledges that permission; it does not obtain it.

```powershell
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
py -u tools/ralph_v56.py --execute --allow-public-assets --data-root "$DataRoot" --epochs 25 --max-rounds 0 --reserve-hours 2
```

Zero max-rounds means no fixed submission count, not unlimited quota.
The default budget is initial verified quota less reserve. Optional
--budget-hours 4 limits session admission to four wall hours.
--max-rounds 1 allows one GPU submission (including public/smoke jobs).
Do not start this command yet if developing or testing V5.7.
If interrupted, run py tools/reconcile_ralph_v2.py --resolve, confirm no
unresolved remote jobs, then resume. If the process was forcibly killed,
inspect active.lock and confirm no controller process is running before
removing stale lock manually. Do not delete remote-state records.

Main branch and baseline checkpoints unchanged. V5.7 will add final
selection/independent evaluation, public asset packaging and strict
submission verification.
