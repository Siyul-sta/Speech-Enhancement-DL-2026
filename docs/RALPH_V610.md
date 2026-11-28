# V6.10 MP-SENet DNS focus
The first five V6 zero-shot jobs showed 3.175741 MP-DNS vs a per-clip
oracle selector 3.179302. This is evidence against investing heavily in
clip-level selection across those particular experts, not an upper bound
on signal-level mixtures.

Additive files only. The existing V6 and V6.9 controllers, trainer source
and accepted preflight remain unchanged. Do not interrupt either job.
The new trainer is generated separately from V6 with exact source
replacement contracts and a distinct experiment identity.

## Focus hypotheses
- phase_derivatives: add 0.08 times multi-resolution, target-energy
  weighted circular instantaneous phase, group delay, and instantaneous
  frequency loss. Original waveform, magnitude and complex losses remain.
- hard_noise_first: initialize hard group probabilities from the disjoint
  Train calibration pool BEFORE the first epoch. Shrink small group
  means toward the pool mean and remove the easy-SNR curriculum for
  challenge specialization. This is one combined noise-sampling hypothesis.
- full_length: 48k samples rather than 16k crop, if not already applied.
- residual: frozen MP-DNS with identity-initialized existing two-stage
  residual corrector. Existing phase/noise recipe flags are inherited.
Only one new hypothesis per submission. Reserved Train oracle phase/magnitude
diagnostics choose the first candidate. No clean input at inference.
No validation targets enter training. PESQ evaluator and external-domain
gates unchanged. Initial parent is the highest verified completed single
V6 checkpoint; if it is not MP-DNS the controller stops for reassessment.

## Use after BOTH currently running V6/V6.9 processes exit
```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
$Py = ".\.venv-v571\Scripts\python.exe"
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
& $Py -m unittest discover -s tests -p "test_v610_focus.py" -v
& $Py -u tools/ralph_v610.py --data-root "$DataRoot"
```
Default command creates a plan, using cached analysis if available;
otherwise runs CPU analysis. No new GPU job.

Only after review of the plan and available quota:
```powershell
& $Py -u tools/ralph_v610.py --data-root "$DataRoot" --execute --allow-public-assets --budget-hours 3 --reserve-hours 2 --epochs 1
```
This takes the shared exclusive lock after stable idle, refuses unresolved
remote states, verifies parent identity and previous base preflight, then
checks full 48k inference and actual Train gradient of the new objective.
It runs smoke (one epoch) and at most one continuation block.
It does not wait for future quota; if estimates exceed available budget
it stops. Billed time may exceed estimates. Existing best weights and
evaluation are preserved with epoch-zero fallback. No optimizer-state
restore. Optional --mix 512 --public-mix 128 can cap augmentation size
when budget is low; this changes the training recipe in addition to the
new objective and is not a controlled single-factor comparison.

After a successful run the result and updated PCM16 comparison are saved
under ignored runs/v6_refine/. No data, models or private outputs are
uploaded to GitHub. No new measured score until real execution.
