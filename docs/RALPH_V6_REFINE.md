# V6 evidence-driven refinement (V6.9)

This is an additive, read-only CPU analysis plus bounded continuation path.
Do NOT stop the currently running V6 controller. The trainer and source
hash are unchanged. Existing V6 preflight remains applicable.

## While V6 is running

Open a separate PowerShell, leave the current window alone.

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
$Py = ".\.venv-v571\Scripts\python.exe"
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
& $Py -m unittest discover -s tests -p "test_v6_refine.py" -v
& $Py -u tools/ralph_v6_refine.py --data-root "$DataRoot" --top 3 --threads 2
```

This command only uses local CPU. No new Kaggle submission, no remote
cancellation, no lock acquisition, no changes to the dataset or historical
results. It takes a snapshot of hash-verified COMPLETE non-smoke V6
results, analyzes all 164 PESQ rows, noise and input-SNR groups, pairwise
failure overlap and per-clip oracle selector diagnostic. It then reconstructs
up to three diverse top experts, caches local enhanced waveforms, fits
the gate on reserved Train speakers and selects a frozen recipe on
separate Train speakers. Actual PCM16 scoring compares the selected
recipe with each single expert on all 164 validation clips. A fresh
offline process reproduces the selected package and validates its hash.
Validation clean is used only to score/select fixed candidates.
The oracle diagnostics are NOT allowed at inference and NOT a guarantee.

The first quick report arrives before the more expensive CPU inference.
The latter may be slow, especially portable SEMamba/ROSE. Cached
predictions are keyed by exact inputs, model/source hash and environment.
All outputs and private cached waveforms are under ignored runs/v6_refine/.
Never upload them or their model packages to a public repository.

## Optional automatic continuation after existing V6 ends

Use this command INSTEAD of the previous analysis-only command to do the
entire sequence with explicit GPU authorization after stable idle.

```powershell
& $Py -u tools/ralph_v6_refine.py --data-root "$DataRoot" --top 3 --threads 2 --allow-public-assets --execute-after-idle --budget-hours 3 --reserve-hours 2 --epochs 1
```

It waits for the existing controller and offline finalizer to release
their lock for at least 90 seconds. No lock deletion, no duplicate GPU
submission. Once it acquires the shared lock, it verifies all prior
remote states, source and preflight identity. If any new checkpoint
has completed, refresh the analysis using cached predictions. It
uses the verified best single checkpoint, not an oracle or ensemble
as the training parent. The diagnostic must be measured on that exact
checkpoint, otherwise no continuation.

The first controlled hypothesis is either:
- full_length: 48000-sample training crops rather than 16000, preserving
  the data recipe and losses; or
- hard_noise: activate the existing train-only hard-noise group weighting
  when reserved Train low-SNR performance is substantially worse and
  target-informed magnitude correction shows greater diagnostic gain.

These are hypothesis-selection heuristics, not causal proof. The
existing V6 phase-aware loss and data engine are unchanged; no claim
of new phase-loss architecture. Training uses a lower continuation LR
and a fresh optimizer, baseline replay and the original external-domain
preservation gate. Smoke first, then at most one bounded continuation
of 1-3 epochs, no unlimited new jobs. If insufficient budget, an unresolved
job, failed smoke, or audit seal exists, no further GPU submission.
Estimates are not hard guarantees of billed GPU time; 2h reserve is
checked before each submission. No automatic wait for quota refresh.
Do not run another controller concurrently.

On a clean completion, rerun CPU analysis/ensemble with new results.
The report is development performance; private test remains untouched.

## Outputs
- quick_report.json: all completed raw-PESQ results and failure groups.
- raw_scores.csv: aligned per-file raw PESQ.
- pcm16_scores.csv: actual PCM16 individual/ensemble comparisons.
- train_select_scores.csv: reserved Train diagnostic results.
- report.json: winner, ensemble score, oracle diagnostic, hypothesis,
  exact parent, rejected candidates and offline replay evidence.
- package/: portable development winner.
- continuation_*.json: actual continuation result and parent lineage.
- post_*/report.json: refreshed comparison after new training.

No actual private data or GPU job was accessed during code development.
No PESQ improvement is claimed until measured on the user's PC/Kaggle.
