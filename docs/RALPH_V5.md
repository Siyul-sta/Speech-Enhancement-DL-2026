# Ralph V5 — challenge PESQ research

Experimental branch only; main and V1–V4 checkpoints remain unchanged.
No 4.5 achievement is claimed. No private data is uploaded by tests.

## Before GPU use

Recover completed V4 outputs if not yet downloaded. Check that no remote job is running.
The fixed challenge evaluation is all 164 validation clips; external dev metrics remain separate.
External pretrained assets/corpora require instructor permission and license compliance.

```powershell
git status --short
git fetch origin
git switch feature/ralph-v5-pesq-diagnostics
git pull --ff-only
py -m compileall -q src tools tests
py tools/oracle_mask_audit.py --self-test
py -m unittest discover -s tests -p "test_ralph_v4.py" -v
py -m unittest discover -s tests -p "test_v5_improvements.py" -v
```

If the local branch does not yet exist, replace git switch with
`git switch --track origin/feature/ralph-v5-pesq-diagnostics`. Do not force checkout with local changes.

## Oracle diagnosis (local, CPU, train-only first)

```powershell
py tools/oracle_mask_audit.py --root "PATH_TO_DATASET" --split train --limit 64 --out runs/oracle_train_sample
py tools/oracle_mask_audit.py --root "PATH_TO_DATASET" --split validation --out runs/oracle_validation
```

The path must contain train/samples.csv and validation/samples.csv. The validation audit is descriptive only; clean references never become input or weights. Oracle projections minimize complex coefficient error, not PESQ. They are not strict PESQ ceilings. Keep reports private.

## GPU smoke before full trials

```powershell
py -u tools/ralph_v4.py --smoke --focus-architecture residual_relaxed --pesq-aware --execute
py -u tools/ralph_v4.py --smoke --focus-architecture dual_axis --execute
```

Only the first two epochs, no external assets, no leaderboard promotion.
The critic is updated in smoke, but generator metric weight starts at epoch 3 in full runs.

## Challenge-first zero-shot benchmark and controlled suite

```powershell
py -u tools/ralph_v4.py --challenge-focus --execute
py -u tools/ralph_v4.py --loop --v5-suite --codex-research --discover --epochs 25 --execute
```

The suite runs completed-stage-aware controlled trials in order:
bounded control, relaxed phase/gain, challenge remix, PESQ critic, MetricGAN transfer,
CMGAN decoder transfer, dual-axis GRU, SepFormer decoder transfer.
The first four stages share fixed seeds, external profile and synthetic schedule.
They differ in the stated hypothesis; remix also changes training-set size.
After completed stages the loop resumes strategy search, subject to quota and failure guards.
Optional `--max-rounds 1` limits training trials. No PESQ stopping threshold.
Optional `--build-public-cache` attempts the existing separate public-only cache.
Without it public corpus downloads remain required. A failed cache is not silently reused.

The public benchmark scores individual MetricGAN+, DeepFilterNet, SepFormer DNS/WHAM,
and pinned official CMGAN, plus fixed pairwise and triple waveform mixtures.
Selection is global, not per-file reference-based. Public model failures are reported.
The CMGAN adapter uses upstream MIT source, pinned revision
39946005d9b66fa1e824edf4bb6bc9a06088e443 and verified blob hashes, strict weights-only load.
Its original STFT has n_fft=400, hop=100, Hamming window and magnitude compression 0.3.
Source: https://github.com/ruizhecao96/CMGAN
The optional public integration test is `V5_TEST_PUBLIC_CMGAN=1` on a machine with network.
Downloads, GPU memory, dependencies and actual inference need runtime verification.

## Reports and guardrails

Read each run's result.json and complete validation_per_file.csv.
v5_best_challenge.json records the best verified V4/V5 challenge candidate seen by the controller;
v4_best.json retains robust selection. Earlier V1–V3 results must be compared separately.
Repeated development validation selection can overfit; do not tune on hidden test.
Noisy-only inference. No generated executable code unless explicitly opted in.
PESQ critic labels come only from training batches, not validation. The critic is a learned
surrogate, not a differentiable implementation of PESQ. Its benefit needs ablation.
ChallengeRemix assumes approximately additive, aligned pairs; residuals are filtered by
correlation, but residual speech contamination remains possible.
