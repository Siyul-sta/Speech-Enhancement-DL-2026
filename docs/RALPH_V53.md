# V5.3 — controlled pretrained transfer

V5.2 CPU and controller checks passed. This branch adds transfer experiments
without claiming a new PESQ record. Main and existing checkpoints are untouched.

## Transfer modes

- CMGAN: decoder, progressive (decoder -> TSCB3/4 -> TSCB1/2 -> encoder),
  decoder + rank-4 LoRA on compatible Linear weights in TSCB4, progressive
  with frozen original-CMGAN teacher, full trainable model.
- MetricGAN+: affine/learnable activation head vs progressively unfreezing
  recurrent backbone. Actual module types determine the disjoint groups.
- SepFormer DNS/WHAM: decoder vs mask-network last parameterized child,
  remaining mask network, encoder. Gradient flow through frozen modules
  remains enabled when required by an unfrozen upstream module.
- Progressive unlock at epochs 1, 4, 7, 10 (fewer groups may finish earlier).
  Every optimizer parameter appears exactly once. Base learning rate 3e-5,
  group multiplier 0.3 per depth and cosine decay to zero. Frozen weights
  do not receive gradients. No early stopping in this controlled suite.
- LoRA matrices are merged into ordinary Linear weights after selecting the
  best checkpoint; inference need not instantiate parametrizations.
- Teacher anchoring is a 0.05 waveform loss on training-only examples where
  the original teacher improves waveform L1 over noisy input. This is a
  heuristic, not proof of perceptual superiority, and only CMGAN uses it.

## Preservation gate

Every trial evaluates its own initial pretrained model before training,
records all 164 challenge and 64 external PESQ values, and saves it as the
epoch-zero fallback. Later checkpoints can replace it only if challenge
PESQ improves by >0.001 and neither fixed external domain drops by more
than 0.15 relative to the initial model. The margin is an engineering
choice and is not an independently validated forgetting threshold.
External domains are development data, not an independent test.
Final selected checkpoint is re-evaluated and checked against both gates.
The initial model-state SHA-256 is recorded; subsequent controlled trials
of the same model must match the earlier verified initial fingerprint.
CMGAN upstream files are pinned; existing SpeechBrain model downloads
are not independently revision-pinned. A state mismatch fails closed.
All model scores must be interpreted on the same evaluation protocol.

## Local checks before GPU

```powershell
git pull --ff-only
py -m compileall -q src tools tests
$env:V5_TEST_PUBLIC_CMGAN = "1"
py -m unittest discover -s tests -p "test_v53_transfer.py" -v
py -m unittest discover -s tests -p "test_v52_phase_critic.py" -v
py -m unittest discover -s tests -p "test_v51_preflight.py" -v
py -m unittest discover -s tests -p "test_ralph_v4.py" -v
py -m unittest discover -s tests -p "test_v5_improvements.py" -v
py tools/reconcile_ralph_v2.py
```

Tests cover real pinned CMGAN if opted in, and small module fixtures for
MetricGAN/SepFormer gradient paths. The latter are not actual SpeechBrain
integration tests. Full clip CUDA memory and full-backbone training are
unverified. In particular SepFormer full unfreezing may exceed GPU memory.
No private data is used in these unit tests.

## Controlled Ralph suite

```powershell
py -u tools/ralph_v4.py --loop --v53-suite --epochs 12
```

This is dry-run only. No submission without --execute. Minimum 12 epochs
ensures progressive groups have a chance to train. Add --max-rounds 1
when explicitly running a single initial trial, and use the same epoch
budget for comparisons. The finite suite has 11 configurations, skips
verified completed stages at the same epoch budget, and does not proceed
to unbounded adaptive search. Existing --v5-suite / --v52-suite unchanged.

All V5.3 trials use seed 42, 1024 empirical train remixes, 512 public
synthetic pairs, base loss, no PESQ critic, no warm start, no transductive
input. This isolates transfer strategy rather than presuming V5.2
unmeasured changes are superior. Model-family comparisons use different
batch sizes as needed; comparisons within a family use the same batch size.
The automatic sequence does not select a new strategy dynamically;
the final cross-version Ralph integration is deferred to V5.6.
Public pretrained models and corpora require instructor permission and
license compliance. Do not run full suite before the final development
and budget review. Read transfer_history.json, zero_shot_validation.csv,
zero_shot_external.csv and result.json for evidence. No hidden test tuning.
