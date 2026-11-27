# V5.2 — phase reconstruction and calibrated PESQ training

V5.1 observations from the supplied local audit: noisy validation 1.3947,
bounded-phase target-informed projection 4.0758, gain-2/full-phase 4.6089,
clean identity 4.6439 (164 pairs). These are diagnostics, not learned-model
results or strict upper bounds. V5.1 GPU smoke completed with critic MAE
1.1111, Pearson .7695 and a nonzero generator metric gradient. It was not
ranked. The user-reported previous learned best remains 2.4800.

## New options and controls

* ComplexResidualUNet: additive real/imaginary spectrum residual instead of
  only multiplying noisy bins. Zero-initialized new head yields identity.
  Native offline checkpoint inference supports the new architecture.
* Phase loss: base objective plus 0.15 times the mean of three-resolution
  power-compressed complex distance and energy-weighted circular phase loss.
  Avoids a discontinuity at +/- pi; epsilon stabilizes silence.
* Empirical remix: sample actual measured SNR values from eligible training
  pairs, not a fitted uniform distribution. Optional square-root-frequency
  noise-category sampling mildly increases rare-class representation.
  Neither samples nor labels from validation enter training.
* Adaptive critic: train with genuine PESQ labels for current prediction,
  noisy input, halfway waveform and clean target. Keep at most 64 groups in
  CPU memory, never write audio to disk. Use two optimizer updates per label
  group, including replay, with within-reference ranking plus regression.
* Reserve the same 32 training clips for each V5.2 stage, exclude them from
  original supervised batches and both sides of remix. These are disjoint
  clips, not necessarily unseen speakers. External corpus overlap is not
  independently ruled out. Score predictions on that holdout every epoch.
  If prior-epoch MAE > 0.5, Pearson < 0.6, correlation undefined, fewer than
  16 critic optimizer updates, or fewer than two completed epochs, the
  generator metric weight is zero. Otherwise ramp weight toward 0.05.
  These are engineering thresholds, not established universal quality gates.
  Check critic_history.json and metric_generator_steps; zero steps is a
  legitimate gating outcome, not proof of metric-based improvement.

## Validation before GPU
```powershell
git pull --ff-only
py -m compileall -q src tools tests
py -m unittest discover -s tests -p "test_v52_phase_critic.py" -v
py -m unittest discover -s tests -p "test_v51_preflight.py" -v
py -m unittest discover -s tests -p "test_ralph_v4.py" -v
py -m unittest discover -s tests -p "test_v5_improvements.py" -v
py tools/reconcile_ralph_v2.py --resolve
```
Do not submit another job until any previous remote state is terminal.
Fixtures exercise gradients and isolation with mocked PESQ; they do not
establish actual performance, CUDA memory or calibration.

## Controlled Ralph suite (do not start before review)
```powershell
py -u tools/ralph_v4.py --loop --v52-suite --epochs 25
```
Dry-run only. When ready, with instructor permission for public training
corpora, append --execute. Add --max-rounds 1 for one trial. By design this
mode skips the public model benchmark and stops after the seven controlled
stages (or quota / safety guards); it does not silently enter endless
adaptive search. Existing --v5-suite behavior is unchanged.
Keep the same epoch budget across a comparison; early stopping may shorten
individual runs. Completed stages at the same requested epoch budget are
skipped. Failures are logged; rerun to retry failed stages.

| Stage | Change from preceding stage |
|---|---|
| phase-control | relaxed mask baseline, standard remix |
| phase-loss | additional phase/complex loss |
| complex-head | additive spectrum head |
| empirical-remix | empirical SNR sampling |
| balanced-remix | sqrt-frequency noise category sampling |
| fixed-critic | legacy fixed metric objective |
| adaptive-critic | replay, ranking, frequent updates, holdout gate |

All use the same seed, reserved train split, 1024 remixes, 512 external
synthetic pairs, base external profile, FP32, no warm start or transduction.
This is a sequential ablation, not a factorial experiment; some changes
are bundles and one seed cannot establish generalization. The original
164 challenge clips and fixed 64 external development pairs remain evaluation
only; no hidden-test labels or Oracle masks enter inference.
Run V5.1 subgroup audit with --model-scores for a matching manifest.
Do not rank smoke jobs or interpret mocked tests as PESQ gains.
