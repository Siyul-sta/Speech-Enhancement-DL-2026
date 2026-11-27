# V5.4 — noisy-conditioned multistage enhancement

V5.3 real CMGAN transfer/LoRA CPU tests passed on the user's machine.
V5.4 adds independent cascade architectures and controlled depth comparisons.
It does not claim improved measured PESQ or launch a GPU job.

## Architecture and training

1. Coarse: native ComplexResidualUNet (identity initialized) or original
   frozen, pinned CMGAN. CMGAN starts from its actual public weights, not
   from an untested V5.3 fine-tuned checkpoint.
2. Residual: conditioned dilated temporal residual network sees the original
   noisy waveform, prior output and their difference. Correction is bounded
   by 0.5 times noisy RMS; not an exact noise estimate.
3. Complex: original and previous STFT features enter a small U-Net;
   additive complex residual allows new spectral components. Noisy phase
   is not assumed to equal clean phase.
4. Refinement: smaller conditioned temporal residual with 0.2 RMS bound.
   Compare with and without V5.2 calibrated perceptual loss. The perceptual
   loss reaches the whole trainable cascade, not just this last block.

All new refinement heads start at zero residual. Stage 2 trains from epoch
1; stage 3 unlocks at epoch 3; stage 4 unlocks at epoch 5. Native coarse
trains from epoch 1, CMGAN coarse remains frozen in eval/no-grad mode for
the entire run. The original noisy input is available at every refinement
stage. Inference accepts only noisy audio. Earlier trainable outputs receive
a small auxiliary waveform L1 loss, and the final output receives the V5.2
phase/complex objective. No target-informed routing or clean-driven
per-clip selection exists at inference. There is no guarantee of monotonic
PESQ improvement at each stage.

## Controlled comparison
All trials use seed 42, batch 1, FP32, 1024 empirical train remixes,
512 external synthetic pairs, and the same reserved 32 train clips (excluded
from supervised training and both sides of remix). The metric variants
calibrate the critic on this train-only reserve; the generator metric weight
can activate at epoch 5 or later only if the V5.2 gate passes. Zero metric
steps is a valid gate outcome. No early stopping or warm-start in this suite.
Depth comparisons change parameter count and stage unlock timing; they do
not isolate depth from compute. One seed is not enough to establish robustness.

| Stage | Architecture | PESQ critic |
| --- | --- | --- |
| native-two | native coarse + residual | off |
| native-three | + complex | off |
| native-four | + refinement | off |
| native-perceptual | same 4 stages | adaptive |
| cmgan-two | frozen CMGAN + residual | off |
| cmgan-three | + complex | off |
| cmgan-four | + refinement | off |
| cmgan-perceptual | same 4 stages | adaptive |

Every trial records the epoch-zero output on 164 challenge and 64 external
development pairs. It retains that checkpoint unless a candidate improves
challenge PESQ by >0.001 and both external domain PESQs remain within
0.15 of their own epoch-zero baselines. These are heuristic thresholds,
not guarantees against forgetting. CMGAN coarse state SHA-256 is checked
before and after training and matched between verified CMGAN cascade trials.
No hidden test evaluation occurs.

Final selected checkpoint outputs are separately scored at each stage;
cascade_scores in result.json, cascade_history.json, and per-stage
cascade_*_validation.csv / cascade_*_external.csv are required artifacts.
The controller rejects incomplete stage rows or mismatched final scores.
All scores are development metrics. The final checkpoint may be epoch zero
and intermediate stages may underperform each other.

## Local verification (no GPU)
```powershell
git pull --ff-only
py -m compileall -q src tools tests
$env:V5_TEST_PUBLIC_CMGAN = "1"
py -m unittest discover -s tests -p "test_v54_cascade.py" -v
py -m unittest discover -s tests -p "test_v53_transfer.py" -v
py -m unittest discover -s tests -p "test_v52_phase_critic.py" -v
py -m unittest discover -s tests -p "test_ralph_v4.py" -v
```
The new tests verify identity initialization, gradients at scheduled
unlocks, original-input conditioning, stage score order, strict native
checkpoint loading, artifact checks, controller round limits and real
CMGAN freezing. Mock PESQ tests are not performance evidence.
The real CMGAN integration uses short CPU clips; full-length CUDA memory
and actual training speed/PESQ remain unverified.

## Ralph suite, dry run
```powershell
py -u tools/ralph_v4.py --loop --v54-suite --epochs 8
```
No GPU without --execute. Minimum 8 epochs ensures refinement unlocks.
Add --max-rounds 1 to limit a future execution to one cascade training
trial. The finite suite skips completed stages at the same requested
epoch budget and stops after eight configurations or safety/quota guards.
No automatic public benchmark or endless adaptive search in this mode.
Final integration into the ongoing Ralph controller is deferred to V5.6.
Use external data and pretrained assets only with instructor permission.

Native cascade families are supported by the strict offline V5 submission
loader. CMGAN cascade families work via infer_v4 with pinned upstream
asset loading, but are NOT admitted by submit_v5 offline bundle loader
until public assets are packaged/verified in V5.7. Existing checkpoints
and main branch remain unchanged.
