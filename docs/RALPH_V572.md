# V5.7.2 — SepFormer output-scale calibration

V5.7.1 actual CPU preflight passed all seven model families and all
three CMGAN/native mixture modes on one local training pair.
The modified-checkpoint DNS/WHAM outputs had peak amplitudes 117.62/9.08
and 79.59%/38.94% of samples above absolute amplitude 1. These are not
unmodified pretrained scores or full-validation statistics.

This release first checks **unmodified** pretrained DNS and WHAM
using 12 train-selection and 12 train-holdout clips (default), disjoint
known speaker groups, fixed seed, distributed over speakers. No
validation clean labels are used. One raw inference per model/clip
is reused for all nine modes. Only metrics, not waveforms, are saved.

Modes: raw (legacy hard-clipped PCM16 baseline), peak attenuation only,
input RMS ratios 0.25/0.50/0.75/1.00 with peak guard, nonnegative
least-squares projection against noisy input with peak guard, soft
limiting, RMS 0.50 with soft limiting. All non-raw modes bound peak
to 0.98 and are differentiable tensor operations. No learned gain is
implemented yet. Selection uses maximum mean PCM16 PESQ among bounded
modes on selection clips. The holdout can veto, never reselect. Both
groups must have no clipping and selected PESQ within 0.05 of baseline
or better. Failed PESQ values are recorded and rejected. This small
train-only experiment is not independent generalization evidence;
RMS/LS can also alter useful speech amplitude and must be evaluated.

The selected family-specific policy is frozen in GPU smoke, pilot,
promotion and new public benchmark configurations, experiment IDs,
remote result verification and archived offline source. Legacy
checkpoints without a policy retain raw behavior. New public recipe
experts carry their own selected policy, restored on packaging.
No evaluator score is changed. The policy operates in the model's
forward, so training and offline inference use the same operation.
It can change loss/gradient conditioning; GPU smoke/full experiments
remain required. The V5.6 scheduler refuses to run without a current
PASS scale report, matching train manifest and current V5.7.1 report
with the same selected policies.

## Local CPU execution (PowerShell)

Use the already verified .venv-v571 environment (Torch/TorchAudio
2.11.0+cpu, pinned SpeechBrain/HyperPyYAML/ruamel.yaml). Do not reinstall
working packages. No GPU submission in these commands.

```powershell
git pull --ff-only
$Py = ".\.venv-v571\Scripts\python.exe"
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
& $Py -m compileall -q src tools tests
& $Py -m unittest discover -s tests -p "test_v572_scale.py" -v
& $Py -m unittest discover -s tests -p "test_v571_preflight.py" -v
& $Py -m unittest discover -s tests -p "test_v56_scheduler.py" -v
& $Py -m unittest discover -s tests -p "test_ralph_v4.py" -v
& $Py -u tools/v572_scale_audit.py --data-root "$DataRoot"
```

Inspect runs/v572_scale_audit/report.json. A FAIL means stop and
review the real metrics; do not switch to a holdout winner. The
report includes per-clip raw peak, clipping and PESQ by mode.
If PASS, re-run all actual offline compatibility checks because the
model source and policies changed:

```powershell
& $Py -u tools/v571_preflight.py --data-root "$DataRoot"
```

The two reports must both PASS before the one-job live Kaggle
integration test. Existing external asset permission, quota, and
independent audit constraints remain. Main branch untouched. No actual
scale audit, new performance score, or GPU submission was performed
by this code development.
