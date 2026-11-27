# V5.7.1 — mandatory real public model CPU preflight

Do not start GPU training until the current CPU preflight report says PASS.
V5.7 initial CI passed; this new version requires a separate real public
asset integration check on the local machine.

## Dependencies

Keep the existing CPU torch installation. Install the matching CPU
torchaudio version (do not blindly upgrade torch), and pinned SpeechBrain
and HyperPyYAML. The following PowerShell commands use the current Python:

```powershell
git pull --ff-only
py -m compileall -q src tools tests
$TorchVersion = (py -c "import torch; print(torch.__version__.split('+')[0])").Trim()
py -m pip install --index-url https://download.pytorch.org/whl/cpu "torchaudio==$TorchVersion"
py -m pip install "speechbrain==1.0.3" "hyperpyyaml==1.2.2" "huggingface_hub<1" einops
py -m unittest discover -s tests -p "test_v571_preflight.py" -v
$env:V5_TEST_PUBLIC_CMGAN = "1"
py -m unittest discover -s tests -p "test_v57_final.py" -v
py -m unittest discover -s tests -p "test_v56_scheduler.py" -v
py -m unittest discover -s tests -p "test_v55_ensemble.py" -v
py -m unittest discover -s tests -p "test_ralph_v4.py" -v
```

If no matching torchaudio wheel exists, stop and inspect the installed
Python and torch versions. Do not start GPU work or silently substitute
incompatible versions. Use a separate compatible Python environment if
needed; the preflight and later local finalization must use that same
environment. Existing code handles Windows local copy strategy and the
removed TorchAudio backend diagnostic in SpeechBrain 1.0.3; this does
not claim support for arbitrary SpeechBrain audio I/O.

## Real CPU test

```powershell
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
py -u tools/v571_preflight.py --data-root "$DataRoot"
```

The first local run downloads only pinned public model assets. It tests
seven supported public-based architectures: CMGAN, three CMGAN cascade
depths, MetricGAN+, SepFormer DNS and SepFormer WHAM. Each loads actual
pretrained weights, evaluates one full 48000-sample fixture, modifies a
parameter to emulate a changed checkpoint (not actual fine-tuning),
exports the checkpoint, then reconstructs it in a separate Python
process with the network/subprocess guard. Raw outputs must match within
atol=2e-5 and rtol=2e-4; actual PCM16 PESQ difference must be <=0.01.
The mixed public/native smoke compares fixed, clip gate and frequency
gate using CMGAN and a Native expert. No claim of performance improvement.
The optional local train fixture uses only the first training pair;
temporary arrays are removed when the test exits normally. No private
audio is uploaded. Omitting --data-root uses a synthetic fixture.
RAM and disk use may be substantial, especially SepFormer.
Actual GPU training gradients and fine-tuned output must still be
verified on completed real runs.

A report at runs/v57_public_preflight/report.json records all outcomes,
source hashes and package versions. The integrated scheduler refuses
GPU submission if missing, failed, incomplete or stale. The successful
report also enables automatic public/native ensemble candidates; the
CMGAN source cache is reused for final packaging. There is no override
flag for this gate in ralph_v56.py. Dry-run still works without a report.

## Zero-shot benchmark packaging

V5.6 now uses a packageable four-model public benchmark, with exact
pinned adapters and full state snapshots of successful experts.
It keeps the same evaluation grid for the remaining models and stores
the fixed selected recipe, including noisy-identity blends.
The finalizer can include newly archived public recipe candidates,
using each expert's clipped output before mixing, matching the benchmark.
Legacy benchmark-only records without state snapshots remain ineligible.
DeepFilterNet is intentionally excluded from the integrated four-model
benchmark until its separate compiled runtime and assets can be verified
offline. The legacy V4 benchmark still supports it; this is a scope
decision, not evidence that it has inferior PESQ.

SpeechBrain pretrained revisions are pinned in the archived source.
New training archives actual YAML and dependency versions.
Public experts in mixed ensembles require a successful preflight,
verified completed checkpoint and local public assets. The finalizer
copies and checks all referenced assets. Actual candidate PCM16
evaluation and a final independent corpus are still required.
Older pre-V5.7 checkpoint records lacking exact assets fail closed.

## Execution after PASS

After instructor permission, all CPU tests and real preflight PASS,
run the existing one-job V5.6 live integration check, not unrestricted
training immediately. No GPU work was submitted by this development.
Independent evaluation data, course submission layout and actual GPU
results remain unresolved. Main branch is unchanged.
