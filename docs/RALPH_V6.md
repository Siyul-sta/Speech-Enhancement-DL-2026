# V6 Maximum PESQ — implementation and runbook

## Scope / status

V6 is an independent experimental path on the existing feature branch. No
changes to main, V5.9 training source, baseline checkpoint, private data or the
fixed V4 PESQ evaluator. Code is designed to run locally with the same Python
venv and Kaggle private dataset. Dry run is the default. No GPU job is
submitted during development/CI.

| Stage | Implementation | Evidence / limitations |
|---|---|---|
| V6.1 | MP-SENet DNS/VB, SEMamba, SEMamba++, ROSE-CD CT; existing CMGAN, MetricGAN+, SepFormer DNS/WHAM and three native families (12 total) | All adapter code integrated. Actual advanced checkpoint and CPU/GPU acceptance must pass before admission. |
| V6.2 | LibriSpeech/MUSAN, pinned Korean FLEURS and Zeroth training, optional DNS AudioSet noise shard, public-only CPU cache | No actual large corpus cache built in this development turn. Corpus terms and provenance must be respected. |
| V6.3 | Real challenge residual cross-mixing, Korean balancing, train-speaker-isolated calibration, SNR curriculum, hard noise group weighting | Lazy indexing supports up to 1M samples; actual training limited by available budget. |
| V6.4 | Multires waveform, log-mag and complex/phase losses; actual PESQ replay/rank critic with held-out calibration gate | Critic may stay disabled if calibration fails; no differentiable exact PESQ. |
| V6.5 | Noisy-only lag/RMS alignment, fixed mixtures, clip/frequency/time-frequency complex gates, train fit/select | Legacy experts may have seen reserved speakers; final validation is development, not independent audit. |
| V6.6 | Frozen coarse checkpoint + 1/2/3 identity-initialized waveform/complex refiners, stage-specific full 164 scoring | Actual GPU cascade smoke/full trials pending. |
| V6.7 | Private single-job scheduler, pilot/extension/ablation, quota/time budget, same-ID network retry/recovery, previous V4 dashboard telemetry | Existing checkpoint lineage validated, optimizer moments reset between blocks. No unconditional resume after 6h timeout. |
| V6.8 | Local finalist selection, offline hash-verified package, full PCM16 WAV evaluation, subgroup metrics, ZIP, independent audit seal | Real private dataset execution and audit still required. |

### Asset evidence

MP-SENet: https://github.com/yxlu-0102/MP-SENet, pinned revision
89932cfe90d1dacb8e170e4a331d762462c21792, MIT. Both official checkpoint blobs
are pinned in src/v6_mpsenet.py. Reviewed model class bodies only; no upstream
training scripts executed. Full-length inference follows upstream RMS and STFT.
The upstream attention axis convention is preserved; utterances are passed
individually. Transfer groups cover decoder, late, early and encoder modules.

SEMamba: https://github.com/RoyChao19477/SEMamba, pinned revision
750ff5d8f0e112557ca039c268cdcc51e267f342; official checkpoint
ckpts/SEMamba_advanced.pth (Git blob c5806e24f83514cf99d2711a1a99c0414e2ac390).
MIT with MP-SENet/CMGAN attribution. src/v6_semamba.py uses reviewed pinned
architecture classes and a portable Mamba1 implementation with exact state
layout, causal convolution, input-dependent SSM, associative affine scan,
RMSNorm, and bidirectional recurrence. Native torch CPU/GPU, no mandatory
compiled extensions. Portable scan may be significantly slower than fused
Mamba. CI tests numerical/gradient parity against direct recurrence and
the real public checkpoint on a synthetic 4096-sample fixture.

SEMamba++: https://github.com/Eddieyjlee/SEMambapp (same original repository
ID 1160517120, owner renamed from infected4098). Pinned revision
cd369c89a8106072c077b468417b765b227afd74; upstream corrected checkpoint from
August 2026: https://huggingface.co/yongjoonlee/semambapp .
src/v6_semambapp.py preserves the official multi-resolution bottleneck,
frequency/global/local FAN, peak normalization and direct magnitude
prediction. The HF model card identifies GPL-2.0 terms; the GitHub source
does not expose a separate LICENSE file. Source downloaded locally at a
fixed commit, not checked into this project. Use locally for evaluation;
do not redistribute upstream source/weights or portable bundle without
reviewing permissions and GPL terms.

ROSE-CD: https://github.com/LiangXu123/ROSE-CD, pinned revision
8270abbf48b016a1a0be370d6c40e67ca8b85fab, MIT. Official CT checkpoint path
logs/CT_pesq5e-4_L2/l6wdouw9/last.ckpt on https://huggingface.co/LIANGXU123/ROSE-CD .
src/v6_rose_cd.py reconstructs the pinned NCSNpp_v2 and OUVE model, strictly
loads dnn weights and folds EMA once. Native upfirdn avoids CUDA compilation.
Uses only weights_only=True, with the inert known data-module class marker,
never arbitrary pickle or Lightning training code. One-step stochastic
prior uses a fixed seed for deterministic evaluation; no clean input.
Supports differentiable head/progressive transfer and local offline config.
Unknown checkpoint architecture, missing EMA or unsafe globals fail closed.

For SEMamba++ and ROSE-CD the first local preflight resolves the exact
official HF revision, filename, LFS SHA256, size and model card license.
The controller propagates those immutable locks to remote jobs; remote
or offline assets must match. No claimed downloaded checkpoint hashes
before actual preflight. Per-family preflight results are persisted.
All three added to the scheduler, transfer groups, cascade, local package
and validation pipeline; actual CPU fixture/full-length and remote training
remain separate acceptance gates. Do not call unverified adapters a PESQ win.

Zeroth Korean: https://www.openslr.org/40/ (CC BY 4.0), pinned parquet mirror
kresnik/zeroth_korean revision 1fe937899f828af822293d05e086200946088bdf.
Only six official train parquet shards, deterministic subset.
FLEURS: existing pinned Korean train path in V4.
DNS: https://github.com/microsoft/DNS-Challenge (DNS5 official public training
AudioSet noise shard, original source terms apply). Optional CPU cache only.
Do not download the whole 39 GB DNS noise corpus by default.
KsponSpeech / gated corpora not fetched; respect access terms.

## PowerShell execution

First ensure old V5.9 jobs are resolved; do not run controllers concurrently.
Keep the same feature branch and local edits. All following commands are
for the user to execute; no GPU submission was performed by code development.

```powershell
cd C:\Users\LG\Speech-Enhancement-DL-2026
git pull --ff-only
$Py = ".\.venv-v571\Scripts\python.exe"
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
& $Py -m compileall -q src tools tests
& $Py -m unittest discover -s tests -p "test_v6_controller.py" -v
& $Py -m unittest discover -s tests -p "test_v6_cpu.py" -v
& $Py -u tools/ralph_v6.py --data-root "$DataRoot"
```

Required local CPU acceptance with actual public MP checkpoints and the
first real train clip (no GPU; downloads verified public assets only).
Existing V5.7.1/V5.7.2 PASS reports must match the local environment. The full 12-model preflight can be slow, especially portable SEMamba/SEMamba++. For individual model diagnostics use --families semamba semambapp rose_cd. Successful checks are retained across runs on identical source/data/environment. Failed families are never silently admitted; explicitly supply --families with accepted models if you want to continue without them.

```powershell
& $Py -u tools/v6_preflight.py --data-root "$DataRoot" --allow-public-download
```

Optional separate public-only CPU cache; no challenge data attached. It can
take substantial network time. Do not run alongside another controller.
DNS flag is optional; leave it out unless the additional archive is desired.
The prior cache path is still supported.

```powershell
& $Py -u tools/ralph_v6.py --prepare-cache --allow-public-assets --data-root "$DataRoot" --cache-korean 512
```

For optional DNS AudioSet shard add --cache-dns-noise. A cache failure
does not authorize re-push until terminal state is resolved.

After successful CPU acceptance and explicit decision to use GPU:

```powershell
py tools/reconcile_ralph_v2.py --resolve
kaggle quota
& $Py -u tools/ralph_v6.py --execute --resume --allow-public-assets --data-root "$DataRoot" --max-rounds 1 --enable-critic --enable-cascade
```

First job: new MP DNS zero-shot, not a training claim. Inspect COMPLETE
and actual PESQ; next jobs benchmark MP VB, SEMamba, SEMamba++, ROSE-CD
when admitted and affordable, then smoke and pilot selected architectures.
Then continuous serial research with budget guard:

```powershell
& $Py -u tools/ralph_v6.py --execute --resume --allow-public-assets --data-root "$DataRoot" --mix 2048 --public-mix 512 --max-mix 32768 --korean-limit 256 --max-epochs 200 --max-rounds 0 --reserve-hours 2 --enable-critic --enable-cascade
```

Training uses ~3s pairs but 1s crops by default for memory/throughput; critic
experiments use full 3s. Validation always full 3s x 164. Max epochs is an
attempted lineage cap, not a guarantee or single-job duration. Budget and
measured throughput may limit data size and epochs; estimates are not actual
billed GPU time. No automatic target-PESQ stop. A model's best checkpoint is
kept if its next epochs regress. External preservation gate is 0.15 PESQ.
Repeated validation-based model selection can bias development scores.

Network interruption: V6 retry same remote reference until per-job wait limit
(default 360 minutes). After a timeout / PC restart, stop old controller and
invoke identical command with --resume. V6 resumes unresolved V6 jobs only;
legacy jobs require reconcile_ralph_v2.py --resolve. Do not delete an active
lock blindly, never resubmit an uncertain job. No automatic infinite quota
reset wait; restart after reset. An already submitted Kaggle job continues
independently of local internet; a powered-off PC cannot schedule new jobs.

Dashboard (read-only; existing UI):
```powershell
py -u tools/dashboard_v4.py --port 8766
```
Open http://127.0.0.1:8766. V6 emits V4 batch/epoch/data telemetry; network
wait also appears in controller ui_state. No new independent dashboard
frontend was built.

## Final selection and submission

After GPU work is stopped and resolved:
```powershell
& $Py -u tools/v6_finalize.py select --data-root "$DataRoot" --out ".\v6_final_001" --top 3 --allow-public-download
```

Or add --finalize-out ".\v6_final_001" to the controller to invoke final
selection after a clean stop. Selection compares a train-only gate candidate
with each individual expert using exactly the same PCM16 164-file metric.
It includes eligible V5 champions as well as V6 models. New output directory
required; previous runs are not changed. Only trust your own archived code.
The result file final_report.json contains the real winning PESQ and path.

```powershell
& $Py ".\v6_final_001\champion\v6_offline.py" --probe
& $Py ".\v6_final_001\champion\v6_offline.py" --input "<noisy-input-folder>" --output ".\v6_submission_001" --zip
```

Replace the placeholder with the actual provided test noisy folder.
Do not evaluate or tune on private test clean. Optional independent audit
requires independently held-out paired data, provenance and explicit
--confirm-heldout. A one-shot shared seal prevents subsequent training or
selection. This audit has not been performed.

The 4.5 target is a research goal, not a promised result. Current known
historical best 2.4800 remains until measured evidence surpasses it.
