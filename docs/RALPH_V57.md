# V5.7 — final candidate selection, portable offline inference and sealed audit

This release completes the initial V5 code roadmap, not the GPU research
campaign. V5.6 controller tests passed before this change. No GPU has been
submitted by V5.7 development and no new PESQ result is claimed.

## Verification before training

```powershell
git pull --ff-only
py -m compileall -q src tools tests
$env:V5_TEST_PUBLIC_CMGAN = "1"
py -m unittest discover -s tests -p "test_v57_final.py" -v
py -m unittest discover -s tests -p "test_v56_scheduler.py" -v
py -m unittest discover -s tests -p "test_v55_ensemble.py" -v
py -m unittest discover -s tests -p "test_ralph_v4.py" -v
py tools/reconcile_ralph_v2.py --resolve
py -u tools/ralph_v56.py
```

Dry run must not submit GPU work. V5.7 tests use synthetic audio and
optional actual pinned CMGAN reconstruction. They do not establish
SpeechBrain end-to-end compatibility or measured speech enhancement PESQ.
The independent evaluation data is not supplied by this repository.

## Research after tests pass

Use V5.6 only with instructor approval for pretrained models and licensed
external data, and after resolving any previous remote job. Start with
one submission to verify the live controller. A public zero-shot job
may be the first job; it is not a trained model.

```powershell
$DataRoot = "C:\Users\LG\OneDrive\Desktop\deep_learning_noise_reduction_challenge"
py -u tools/ralph_v56.py --execute --allow-public-assets --data-root "$DataRoot" --epochs 25 --max-rounds 1 --reserve-hours 2
```

After verifying that actual result, replace 1 with 0 for no fixed number
of submissions. Quota, local wall budget, failure, and unknown-state
guards still apply. FP32 is default. Neither the target score 4.5 nor
any other numerical target guarantees attainability. Do not run multiple
controllers concurrently. Full results go to runs/ralph_v2; optional
native ensembles go to runs/ralph_v2/v56_ensembles.

## Final selection after GPU experiments are stopped

```powershell
py -u tools/v57_finalize.py select --data-root "$DataRoot" --out runs/v57_final --top 5
```

If CMGAN assets were already downloaded by the real CMGAN unit test,
specify the actual local directory containing the pinned LICENSE,
src/models/conformer.py, and src/models/generator.py. Otherwise
--allow-public-download permits retrieval of those pinned public code
files, not private audio or data. Review license and course permission
before redistribution. For example if the directory exists:

```powershell
py -u tools/v57_finalize.py select --data-root "$DataRoot" --out runs/v57_final_cmgan --top 5 --cmgan-root local_test_output/cmgan_assets
```

The command considers up to N verified V4/V5 trained checkpoints and
V5.5 ensembles, ordered by their prior development scores, deduplicates
checkpoint bytes, and re-evaluates each admitted candidate on the same
164 challenge clips. Each candidate is packaged, loaded in a separate
offline-guarded process, probed with a full-length synthetic input,
then scored after writing and reading actual PCM16 WAVs in a temporary
directory. No private evaluation WAVs are persisted by this evaluation.
The selection criterion is highest PCM16 PESQ among the successfully
evaluated shortlist. Rejected candidates are recorded with errors; the
best known historical V1–V3 score (2.4800) is not automatically admitted
because its exact source/checkpoint and common evaluation protocol have
not been independently verified. Public benchmark recipe-only results
are excluded until a checkpoint-compatible offline recipe exists.
The selected model is not guaranteed to exceed these excluded candidates.

Outputs: candidate_report.json, final_report.json, per-candidate
evaluation/scores.csv and summary.json, and champion/ containing
trusted archived source, checkpoint, gate if applicable, dependencies
metadata, runner and SHA-256 manifest. Reports include noisy baseline,
paired improvement, 95% bootstrap interval (speaker-cluster when all
speaker groups are known, otherwise clip-level), paired gains versus
other successful candidates, noise/SNR subgroups, 5th percentile,
degraded counts and worst indices. These intervals are descriptive;
they do not correct for validation reuse, selection bias or multiplicity.

## Public checkpoint packaging

Native and native-gated ensembles are supported. Trained CMGAN and
CMGAN-cascade checkpoints can use the exact pinned upstream source and
license; full selected state dict supplies weights. The loader bypasses
the original network-fetching constructor and strictly loads final
weights. MetricGAN+ and SepFormer checkpoints need the exact local YAML
captured from training and matching SpeechBrain/HyperPyYAML versions.
New training scripts archive those files when available. Old runs
without the archived configuration fail closed. Public adapter
reconstruction must pass the offline probe and full development
evaluation before admission. SpeechBrain compatibility remains
unverified against real models until these checks actually run.
Public model recipe-only winners and public-model gated ensembles are
not yet admitted. No arbitrary generated model source is accepted.
The package ships source/assets and environment version records, not
the Python interpreter or dependency wheels. Install dependencies
before disconnecting. The Python audit hook blocks network and child
process operations, but is not an OS-level sandbox for arbitrary
untrusted Python or YAML.

## Submission generation (only once actual input path is supplied)

```powershell
py runs/v57_final/champion/v57_offline.py --input "PATH_TO_NOISY_WAV_DIRECTORY" --output runs/v57_submission --zip
```

This reads noisy WAVs only and writes same-name mono 16 kHz, 48,000-sample
PCM16 WAVs. It checks output filenames, format, length, finite values and
SHA-256; optional flat WAV-only ZIP is reopened and contents verified.
The course-specific ZIP layout and test file count must be confirmed
against the instructor's actual submission instructions. It does not
access hidden test labels or send files anywhere.

## Independent evaluation, only after final model is frozen

A separate user-provided paired corpus and manifest is required. Each
manifest uses segment_id, noisy_path, clean_path and optional speaker/noise
metadata. The user must verify that the dataset was not used in any
training, public pretraining (where ascertainable) or tuning. Exact
waveform/segment/known speaker overlap with challenge Train/Validation
is rejected. Near duplicates and external-corpus overlap cannot be
ruled out solely by these checks.

```powershell
py tools/v57_finalize.py audit --final runs/v57_final --data-root "$DataRoot" --audit-root "PATH_TO_SEPARATE_HELDOUT_CORPUS" --manifest "PATH_TO_AUDIT_MANIFEST" --confirm-heldout --provenance "Describe independently held-out source"
```

The frozen bundle hash is checked, and a global seal is written before
the one-shot audit. Interruptions leave the seal for review. V5.6
execution and final reselection stop once a seal exists. Do not use
audit results to choose new models. This local path does not consume GPU
quota; large models may be slow or require substantial CPU memory.
The original V4 independent public audit remains a separate mechanism,
not automatically invoked by this finalizer. No independent PESQ
has been measured yet. Main branch and historical checkpoints unchanged.
