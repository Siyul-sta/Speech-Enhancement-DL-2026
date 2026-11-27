# Ralph V5 — 4.5 PESQ-WB research target

## Competition constraints (2026-11-27)

- Hidden evaluation: 483 noisy-only clips, including unseen original noises. Public development: 164 clips; training: 1296 clips.
- External training and pretrained models allowed; record dataset split, source, revision, license, and usage.
- No validation-clean training, per-file oracle selection, or hidden-test tuning. Oracle analysis is diagnostic only.
- Deadline December 13; report December 14. Keep main and previous checkpoints unchanged.
- 4.5 is an aspirational research target, not a predicted achievable score. Preserve GPU quota guard and report failures.

## Execution

In PowerShell from the project directory, after confirming no remote job and protecting local edits:

    git fetch origin
    git switch feature/ralph-v5-pesq-diagnostics
    git pull --ff-only
    py -m compileall -q tools src tests
    py -m unittest discover -s tests -p "test_v5_quality.py" -v
    py -m unittest discover -s tests -p "test_ralph_v4.py" -v

If branch does not yet exist locally use git switch --track origin/feature/ralph-v5-pesq-diagnostics instead.
Install CPU testing dependencies in the active Python environment (torch, numpy, pandas, scipy, soundfile); PESQ needed for actual evaluation.

Recover prior output first, without starting GPU:

    kaggle kernels output siyul1122/research-speech-v420261125093818-job-1 -p runs/recovered_v4_job1

Run diagnostic against local dataset (replace PATH with actual dataset root):

    py tools/oracle_mask_audit.py --self-test
    py tools/oracle_mask_audit.py --root PATH --split train --out runs/oracle_train

Dry-run and then GPU execution (one round initially, not unlimited):

    py -u tools/ralph_v4.py --loop --v5-suite --max-rounds 1
    py -u tools/ralph_v4.py --loop --v5-suite --epochs 15 --max-rounds 1 --execute

The public benchmark is separate and may need external corpus downloads/cache. Only use --build-public-cache after verifying quota, disk/output limits, and previous remote terminal states. If data access or GPU fails, do not assume scores. Continuous suite: omit --max-rounds after verifying the pilot.

## Controlled stages

Bounded control -> relaxed mask -> empirical train remix (1024) -> train-only PESQ critic -> MetricGAN+ transfer -> CMGAN transfer -> dual-axis -> SepFormer DNS transfer -> 4096 remix trials. Public benchmark also tests DeepFilterNet, SepFormer WHAM and fixed global waveform blends. Compare identical data, seed and epoch budgets for ablations; different data counts imply different compute budgets. No validation noisy adaptation in this suite. Keep both challenge and external robustness scores, and use the independent audit for final generalization evidence, never training.

External code/data provenance: see README_V4.md, src/kaggle_v4_external.py and generated external_manifest.json. CMGAN upstream https://github.com/ruizhecao96/CMGAN revision 39946005d9b66fa1e824edf4bb6bc9a06088e443; record actual artifacts and verify rights before redistribution. MetricGAN+ https://huggingface.co/speechbrain/metricgan-plus-voicebank; record resolved revision from actual run. External train data include LibriSpeech train and MUSAN train-noise groups; public dev/test use separate groups; optional Korean FLEURS train, OpenSLR Spanish and simulated RIR profiles only if selected. See actual per-run manifests for exact usage.

## Submission preparation (native models)

From completed, locally recovered run folders containing result.json, source.py and best_model.pt:

    py tools/build_v5_bundle.py --run PATH_TO_RUN --out submission_weights
    py tools/submit_v5.py --config submission_weights/bundle.json --input input --output output
    py tools/submit_v5.py --config submission_weights/bundle.json --validation-root DATASET_ROOT --output runs/submission_eval

For an ensemble, pass multiple --run folders and positive fixed --weights summing to one (or omit for uniform weights). Evaluate the entire exported PCM16 bundle; individual checkpoints' validation scores do not establish the ensemble score.
Native residual, relaxed, dual-axis, temporal and composed models admitted. The script rejects public wrappers until offline assets have been explicitly packaged. A real pretrained winning model is not submission-ready simply because it has a checkpoint or recipe. Do not publish original private recordings or Kaggle credentials. Pin dependencies of the eventual winning architecture, add weights via Git LFS if required, test a fresh checkout with network disabled and record actual PESQ/SNR, commit/tag, and AI use. The builder does not push weights or submit to LearnUs.

## Status

Code-level research changes and tests exist. GPU training, real public model adapters, oracle metrics, actual scores, and clean-environment offline replay require real execution. No new performance claim.
