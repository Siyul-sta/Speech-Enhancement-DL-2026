# Ralph V4 — Generalization research

This branch is deliberately unmerged. No Kaggle execution is initiated by a dry run or CI.
V1–V3 checkpoints and modified local src/kaggle_train.py are not overwritten.

## Capabilities
- Public models: MetricGAN+, DeepFilterNet, SepFormer DNS and WHAM (zero-shot and pairwise blends).
- Mixed-corpus training: challenge train, LibriSpeech, MUSAN, optional OpenSLR 73 Spanish, OpenSLR 28 simulated RIR, and Google FLEURS Korean train speech.
- Model families: residual-phase, temporal CNN, public MetricGAN transfer, SepFormer DNS/WHAM decoder transfer, adaptive neural/DSP mixture, declarative composed networks.
- Optional EMA consistency / RemixIT-style training on training noisy audio or explicitly marked transductive noisy adaptation; public teacher distillation.
- Codex chooses validated data profiles, model topology, objective weights, augmentation and training configuration; metadata discovery from Hugging Face and arXiv.
- Public metadata is not executable code. Unregistered datasets/models need license/schema/adapter review. Arbitrary generated Python is opt-in because AST is not a security sandbox.
- Fixed development evaluation: 164 challenge clips, 32 external normal and 32 external stress clips.
- Robust selection = 0.5 * mean of the three PESQ-WB domain means + 0.5 * worst domain mean.
- Separate frozen final audit on 128 held-out synthetic clips; optional DNSMOS, transcript-based WER, URGENT paired and blind mixed real/simulated evaluation, and Korean FLEURS test-speech synthetic evaluation. Both public recipes and trained checkpoints can be audited. No audit results feed the planner.
- Offline inference for saved checkpoint or public recipe and exact archived source; mono output, overlap-add, input-rate restoration.
- Read-only V4 dashboard at localhost:8766.
- No PESQ cutoff, only quota/uncertain job/failure/optional round limits.

## Local checks (no GPU)
    py -m unittest discover -s tests -p test_ralph_v4.py -v
    py -m unittest discover -s tests -p test_v4_cpu_models.py -v
    py -u tools/ralph_v4.py --loop --codex-research --discover
    py tools/dashboard_v4.py

CPU model tests need torch, numpy, pandas, scipy and soundfile.
GitHub Actions uses no Kaggle credentials or private audio.

## Execution (only after user authorizes GPU)
    py -u tools/ralph_v4.py --loop --codex-research --discover --build-public-cache --epochs 15 --max-rounds 1 --execute

Omit --max-rounds or set 0 for continuous research. The first zero-shot benchmark is separate from the training-round cap. Quota reserve is 2 hours. Unknown remote state blocks resubmission. Do not delete locks or remote state while a job is running.
For final frozen audit:
    py -u tools/ralph_v4.py --final-audit --audit-aux --audit-urgent --audit-korean --execute

Final audit is deliberately separate from the loop; do not use its results to tune.
If a Kaggle job is interrupted, use tools/reconcile_ralph_v2.py, which understands V4 state.

## Remaining runtime risks
Public model dependencies, remote model downloads, real external archives (large),
private Kaggle mount and checkpoint transfer, actual GPU training and optional
DNSMOS/WER dependencies require integration validation. Optional --build-public-cache prepares a separate CPU-only private kernel whose inputs are empty and whose outputs contain only public source data. Later jobs mount that public cache as a kernel source, without downloading the archives repeatedly. Actual Kaggle output-size/mount compatibility remains to be tested. Korean speech is streamed per job. Without the cache, public archives are downloaded per job.
Dataset discovery proposes new resources, but does not automatically execute unknown
repositories or admit unreviewed licenses. The optional extended audit includes Korean and URGENT; URGENT blind is mixed real/simulated and has no reference. It cannot prove generalization to every language/environment or exclude unknown pretrained-corpus overlap.
No performance improvement is claimed until real PESQ evaluation has completed.
