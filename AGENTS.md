# Instructions for Codex: speech enhancement

Goal: improve arithmetic mean PESQ-WB on **all 164 validation clips** beyond the CNN 1.6961209534 baseline while respecting held-out private test conditions.

Rules:
- Training uses train/noisy and train/clean paired speech only.
- Validation clean is allowed **for evaluation/model selection**, not as training data.
- Inference uses ONLY noisy audio, stored weights and fixed settings; no clean or noise reference.
- Do NOT access or publish private audio datasets, Kaggle tokens, secrets, or personal information.
- Do NOT change the existing CNN checkpoint or baseline scripts while experimenting.
- Do NOT promote an experimental checkpoint before complete 164-file validation and compatible inference checks.
- Keep every experiment named, reproducible, fixed seed, tracked history, and comparable metric protocol.
- Do not claim test PESQ from validation PESQ; do not tune on private test.
- Prefer changing one major hypothesis at a time.
- For free GPUs, use smoke runs before full training, and honor hard experiment/time budgets.
- Model changes must pass python self-tests, shape/dtype, finite gradients, and full 16 kHz mono sample-count checks.
- No unsupervised uploads of dataset files and no paid GPU provisioning. Adaptive controller may run without a wall-clock or round cap ONLY with Kaggle quota checks before every submission, exclusive local lock, timeout/error guards, and no concurrent jobs.
- Kaggle kernel and private dataset must stay PRIVATE.
- Codes in src/kaggle_train.py are self-contained because Kaggle kernels push uploads the single code file as the kernel.

Current state: V1 Ralph has reached verified 164-file PESQ-WB 2.4043. V2 complex mask trainer completed 25-epoch trial at 2.3125. Three architecture templates now exist: src/kaggle_train.py (magnitude), src/kaggle_train_v2.py (complex mask), and src/kaggle_train_v3.py (separate gain and phase correction). Adaptive controller tools/ralph_v2.py has been authored, but the new controller has NOT yet passed user-side offline tests or an end-to-end Windows/Kaggle trial.

**For adaptive Ralph V2 only:** Codex must edit exactly one *scratch* candidate.py under runs/ralph_v2/run_*/round_*/, NOT any tracked src/ trainer file. Within that candidate only existing ConvBlock, the selected family model class (SpectrogramUNet, ComplexMaskUNet, or ResidualPhaseUNet) and enhancement_loss may change. No changes to top-level imports, any other functions, dataset access, training optimizer loop, 164-file validation PESQ, early stopping, checkpoints or metrics. Codex may change architecture inside those defined classes, and it must not run GPU jobs. The controller validates candidate AST before submission and switches model families when progress stagnates. The prior V1/V2 scripts retain their own restrictions. Never modify evaluation logic to inflate scores.
When reviewing results:
1. Read runs/<experiment>/result.json and history.csv.
2. Compare valid complete 164-file PESQ-WB to 1.6961209534 and prior best.
3. Identify specific next hypothesis with evidence and expected risk.
4. Design a bounded experiment with reproducible settings.
5. Never leak speech or identifiers from private data.


## Ralph V4 scope (only tools/ralph_v4.py and src/kaggle_v4_*.py)

V4 permits public pretrained weights, licensed external training speech/noise,
train-only augmentation, transfer, distillation and noisy-only consistency.
The earlier train-only challenge restriction applies to V1–V3, not V4.
Validation/test clean is NEVER used as a training target. Optional transductive
noisy adaptation is separately recorded and excluded from inductive final audit.
Do not expose private recordings, credentials, or raw local files to public APIs.
Unknown external assets are research proposals until license, schema and adapter
are reviewed. Prefer validated declarative model/loss descriptions; experimental
generated Python is opt-in, not sandboxed by AST checks. Fixed evaluator, data
splits and metric protocol may not be modified by an experiment proposal.
Stop for quota/uncertain GPU state or failure budget, never a target PESQ.
