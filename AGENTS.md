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
- No unsupervised uploads of dataset files, no paid GPU provisioning and no unbounded loops.
- Kaggle kernel and private dataset must stay PRIVATE.
- Codes in src/kaggle_train.py are self-contained because Kaggle kernels push uploads the single code file as the kernel.

Current state: the basic E02/E03 experiment queue exists. Dynamic AI-controlled code changes are NOT enabled yet.
When reviewing results:
1. Read runs/<experiment>/result.json and history.csv.
2. Compare valid complete 164-file PESQ-WB to 1.6961209534 and prior best.
3. Identify specific next hypothesis with evidence and expected risk.
4. Design a bounded experiment with reproducible settings.
5. Never leak speech or identifiers from private data.
