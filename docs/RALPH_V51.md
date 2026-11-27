# V5.1 — preflight, data diagnosis and reliable smoke
Keep existing checkpoints and local edits. This patch is on the unmerged V5 branch.
No score improvement is claimed by these engineering checks.

## 1. Synchronize and run CPU checks
```powershell
git pull --ff-only
py -m compileall -q src tools tests
py -m unittest discover -s tests -p "test_ralph_v4.py" -v
py -m unittest discover -s tests -p "test_v51_preflight.py" -v
$env:V5_TEST_PUBLIC_CMGAN = "1"
py -m unittest discover -s tests -p "test_v5_improvements.py" -v
```
The V5.1 fixture tests verify exact overlap, SNR, model-to-critic gradient,
finite calibration metrics and that no running job is downloaded. Fake metrics
in unit tests are not performance scores. The public CMGAN test needs network.

## 2. Recover the already completed V4 output
```powershell
py tools/v51_recover.py --ref siyul1122/research-speech-v420261125093818-job-1 --out runs/recovered_v4_job1_v51
```
Read recovery.json, result.json and history.csv if present.
The script checks authenticated COMPLETE state and all available protocol-required
score rows and checkpoint; it does not create a GPU job, modify remote state,
or add a result to the controller leaderboard. Exact archived source is required
for inference and must be independently matched to the job. If already downloaded,
use a new output directory or inspect existing verified artifacts.

## 3. Diagnose the actual local dataset
The dataset root must contain both train/samples.csv and validation/samples.csv.
Use the original local data; do not upload private WAVs or manifests to GitHub.
```powershell
$DataRoot = "REPLACE_WITH_ACTUAL_DATASET_FOLDER"
py tools/v51_data_audit.py --root "$DataRoot" --out runs/v51_data
```
This requires numpy and soundfile, not PESQ. It checks 1296/164 pairs,
16 kHz mono 48000 samples, SNR, residual correlation, clipping, silence,
exact waveform overlap, duplicate IDs, and metadata-derived groups.
Exact hashes do not detect all near-duplicates or prove speaker independence.
Noise classes are unknown if labels are absent. Residual may contain speech.
Speaker identifiers are hashed in the report but the report should remain private.

For full noisy PESQ, Oracle, and optional per-model analysis, the local
Python environment must have pesq==0.0.4 installed:
```powershell
py -m pip install pesq==0.0.4
py tools/v51_data_audit.py --root "$DataRoot" --out runs/v51_oracle --oracle
```
The optional model score argument is `--model-scores <validation_per_file.csv>`.
Use only a CSV from the exact same manifest/order; the script checks count/index,
not semantic correspondence. Oracle uses the first 64 train rows (exploratory,
not random) and all 164 validation rows; use --oracle-train-limit 1296 for full
train. Oracle is target-informed diagnosis, not a deployable method or strict PESQ bound.
If Windows/Python cannot build pesq, complete the no-PESQ audit first and
run the metric audit on a supported Python/CPU environment. Do not skip validation rows.

## 4. Real GPU metric smoke
After CPU checks and reconciling prior jobs, execute only this smoke:
```powershell
py tools/reconcile_ralph_v2.py
py -u tools/ralph_v4.py --smoke --focus-architecture residual_relaxed --pesq-aware --execute
```
The ordinary smoke remains 2 epochs. The metric-aware smoke now runs 3 epochs
so generator metric loss is active, requires positive finite gradient from that
loss, logs actual critic update and generator metric step counts, and reports
MAE/RMSE/Pearson on 64 train-derived engineering fixtures from the final epoch.
Correlation may be undefined when scores have no variance. Calibration is not a
generalization claim and never selects checkpoint or enters the leaderboard.
PESQ labels are from training batches only. Smoke uses FP32, not AMP.
Existing GPU quota and exclusive local lock checks remain in effect.
Only after this passes, inspect actual diagnostic reports before starting V5.2.
