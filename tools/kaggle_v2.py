"""V2 controlled architecture/loss experiments: complex ratio mask U-Net.

Run only after stopping the old Ralph Loop and verifying its latest remote
Kaggle job is complete, cancelled, or safely finalized. Uses separate private
Kaggle job IDs and separate output folders; V1 checkpoints are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "src" / "kaggle_train_v2.py"
EXPERIMENTS = ("V2_CRM_PHASE", "V2_CRM_SISDR")
BASELINE_PESQ = 1.6961209534144983


def run_command(args, capture=False):
    print("$", " ".join(str(x) for x in args), flush=True)
    result = subprocess.run(args, text=True, capture_output=capture, check=True)
    if capture:
        return (result.stdout + "\n" + result.stderr).strip()
    return ""


def package_job(username, dataset, variant, epochs, batch_size, patience, tag):
    slug = "speech-enh-v2-" + tag + "-" + variant.lower().replace("_", "-")
    ref = username + "/" + slug
    folder = ROOT / "kaggle_run" / ("v2_" + variant.lower())
    folder.mkdir(parents=True, exist_ok=True)

    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    config = {
        "id": variant, "epochs": epochs, "batch_size": batch_size,
        "patience": patience, "seed": 42,
    }
    source, config_count = re.subn(
        r"^EXPERIMENT_CONFIG = .+$", "EXPERIMENT_CONFIG = " + repr(config),
        source, flags=re.MULTILINE,
    )
    source, data_count = re.subn(
        r'^DATASET_SLUG = .+$', "DATASET_SLUG = " + repr(dataset),
        source, flags=re.MULTILINE,
    )
    if config_count != 1 or data_count != 1:
        raise RuntimeError("Failed to inject Kaggle experiment configuration")
    (folder / "run.py").write_text(source, encoding="utf-8")

    metadata = {
        "id": ref,
        "title": "Speech Enh V2 " + tag + " " + variant.replace("_", " "),
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return folder, ref


def show_kernel_logs(kaggle, ref):
    """Print the latest Kaggle logs automatically if a job fails."""
    try:
        logs = run_command([kaggle, "kernels", "logs", ref], capture=True)
        print("\n=== KAGGLE ERROR LOG (last 24,000 chars) ===", flush=True)
        print(logs[-24000:], flush=True)
        print("=== END KAGGLE ERROR LOG ===\n", flush=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Unable to retrieve logs for {ref}: {exc}", flush=True)


def wait_for_kernel(kaggle, ref, interval, max_wait):
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        status = run_command([kaggle, "kernels", "status", ref], capture=True)
        print(f"[{ref}] {status}", flush=True)
        low = status.lower()
        # Errors are terminal; do not trigger further GPU jobs after an error.
        if re.search(r"\b(error|failed|cancelled|canceled)\b", low):
            show_kernel_logs(kaggle, ref)
            raise RuntimeError(f"Kaggle run failed: {ref}\n{status}")
        if re.search(r"\bcomplete\b", low):
            return
        if not re.search(r"\b(queued|running|pending)\b", low):
            show_kernel_logs(kaggle, ref)
            raise RuntimeError(f"Unrecognized Kaggle status: {status}")
        time.sleep(interval)
    show_kernel_logs(kaggle, ref)
    raise TimeoutError(
        f"Kaggle job still unresolved after max_wait; check manually: {ref}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Controlled phase-aware complex U-Net V2 experiments"
    )
    parser.add_argument("--username", required=True, help="Kaggle username (NOT GitHub)")
    parser.add_argument("--dataset", required=True, help="Approved PRIVATE Kaggle dataset: user/slug")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--poll-seconds", type=int, default=90)
    parser.add_argument("--max-wait-minutes", type=int, default=180)
    parser.add_argument("--max-total-gpu-minutes", type=int, default=180)
    parser.add_argument("--min-gpu-quota-hours", type=float, default=2.0)
    parser.add_argument("--smoke", action="store_true", help="Just 2 epochs / 1 experiment")
    parser.add_argument("--execute", action="store_true", help="Submit Kaggle jobs; default is prepare-only")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.username):
        parser.error("Invalid username")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", args.dataset):
        parser.error("Use dataset in owner/slug form")
    if not (1 <= args.epochs <= 100 and 1 <= args.batch_size <= 16 and 1 <= args.patience <= 30):
        parser.error("Epochs/batch size/patience outside safe range")
    if args.max_total_gpu_minutes <= 0:
        parser.error("GPU budget must be positive")
    kaggle = shutil.which("kaggle")
    if args.execute and not kaggle:
        parser.error("Kaggle CLI missing: python -m pip install -U kaggle")
    total_gpu_seconds = 0.0
    reference = ROOT / "runs" / "e02_unet" / "result.json"
    baseline = BASELINE_PESQ
    if reference.is_file():
        data = json.loads(reference.read_text(encoding="utf-8"))
        if int(data.get("validation_files", 0)) == 164:
            baseline = float(data["best_validation_pesq_wb"])
    best = baseline
    tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    results_dir = ROOT / "runs" / "v2" / tag
    results_dir.mkdir(parents=True, exist_ok=False)
    print(f"V2 run={tag}, V1 reference PESQ-WB={baseline:.4f}")
    variants = args.experiments[:1] if args.smoke else args.experiments
    for variant in variants:
        if total_gpu_seconds >= args.max_total_gpu_minutes * 60:
            print("GPU wall-time budget reached. Stopping.")
            break
        epochs = 2 if args.smoke else args.epochs
        folder, ref = package_job(
            args.username, args.dataset, variant, epochs, args.batch_size, args.patience, tag
        )
        print(f"Prepared private GPU job: {ref}, epochs={epochs}, path={folder}")
        if not args.execute:
            continue
        quota = run_command([kaggle, "quota"], capture=True)
        for line in quota.splitlines():
            fields = line.strip().split()
            if fields and fields[0] == "GPU" and len(fields) >= 3:
                remaining = float(fields[2].rstrip("h"))
                print(f"GPU quota remaining={remaining:.2f}h", flush=True)
                if remaining < args.min_gpu_quota_hours:
                    raise RuntimeError("Not enough free Kaggle GPU quota for safe V2 run")
                break
        else:
            raise RuntimeError("Cannot verify Kaggle GPU quota; refusing submission")
        started = time.monotonic()
        run_command([kaggle, "kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4"])
        wait_for_kernel(kaggle, ref, args.poll_seconds, args.max_wait_minutes * 60)
        output_dir = results_dir / variant.lower()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_command([kaggle, "kernels", "output", ref, "-p", str(output_dir), "-o"])
        result_path = output_dir / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"No result.json downloaded from completed job: {ref}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        score = float(result["best_validation_pesq_wb"])
        if not math.isfinite(score) or int(result["validation_files"]) != 164:
            raise RuntimeError(f"Invalid validation result: {result}")
        checkpoint = output_dir / "best_model.pt"
        per_file = output_dir / "validation_per_file.csv"
        if not checkpoint.is_file() or not per_file.is_file():
            raise RuntimeError("Missing checkpoint or per-file validation results")
        if len(per_file.read_text(encoding="utf-8").splitlines()) != 165:
            raise RuntimeError("Expected exactly 164 validation rows")
        total_gpu_seconds += time.monotonic() - started
        best = max(best, score)
        print(
            f"RESULT {variant}: PESQ-WB={score:.4f}, "
            f"delta_vs_v1_E02={score - baseline:+.4f}, "
            f"best_so_far={best:.4f}"
        )
        with (results_dir / "experiments.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        # Compare all scheduled variants; one successful score is not a stop
        # condition. Validation results inform selection, never training.
    if not args.execute:
        print("Prepare-only completed. Add --execute once Kaggle credentials and dataset are ready.")
    else:
        print("V2 experiment batch finished; inspect runs/v2/ and compare with V1.")


if __name__ == "__main__":
    main()
