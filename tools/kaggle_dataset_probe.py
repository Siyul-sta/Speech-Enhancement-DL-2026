"""Check whether a PRIVATE Kaggle dataset is mounted, without a GPU or model training.

Usage:
  py tools/kaggle_dataset_probe.py --username siyul1122 --dataset siyul1122/speech-data-private
  py tools/kaggle_dataset_probe.py --username siyul1122 --dataset siyul1122/speech-data-private --execute

Prints dataset directory names and counts only, never speech/audio content.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cli_call(argv: list[str], *, capture: bool = False) -> str:
    command = [str(x) for x in argv]
    print("$", " ".join(command), flush=True)
    output = subprocess.run(command, capture_output=capture, text=True, check=True)
    return (output.stdout + "\n" + output.stderr).strip() if capture else ""


def main():
    parser = argparse.ArgumentParser(description="CPU-only Kaggle dataset mount diagnostic")
    parser.add_argument("--username", required=True)
    parser.add_argument("--dataset", required=True, help="Kaggle user/dataset-slug")
    parser.add_argument("--execute", action="store_true", help="Submit diagnostic CPU kernel")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        parser.error("Invalid username")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", args.dataset):
        parser.error("Invalid dataset slug")

    folder = ROOT / "kaggle_run" / "data_probe"
    folder.mkdir(parents=True, exist_ok=True)
    ref = f"{args.username}/speech-enhancement-data-probe"
    probe = r'''
from pathlib import Path

INPUT = Path("/kaggle/input")
print("DATA_PROBE_START", flush=True)
print("INPUT_EXISTS =", INPUT.is_dir(), flush=True)
if INPUT.is_dir():
    for p in sorted(INPUT.iterdir())[:30]:
        print("TOP_LEVEL_INPUT =", str(p), "is_dir =", p.is_dir(), flush=True)
    matches = sorted(INPUT.rglob("train/samples.csv"))
    print("TRAIN_MANIFEST_COUNT =", len(matches), flush=True)
    matched = False
    for path in matches[:10]:
        root = path.parent.parent
        val = root / "validation/samples.csv"
        train_wav_dir = root / "train/noisy"
        val_wav_dir = root / "validation/noisy"
        print("CANDIDATE_ROOT =", str(root), flush=True)
        print("VAL_MANIFEST_EXISTS =", val.is_file(), flush=True)
        print("TRAIN_NOISY_DIR_EXISTS =", train_wav_dir.is_dir(), flush=True)
        print("VALIDATION_NOISY_DIR_EXISTS =", val_wav_dir.is_dir(), flush=True)
        print("TRAIN_NOISY_WAV_COUNT =", len(list(train_wav_dir.glob("*.wav"))), flush=True)
        print("VALIDATION_NOISY_WAV_COUNT =", len(list(val_wav_dir.glob("*.wav"))), flush=True)
        if val.is_file() and train_wav_dir.is_dir() and val_wav_dir.is_dir():
            matched = True
    print("DATA_PROBE_RESULT =", "MOUNTED" if matched else "NOT_MOUNTED", flush=True)
else:
    print("DATA_PROBE_RESULT = NOT_MOUNTED", flush=True)
print("DATA_PROBE_END", flush=True)
'''
    (folder / "probe.py").write_text(probe.lstrip(), encoding="utf-8")
    metadata = {
        "id": ref,
        "title": "Speech Enhancement Data Probe",
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": [args.dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Prepared CPU-only private diagnostic:", ref, flush=True)
    if not args.execute:
        print("Prepare-only; add --execute to submit. GPU is disabled in metadata.", flush=True)
        return
    kaggle = shutil.which("kaggle")
    if not kaggle:
        parser.error("Kaggle CLI not found; install with py -m pip install -U kaggle")
    cli_call([kaggle, "kernels", "push", "-p", folder])
    deadline = time.monotonic() + 360
    finished = False
    while time.monotonic() < deadline:
        status = cli_call([kaggle, "kernels", "status", ref], capture=True)
        print(status, flush=True)
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", status)
        if match and match.group(1) in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
            finished = True
            break
        time.sleep(12)
    print("\n=== DATA PROBE LOGS ===", flush=True)
    try:
        print(cli_call([kaggle, "kernels", "logs", ref], capture=True)[-20000:], flush=True)
    except subprocess.CalledProcessError as exc:
        print(f"Could not fetch logs: {exc}", flush=True)
    if not finished:
        print("Diagnostic still running. Inspect Kaggle job before trying again.", flush=True)


if __name__ == "__main__":
    main()
