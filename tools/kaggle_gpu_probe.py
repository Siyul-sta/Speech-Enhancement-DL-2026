"""Submit a tiny PRIVATE Kaggle T4 GPU probe, without attaching speech data.

Usage:
  py tools/kaggle_gpu_probe.py --username siyul1122 --execute

This is NOT model training and does not need a Kaggle dataset.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args, *, capture=False):
    cmd = [str(x) for x in args]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=capture, text=True, check=True)
    if capture:
        return (proc.stdout + "\n" + proc.stderr).strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=360)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        parser.error("Invalid Kaggle username")
    folder = ROOT / "kaggle_run" / "gpu_probe"
    folder.mkdir(parents=True, exist_ok=True)
    ref = args.username + "/speech-enhancement-gpu-diagnostic"
    probe = r'''
from __future__ import annotations
import os
import subprocess
import sys

print("GPU_PROBE_START", flush=True)
print("python =", sys.version.split()[0], flush=True)
print("CUDA_VISIBLE_DEVICES =", repr(os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)
print("NVIDIA_VISIBLE_DEVICES =", repr(os.environ.get("NVIDIA_VISIBLE_DEVICES")), flush=True)

for command in (["nvidia-smi", "-L"], ["nvidia-smi"]):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=15)
        print("COMMAND =", " ".join(command), "return_code =", p.returncode, flush=True)
        print("STDOUT =", p.stdout[:3000], flush=True)
        print("STDERR =", p.stderr[:1000], flush=True)
    except Exception as exc:
        print("COMMAND_FAILED =", type(exc).__name__, str(exc), flush=True)

try:
    import torch
    print("torch =", torch.__version__, flush=True)
    print("torch.version.cuda =", torch.version.cuda, flush=True)
    print("torch.cuda.device_count =", torch.cuda.device_count(), flush=True)
    ok = torch.cuda.is_available()
    print("torch.cuda.is_available =", ok, flush=True)
    if ok:
        print("torch.cuda.get_device_name =", torch.cuda.get_device_name(0), flush=True)
        test = torch.ones((8, 8), device="cuda")
        print("CUDA_TENSOR_TEST =", float(test.sum()), flush=True)
    print("GPU_PROBE_RESULT =", "AVAILABLE" if ok else "NOT_AVAILABLE", flush=True)
except Exception as exc:
    print("TORCH_PROBE_ERROR =", type(exc).__name__, str(exc), flush=True)
    print("GPU_PROBE_RESULT = ERROR", flush=True)

print("GPU_PROBE_END", flush=True)
'''
    (folder / "probe.py").write_text(probe.lstrip(), encoding="utf-8")
    metadata = {
        "id": ref,
        "title": "Speech Enhancement GPU Diagnostic",
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Prepared private no-data GPU probe:", folder)
    if not args.execute:
        print("No GPU job submitted. Use --execute to run a minimal GPU diagnostic.")
        return
    kaggle = shutil.which("kaggle")
    if not kaggle:
        parser.error("Kaggle CLI not installed")
    run([kaggle, "kernels", "push", "-p", folder, "--accelerator", "NvidiaTeslaT4"])
    until = time.monotonic() + args.max_wait_seconds
    while time.monotonic() < until:
        status = run([kaggle, "kernels", "status", ref], capture=True)
        print(status, flush=True)
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", status)
        if not match:
            print("Unrecognized status; check notebook page:", ref)
            break
        if match.group(1) in ("COMPLETE", "ERROR", "CANCELLED", "CANCELED"):
            break
        time.sleep(12)
    print("\n=== Kaggle GPU probe logs ===", flush=True)
    try:
        print(run([kaggle, "kernels", "logs", ref], capture=True), flush=True)
    except subprocess.CalledProcessError as e:
        print("Could not retrieve logs. View the private Kaggle notebook:", ref, flush=True)


if __name__ == "__main__":
    main()
