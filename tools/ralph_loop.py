"""Bounded local Codex -> Kaggle -> PESQ -> Codex experiment loop.

Run only AFTER the existing E02/E03 batch has finished. Does NOT require
GitHub Actions or a cloud controller. PC and PowerShell must stay running.

Dry run:
  py tools/ralph_loop.py --username siyul1122 --dataset siyul1122/speech-data-private
Live (opt-in):
  py tools/ralph_loop.py --username siyul1122 --dataset siyul1122/speech-data-private --rounds 3 --epochs 25 --execute

Private Kaggle data remains on Kaggle. This script never uploads raw WAVs.
Codex edits only src/kaggle_train.py, and each candidate runs a distinct PRIVATE kernel.
"""
from __future__ import annotations

import argparse
import hashlib
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
SOURCE = ROOT / "src" / "kaggle_train.py"
OUTPUT = ROOT / "runs" / "ralph"
BASELINE = 1.6961209534144983


def command(argv, *, capture=False, timeout=None):
    argv = [str(arg) for arg in argv]
    print("$ " + " ".join(argv[:7]) + (" ..." if len(argv) > 7 else ""), flush=True)
    p = subprocess.run(argv, cwd=ROOT, check=False, text=True, capture_output=capture, timeout=timeout)
    if p.returncode:
        print((p.stdout or "")[-6000:] + (p.stderr or "")[-8000:], flush=True)
        raise RuntimeError(f"Command failed (exit {p.returncode}): {argv[:4]}")
    return (p.stdout + "\n" + p.stderr).strip() if capture else ""


def sha256(payload):
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_modified_paths():
    tracked = command(["git", "diff", "--name-only"], capture=True)
    staged = command(["git", "diff", "--cached", "--name-only"], capture=True)
    return {p.strip().replace("\\", "/") for p in (tracked + "\n" + staged).splitlines() if p.strip()}


def initial_best():
    best = {"score": BASELINE, "origin": "original_cnn", "round": 0}
    for exp in ("e02_unet", "e03_mrstft"):
        path = ROOT / "runs" / exp / "result.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            score = float(data["best_validation_pesq_wb"])
            if int(data["validation_files"]) == 164 and math.isfinite(score) and score > best["score"]:
                best = {"score": score, "origin": exp, "round": 0}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            print("WARN: invalid previous result ignored:", path)
    return best


def require_quota(kaggle, min_remaining_hours):
    # Do not submit another job if we cannot verify remaining free GPU quota.
    output = command([kaggle, "quota"], capture=True)
    for line in output.splitlines():
        cells = line.strip().split()
        if len(cells) >= 4 and cells[0] == "GPU":
            remaining = float(cells[2].rstrip("h"))
            print(f"Kaggle GPU hours remaining: {remaining:.2f}", flush=True)
            if remaining < min_remaining_hours:
                raise RuntimeError("Insufficient verified Kaggle GPU quota; stopping")
            return
    raise RuntimeError("Could not read Kaggle GPU quota; refusing to submit")


def codex_edit(codex, score, round_n, recent, output_folder):
    prompt = (
        "You are performing ONE bounded iteration of a traditional speech enhancement "
        "research experiment, inside the current git repository. "
        "READ AGENTS.md. Current best 164-file PESQ-WB mean is "
        f"{score:.8f}; last experiment summary: {json.dumps(recent, ensure_ascii=False)[:2000]}. "
        "Edit ONLY src/kaggle_train.py. Improve the model architecture or the loss, "
        "with ONE clear hypothesis and the smallest safe change. Keep "
        "EXPERIMENT_CONFIG and DATASET_SLUG assignment lines exactly present; "
        "keep SpectrogramUNet, SpeechPairs, run, run() entrypoint, E02_UNET accepted, "
        "valid 16 kHz mono 48000-sample outputs, deterministic seed, and "
        "PESQ on all 164 validation files; validation must not be used for training. "
        "Do not load/upload raw datasets, tokens, secrets, or any private audio to "
        "external services. No Internet/network operations, no git commits, no "
        "GPU submissions, no modifications to any other file. "
        "Do not change baseline metrics, early stopping or result schema. "
        "Do not delete existing functionality. Explain your hypothesis in the final message."
    )
    summary = output_folder / "codex_summary.txt"
    stdout_log = output_folder / "codex_log.txt"
    args = [
        codex, "exec", "--sandbox", "workspace-write",
        "-C", str(ROOT), "-o", str(summary), prompt,
    ]
    print(f"ROUND {round_n}: Codex editing a candidate source ...", flush=True)
    with stdout_log.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, timeout=20 * 60, check=False,
        )
    if proc.returncode != 0:
        print(stdout_log.read_text(encoding="utf-8")[-4000:], flush=True)
        raise RuntimeError(f"Codex failed (exit {proc.returncode})")
    return summary.read_text(encoding="utf-8") if summary.exists() else ""


def validate_candidate(candidate, before):
    if candidate == before:
        raise RuntimeError("Codex made no source change; GPU job not submitted")
    changed = current_modified_paths()
    if changed != {"src/kaggle_train.py"}:
        raise RuntimeError(f"Unexpected tracked changes from Codex: {sorted(changed)}")
    compile(candidate, str(SOURCE), "exec")
    for name in ("EXPERIMENT_CONFIG", "DATASET_SLUG", "SpectrogramUNet",
                 "SpeechPairs", "def run(", 'if __name__ == "__main__"'):
        if name not in candidate:
            raise RuntimeError(f"Required source symbol missing: {name}")
    if re.search(r"^EXPERIMENT_CONFIG = .+$", candidate, re.M) is None:
        raise RuntimeError("No configurable EXPERIMENT_CONFIG assignment")
    if re.search(r"^DATASET_SLUG = .+$", candidate, re.M) is None:
        raise RuntimeError("No configurable DATASET_SLUG assignment")
    print("Candidate Python syntax and experiment interface: PASS", flush=True)


def package(round_number, dataset, username, epochs, batch_size, source):
    code = re.sub(
        r"^EXPERIMENT_CONFIG = .+$",
        "EXPERIMENT_CONFIG = " + repr({
            "id": "E02_UNET", "epochs": epochs, "batch_size": batch_size,
            "patience": 7, "seed": 42,
        }),
        source, count=1, flags=re.M,
    )
    code = re.sub(
        r"^DATASET_SLUG = .+$",
        "DATASET_SLUG = " + repr(dataset),
        code, count=1, flags=re.M,
    )
    slug = f"speech-enhancement-ralph-r{round_number:02d}"
    ref = username + "/" + slug
    folder = ROOT / "kaggle_run" / f"ralph_r{round_number:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.py").write_text(code, encoding="utf-8")
    metadata = {
        "id": ref,
        "title": "Speech Enhancement Ralph R" + f"{round_number:02d}",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",  # existing ensure_pesq() installs via pip
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return folder, ref


def run_kaggle(kaggle, folder, ref, result_folder, max_wait_seconds):
    # A distinct kernel per candidate avoids overwriting the ongoing E02/E03 jobs.
    command([kaggle, "kernels", "push", "-p", folder, "--accelerator", "NvidiaTeslaT4"])
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        state = command([kaggle, "kernels", "status", ref], capture=True)
        print(state, flush=True)
        status = re.search(r"KernelWorkerStatus\.([A-Z_]+)", state)
        if not status:
            raise RuntimeError("Unknown Kaggle status; stopping")
        if status.group(1) == "COMPLETE":
            command([kaggle, "kernels", "output", ref, "-p", result_folder, "-o"])
            result = result_folder / "result.json"
            if not result.is_file():
                raise RuntimeError("Completed Kaggle kernel has no result.json")
            return json.loads(result.read_text(encoding="utf-8"))
        if status.group(1) in ("ERROR", "CANCELLED", "CANCELED"):
            try:
                print(command([kaggle, "kernels", "logs", ref], capture=True)[-16000:])
            except Exception as exc:
                print("Could not fetch Kaggle logs:", exc, flush=True)
            raise RuntimeError(f"Kaggle kernel failed: {ref}")
        time.sleep(60)
    raise TimeoutError(
        f"Kernel still running after local wait deadline. Inspect {ref}; "
        "Kaggle GPU job was NOT cancelled automatically."
    )


def validate_result(result):
    score = float(result["best_validation_pesq_wb"])
    if not math.isfinite(score) or int(result["validation_files"]) != 164:
        raise RuntimeError("Untrustworthy PESQ result or incomplete validation set")
    if int(result["epochs_completed"]) < 1 or not result.get("checkpoint"):
        raise RuntimeError("Missing completed training/checkpoint")
    return score


def main():
    parser = argparse.ArgumentParser(description="Local bounded Codex + Kaggle research loop")
    parser.add_argument("--username", required=True)
    parser.add_argument("--dataset", required=True, help="Approved private Kaggle dataset user/slug")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-total-minutes", type=int, default=180)
    parser.add_argument("--max-job-minutes", type=int, default=80)
    parser.add_argument("--min-gpu-quota-hours", type=float, default=2.0)
    parser.add_argument("--stop-at-pesq", type=float, default=3.0)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    parser.add_argument("--max-stagnant", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.username) or not re.fullmatch(
        r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", args.dataset
    ):
        parser.error("Invalid Kaggle username or private dataset ref")
    if not (1 <= args.rounds <= 10 and 2 <= args.epochs <= 100 and
            1 <= args.batch_size <= 16 and 15 <= args.max_job_minutes <= 180 and
            20 <= args.max_total_minutes <= 600 and args.max_stagnant >= 1):
        parser.error("Experiment limits out of safe range")
    if not SOURCE.exists():
        parser.error("Missing src/kaggle_train.py")
    current = SOURCE.read_text(encoding="utf-8")
    best = initial_best()
    print(f"Starting reference: {best['origin']} PESQ-WB={best['score']:.4f}")
    print(
        f"Plan: at most {args.rounds} Codex revisions + {args.rounds} Kaggle GPU jobs, "
        f"epochs <= {args.epochs}, local duration limit ~{args.max_total_minutes} min."
    )
    print("Existing E02/E03 batch must be COMPLETE before starting this controller.")
    if not args.execute:
        print("DRY RUN ONLY. No Codex request and no Kaggle GPU job submitted.")
        return
    if current_modified_paths():
        parser.error("Clean tracked git changes first; preserving current work")
    codex = shutil.which("codex") or shutil.which("codex.cmd")
    kaggle = shutil.which("kaggle")
    if not codex or not kaggle:
        parser.error("codex and kaggle CLI must both be available and logged in")
    command([codex, "login", "status"])
    require_quota(kaggle, args.min_gpu_quota_hours)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    champion_source = current
    (OUTPUT / "best_source.py").write_text(champion_source, encoding="utf-8")
    (OUTPUT / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    started = time.monotonic()
    recent = {"best": best}
    stagnant = 0
    try:
        for round_n in range(1, args.rounds + 1):
            if time.monotonic() - started >= args.max_total_minutes * 60:
                print("Total wall-time budget reached. Stopping.", flush=True)
                break
            if best["score"] >= args.stop_at_pesq:
                print("Target PESQ reached. Stopping.", flush=True)
                break
            require_quota(kaggle, args.min_gpu_quota_hours)
            folder = OUTPUT / f"round_{round_n:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            SOURCE.write_text(champion_source, encoding="utf-8")
            try:
                explanation = codex_edit(codex, best["score"], round_n, recent, folder)
                candidate = SOURCE.read_text(encoding="utf-8")
                validate_candidate(candidate, champion_source)
                (folder / "candidate_source.py").write_text(candidate, encoding="utf-8")
                (folder / "hypothesis.txt").write_text(explanation, encoding="utf-8")
                job_folder, ref = package(round_n, args.dataset, args.username, args.epochs, args.batch_size, candidate)
                print(f"SUBMIT {ref} source_hash={sha256(candidate)[:12]}", flush=True)
                result = run_kaggle(kaggle, job_folder, ref, folder, args.max_job_minutes * 60)
                score = validate_result(result)
                recent = result
                print(f"Round {round_n} PESQ-WB={score:.4f}, champion={best['score']:.4f}", flush=True)
                if score >= best["score"] + args.min_improvement:
                    champion_source = candidate
                    best = {
                        "score": score, "origin": ref, "round": round_n,
                        "source_hash": sha256(candidate),
                        "checkpoint": str(folder / "best_model.pt"),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    (OUTPUT / "best_source.py").write_text(champion_source, encoding="utf-8")
                    (OUTPUT / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
                    stagnant = 0
                    print("IMPROVED: new champion saved", flush=True)
                else:
                    stagnant += 1
                    print(f"NO IMPROVEMENT: rolling back; stagnant={stagnant}", flush=True)
                    SOURCE.write_text(champion_source, encoding="utf-8")
                if stagnant >= args.max_stagnant:
                    print("Too many non-improving rounds. Stopping.", flush=True)
                    break
            except Exception:
                SOURCE.write_text(champion_source, encoding="utf-8")
                print("Candidate source restored to last champion after failure.", flush=True)
                raise
    finally:
        SOURCE.write_text(champion_source, encoding="utf-8")
        print(f"FINAL best PESQ-WB={best['score']:.4f} ({best['origin']})", flush=True)
        print("Results and candidate sources: runs/ralph/ (not committed to GitHub)", flush=True)


if __name__ == "__main__":
    main()
