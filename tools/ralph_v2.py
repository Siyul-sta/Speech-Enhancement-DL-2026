"""Adaptive Ralph V2: automatically switch model families after PESQ stagnation.

Independent candidates in ignored runs/ directory; tracked trainer templates unchanged.
Protects 164-file validation, paired training split, best checkpoints, private Kaggle
dataset, and avoids submitting remote jobs if a prior submission is uncertain.

Dry run:
  py tools/ralph_v2.py --username siyul1122 --dataset siyul1122/speech-data-private
Live (ONLY after old local loop and Kaggle jobs are stopped/complete):
  py tools/ralph_v2.py --username siyul1122 --dataset siyul1122/speech-data-private --execute
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import ralph_loop as legacy

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "ralph_v2"
FAMILIES = ("residual_phase", "magnitude", "complex")
TEMPLATES = {
    "magnitude": ("src/kaggle_train.py", "SpectrogramUNet", "E02_UNET"),
    "residual_phase": ("src/kaggle_train_v3.py", "ResidualPhaseUNet", "V3_RESIDUAL_PHASE"),
    "complex": ("src/kaggle_train_v2.py", "ComplexMaskUNet", "V2_CRM_PHASE"),
}
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "getattr", "setattr",
    "system", "popen", "run", "load", "save", "read_text", "write_text",
    "unlink", "remove", "rmtree", "urlopen", "request", "post",
    "put", "delete", "connect", "socket",
}


def hash_source(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def valid_full_result(folder: Path, *, min_epochs: int = 5) -> dict | None:
    path = folder / "result.json"
    checkpoint = folder / "best_model.pt"
    rows = folder / "validation_per_file.csv"
    if not all(x.is_file() for x in (path, checkpoint, rows)):
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        score = float(result["best_validation_pesq_wb"])
        if (int(result["validation_files"]) != 164
                or int(result["epochs_completed"]) < min_epochs
                or not math.isfinite(score)
                or len(rows.read_text(encoding="utf-8").splitlines()) != 165):
            return None
        return result
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
        return None


def accepted_record(source: str, folder: Path, *, name: str, variant: str) -> dict | None:
    result = valid_full_result(folder)
    if result is None:
        return None
    return {
        "score": float(result["best_validation_pesq_wb"]),
        "origin": name,
        "variant": variant,
        "checkpoint": str((folder / "best_model.pt").resolve()),
        "results": str(folder.resolve()),
        "source_hash": hash_source(source),
        "source": source,
    }


def recover_prior_family(family: str) -> dict | None:
    """Recover best family source/checkpoint from prior adaptive runs if verified."""
    strongest = None
    if not OUTPUT.exists():
        return None
    for path in sorted(OUTPUT.glob("run_*/families/" + family + "/best.json")):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            source_path = path.with_name("best_source.py")
            source = source_path.read_text(encoding="utf-8")
            if hash_source(source) != info["source_hash"]:
                continue
            folder = Path(info["checkpoint"]).parent
            candidate = accepted_record(
                source, folder, name=info["origin"], variant=info["variant"]
            )
            if candidate and abs(candidate["score"] - float(info["score"])) < 1e-6:
                if strongest is None or candidate["score"] > strongest["score"]:
                    strongest = candidate
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return strongest


def initial_families() -> dict:
    """Use V1 verified champion, completed V2 trials, and prior adaptive winners."""
    best_v1 = legacy.initial_best(require_complete=True)
    v1_source = (
        Path(best_v1["source_path"]).read_text(encoding="utf-8")
        if best_v1.get("source_path")
        else (ROOT / TEMPLATES["magnitude"][0]).read_text(encoding="utf-8")
    )
    v1_folder = Path(best_v1["checkpoint"]).parent
    v1 = accepted_record(
        v1_source, v1_folder, name=best_v1["origin"], variant=best_v1["variant"]
    )
    if v1 is None or abs(v1["score"] - best_v1["score"]) >= 1e-6:
        raise RuntimeError("Verified V1 champion source/checkpoint/PESQ files not consistent")
    states = {"magnitude": v1}
    for family in ("complex", "residual_phase"):
        template, _, variant = TEMPLATES[family]
        source = (ROOT / template).read_text(encoding="utf-8")
        strongest = {
            "score": float("-inf"), "origin": "untrained_template",
            "variant": variant, "checkpoint": None,
            "source_hash": hash_source(source), "source": source,
        }
        if family == "complex":
            for folder in (ROOT / "runs" / "v2").glob("*/v2_crm_phase"):
                candidate = accepted_record(
                    source, folder, name=str(folder), variant=variant
                )
                if candidate and candidate["score"] > strongest["score"]:
                    strongest = candidate
        states[family] = strongest
    for family in FAMILIES:
        prior = recover_prior_family(family)
        if prior and prior["score"] > states[family]["score"]:
            states[family] = prior
    return states


def recover_completed_smoke(states: dict) -> dict | None:
    """Reuse a verified 2-epoch smoke candidate when its full run never started.

    Read-only recovery. Reject any candidate that edits protected validation
    or trainer sections, has a different source than the prior smoke run,
    or has an ambiguous subsequent full-job directory.
    """
    eligible = []
    for state_file in sorted(OUTPUT.glob("run_*/round_*/smoke/remote_state.json")):
        smoke_dir = state_file.parent
        round_dir = smoke_dir.parent
        try:
            info = json.loads(state_file.read_text(encoding="utf-8"))
            if info.get("status") != "COMPLETE":
                continue
            # A previously started full job must never be re-submitted.
            if (round_dir / "full").exists():
                continue
            result = valid_full_result(smoke_dir, min_epochs=1)
            if result is None or int(result["epochs_completed"]) != 2:
                continue
            variant = str(result["experiment"])
            family = next(
                (name for name, cfg in TEMPLATES.items() if cfg[2] == variant),
                None,
            )
            if family is None or states[family]["checkpoint"] is not None:
                continue
            source_path = round_dir / "candidate.py"
            base_path = round_dir.parent / "families" / family / "best_source.py"
            if not source_path.is_file() or not base_path.is_file():
                continue
            source = source_path.read_text(encoding="utf-8")
            previous = base_path.read_text(encoding="utf-8")
            validate_source(source, previous, family)
            if result.get("model") != TEMPLATES[family][1]:
                continue
            eligible.append({
                "source": source,
                "family": family,
                "variant": variant,
                "source_hash": hash_source(source),
                "smoke_dir": str(smoke_dir),
                "prior_run": str(round_dir),
            })
        except (OSError, ValueError, KeyError, TypeError, StopIteration,
                SyntaxError, json.JSONDecodeError) as exc:
            print(f"Skipping unsafe recovered smoke {smoke_dir}: {exc}", flush=True)
            continue
    if not eligible:
        return None
    selected = eligible[-1]
    print(
        f"RECOVERED VERIFIED SMOKE: {selected['prior_run']} "
        f"family={selected['family']}; skipping repeat 2-epoch GPU job",
        flush=True,
    )
    return selected


def recovered_full_runs(states: dict) -> list[dict]:
    """Import verified full Kaggle jobs whose scoring was interrupted locally.

    Requires local 164-file output, the exact originally submitted run.py,
    the metadata ID, and the editable-only source AST policy. An unresolved
    remote_state is NOT accepted even if files appear on disk.
    """
    recovered = []
    for state_path in sorted(OUTPUT.glob("run_*/round_*/full/remote_state.json")):
        folder = state_path.parent
        round_dir = folder.parent
        run_dir = round_dir.parent
        try:
            info = json.loads(state_path.read_text(encoding="utf-8"))
            if info.get("status") != "COMPLETE":
                continue
            result = valid_full_result(folder)
            if result is None:
                continue
            variant = str(result["experiment"])
            family = next((name for name in FAMILIES
                           if TEMPLATES[name][2] == variant), None)
            if family is None:
                continue
            candidate_path = round_dir / "candidate.py"
            base_path = run_dir / "families" / family / "best_source.py"
            if not candidate_path.is_file() or not base_path.is_file():
                continue
            candidate = candidate_path.read_text(encoding="utf-8")
            reference = base_path.read_text(encoding="utf-8")
            if candidate != reference:
                validate_source(candidate, reference, family)
            else:
                ast.parse(candidate)
            tag = run_dir.name.removeprefix("run_")
            round_number = int(round_dir.name.removeprefix("round_"))
            job_folder = ROOT / "kaggle_run" / "ralph_v2" / tag / f"job_{2 * round_number:04d}"
            packed = (job_folder / "run.py").read_text(encoding="utf-8")
            metadata = json.loads((job_folder / "kernel-metadata.json").read_text(encoding="utf-8"))
            if metadata.get("id") != info.get("ref"):
                continue
            # Compare the exact submitted source, allowing only original
            # run packaging to inject experiment config and dataset slug.
            expected = candidate
            for field in ("EXPERIMENT_CONFIG", "DATASET_SLUG"):
                found = re.search(r"^" + field + r" = .+$", packed, flags=re.M)
                if found is None:
                    raise ValueError(f"Packaged script missing {field}")
                expected, count = re.subn(
                    r"^" + field + r" = .+$",
                    lambda _: found.group(0), expected, count=1, flags=re.M,
                )
                if count != 1:
                    raise ValueError(f"Candidate script missing {field}")
            if expected != packed:
                print(f"RECOVERY SKIPPED: source mismatch for {state_path}", flush=True)
                continue
            if result.get("model") != TEMPLATES[family][1]:
                continue
            record = accepted_record(
                candidate, folder, name=str(info["ref"]), variant=variant
            )
            if record is None:
                continue
            entry = {
                "status": "RECOVERED_COMPLETE",
                "family": family,
                "score": record["score"],
                "hypothesis": f"Recovered completed GPU job {info['ref']} (no resubmission)",
                "source_hash": record["source_hash"],
                "round": round_number,
                "result_dir": str(folder),
            }
            recovered.append(entry)
            if record["score"] > states[family]["score"]:
                states[family] = record
                print(
                    f"RECOVERED FAMILY BEST: {family} PESQ-WB={record['score']:.4f} "
                    f"from {info['ref']}",
                    flush=True,
                )
        except (OSError, KeyError, ValueError, TypeError, SyntaxError,
                json.JSONDecodeError) as exc:
            print(f"RECOVERY SKIPPED: {state_path}: {exc}", flush=True)
    return recovered


def paired_diagnostics(candidate_folder: Path, reference_folder: Path) -> dict:
    """Clip-by-clip PESQ improvement against the globally strongest baseline."""
    def load_scores(path):
        with path.open("r", newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        scores = {int(row["index"]): float(row["pesq_wb"]) for row in records}
        if len(records) != 164 or set(scores) != set(range(164)):
            raise ValueError("Paired PESQ results must contain exactly 164 aligned clips")
        return scores

    current = load_scores(candidate_folder / "validation_per_file.csv")
    reference = load_scores(reference_folder / "validation_per_file.csv")
    deltas = sorted(current[i] - reference[i] for i in range(164))
    return {
        "win_rate": round(100 * sum(x > 0 for x in deltas) / 164, 2),
        "worst10_mean_delta": round(sum(deltas[:16]) / 16, 5),
        "mean_paired_delta": round(sum(deltas) / 164, 5),
    }


def record_for_json(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "source"}


def validate_source(candidate: str, before: str, family: str) -> list[str]:
    """AST structural lock: data, training, scoring, eval, imports unchanged."""
    if candidate == before:
        raise ValueError("No candidate changes; skip GPU")
    old = ast.parse(before)
    new = ast.parse(candidate)
    if len(old.body) != len(new.body):
        raise ValueError("Top-level shape changed")
    model_class = TEMPLATES[family][1]
    permitted = {"ConvBlock", model_class, "enhancement_loss"}
    modified = []
    for old_node, new_node in zip(old.body, new.body):
        if (type(old_node) is not type(new_node)
                or getattr(old_node, "name", None) != getattr(new_node, "name", None)):
            raise ValueError("Top-level definitions reordered or renamed")
        name = getattr(new_node, "name", None)
        different = ast.dump(old_node, include_attributes=False) != ast.dump(
            new_node, include_attributes=False
        )
        if not different:
            continue
        if name not in permitted:
            raise ValueError(f"Protected training/evaluation/data section modified: {name}")
        for item in ast.walk(new_node):
            if isinstance(item, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                raise ValueError(f"Unsupported import/global statement in {name}")
            if isinstance(item, ast.Call):
                fn = item.func
                call = fn.id if isinstance(fn, ast.Name) else (
                    fn.attr if isinstance(fn, ast.Attribute) else ""
                )
                if call in BLOCKED_CALLS:
                    raise ValueError(f"Forbidden method call in {name}: {call}")
        modified.append(name)
    if not modified:
        raise ValueError("Only comments/no model changes")
    return modified


def read_history() -> list[dict]:
    """Persistent experiment memory from every previous adaptive session."""
    history = []
    if OUTPUT.exists():
        for file in sorted(OUTPUT.glob("run_*/experiments.jsonl")):
            for line in file.read_text(encoding="utf-8").splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return history


def append_history(file: Path, entry: dict) -> None:
    with file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def codex_propose(codex: str, *, workfile: Path, family: str,
                  champion: dict, family_state: dict, history: list[dict],
                  round_dir: Path, model: str) -> str:
    short_history = [
        {k: entry.get(k) for k in ("round", "family", "score", "status", "hypothesis", "win_rate", "worst10_mean_delta", "mean_paired_delta")}
        for entry in history[-15:]
    ]
    relative = workfile.relative_to(ROOT).as_posix()
    instructions = (
        "You are the architecture search agent in a speech enhancement experiment. "
        "Read AGENTS.md. You MUST edit exactly ONE file: " + relative + ". "
        "This is an isolated scratch copy of a full trainer. "
        f"Current model family is {family}; its editable network class is {TEMPLATES[family][1]}. "
        f"Global best complete 164-file PESQ-WB={champion['score']:.5f} "
        f"({champion['origin']}). "
        f"Best for this family={family_state['score']:.5f}. "
        f"Past attempts (avoid repeat hypotheses): {json.dumps(short_history, ensure_ascii=False)[-8000:]}. "
        "Change ONE substantive, testable architecture or signal-domain loss hypothesis, "
        "INSIDE existing ConvBlock, the family model class, and/or enhancement_loss. "
        "Do not change imports or any other top-level definition. "
        "NEVER change dataset reading, validation, PESQ score, optimizer/train loop, "
        "checkpoint format, 164-clip coverage, or fixed random seed. "
        "Do NOT introduce torch load, file access, network access or external data. "
        "Validation clean must never be used in inference or training. "
        "Do not edit tracked repository files, AGENTS.md, or create any new files. "
        "Do not call Kaggle, train a model, or commit changes. "
        "Explain in final output one expected benefit, one risk, and what changed."
    )
    summary = round_dir / "codex_summary.txt"
    logfile = round_dir / "codex_log.txt"
    argv = [
        codex, "exec", "--model", model, "--sandbox", "workspace-write",
        "-C", str(ROOT), "-o", str(summary), instructions,
    ]
    with logfile.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(
            argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, timeout=20 * 60, check=False,
        )
    if proc.returncode:
        raise RuntimeError(
            f"Codex exit={proc.returncode}: {logfile.read_text(encoding='utf-8')[-2000:]}"
        )
    return summary.read_text(encoding="utf-8") if summary.exists() else ""


def cpu_check(candidate_file: Path):
    if importlib.util.find_spec("torch") is None:
        print("No local torch; Kaggle 2-epoch smoke test required on new families.", flush=True)
        return
    proc = subprocess.run(
        [sys.executable, str(candidate_file), "--self-test"],
        cwd=ROOT, text=True, capture_output=True, timeout=120, check=False,
    )
    if proc.returncode:
        raise RuntimeError("Candidate CPU self-test failed: " + (proc.stdout + proc.stderr)[-4000:])
    print("Candidate CPU self-test passed", flush=True)


def package_candidate(*, source: str, variant: str, args, run_tag: str, job_id: int, epochs: int):
    """Write an isolated PRIVATE Kaggle job without overwriting any V1 packages."""
    config = {
        "id": variant, "epochs": epochs, "batch_size": args.batch_size,
        "patience": 7, "seed": 42,
    }
    code, config_count = re.subn(
        r"^EXPERIMENT_CONFIG = .+$",
        "EXPERIMENT_CONFIG = " + repr(config),
        source, count=1, flags=re.MULTILINE,
    )
    code, dataset_count = re.subn(
        r"^DATASET_SLUG = .+$", "DATASET_SLUG = " + repr(args.dataset),
        code, count=1, flags=re.MULTILINE,
    )
    if config_count != 1 or dataset_count != 1:
        raise RuntimeError("Cannot inject Kaggle private-data or experiment configuration")
    folder = ROOT / "kaggle_run" / "ralph_v2" / run_tag / f"job_{job_id:04d}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "run.py").write_text(code, encoding="utf-8")
    # Kaggle normalizes the notebook TITLE to its real URL slug.
    # Keep the metadata id aligned with that slug (verified from --mine).
    title = f"Adaptive Speech V2 {run_tag} job {job_id}"
    ref = f"{args.username}/adaptive-speech-v2-{run_tag}-job-{job_id}"
    metadata = {
        "id": ref,
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [args.dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    json_write(folder / "kernel-metadata.json", metadata)
    return folder, ref


def poll_adaptive_kernel(kaggle: str, folder: Path, ref: str,
                         result_folder: Path, max_wait_seconds: int,
                         *, sleep_seconds: float = 60,
                         max_consecutive_status_errors: int = 5):
    """Submit ONCE, retry transient Kaggle status failures, never duplicate push.

    An unresolved status is NEVER treated as a cancelled job. After exhausting
    retries we stop and keep remote_state.json unresolved for manual recovery.
    """
    legacy.command([kaggle, "kernels", "push", "-p", folder, "--accelerator", "NvidiaTeslaT4"])
    deadline = time.monotonic() + max_wait_seconds
    failed = 0
    while time.monotonic() < deadline:
        try:
            proc = subprocess.run(
                [str(kaggle), "kernels", "status", ref],
                cwd=ROOT, text=True, capture_output=True,
                timeout=90, check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            failed += 1
            details = repr(exc)
            output = ""
        else:
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            details = output[-2500:].strip()
            if proc.returncode != 0:
                failed += 1
            else:
                status = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
                if not status:
                    failed += 1
                    details = "Unrecognized successful Kaggle status: " + details
                elif status.group(1) == "COMPLETE":
                    print(f"REMOTE COMPLETE: {ref}", flush=True)
                    legacy.command([kaggle, "kernels", "output", ref,
                                    "-p", result_folder, "-o"])
                    result_file = result_folder / "result.json"
                    if not result_file.exists():
                        raise RuntimeError("Kaggle COMPLETE but result.json missing")
                    return json.loads(result_file.read_text(encoding="utf-8"))
                elif status.group(1) in ("ERROR", "CANCELLED", "CANCELED"):
                    try:
                        print(legacy.command([kaggle, "kernels", "logs", ref],
                                             capture=True)[-12000:], flush=True)
                    except Exception as exc:
                        print("Could not download Kaggle error logs:", exc, flush=True)
                    raise legacy.KaggleTerminalFailure(
                        f"Kaggle job ended {status.group(1)}: {ref}"
                    )
                else:
                    failed = 0
                    print(f"Kaggle remote status={status.group(1)}: {ref}", flush=True)
        if failed:
            print(
                f"Kaggle STATUS QUERY FAILED ({failed}/{max_consecutive_status_errors}): "
                f"{details}",
                flush=True,
            )
            if failed >= max_consecutive_status_errors:
                raise RuntimeError(
                    f"Kaggle state unknown after {failed} status attempts: {ref}. "
                    "Do NOT start another GPU job. Inspect private Kaggle notebook."
                )
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(float(sleep_seconds if not failed else 20), remaining))
    raise TimeoutError(
        f"Kaggle status wait timeout: {ref}; remote job NOT cancelled."
    )


def remote_job(kaggle: str, *, source: str, variant: str, args, run_tag: str,
               job_id: int, result_dir: Path, is_smoke: bool):
    epochs = 2 if is_smoke else args.epochs
    job_dir, ref = package_candidate(
        source=source, variant=variant, args=args, run_tag=run_tag,
        job_id=job_id, epochs=epochs,
    )
    state = result_dir / "remote_state.json"
    json_write(state, {
        "ref": ref, "status": "SUBMIT_PENDING", "epochs": epochs,
        "created_at": datetime.now(timezone.utc).isoformat()
    })
    # If the local controller crashes after 'push', a new controller must
    # NOT resubmit before explicitly checking this remote notebook.
    result = poll_adaptive_kernel(
        kaggle, job_dir, ref, result_dir, args.max_job_minutes * 60
    )
    json_write(state, {"ref": ref, "status": "COMPLETE", "epochs": epochs})
    if valid_full_result(result_dir, min_epochs=1 if is_smoke else 5) is None:
        raise RuntimeError("Remote job output invalid (missing full PESQ/checkpoint)")
    return result, ref


def no_uncertain_prior_jobs() -> None:
    for path in OUTPUT.glob("run_*/round_*/**/remote_state.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RuntimeError(f"Unparseable prior remote state: {path}")
        if value.get("status") not in ("COMPLETE", "ERROR_TERMINAL"):
            raise RuntimeError(
                f"Unresolved previous Kaggle job {value.get('ref')} ({path}); "
                "inspect Kaggle manually before starting new work."
            )


def save_state(folder: Path, states: dict, champion: dict) -> None:
    for family, item in states.items():
        base = folder / "families" / family
        base.mkdir(parents=True, exist_ok=True)
        (base / "best_source.py").write_text(item["source"], encoding="utf-8")
        json_write(base / "best.json", record_for_json(item))
    json_write(folder / "global_best.json", record_for_json(champion))


def choose_family(family_index: int, stagnation: int, threshold: int) -> tuple[int, int]:
    if stagnation >= threshold:
        return (family_index + 1) % len(FAMILIES), 0
    return family_index, stagnation


def snapshot_existing_tracked_edits() -> dict[str, str]:
    """Preserve the existing V1 trainer edit without touching or trusting its code.

    Adaptive V2 builds models from validated run checkpoint/source pairs. A
    historical local V1 trainer edit may remain in the worktree; allow ONLY
    that path, recording a byte-for-byte digest to detect any later changes.
    """
    modified = legacy.current_modified_paths()
    disallowed = modified - {"src/kaggle_train.py"}
    if disallowed:
        raise RuntimeError(
            f"Unrelated tracked changes {sorted(disallowed)}; refusing to touch user work"
        )
    snapshot = {}
    if "src/kaggle_train.py" in modified:
        path = ROOT / "src" / "kaggle_train.py"
        if not path.is_file():
            raise RuntimeError("Modified V1 trainer is missing; refusing to continue")
        snapshot["src/kaggle_train.py"] = hashlib.sha256(path.read_bytes()).hexdigest()
        print("Found existing modified V1 trainer; preserving exact bytes.", flush=True)
    return snapshot


def assert_tracked_unchanged(snapshot: dict[str, str]) -> None:
    """Abort if Codex (or another program) alters ANY tracked working-tree file."""
    current = snapshot_existing_tracked_edits()
    if current != snapshot:
        raise RuntimeError(
            f"Tracked working-tree changes detected: before={snapshot}, now={current}. "
            "Stopping to protect existing files."
        )


def preflight(args):
    initial_tracked = snapshot_existing_tracked_edits()
    if not all((ROOT / cfg[0]).is_file() for cfg in TEMPLATES.values()):
        raise RuntimeError("Missing family trainer template. Run git pull --ff-only.")
    no_uncertain_prior_jobs()
    lock = OUTPUT / "active.lock"
    if lock.exists():
        raise RuntimeError(
            f"Another adaptive controller may be running ({lock}); "
            "inspect processes before manual lock removal."
        )
    if not args.execute:
        return None, None, None, initial_tracked
    codex = shutil.which("codex") or shutil.which("codex.cmd")
    kaggle = shutil.which("kaggle")
    if not codex or not kaggle:
        raise RuntimeError("Both Codex CLI and Kaggle CLI must be installed and logged in")
    legacy.command([codex, "login", "status"])
    legacy.require_quota(kaggle, args.min_gpu_quota_hours)
    return codex, kaggle, lock, initial_tracked


def main():
    p = argparse.ArgumentParser(description="Stagnation-triggered architecture-switching Ralph V2")
    p.add_argument("--username", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--codex-model", default="gpt-6.1-sol")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=0, help="0 = until GPU quota/safety stop")
    p.add_argument("--stagnation-rounds", type=int, default=2)
    p.add_argument("--min-improvement", type=float, default=0.005)
    p.add_argument("--min-gpu-quota-hours", type=float, default=2.0)
    p.add_argument("--max-job-minutes", type=int, default=80)
    p.add_argument("--max-errors", type=int, default=3)
    p.add_argument("--execute", action="store_true", help="Submit private Kaggle jobs; default is dry run")
    args = p.parse_args()
    if not (re.fullmatch(r"[a-zA-Z0-9_-]+", args.username)
            and re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", args.dataset)
            and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,79}", args.codex_model)):
        p.error("Invalid Kaggle identifiers or Codex model name")
    if not (2 <= args.epochs <= 100 and 1 <= args.batch_size <= 16
            and 0 <= args.max_rounds <= 1000
            and 1 <= args.stagnation_rounds <= 20
            and 0 < args.min_improvement <= 0.25
            and 1 <= args.min_gpu_quota_hours <= 20
            and 15 <= args.max_job_minutes <= 180
            and 1 <= args.max_errors <= 10):
        p.error("Experiment configuration outside safety bounds")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    states = initial_families()
    recovered_full = recovered_full_runs(states)
    recovered_smoke = recover_completed_smoke(states)
    champion = max(
        (v for v in states.values() if math.isfinite(v["score"])),
        key=lambda v: v["score"],
    )
    print(f"GLOBAL CHAMPION: {champion['score']:.4f} / {champion['origin']}", flush=True)
    for name, state in states.items():
        print(f"  family={name}, best={state['score']:.4f}", flush=True)
    print(
        f"Stagnation trigger: {args.stagnation_rounds} non-improving full experiments "
        f"-> change model family {FAMILIES}; "
        f"max rounds={args.max_rounds or 'quota/safety only'}",
        flush=True,
    )
    if not args.execute:
        print("DRY RUN: no Codex edits and no Kaggle GPU submission", flush=True)
        return

    codex, kaggle, lock, initial_tracked = preflight(args)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\n")
    except FileExistsError:
        raise RuntimeError("Adaptive Ralph already running or stale lock: check processes")
    tag = "v2" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_dir = OUTPUT / ("run_" + tag)
    run_dir.mkdir(parents=True, exist_ok=False)
    history = read_history()
    history.extend(recovered_full)
    rounds = range(1, args.max_rounds + 1) if args.max_rounds else itertools.count(1)
    family_index, stagnant, errors = (
        FAMILIES.index(recovered_smoke["family"]) if recovered_smoke else 0,
        0, 0,
    )
    tried = set()
    if "src/kaggle_train.py" in initial_tracked:
        # This is a copy, never a checkout/reset/overwrite of the V1 trainer.
        source = ROOT / "src" / "kaggle_train.py"
        backup = run_dir / "preserved_v1_trainer.py"
        backup.write_bytes(source.read_bytes())
        assert_tracked_unchanged(initial_tracked)
        if hashlib.sha256(backup.read_bytes()).hexdigest() != initial_tracked["src/kaggle_train.py"]:
            raise RuntimeError("V1 backup integrity check failed; stopping")
        print(f"Preserved V1 trainer at {backup}", flush=True)
    save_state(run_dir, states, champion)
    print(f"RUN DIRECTORY: {run_dir}", flush=True)
    try:
        for i in rounds:
            assert_tracked_unchanged(initial_tracked)
            try:
                legacy.require_quota(kaggle, args.min_gpu_quota_hours)
            except legacy.QuotaGuardReached as exc:
                print("STOP: " + str(exc), flush=True)
                break

            family = FAMILIES[family_index]
            current = states[family]
            variant = current["variant"]
            round_dir = run_dir / f"round_{i:04d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            candidate_file = round_dir / "candidate.py"
            candidate_file.write_text(current["source"], encoding="utf-8")
            print(
                f"ROUND {i}: model family={family}, global={champion['score']:.4f}, "
                f"family best={current['score']:.4f}",
                flush=True,
            )
            submitted = False
            entry = {
                "round": i, "family": family,
                "global_before": champion["score"],
                "family_before": current["score"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                if i == 1 and recovered_smoke:
                    # Use the EXACT source that previously passed the complete
                    # 164-clip smoke validation; no new Codex edit and no smoke.
                    candidate = recovered_smoke["source"]
                    candidate_file.write_text(candidate, encoding="utf-8")
                    summary = (
                        "Continue verified prior 2-epoch smoke candidate from "
                        + recovered_smoke["prior_run"]
                    )
                    entry["recovered_smoke"] = recovered_smoke["smoke_dir"]
                    entry["smoke"] = "verified_prior_run"
                    print("Using previously validated smoke candidate; no duplicate smoke GPU", flush=True)
                else:
                    summary = codex_propose(
                        codex, workfile=candidate_file, family=family,
                        champion=champion, family_state=current, history=history,
                        round_dir=round_dir, model=args.codex_model,
                    )
                    candidate = candidate_file.read_text(encoding="utf-8")
                entry["edits"] = validate_source(candidate, current["source"], family)
                assert_tracked_unchanged(initial_tracked)
                entry["source_hash"] = hash_source(candidate)
                entry["hypothesis"] = summary[:1000]
                cpu_check(candidate_file)
                (round_dir / "hypothesis.txt").write_text(summary, encoding="utf-8")

                # First experiment for an untrained architecture receives a
                # 2-epoch smoke test. Smoke scores can never become champions.
                if family not in tried and current["checkpoint"] is None and not (
                    i == 1 and recovered_smoke
                ):
                    print(f"ROUND {i} new architecture: 2-epoch smoke first", flush=True)
                    smoke_dir = round_dir / "smoke"
                    smoke_dir.mkdir()
                    submitted = True
                    remote_job(
                        kaggle, source=candidate, variant=variant, args=args,
                        run_tag=tag, job_id=2 * i - 1,
                        result_dir=smoke_dir, is_smoke=True,
                    )
                    entry["smoke"] = "passed"
                tried.add(family)
                legacy.require_quota(kaggle, args.min_gpu_quota_hours)
                full_dir = round_dir / "full"
                full_dir.mkdir()
                submitted = True
                result, ref = remote_job(
                    kaggle, source=candidate, variant=variant, args=args,
                    run_tag=tag, job_id=2 * i,
                    result_dir=full_dir, is_smoke=False,
                )
                score = float(result["best_validation_pesq_wb"])
                entry.update({
                    "status": "COMPLETE", "score": score,
                    "kernel": ref, "best_epoch": result.get("best_epoch"),
                    "epochs_completed": result.get("epochs_completed"),
                })
                entry.update(paired_diagnostics(
                    full_dir, Path(champion["results"])
                ))
                new_record = accepted_record(
                    candidate, full_dir, name=ref, variant=variant
                )
                if new_record is None:
                    raise RuntimeError("Completed full job has invalid artifacts")
                if score >= current["score"] + args.min_improvement:
                    states[family] = new_record
                    entry["family_improved"] = True
                    print(f"FAMILY IMPROVED: {family} {score:.4f}", flush=True)
                else:
                    entry["family_improved"] = False
                if score >= champion["score"] + args.min_improvement:
                    champion = new_record
                    entry["global_improved"] = True
                    stagnant = 0
                    print(f"NEW GLOBAL CHAMPION PESQ-WB={score:.4f}", flush=True)
                else:
                    entry["global_improved"] = False
                    stagnant += 1
                    print(f"GLOBAL PLATEAU: {stagnant}/{args.stagnation_rounds}", flush=True)
                errors = 0
                save_state(run_dir, states, champion)
            except legacy.QuotaGuardReached:
                print("GPU quota guard reached; preserving current champion", flush=True)
                break
            except (TimeoutError, KeyboardInterrupt):
                entry["status"] = "REMOTE_UNCERTAIN"
                append_history(run_dir / "experiments.jsonl", entry)
                raise
            except Exception as exc:
                entry.update({"status": "ERROR", "error": str(exc)})
                (round_dir / "error.txt").write_text(repr(exc), encoding="utf-8")
                if isinstance(exc, legacy.KaggleTerminalFailure):
                    # Kaggle returned a terminal ERROR/CANCELLED state: safe to
                    # retry a different hypothesis without an orphan-job alarm.
                    for status_file in round_dir.glob("*/remote_state.json"):
                        status = json.loads(status_file.read_text(encoding="utf-8"))
                        if status.get("status") == "SUBMIT_PENDING":
                            status["status"] = "ERROR_TERMINAL"
                            json_write(status_file, status)
                if submitted and not isinstance(exc, legacy.KaggleTerminalFailure):
                    append_history(run_dir / "experiments.jsonl", entry)
                    print(
                        "Remote job may still run or have ambiguous state. STOP to avoid overlap.",
                        flush=True,
                    )
                    raise
                try:
                    assert_tracked_unchanged(initial_tracked)
                except RuntimeError:
                    append_history(run_dir / "experiments.jsonl", entry)
                    print("Tracked files changed; stop to protect user work", flush=True)
                    raise
                errors += 1
                stagnant += 1
                print(f"ROUND {i} failed ({errors}/{args.max_errors}): {exc}", flush=True)

            append_history(run_dir / "experiments.jsonl", entry)
            history.append(entry)
            if errors >= args.max_errors:
                print("STOP: too many consecutive local/model errors", flush=True)
                break
            next_index, next_stagnant = choose_family(
                family_index, stagnant, args.stagnation_rounds
            )
            if next_index != family_index:
                print(
                    f"MODEL SWITCH: {FAMILIES[family_index]} -> {FAMILIES[next_index]}",
                    flush=True,
                )
            family_index, stagnant = next_index, next_stagnant
    finally:
        save_state(run_dir, states, champion)
        if lock.exists():
            lock.unlink()
        print(
            f"FINAL CHAMPION PESQ-WB={champion['score']:.4f} "
            f"({champion['origin']})",
            flush=True,
        )
        print(f"Experiment history and candidate sources in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
