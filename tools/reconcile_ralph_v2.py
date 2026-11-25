"""Inspect and safely reconcile unresolved Ralph V2 Kaggle remote submissions.

Default mode is READ ONLY: reports current Kaggle status. --resolve downloads
COMPLETED outputs and marks only verified terminal work as resolved.
This never calls kaggle kernels push, does not create GPU jobs, and does not
delete or replace any model/source/checkpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import ralph_v2 as adaptive

ROOT = Path(__file__).resolve().parents[1]


def find_actual_ref(kaggle: str, recorded_ref: str) -> tuple[str | None, str]:
    """Find the canonical Kaggle notebook slug from an authenticated --mine listing.

    Older Ralph V2 metadata used a ref that disagreed with its actual title.
    Require an exact match to the historically observed title-derived slug.
    Do not treat missing or ambiguous entries as cancelled kernels.
    """
    match = re.fullmatch(
        r"([A-Za-z0-9_-]+)/speech-enhancement-auto-(v2[0-9]{14})-j([0-9]{4})",
        recorded_ref,
    )
    if match is None:
        return None, "No recognized legacy slug mapping"
    owner, tag, job_number = match.groups()
    expected = f"{owner}/adaptive-speech-v2-{tag}-job-{int(job_number)}"
    try:
        listing = subprocess.run(
            [str(kaggle), "kernels", "list", "--mine"],
            text=True, capture_output=True, cwd=ROOT,
            timeout=90, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"Cannot list owned kernels: {exc}"
    if listing.returncode:
        return None, ((listing.stdout or "") + "\n" + (listing.stderr or ""))[-2500:]
    refs = [
        row.split()[0] for row in listing.stdout.splitlines()
        if row.split() and re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", row.split()[0])
    ]
    matches = [ref for ref in refs if ref == expected]
    if len(matches) == 1:
        return expected, "Verified exact canonical notebook ref from kaggle kernels list --mine"
    return None, f"Expected {expected} not uniquely found in owned kernels"


def status(kaggle: str, ref: str) -> tuple[str | None, str]:
    result = subprocess.run(
        [str(kaggle), "kernels", "status", ref],
        text=True, capture_output=True, cwd=ROOT, timeout=90, check=False,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode:
        return None, combined.strip()
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", combined)
    return (match.group(1) if match else None), combined.strip()


def pending_states():
    for path in sorted(adaptive.OUTPUT.glob("run_*/round_*/**/remote_state.json")):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"INVALID remote state: {path}")
            continue
        if info.get("status") not in ("COMPLETE", "ERROR_TERMINAL"):
            yield path, info


def reconcile(kaggle: str, path: Path, info: dict, *, resolve: bool):
    recorded_ref = info["ref"]
    ref = recorded_ref
    worker, message = status(kaggle, ref)
    if not worker:
        actual_ref, hint = find_actual_ref(kaggle, recorded_ref)
        if actual_ref:
            print(f"LEGACY SLUG MISMATCH: {recorded_ref} -> {actual_ref}")
            ref = actual_ref
            worker, message = status(kaggle, ref)
        else:
            print(f"Could not verify alternate Kaggle slug: {hint}")
    print(f"\nKAGGLE: {ref}")
    print(f"local={info.get('status')} remote={worker or 'UNKNOWN'}")
    if not worker:
        print("Kaggle CLI diagnostic:", message[-4000:])
        print("NOT RESOLVED. Do not restart automatic GPU experiments.")
        return
    if resolve and ref != recorded_ref:
        info["original_ref"] = recorded_ref
        info["ref"] = ref
        adaptive.json_write(path, info)
        print("Verified and saved corrected actual Kaggle notebook ref.")
    if worker in ("RUNNING", "QUEUED", "PENDING", "INITIALIZING", "CANCEL_REQUESTED"):
        print("Remote GPU may still be in use. NOT RESOLVED.")
        return
    if worker == "COMPLETE":
        if not resolve:
            print("Remote job COMPLETE. Re-run --resolve to fetch/check artifacts.")
            return
        result_dir = path.parent
        cmd = [str(kaggle), "kernels", "output", ref, "-p", str(result_dir), "-o"]
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
        if proc.returncode:
            print("Download failed; leave unresolved.")
            return
        min_epochs = 1 if result_dir.name == "smoke" else 5
        if info.get("kind") in ("v4_public", "v4_external"):
            try:
                import ralph_v4
                ralph_v4.verify(result_dir, info["kind"])
                valid = True
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
        else:
            valid = adaptive.valid_full_result(result_dir, min_epochs=min_epochs) is not None
        if not valid:
            print("Missing/invalid 164-score result, checkpoint or training metadata. "
                  "Keep unresolved for inspection.")
            return
        info["status"] = "COMPLETE"
        info["reconciled"] = True
        adaptive.json_write(path, info)
        print("RESOLVED COMPLETE. Verified checkpoint, result, 164 PESQ rows.")
        return
    if worker in ("ERROR", "CANCELLED", "CANCELED", "CANCEL_ACKNOWLEDGED"):
        if resolve:
            info["status"] = "ERROR_TERMINAL"
            info["reconciled"] = True
            info["terminal_remote_status"] = worker
            adaptive.json_write(path, info)
            print("RESOLVED TERMINAL ERROR; no running notebook reported.")
        else:
            print("Terminal error. Re-run --resolve to mark safe.")
        return
    print("Unknown Kaggle state; NOT RESOLVED.")


def main():
    p = argparse.ArgumentParser(description="Read-only by default; --resolve for verified remote terminal states")
    p.add_argument("--resolve", action="store_true")
    args = p.parse_args()
    kaggle = shutil.which("kaggle")
    if not kaggle:
        p.error("kaggle CLI not found")
    entries = list(pending_states())
    if not entries:
        print("No unresolved Ralph V2 remote jobs recorded.")
        return
    for path, info in entries:
        reconcile(kaggle, path, info, resolve=args.resolve)


if __name__ == "__main__":
    main()
