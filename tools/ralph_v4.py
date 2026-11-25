"""V4 public model benchmark launcher. No raw audio is uploaded by this controller."""
import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import ralph_v2 as core
import ralph_loop as legacy

ROOT = Path(__file__).resolve().parents[1]

def verify(folder):
    r = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    with (folder / "validation_per_file.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = [float(row["pesq_wb"]) for row in rows]
    if len(scores) != 164 or [int(row["index"]) for row in rows] != list(range(164)):
        raise ValueError("Incomplete validation")
    mean = sum(scores) / 164
    if not all(math.isfinite(v) for v in scores) or r["validation_files"] != 164 or abs(mean - float(r["best_validation_pesq_wb"])) > 1e-6:
        raise ValueError("Invalid score")
    return r

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    src = (ROOT / "src/kaggle_v4_public.py").read_text(encoding="utf-8")
    compile(src, "kaggle_v4_public.py", "exec")
    core.no_uncertain_prior_jobs()
    print("V4 public benchmark: MetricGAN+, DeepFilterNet, fixed global blends")
    if not args.execute:
        print("DRY RUN; no submission")
        return
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise RuntimeError("Kaggle CLI missing")
    legacy.require_quota(kaggle, 2)
    lock = core.OUTPUT / "active.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as f:
        import os
        f.write(str(os.getpid()))
    try:
        tag = "v4" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        # Shared remote-state tree preserves the existing duplicate-job guard.
        folder = core.OUTPUT / ("run_" + tag) / "round_0001" / "public"
        folder.mkdir(parents=True)
        job = ROOT / "kaggle_run" / "ralph_v4" / tag
        job.mkdir(parents=True)
        title = "Public Speech " + tag
        ref = "siyul1122/public-speech-" + tag
        src = src.replace('DATASET_SLUG = "PRIVATE_DATASET_SLUG"', 'DATASET_SLUG = "siyul1122/speech-data-private"')
        (job / "run.py").write_text(src, encoding="utf-8")
        core.json_write(job / "kernel-metadata.json", {"id":ref,"title":title,"code_file":"run.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,"enable_internet":True,"dataset_sources":["siyul1122/speech-data-private"],"competition_sources":[],"kernel_sources":[]})
        state = folder / "remote_state.json"
        core.json_write(state, {"ref":ref,"status":"SUBMIT_PENDING","kind":"v4_public"})
        try:
            core.poll_adaptive_kernel(kaggle, job, ref, folder, 80*60)
        except legacy.KaggleTerminalFailure:
            core.json_write(state, {"ref":ref,"status":"ERROR_TERMINAL","kind":"v4_public"})
            raise
        r = verify(folder)
        core.json_write(state, {"ref":ref,"status":"COMPLETE","kind":"v4_public"})
        print(json.dumps(r, indent=2))
        print("Public benchmark complete. This is an inference recipe, not a trained local checkpoint.")
    finally:
        lock.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
