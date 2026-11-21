"""Compare 164-file V2 PESQ against V1 champion without running GPU.

Usage:
  py tools/compare_v2.py
Uses score-index alignment from the unchanged SpeechPairs validation manifest.
Validation clean is NEVER used for training; no private WAV files are opened.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def per_file(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 164:
        raise ValueError(f"Need 164 validation scores: {path} has {len(rows)}")
    result = {int(row["index"]): float(row["pesq_wb"]) for row in rows}
    if set(result) != set(range(164)):
        raise ValueError(f"Validation indices must be 0..163 exactly: {path}")
    return result


def candidates():
    pairs = [("V1 E02", ROOT / "runs" / "e02_unet"),
             ("V1 E03", ROOT / "runs" / "e03_mrstft")]
    for run in sorted((ROOT / "runs" / "ralph").glob("run_*")):
        record = run / "best.json"
        if not record.is_file():
            continue
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
            if int(data.get("round", 0)) <= 0:
                continue
            folder = Path(data["checkpoint"]).parent
            pairs.append((f"Ralph best ({run.name})", folder))
        except (ValueError, KeyError, TypeError, OSError):
            continue
    for run in sorted((ROOT / "runs" / "v2").glob("*")):
        if not run.is_dir():
            continue
        for folder in run.iterdir():
            if folder.is_dir():
                pairs.append((f"V2 {folder.name} ({run.name})", folder))
    for name, folder in pairs:
        result_path = folder / "result.json"
        score_path = folder / "validation_per_file.csv"
        if not result_path.is_file() or not score_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            scores = per_file(score_path)
            yield name, float(result["best_validation_pesq_wb"]), scores
        except (ValueError, KeyError, TypeError, OSError) as exc:
            print(f"Skipping invalid {name}: {exc}")


def main():
    records = list(candidates())
    if not records:
        print("No completed 164-file validation results found.")
        return
    # Use the strongest verified pre-V2 result as comparison baseline.
    original = [(n, score, rows) for n, score, rows in records
                if n.startswith("V1") or n.startswith("Ralph")]
    if not original:
        print("Missing prior V1 reference PESQ.")
        return
    reference = max(original, key=lambda x: x[1])
    print(f"Reference: {reference[0]}, PESQ-WB={reference[1]:.4f}")
    print("Model                                             PESQ    Delta    WinRate    Worst10%")
    for name, score, rows in sorted(records, key=lambda x: -x[1]):
        delta = [rows[i] - reference[2][i] for i in range(164)]
        win_rate = 100 * sum(d > 0 for d in delta) / 164
        weak = sorted(delta)[:16]
        print(f"{name[:48]:48} {score:7.4f} {statistics.mean(delta):+8.4f}"
              f" {win_rate:8.1f}% {statistics.mean(weak):+10.4f}")
    print("Worst10% = mean delta over the 16 clips most harmed vs V1 reference.")
    print("A positive overall PESQ does not imply no individual clips were harmed.")


if __name__ == "__main__":
    main()
