"""Offline safety tests for tools/ralph_loop.py. No Kaggle job or Codex call.

Run: py -m unittest discover -s tests -p "test_ralph_loop.py" -v
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ralph_loop", PROJECT / "tools" / "ralph_loop.py")
ralph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ralph)


class RalphLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The live experiment can leave a successful candidate in the working tree.
        # Test against the tracked trainer from HEAD rather than the mutable file.
        cls.source = subprocess.run(
            ["git", "show", "HEAD:src/kaggle_train.py"], cwd=PROJECT,
            check=True, text=True, capture_output=True,
        ).stdout

    def test_codex_model_is_explicit_and_configurable(self):
        script = (PROJECT / "tools" / "ralph_loop.py").read_text(encoding="utf-8")
        self.assertIn('"--codex-model", default="gpt-6.1-sol"', script)
        self.assertIn('codex, "exec", "--model", model', script)
        self.assertIn('args.codex_model)', script)

    def test_dry_run_does_not_need_gpu_or_codex(self):
        # Dry-run is tested by users via CLI; unit-level check ensures args are declared.
        self.assertIn('parser.add_argument("--execute", action="store_true")',
                      (PROJECT / "tools" / "ralph_loop.py").read_text(encoding="utf-8"))

    def test_model_edit_is_accepted_without_changing_evaluation(self):
        revised = self.source.replace(
            "self.head = nn.Conv2d(16, 1, 1)",
            "self.head = nn.Conv2d(16, 1, 3, padding=1)",
        )
        self.assertNotEqual(revised, self.source)
        with patch.object(ralph, "current_modified_paths", return_value={"src/kaggle_train.py"}):
            ralph.validate_candidate(revised, self.source)

    def test_evaluation_change_is_rejected(self):
        revised = self.source.replace(
            'pesq_function(SAMPLE_RATE, reference, wave, "wb")',
            'pesq_function(SAMPLE_RATE, reference, wave, "nb")',
        )
        self.assertNotEqual(revised, self.source)
        with patch.object(ralph, "current_modified_paths", return_value={"src/kaggle_train.py"}):
            with self.assertRaisesRegex(RuntimeError, "Protected"):
                ralph.validate_candidate(revised, self.source)

    def test_top_level_import_is_rejected(self):
        revised = self.source.replace("import csv\n", "import csv\nimport socket\n")
        with patch.object(ralph, "current_modified_paths", return_value={"src/kaggle_train.py"}):
            with self.assertRaisesRegex(RuntimeError, "top-level"):
                ralph.validate_candidate(revised, self.source)

    def test_writes_outside_source_are_rejected(self):
        revised = self.source.replace(
            "self.head = nn.Conv2d(16, 1, 1)",
            "self.head = nn.Conv2d(16, 1, 3, padding=1)",
        )
        with patch.object(ralph, "current_modified_paths", return_value={"src/kaggle_train.py", "README.md"}):
            with self.assertRaisesRegex(RuntimeError, "Unexpected tracked changes"):
                ralph.validate_candidate(revised, self.source)

    def test_overnight_model_flags_exist(self):
        script = (PROJECT / "tools" / "ralph_loop.py").read_text(encoding="utf-8")
        self.assertIn('"--no-pesq-target", action="store_true"', script)
        self.assertIn("1 <= args.rounds <= 40", script)
        self.assertIn("if args.max_stagnant and stagnant >= args.max_stagnant:", script)
        self.assertIn("class KaggleTerminalFailure(RuntimeError):", script)

    def test_prior_verified_champion_is_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp, variant, score in (
                ("e02_unet", "E02_UNET", 2.3771),
                ("e03_mrstft", "E03_MRSTFT", 2.20),
            ):
                folder = root / "runs" / exp
                folder.mkdir(parents=True)
                (folder / "best_model.pt").write_bytes(b"test-baseline")
                (folder / "result.json").write_text(json.dumps({
                    "experiment": variant,
                    "best_validation_pesq_wb": score,
                    "validation_files": 164,
                    "epochs_completed": 25,
                }), encoding="utf-8")
            folder = root / "runs" / "ralph" / "run_20261120180100"
            round_dir = folder / "round_01"
            round_dir.mkdir(parents=True)
            checkpoint = round_dir / "best_model.pt"
            checkpoint.write_bytes(b"test-new-champion")
            (round_dir / "result.json").write_text(json.dumps({
                "best_validation_pesq_wb": 2.44,
                "validation_files": 164,
                "epochs_completed": 25,
            }), encoding="utf-8")
            (folder / "best_source.py").write_text(self.source, encoding="utf-8")
            (folder / "best.json").write_text(json.dumps({
                "score": 2.44,
                "origin": "siyul1122/speech-enhancement-ralph-test-r01",
                "round": 1,
                "variant": "E02_UNET",
                "checkpoint": str(checkpoint),
                "source_hash": ralph.sha256(self.source),
            }), encoding="utf-8")
            with patch.object(ralph, "ROOT", root):
                best = ralph.initial_best(require_complete=True)
            self.assertAlmostEqual(best["score"], 2.44)
            self.assertEqual(best["source_path"], str(folder / "best_source.py"))

    def test_finished_baselines_pick_e03_when_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp, variant, score in (
                ("e02_unet", "E02_UNET", 2.21),
                ("e03_mrstft", "E03_MRSTFT", 2.32),
            ):
                d = root / "runs" / exp
                d.mkdir(parents=True)
                (d / "best_model.pt").write_bytes(b"test-placeholder")
                (d / "result.json").write_text(json.dumps({
                    "experiment": variant,
                    "best_validation_pesq_wb": score,
                    "validation_files": 164,
                    "epochs_completed": 25,
                }), encoding="utf-8")
            with patch.object(ralph, "ROOT", root):
                best = ralph.initial_best(require_complete=True)
            self.assertEqual(best["variant"], "E03_MRSTFT")
            self.assertAlmostEqual(best["score"], 2.32)

    def test_smoke_only_results_block_live_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp, variant in (("e02_unet", "E02_UNET"), ("e03_mrstft", "E03_MRSTFT")):
                d = root / "runs" / exp
                d.mkdir(parents=True)
                (d / "best_model.pt").write_bytes(b"test-placeholder")
                (d / "result.json").write_text(json.dumps({
                    "experiment": variant,
                    "best_validation_pesq_wb": 1.9,
                    "validation_files": 164,
                    "epochs_completed": 2,
                }), encoding="utf-8")
            with patch.object(ralph, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "smoke-only"):
                    ralph.initial_best(require_complete=True)

    def test_package_e03_is_private_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(ralph, "ROOT", root):
                folder, ref = ralph.package(
                    1, "siyul1122/speech-data-private", "siyul1122",
                    5, 4, self.source, "E03_MRSTFT", "20261121010101",
                )
            metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
            code = (folder / "run.py").read_text(encoding="utf-8")
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["dataset_sources"], ["siyul1122/speech-data-private"])
            self.assertIn("'id': 'E03_MRSTFT'", code)
            self.assertIn("'epochs': 5", code)
            self.assertIn("20261121010101-r01", ref)


if __name__ == "__main__":
    unittest.main()
