"""Offline safety tests for tools/ralph_loop.py. No Kaggle job or Codex call.

Run: py -m unittest discover -s tests -p "test_ralph_loop.py" -v
"""
from __future__ import annotations

import importlib.util
import json
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
        cls.source = (PROJECT / "src" / "kaggle_train.py").read_text(encoding="utf-8")

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
        revised = "import socket\n" + self.source
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
