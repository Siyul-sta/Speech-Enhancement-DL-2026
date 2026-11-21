"""Offline test suite for the adaptive architecture-switching Ralph V2 controller.

Run: py -m unittest discover -s tests -p "test_ralph_v2.py" -v
No Kaggle GPU submissions, no Codex execution, no private data needed.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("ralph_v2", ROOT / "tools" / "ralph_v2.py")
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


class AdaptiveRalphTests(unittest.TestCase):
    def test_three_distinct_model_families(self):
        self.assertEqual(
            adaptive.FAMILIES, ("residual_phase", "magnitude", "complex")
        )
        models = [adaptive.TEMPLATES[x][1] for x in adaptive.FAMILIES]
        self.assertEqual(len(set(models)), 3)

    def test_switch_trigger_only_at_threshold(self):
        self.assertEqual(adaptive.choose_family(0, 1, 2), (0, 1))
        self.assertEqual(adaptive.choose_family(0, 2, 2), (1, 0))
        self.assertEqual(adaptive.choose_family(1, 2, 2), (2, 0))
        self.assertEqual(adaptive.choose_family(2, 2, 2), (0, 0))

    def test_unchanged_and_evaluation_edits_rejected(self):
        before = (ROOT / "src" / "kaggle_train_v2.py").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "No candidate"):
            adaptive.validate_source(before, before, "complex")
        candidate = before.replace(
            'pesq_function(SAMPLE_RATE, reference, wave, "wb")',
            'pesq_function(SAMPLE_RATE, reference, wave, "nb")'
        )
        self.assertNotEqual(candidate, before)
        with self.assertRaisesRegex(ValueError, "Protected"):
            adaptive.validate_source(candidate, before, "complex")

    def test_architecture_edit_allowed_and_imports_rejected(self):
        before = (ROOT / "src" / "kaggle_train_v3.py").read_text(encoding="utf-8")
        good = before.replace(
            "phase_delta = 0.5 * torch.tanh",
            "phase_delta = 0.4 * torch.tanh",
        )
        self.assertNotEqual(good, before)
        names = adaptive.validate_source(good, before, "residual_phase")
        self.assertIn("ResidualPhaseUNet", names)
        unsafe = before.replace(
            "phase_delta = 0.5 * torch.tanh",
            "import socket\n        phase_delta = 0.4 * torch.tanh",
        )
        with self.assertRaisesRegex(ValueError, "import/global"):
            adaptive.validate_source(unsafe, before, "residual_phase")

    def test_complex_and_residual_template_syntax(self):
        for family in adaptive.FAMILIES:
            template, model, variant = adaptive.TEMPLATES[family]
            source = (ROOT / template).read_text(encoding="utf-8")
            import ast
            ast.parse(source)
            self.assertIn(f"class {model}(", source)
            self.assertIn('"validation_files": len(val_set)', source)
            self.assertIn('pesq_function(SAMPLE_RATE, reference, wave, "wb")', source)
            self.assertIn("length=noisy.shape[-1]", source)

    def test_kaggle_jobs_are_private_and_separate_from_v1(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(adaptive, "ROOT", Path(temp)):
                source = (ROOT / "src" / "kaggle_train_v3.py").read_text(encoding="utf-8")
                args = SimpleNamespace(
                    username="siyul1122", dataset="siyul1122/speech-data-private",
                    batch_size=4,
                )
                folder, ref = adaptive.package_candidate(
                    source=source, variant="V3_RESIDUAL_PHASE", args=args,
                    run_tag="v220261121111111", job_id=2, epochs=2,
                )
                meta = json.loads(
                    (folder / "kernel-metadata.json").read_text(encoding="utf-8")
                )
                self.assertTrue(meta["is_private"] == "true")
                self.assertEqual(meta["dataset_sources"], [args.dataset])
                self.assertIn("kaggle_run", folder.parts)
                self.assertIn("ralph_v2", folder.parts)
                self.assertNotIn("ralph_r02", str(folder))
                self.assertEqual(ref, meta["id"])
                self.assertIn("'epochs': 2", (folder / "run.py").read_text(encoding="utf-8"))

    def test_completed_score_requires_checkpoint_and_all_164_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "result.json").write_text(
                json.dumps({
                    "best_validation_pesq_wb": 2.5,
                    "validation_files": 164, "epochs_completed": 25
                }), encoding="utf-8"
            )
            (folder / "best_model.pt").write_bytes(b"placeholder")
            (folder / "validation_per_file.csv").write_text(
                "index,pesq_wb\n" + "".join(f"{i},2.5\n" for i in range(164)),
                encoding="utf-8",
            )
            self.assertIsNotNone(adaptive.valid_full_result(folder))
            (folder / "validation_per_file.csv").write_text(
                "index,pesq_wb\n0,2.5\n", encoding="utf-8"
            )
            self.assertIsNone(adaptive.valid_full_result(folder))

    def test_smoke_results_never_promotable(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "result.json").write_text(
                json.dumps({
                    "best_validation_pesq_wb": 4.0,
                    "validation_files": 164, "epochs_completed": 2
                }), encoding="utf-8"
            )
            (folder / "best_model.pt").write_bytes(b"placeholder")
            (folder / "validation_per_file.csv").write_text(
                "index,pesq_wb\n" + "".join(f"{i},4.0\n" for i in range(164)),
                encoding="utf-8",
            )
            self.assertIsNone(adaptive.valid_full_result(folder))
            self.assertIsNotNone(adaptive.valid_full_result(folder, min_epochs=1))

    def test_uncertain_previous_remote_submission_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run_test" / "round_0001" / "full" / "remote_state.json"
            adaptive.json_write(
                path, {"ref": "siyul1122/speech-enh-ralph-test", "status": "SUBMIT_PENDING"}
            )
            with patch.object(adaptive, "OUTPUT", Path(temp)):
                with self.assertRaisesRegex(RuntimeError, "Unresolved previous Kaggle job"):
                    adaptive.no_uncertain_prior_jobs()
                adaptive.json_write(path, {"ref": "test", "status": "COMPLETE"})
                adaptive.no_uncertain_prior_jobs()

    def test_preexisting_modified_v1_trainer_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = root / "src" / "kaggle_train.py"
            trainer.parent.mkdir(parents=True)
            trainer.write_text("modified historical champion source", encoding="utf-8")
            with patch.object(adaptive, "ROOT", root), patch.object(
                adaptive.legacy, "current_modified_paths", return_value={"src/kaggle_train.py"}
            ):
                snapshot = adaptive.snapshot_existing_tracked_edits()
                self.assertEqual(set(snapshot), {"src/kaggle_train.py"})
                adaptive.assert_tracked_unchanged(snapshot)
                self.assertEqual(trainer.read_text(encoding="utf-8"),
                                 "modified historical champion source")
                trainer.write_text("unexpected external edit", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Tracked working-tree changes"):
                    adaptive.assert_tracked_unchanged(snapshot)

    def test_unrelated_tracked_edits_are_still_rejected(self):
        with patch.object(
            adaptive.legacy, "current_modified_paths",
            return_value={"src/kaggle_train.py", "README.md"}
        ):
            with self.assertRaisesRegex(RuntimeError, "Unrelated tracked changes"):
                adaptive.snapshot_existing_tracked_edits()
        with patch.object(adaptive.legacy, "current_modified_paths", return_value=set()):
            self.assertEqual(adaptive.snapshot_existing_tracked_edits(), {})
            adaptive.assert_tracked_unchanged({})

    def test_result_source_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = "print('model')\n"
            record = {
                "score": 2.4, "variant": "V3_RESIDUAL_PHASE",
                "checkpoint": str(root / "model.pt"),
                "origin": "test", "source_hash": adaptive.hash_source(source),
                "source": source,
            }
            adaptive.save_state(root, {"residual_phase": record}, record)
            info = json.loads(
                (root / "families" / "residual_phase" / "best.json").read_text(encoding="utf-8")
            )
            self.assertEqual(info["score"], 2.4)
            self.assertNotIn("source", info)
            self.assertEqual(
                (root / "families" / "residual_phase" / "best_source.py").read_text(encoding="utf-8"),
                source,
            )


if __name__ == "__main__":
    unittest.main()
