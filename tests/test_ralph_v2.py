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
                self.assertEqual(ref, "siyul1122/adaptive-speech-v2-v220261121111111-job-2")
                self.assertEqual(meta["title"], "Adaptive Speech V2 v220261121111111 job 2")
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

    def test_kaggle_status_transient_404_is_retried_without_resubmitting(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "job"
            folder.mkdir()
            result_dir = root / "output"
            result_dir.mkdir()
            statuses = [
                SimpleNamespace(returncode=1, stdout="", stderr="Not Found"),
                SimpleNamespace(returncode=0, stdout="KernelWorkerStatus.RUNNING", stderr=""),
                SimpleNamespace(returncode=0, stdout="KernelWorkerStatus.COMPLETE", stderr=""),
            ]
            pushes = []
            def command(argv, **kwargs):
                if "push" in argv:
                    pushes.append(argv)
                if "output" in argv:
                    (result_dir / "result.json").write_text(
                        '{"best_validation_pesq_wb": 2.5}', encoding="utf-8"
                    )
                return ""
            with patch.object(adaptive.legacy, "command", side_effect=command), \
                 patch.object(adaptive.subprocess, "run", side_effect=statuses) as fake_status, \
                 patch.object(adaptive.time, "sleep"):
                result = adaptive.poll_adaptive_kernel(
                    "kaggle", folder, "owner/ref", result_dir,
                    max_wait_seconds=30, sleep_seconds=0,
                )
            self.assertEqual(fake_status.call_count, 3)
            self.assertEqual(len(pushes), 1)
            self.assertEqual(result["best_validation_pesq_wb"], 2.5)

    def test_kaggle_status_repeated_failure_blocks_new_submission(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            statuses = [
                SimpleNamespace(returncode=1, stdout="", stderr="HTTP 404")
                for _ in range(5)
            ]
            pushes = []
            def command(argv, **kwargs):
                if "push" in argv:
                    pushes.append(argv)
                return ""
            with patch.object(adaptive.legacy, "command", side_effect=command), \
                 patch.object(adaptive.subprocess, "run", side_effect=statuses), \
                 patch.object(adaptive.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "unknown after 5"):
                    adaptive.poll_adaptive_kernel(
                        "kaggle", result_dir, "owner/ref", result_dir,
                        max_wait_seconds=30, sleep_seconds=0,
                    )
            self.assertEqual(len(pushes), 1)

    def test_reconcile_unknown_kaggle_state_never_marks_safe(self):
        import reconcile_ralph_v2 as recon
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            state = folder / "remote_state.json"
            info = {"ref": "owner/kernel", "status": "SUBMIT_PENDING"}
            adaptive.json_write(state, info)
            with patch.object(recon, "status", return_value=(None, "HTTP 404")):
                recon.reconcile("kaggle", state, info, resolve=True)
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "SUBMIT_PENDING")

    def test_reconcile_complete_needs_full_164_file_evidence(self):
        import reconcile_ralph_v2 as recon
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = root / "smoke"
            smoke.mkdir()
            state = smoke / "remote_state.json"
            info = {"ref": "owner/kernel", "status": "SUBMIT_PENDING"}
            adaptive.json_write(state, info)
            # An API COMPLETE without downloaded verified result is insufficient.
            with patch.object(recon, "status", return_value=("COMPLETE", "complete")), \
                 patch.object(recon.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                recon.reconcile("kaggle", state, info, resolve=True)
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8"))["status"], "SUBMIT_PENDING"
            )
            (smoke / "result.json").write_text(json.dumps({
                "best_validation_pesq_wb": 1.9, "validation_files": 164,
                "epochs_completed": 2,
            }), encoding="utf-8")
            (smoke / "best_model.pt").write_bytes(b"fake checkpoint")
            (smoke / "validation_per_file.csv").write_text(
                "index,pesq_wb\n" + "".join(f"{i},1.9\n" for i in range(164)),
                encoding="utf-8"
            )
            with patch.object(recon, "status", return_value=("COMPLETE", "complete")), \
                 patch.object(recon.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                recon.reconcile("kaggle", state, info, resolve=True)
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8"))["status"], "COMPLETE"
            )

    def test_legacy_wrong_slug_is_resolved_only_from_owned_kaggle_list(self):
        import reconcile_ralph_v2 as recon
        from types import SimpleNamespace
        old = "siyul1122/speech-enhancement-auto-v220261121052503-j0001"
        actual = "siyul1122/adaptive-speech-v2-v220261121052503-job-1"
        owned = (
            "ref       title\n---\n" +
            actual + "     Adaptive Speech V2 v220261121052503 job 1\n"
        )
        with patch.object(
            recon.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout=owned, stderr="")
        ) as mocked:
            found, details = recon.find_actual_ref("kaggle", old)
        self.assertEqual(found, actual)
        self.assertIn("Verified exact", details)
        self.assertEqual(mocked.call_args.args[0], ["kaggle", "kernels", "list", "--mine"])
        with patch.object(
            recon.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="ref  title\n", stderr="")
        ):
            found, _ = recon.find_actual_ref("kaggle", old)
        self.assertIsNone(found)

    def test_legacy_reconcile_read_only_uses_canonical_ref_without_mutating(self):
        import reconcile_ralph_v2 as recon
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "remote_state.json"
            old = "siyul1122/speech-enhancement-auto-v220261121052503-j0001"
            actual = "siyul1122/adaptive-speech-v2-v220261121052503-job-1"
            info = {"ref": old, "status": "SUBMIT_PENDING"}
            adaptive.json_write(path, info)
            def mocked_status(kaggle, ref):
                if ref == old:
                    return None, "Permission kernels.get denied"
                if ref == actual:
                    return "RUNNING", "KernelWorkerStatus.RUNNING"
                raise AssertionError("Unexpected ref: " + ref)
            with patch.object(recon, "status", side_effect=mocked_status), \
                 patch.object(recon, "find_actual_ref", return_value=(actual, "unique")):
                recon.reconcile("kaggle", path, info, resolve=False)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"ref": old, "status": "SUBMIT_PENDING"}
            )

    def test_recovered_completed_smoke_skips_duplicate_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run_v220261121052503"
            round_dir = run / "round_0001"
            smoke = round_dir / "smoke"
            smoke.mkdir(parents=True)
            bestsource = run / "families" / "residual_phase" / "best_source.py"
            bestsource.parent.mkdir(parents=True)
            baseline = (ROOT / "src" / "kaggle_train_v3.py").read_text(encoding="utf-8")
            candidate = baseline.replace(
                "phase_delta = 0.5 * torch.tanh",
                "phase_delta = 0.4 * torch.tanh",
            )
            self.assertNotEqual(candidate, baseline)
            bestsource.write_text(baseline, encoding="utf-8")
            (round_dir / "candidate.py").write_text(candidate, encoding="utf-8")
            adaptive.json_write(smoke / "remote_state.json", {
                "ref": "siyul1122/adaptive-speech-v2-v220261121052503-job-1",
                "status": "COMPLETE", "epochs": 2
            })
            adaptive.json_write(smoke / "result.json", {
                "best_validation_pesq_wb": 1.8654,
                "experiment": "V3_RESIDUAL_PHASE",
                "model": "ResidualPhaseUNet",
                "validation_files": 164,
                "epochs_completed": 2,
            })
            (smoke / "best_model.pt").write_bytes(b"test")
            (smoke / "validation_per_file.csv").write_text(
                "index,pesq_wb\n" +
                "".join(f"{i},1.8654\n" for i in range(164)), encoding="utf-8"
            )
            state = {
                "residual_phase": {
                    "checkpoint": None, "score": float("-inf"),
                    "source": baseline,
                    "variant": "V3_RESIDUAL_PHASE"
                }
            }
            with patch.object(adaptive, "OUTPUT", root):
                reused = adaptive.recover_completed_smoke(state)
                self.assertIsNotNone(reused)
                self.assertEqual(reused["source"], candidate)
                self.assertEqual(reused["family"], "residual_phase")
                # A known completed smoke must not be selected if a full
                # run was even prepared, since its status may be ambiguous.
                (round_dir / "full").mkdir()
                self.assertIsNone(adaptive.recover_completed_smoke(state))

    def test_smoke_requires_confirmed_completion_not_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            smoke = Path(temp) / "run_test" / "round_0001" / "smoke"
            smoke.mkdir(parents=True)
            adaptive.json_write(
                smoke / "remote_state.json",
                {"ref": "owner/smoke", "status": "SUBMIT_PENDING", "epochs": 2}
            )
            with patch.object(adaptive, "OUTPUT", Path(temp)):
                self.assertIsNone(adaptive.recover_completed_smoke({}))

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
