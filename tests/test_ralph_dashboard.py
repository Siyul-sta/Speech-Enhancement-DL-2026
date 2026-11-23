"""Offline tests for the local read-only speech research dashboard.

Run: py -m unittest discover -s tests -p "test_ralph_dashboard.py" -v
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ralph_dashboard", ROOT / "tools" / "ralph_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class DashboardTests(unittest.TestCase):
    def test_static_ui_has_expected_sections_and_safe_refresh(self):
        ui = dashboard.HTML
        for text in (
            "GLOBAL CHAMPION", "실험 진행 현황", "PESQ 실험 추이",
            "최근 실험 기록", "모델 계열별 최고 성능",
            "fetch(\"/api/status\"", "setInterval(sync,4000)"
        ):
            self.assertIn(text, ui)

    def test_no_run_is_safe_and_returns_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(dashboard, "ROOT", root), \
                 patch.object(dashboard, "OUTPUT", root / "runs" / "ralph_v2"):
                result = dashboard.collect()
            self.assertFalse(result["alive"])
            self.assertIsNone(result["best"])
            self.assertEqual(result["points"], [])
            self.assertEqual(len(result["families"]), 3)
            json.dumps(result, allow_nan=False)

    def test_champion_and_family_scores_use_real_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs" / "ralph_v2"
            run = output / "run_v220261121052503"
            run.mkdir(parents=True)
            past = root / "runs" / "ralph" / "run_20261120180226"
            past.mkdir(parents=True)
            (past / "best.json").write_text(
                json.dumps({"score": 2.4043}), encoding="utf-8"
            )
            (run / "global_best.json").write_text(
                json.dumps({"score": 2.329}), encoding="utf-8"
            )
            f = run / "families" / "residual_phase"
            f.mkdir(parents=True)
            (f / "best.json").write_text(
                json.dumps({"score": 2.329}), encoding="utf-8"
            )
            (run / "experiments.jsonl").write_text(
                json.dumps({
                    "round": 1, "family": "residual_phase",
                    "score": 2.329, "status": "COMPLETE",
                }) + "\n", encoding="utf-8"
            )
            with patch.object(dashboard, "ROOT", root), \
                 patch.object(dashboard, "OUTPUT", output):
                result = dashboard.collect()
            self.assertAlmostEqual(result["best"], 2.4043)
            self.assertAlmostEqual(result["points"][0]["score"], 2.329)
            self.assertAlmostEqual(next(
                x["score"] for x in result["families"]
                if x["id"] == "residual_phase"
            ), 2.329)

    def test_heartbeat_freshness_is_real_not_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs" / "ralph_v2"
            run = output / "run_v220261121123000"
            run.mkdir(parents=True)
            dashboard_state = run / "ui_state.json"
            with patch.object(dashboard, "ROOT", root), \
                 patch.object(dashboard, "OUTPUT", output):
                dashboard_state.write_text(
                    json.dumps({
                        "stage": "training", "round": 4,
                        "family": "complex",
                        "updated_at": (datetime.now(timezone.utc)
                                       - timedelta(minutes=10)).isoformat(),
                    }), encoding="utf-8"
                )
                stale = dashboard.collect()
                self.assertFalse(stale["alive"])
                self.assertEqual(stale["stage"], "offline")
                dashboard_state.write_text(
                    json.dumps({
                        "stage": "training", "round": 4,
                        "family": "complex",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }), encoding="utf-8"
                )
                live = dashboard.collect()
            self.assertTrue(live["alive"])
            self.assertEqual(live["stage"], "training")
            self.assertEqual(live["round"], 4)

    def test_reconciled_completed_job_is_visible_without_history_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs" / "ralph_v2"
            full = output / "run_v220261121054951" / "round_0005" / "full"
            full.mkdir(parents=True)
            (full / "result.json").write_text(json.dumps({
                "validation_files": 164, "best_validation_pesq_wb": 2.45,
                "experiment": "V3_RESIDUAL_PHASE"
            }), encoding="utf-8")
            (full / "remote_state.json").write_text(
                json.dumps({"status": "COMPLETE"}), encoding="utf-8"
            )
            with patch.object(dashboard, "ROOT", root), \
                 patch.object(dashboard, "OUTPUT", output):
                result = dashboard.collect()
            self.assertEqual(len(result["points"]), 1)
            self.assertEqual(result["points"][0]["status"], "RECOVERED_COMPLETE")
            self.assertEqual(result["points"][0]["round"], 5)

    def test_legacy_controller_pending_job_shows_verified_kaggle_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs" / "ralph_v2"
            trial = output / "run_v220261123064255" / "round_0001"
            (trial / "full").mkdir(parents=True)
            (trial / "candidate.py").write_text(
                "class ComplexMaskUNet(object):\n    pass\n", encoding="utf-8"
            )
            (trial / "full" / "remote_state.json").write_text(
                json.dumps({
                    "ref": "siyul1122/adaptive-speech-v2-v220261123064255-job-2",
                    "status": "SUBMIT_PENDING", "epochs": 25,
                }), encoding="utf-8"
            )
            with patch.object(dashboard, "ROOT", root), \
                 patch.object(dashboard, "OUTPUT", output), \
                 patch.object(dashboard, "kaggle_worker_status", return_value="RUNNING"), \
                 patch.object(dashboard, "kaggle_live_epoch", return_value={
                     "epoch": 7, "pesq": 2.204
                 }):
                result = dashboard.collect()
            self.assertFalse(result["alive"])
            self.assertTrue(result["remoteLive"])
            self.assertEqual(result["stage"], "training")
            self.assertEqual(result["round"], 1)
            self.assertEqual(result["family"], "complex")
            self.assertEqual(result["remote"], "RUNNING")
            self.assertEqual(result["epoch"], 7)
            self.assertEqual(result["total"], 25)
            self.assertEqual(result["livePESQ"], 2.204)

    def test_legacy_pending_remote_complete_does_not_fake_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs" / "ralph_v2"
            full = output / "run_v220261123064255" / "round_0001" / "full"
            full.mkdir(parents=True)
            (full / "remote_state.json").write_text(
                json.dumps({"ref": "siyul1122/kernel", "status": "SUBMIT_PENDING"}),
                encoding="utf-8"
            )
            with patch.object(dashboard, "ROOT", root), \
                 patch.object(dashboard, "OUTPUT", output), \
                 patch.object(dashboard, "kaggle_worker_status", return_value="COMPLETE"):
                result = dashboard.collect()
            self.assertFalse(result["alive"])
            self.assertFalse(result["remoteLive"])
            self.assertEqual(result["stage"], "recover")
            self.assertEqual(result["remote"], "COMPLETE")

    def test_legacy_pending_unknown_does_not_claim_idle_or_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs" / "ralph_v2"
            full = output / "run_v220261123064255" / "round_0001" / "full"
            full.mkdir(parents=True)
            (full / "remote_state.json").write_text(
                json.dumps({"ref": "siyul1122/kernel", "status": "SUBMIT_PENDING"}),
                encoding="utf-8"
            )
            with patch.object(dashboard, "ROOT", root), \
                 patch.object(dashboard, "OUTPUT", output), \
                 patch.object(dashboard, "kaggle_worker_status", return_value=None):
                result = dashboard.collect()
            self.assertFalse(result["alive"])
            self.assertFalse(result["remoteLive"])
            self.assertEqual(result["stage"], "checking")

    def test_cached_kaggle_status_is_read_only_and_throttled(self):
        from types import SimpleNamespace
        status = "owner/kernel"
        response = SimpleNamespace(returncode=0, stdout="KernelWorkerStatus.RUNNING", stderr="")
        with patch.object(dashboard.shutil, "which", return_value="kaggle"), \
             patch.object(dashboard.subprocess, "run", return_value=response) as invoked:
            dashboard._STATUS_CACHE.clear()
            self.assertEqual(dashboard.kaggle_worker_status(status), "RUNNING")
            self.assertEqual(dashboard.kaggle_worker_status(status), "RUNNING")
            self.assertEqual(invoked.call_count, 1)
            command = invoked.call_args.args[0]
            self.assertEqual(command, ["kaggle", "kernels", "status", status])
            self.assertNotIn("push", command)

    def test_unknown_and_sensitive_logs_are_not_exposed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "adaptive_test.log").write_text(
                "hello\nAuthorization: Bearer something\nGPU training started\n",
                encoding="utf-8",
            )
            with patch.object(dashboard, "OUTPUT", output):
                lines = dashboard.log_tail()
            self.assertIn("hello", lines)
            self.assertIn("GPU training started", lines)
            self.assertFalse(any("Authorization" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
