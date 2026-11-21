"""No GPU submission: structural and packaging tests for V2 complex U-Net.

Run: py -m unittest discover -s tests -p "test_kaggle_v2.py" -v
Local torch is optional; the model self-test is run only if installed.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src" / "kaggle_train_v2.py"
RUNNER = ROOT / "tools" / "kaggle_v2.py"
spec = importlib.util.spec_from_file_location("kaggle_v2", RUNNER)
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


class V2Tests(unittest.TestCase):
    def test_trainer_syntax_and_clean_manifest_separation(self):
        code = TRAINER.read_text(encoding="utf-8")
        ast.parse(code)
        self.assertIn('train_set = SpeechPairs(root, "train")', code)
        self.assertIn('val_set = SpeechPairs(root, "validation")', code)
        self.assertIn('if index != len(loader.dataset):', code)
        self.assertIn('pesq_function(SAMPLE_RATE, reference, wave, "wb")', code)
        self.assertIn('"validation_files": len(val_set)', code)

    def test_complex_mask_is_not_magnitude_only(self):
        code = TRAINER.read_text(encoding="utf-8")
        self.assertIn('class ComplexMaskUNet(nn.Module):', code)
        self.assertIn('mask_real = 1.0 + torch.tanh', code)
        self.assertIn('mask_imag = torch.tanh', code)
        self.assertIn('torch.complex(mask_real, mask_imag)', code)
        self.assertIn('length=noisy.shape[-1]', code)
        self.assertIn('nn.init.zeros_(self.head.weight)', code)

    def test_loss_variants_change_single_hypothesis(self):
        code = TRAINER.read_text(encoding="utf-8")
        self.assertIn('"V2_CRM_PHASE", "V2_CRM_SISDR"', code)
        self.assertIn('if variant == "V2_CRM_SISDR":', code)
        self.assertIn('torch.view_as_real(a - b)', code)
        self.assertIn('CosineAnnealingLR(', code)

    def test_package_job_is_unique_and_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(v2, "ROOT", Path(tmp)):
                folder, ref = v2.package_job(
                    "siyul1122", "siyul1122/speech-data-private",
                    "V2_CRM_PHASE", 2, 4, 7, "20261121123456"
                )
                metadata = json.loads(
                    (folder / "kernel-metadata.json").read_text(encoding="utf-8")
                )
                code = (folder / "run.py").read_text(encoding="utf-8")
            self.assertEqual(metadata["id"], ref)
            self.assertEqual(metadata["is_private"], "true")
            self.assertEqual(metadata["dataset_sources"], ["siyul1122/speech-data-private"])
            self.assertIn("'epochs': 2", code)
            self.assertIn("'id': 'V2_CRM_PHASE'", code)
            expected_slug = re.sub(
                r"[^a-z0-9]+", "-", metadata["title"].lower()
            ).strip("-")
            self.assertEqual(ref.split("/", 1)[1], expected_slug)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable locally")
    def test_cpu_model_forward_backward(self):
        p = subprocess.run(
            [sys.executable, str(TRAINER), "--self-test"],
            cwd=ROOT, text=True, capture_output=True, timeout=90,
        )
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("SELF_TEST_PASSED", p.stdout)


if __name__ == "__main__":
    unittest.main()
