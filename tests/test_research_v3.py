import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_v3 as policy

class ResearchV3Tests(unittest.TestCase):
    def setUp(self):
        self.old = (ROOT / "src/kaggle_train_v3.py").read_text(encoding="utf-8")
        self.base = policy.upgrade(self.old, "ResidualPhaseUNet")

    def test_upgrade_all_families(self):
        for path, name in [("kaggle_train.py", "SpectrogramUNet"), ("kaggle_train_v2.py", "ComplexMaskUNet"), ("kaggle_train_v3.py", "ResidualPhaseUNet")]:
            old = (ROOT / "src" / path).read_text(encoding="utf-8")
            new = policy.upgrade(old, name)
            compile(new, path, "exec")
            self.assertEqual(new, policy.upgrade(new, name))
            _, before = policy.nodes(old)
            _, after = policy.nodes(new)
            for key in ("evaluate", "SpeechPairs", "ensure_pesq", "locate_data_root"):
                self.assertEqual(policy.dump(before[key]), policy.dump(after[key]))
            self.assertTrue(policy.HOOKS <= set(after))
    def test_optimizer_and_augmentation(self):
        new = self.base.replace("lr=0.00035", "lr=0.0002").replace("return noisy, clean", "return noisy * 0.9, clean * 0.9")
        self.assertIn("build_optimizer", policy.validate(new, self.base, "ResidualPhaseUNet"))
        self.assertIn("augment_train", policy.validate(new, self.base, "ResidualPhaseUNet"))
    def test_new_neural_class(self):
        new = self.base.replace("def build_optimizer", "class ExtraBlock(nn.Module):\n    def forward(self, x):\n        return x.relu()\n\ndef build_optimizer")
        self.assertIn("ExtraBlock", policy.validate(new, self.base, "ResidualPhaseUNet"))
    def test_metric_immutable(self):
        new = self.base.replace('reference, wave, "wb"', 'reference, wave, "nb"')
        with self.assertRaises(ValueError):
            policy.validate(new, self.base, "ResidualPhaseUNet")
    def test_validation_immutable(self):
        new = self.base.replace('SpeechPairs(root, "validation")', 'SpeechPairs(root, "train")')
        with self.assertRaises(ValueError):
            policy.validate(new, self.base, "ResidualPhaseUNet")
    def test_no_file_access(self):
        new = self.base.replace("return noisy, clean", 'open("secret"); return noisy, clean')
        with self.assertRaises(ValueError):
            policy.validate(new, self.base, "ResidualPhaseUNet")
    def test_no_dependency_shadow(self):
        new = self.base.replace("def build_optimizer", "import math as np\n\ndef build_optimizer")
        with self.assertRaises(ValueError):
            policy.validate(new, self.base, "ResidualPhaseUNet")
    def test_no_runner_changes(self):
        new = self.base.replace('len(val_set) != 164', 'len(val_set) != 1')
        with self.assertRaises(ValueError):
            policy.validate(new, self.base, "ResidualPhaseUNet")
    def test_noop(self):
        with self.assertRaises(ValueError):
            policy.validate(self.base, self.base, "ResidualPhaseUNet")
