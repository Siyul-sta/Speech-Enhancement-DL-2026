"""Colab adapter tests; no network, no GPU allocation, no private audio."""
import ast
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"src")]
import v71_colab_worker as w
import v7_search as search
import ralph_v6 as ctl

class ColabTests(unittest.TestCase):
    def test_source_only_path_and_runtime_adaptations(self):
        base=search.source()
        cfg=ctl.config("mp_dns","train",search.digest(base),{},epochs=1)
        src=w.colab_source(base,cfg,"/content/drive/MyDrive/output/trial_00/train","owner/data","/content/bundle/data","/content/bundle/parent/best_model.pt","/content/v71_public_cache")
        def nodes(s):return {n.name:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
        a,b=nodes(base),nodes(src)
        for name in ("evaluate","v6_eval","v6_loaders","v6_loss","v6_step","v6_split","v6_mix"):
            self.assertEqual(a[name],b[name],name)
        self.assertIn("Colab GPU required",src)
        self.assertIn("Colab parent checkpoint hash mismatch",src)
        self.assertIn("/content/v71_public_cache",src)
    def test_fail_closed_on_changed_source(self):
        base=search.source().replace('if Path("/kaggle").exists() and device.type!="cuda": raise RuntimeError("GPU required")',"pass")
        cfg=ctl.config("mp_dns","train",search.digest(base),{},epochs=1)
        with self.assertRaises(ValueError):
            w.colab_source(base,cfg,"/content/out","owner/data","/content/data","/content/parent","/content/cache")
if __name__=="__main__":unittest.main()
