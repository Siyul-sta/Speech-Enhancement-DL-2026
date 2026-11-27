"""V5.7.1 CPU controller contracts; actual weights run via v571_preflight.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"src")]
import v571_preflight as p
import v57_finalize as final
import v57_public_assets as assets
import ralph_v4 as v4

class V571Tests(unittest.TestCase):
    def test_missing_stale_and_failed_report(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"report.json"
            with self.assertRaises(RuntimeError): p.require_report(path)
            report={"protocol":"v571-public-cpu-v1","status":"PASS","identity":{"hash":"old"},"families":{x:{"status":"PASS"} for x in p.FAMILIES},"mixed_ensemble":{"status":"PASS"}}
            path.write_text(json.dumps(report),encoding="utf-8")
            with patch.object(p,"identity",return_value={"hash":"new"}):
                with self.assertRaises(RuntimeError): p.require_report(path)
            report["identity"]={"hash":"new"};report["families"]["metricgan"]["status"]="FAIL"
            path.write_text(json.dumps(report),encoding="utf-8")
            with patch.object(p,"identity",return_value={"hash":"new"}):
                with self.assertRaises(RuntimeError): p.require_report(path)

    def test_recipe_exact_weights_and_validation(self):
        self.assertEqual(final.recipe_weights("cmgan:0.85"),[("cmgan",.85),("noisy_identity",1-.85)])
        self.assertEqual(final.recipe_weights("blend:metricgan:sepformer_dns:0.25"),[("metricgan",.25),("sepformer_dns",.75)])
        self.assertEqual(final.recipe_weights("mean3:cmgan:metricgan:sepformer_wham"),[("cmgan",1/3),("metricgan",1/3),("sepformer_wham",1/3)])
        for recipe in ("cmgan:nan","cmgan:1.5","blend:metricgan:metricgan:.5","deepfilter:1"):
            with self.assertRaises(ValueError): final.recipe_weights(recipe)

    def test_shared_archives_compile(self):
        source=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
        public=(ROOT/"src/kaggle_v4_public.py").read_text(encoding="utf-8")
        combined=v4.public_with_external(public,source)
        compile(combined,"public.py","exec")
        for name in ("class CMGANTransfer","class MetricGANTransfer","class SepformerTransfer","def archive_speechbrain_config","def export_recipe_expert"):
            self.assertIn(name,combined)

    def test_bounded_sepformer_roundoff(self):
        import numpy as np
        reference=np.zeros(48000,dtype=np.float32)
        reference[0]=8.
        actual=reference.copy()
        actual[10:20]=8.7e-5
        with self.assertRaises(RuntimeError): p.waveform_equivalence(actual,reference)
        stats=p.waveform_equivalence(actual,reference,True)
        self.assertEqual(stats["numeric_comparison"],"bounded_sepformer_fp32_roundoff")
        self.assertLessEqual(stats["pcm16_max_lsb_error"],4)
        self.assertGreater(stats["reference_clipped_fraction"],0)
        actual[10]=.001
        with self.assertRaises(RuntimeError): p.waveform_equivalence(actual,reference,True)
        actual[10]=np.nan
        with self.assertRaises(ValueError): p.waveform_equivalence(actual,reference,True)

    def test_fixture(self):
        import numpy as np
        x,y=p.fixture()
        self.assertEqual(x.shape,(48000,))
        self.assertEqual(y.shape,(48000,))
        self.assertTrue(np.isfinite(x).all())
        self.assertFalse(np.array_equal(x,y))

if __name__=="__main__": unittest.main()
