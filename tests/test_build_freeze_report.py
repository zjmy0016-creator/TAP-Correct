import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_freeze_report import build_report, load_backbone_summary


class TestBuildFreezeReport(unittest.TestCase):
    def test_load_backbone_summary_contract(self):
        summary = load_backbone_summary(ROOT)
        self.assertEqual(set(summary["backbone"]), {"vitb32", "vitb16", "vitl14"})
        required = {
            "backbone",
            "v1_fp_at_b5cov",
            "b5_fp_at_defer20",
            "v1_pick_recall_at_b5cov",
            "b5_pick_recall_at_defer20",
            "v1_beats_b5family_share",
            "v1_max_coverage",
            "b5family_max_coverage",
        }
        self.assertTrue(required.issubset(set(summary.columns)))

    def test_build_report_writes_formal_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = build_report(ROOT, Path(tmp))
            self.assertTrue(all(path.exists() for path in artifacts.values()))
            summary = pd.read_csv(artifacts["summary_csv"])
            self.assertEqual(len(summary), 3)
            report_text = artifacts["report_md"].read_text(encoding="utf-8")
            self.assertIn("Frozen protocol", report_text)
            self.assertIn("same-coverage recall gap", report_text)
            self.assertIn("Laboro Tomato", report_text)

    def test_load_backbone_summary_prefers_calibrated_b5_point_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe_root = root / "outputs" / "probe_512d_endpoint"
            for backbone in ("vitb32", "vitb16", "vitl14"):
                out_dir = probe_root / backbone
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [
                        {"defer": 0.0, "false_pick_rate": 0.20, "actual_coverage": 0.40, "pick_recall": 0.40, "variant": "V1_E_512D_endpoint"},
                        {"defer": 0.6, "false_pick_rate": 0.05, "actual_coverage": 0.80, "pick_recall": 0.80, "variant": "V1_E_512D_endpoint"},
                        {"defer": 0.2, "false_pick_rate": 0.12, "actual_coverage": 0.50, "pick_recall": 0.50, "variant": "B5family_argmax_margin"},
                        {"defer": 0.6, "false_pick_rate": 0.08, "actual_coverage": 0.40, "pick_recall": 0.40, "variant": "B5family_argmax_margin"},
                        {"defer": 0.2, "false_pick_rate": 0.22, "actual_coverage": 0.75, "pick_recall": 0.75, "variant": "B5_calibrated_argmax_margin"},
                    ]
                ).to_csv(out_dir / f"frontier_{backbone}_K16.csv", index=False)
            pd.DataFrame(
                [{"method": "V1_TAP_512D_endpoint", "backbone": "vitb32", "k": 16, "false_pick_rate": 0.1, "pick_precision": 0.9, "pick_recall": 0.8, "revisit_burden": 0.2}]
            ).to_csv(probe_root / "V1_headline_vitb32_K16.csv", index=False)
            summary = load_backbone_summary(root)
        self.assertTrue((summary["b5_coverage_at_defer20"] == 0.75).all())
        self.assertTrue((summary["b5_fp_at_defer20"] == 0.22).all())
        self.assertTrue((summary["b5_calibration_source"] == "calibration_margin_percentile").all())


if __name__ == "__main__":
    unittest.main()
