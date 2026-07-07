# -*- coding: utf-8 -*-
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v1_freeze_report import build_report, load_backbone_summary


class TestV1FreezeReport(unittest.TestCase):
    def test_load_backbone_summary_contract(self):
        summary = load_backbone_summary(ROOT)

        self.assertEqual(set(summary["backbone"]), {"vitb32", "vitb16", "vitl14"})
        required = {
            "backbone",
            "v0_fp_at_b5cov",
            "v1_fp_at_b5cov",
            "b5_fp_at_defer20",
            "v1_pick_recall_at_b5cov",
            "b5_pick_recall_at_defer20",
            "v1_beats_b5family_share",
            "v1_max_coverage",
            "b5family_max_coverage",
        }
        self.assertTrue(required.issubset(set(summary.columns)))

        vitb32 = summary[summary["backbone"] == "vitb32"].iloc[0]
        self.assertLess(vitb32["v1_fp_at_b5cov"], vitb32["v0_fp_at_b5cov"])

        strong = summary[summary["backbone"].isin(["vitb16", "vitl14"])]
        self.assertTrue((strong["v1_fp_at_b5cov"] < strong["b5_fp_at_defer20"]).all())

    def test_build_report_writes_formal_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            artifacts = build_report(ROOT, out_dir)

            summary_csv = artifacts["summary_csv"]
            headline_csv = artifacts["headline_csv"]
            claim_csv = artifacts["claim_csv"]
            report_md = artifacts["report_md"]

            self.assertTrue(summary_csv.exists())
            self.assertTrue(headline_csv.exists())
            self.assertTrue(claim_csv.exists())
            self.assertTrue(report_md.exists())

            summary = pd.read_csv(summary_csv)
            self.assertEqual(len(summary), 3)

            report_text = report_md.read_text(encoding="utf-8")
            self.assertIn("Formal V1 frozen protocol", report_text)
            self.assertIn("post-hoc method selection", report_text)
            self.assertIn("same-coverage recall trade-off", report_text)


if __name__ == "__main__":
    unittest.main()
