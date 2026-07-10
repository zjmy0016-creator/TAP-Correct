import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.official_evaluation import build_official_evaluation


class TestOfficialEvaluation(unittest.TestCase):
    def test_official_evaluation_reproduces_headline_and_writes_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = build_official_evaluation(ROOT, Path(tmp), k=16, n_bootstrap=50, seed=7)
            metrics = pd.read_csv(artifacts["metrics_csv"]).iloc[0]
            self.assertAlmostEqual(0.091359, metrics["false_pick_rate"], places=6)
            self.assertAlmostEqual(0.908641, metrics["pick_precision"], places=6)
            self.assertAlmostEqual(0.805964, metrics["pick_recall"], places=6)
            self.assertAlmostEqual(0.191872, metrics["revisit_burden"], places=6)
            for key in ("decisions_npz", "metrics_csv", "ci_csv", "turning_csv", "paired_b5_csv", "summary_md"):
                self.assertTrue(artifacts[key].exists(), key)
            confidence_intervals = pd.read_csv(artifacts["ci_csv"])
            self.assertEqual(
                {"false_pick_rate", "pick_precision", "pick_recall", "revisit_burden"},
                set(confidence_intervals["metric"]),
            )


if __name__ == "__main__":
    unittest.main()
