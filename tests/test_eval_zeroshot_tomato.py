import unittest

import numpy as np

from scripts.eval_zeroshot_tomato import class_metric_rows, find_collapsed_class


class TestTomatoZeroShotSummary(unittest.TestCase):
    def test_detects_worst_collapsed_class(self):
        classes = ["immature", "mature", "turning"]
        true = np.array(
            ["immature"] * 3
            + ["mature"] * 4
            + ["turning"] * 3
        )
        pred = np.array(
            ["immature", "immature", "turning"]
            + ["turning", "turning", "immature", "immature"]
            + ["turning", "immature", "turning"]
        )

        rows = class_metric_rows(true, pred, classes)
        collapsed = find_collapsed_class(rows)

        self.assertEqual(collapsed["class"], "mature")
        self.assertEqual(collapsed["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
