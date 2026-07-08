import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_zs_temp_selective_baseline import (
    calibrate_temperature,
    load_classes,
    run,
    text_prototypes,
)


class TestRunZsTempSelectiveBaseline(unittest.TestCase):
    def test_calibrate_temperature_returns_grid_value(self):
        logits = np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
        labels = np.array(["a", "b", "a"])
        temp, nll = calibrate_temperature(logits, labels, ["a", "b"])
        self.assertIn(temp, {0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0})
        self.assertGreaterEqual(nll, 0.0)

    def test_text_prototypes_load_dynamic_class_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.npz"
            np.savez(
                path,
                image_feats=np.eye(2),
                classes=np.array(["mature", "immature"]),
                splits=np.array(["train", "test"]),
                text_mature=np.array([[1.0, 0.0], [1.0, 0.0]]),
                text_immature=np.array([[0.0, 1.0], [0.0, 1.0]]),
            )
            with np.load(path, allow_pickle=True) as data:
                classes = load_classes(data)
                protos = text_prototypes(data, classes)
            self.assertEqual(classes, ["immature", "mature"])
            self.assertEqual(protos.shape, (2, 2))

    def test_run_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            npz = tmp / "features_toy_vitb16.npz"
            out = tmp / "summary.csv"
            np.savez(
                npz,
                image_feats=np.array(
                    [
                        [1.0, 0.0],
                        [0.9, 0.1],
                        [0.0, 1.0],
                        [0.1, 0.9],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    dtype=float,
                ),
                classes=np.array(["mature", "mature", "immature", "immature", "mature", "immature"]),
                splits=np.array(["train", "train", "train", "train", "test", "test"]),
                text_mature=np.array([[1.0, 0.0], [1.0, 0.0]]),
                text_immature=np.array([[0.0, 1.0], [0.0, 1.0]]),
            )

            rows = run(npz, out, dataset="toy", backbone="toy")

            self.assertTrue(out.exists())
            self.assertEqual(len(rows), 19)
            self.assertEqual(rows[0]["baseline"], "ZS-hard")


if __name__ == "__main__":
    unittest.main()