import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_proto_adapter_selective_baseline import (
    PROTO_WEIGHT_GRID,
    calibrate_proto_weight,
    fused_proto_logits,
    run,
)


class TestRunProtoAdapterSelectiveBaseline(unittest.TestCase):
    def test_fused_proto_logits_adds_weighted_prototype_signal(self):
        zero_shot = np.array([[0.2, 0.1], [0.0, 0.3]])
        proto = np.array([[1.0, 0.0], [0.0, 1.0]])

        fused = fused_proto_logits(zero_shot, proto, proto_weight=2.0)

        np.testing.assert_allclose(fused, np.array([[2.2, 0.1], [0.0, 2.3]]))

    def test_calibrate_proto_weight_returns_grid_value(self):
        zero_shot = np.array([[0.2, 0.1], [0.1, 0.2]])
        proto = np.array([[2.0, 0.0], [0.0, 2.0]])
        labels = np.array(["a", "b"])

        weight, nll = calibrate_proto_weight(
            zero_shot_logits=zero_shot,
            proto_logits=proto,
            labels=labels,
            classes=["a", "b"],
        )

        self.assertIn(weight, PROTO_WEIGHT_GRID)
        self.assertGreaterEqual(nll, 0.0)

    def test_run_writes_episode_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            npz = tmp / "features_toy_vitb16.npz"
            out = tmp / "proto.csv"
            np.savez(
                npz,
                image_feats=np.array(
                    [
                        [1.0, 0.0],
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [0.0, 1.0],
                        [0.1, 0.9],
                        [0.2, 0.8],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ],
                    dtype=float,
                ),
                classes=np.array(
                    [
                        "mature",
                        "mature",
                        "mature",
                        "immature",
                        "immature",
                        "immature",
                        "mature",
                        "immature",
                    ]
                ),
                splits=np.array(["train", "train", "train", "train", "train", "train", "test", "test"]),
                text_mature=np.array([[1.0, 0.0], [1.0, 0.0]]),
                text_immature=np.array([[0.0, 1.0], [0.0, 1.0]]),
            )

            rows = run(
                npz_path=npz,
                out_csv=out,
                dataset="toy",
                backbone="toy",
                k=1,
                n_episodes=2,
                base_seed=11,
            )

            self.assertTrue(out.exists())
            self.assertEqual(len(rows), 38)
            self.assertEqual(rows[0]["baseline"], "ProtoAdapter-hard")


if __name__ == "__main__":
    unittest.main()
