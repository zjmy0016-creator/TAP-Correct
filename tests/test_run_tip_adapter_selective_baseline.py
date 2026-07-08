import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_tip_adapter_selective_baseline import (
    ALPHA_GRID,
    BETA_GRID,
    calibrate_alpha_beta,
    run,
    sample_k_support,
)


class TestRunTipAdapterSelectiveBaseline(unittest.TestCase):
    def test_sample_k_support_balances_classes(self):
        labels = np.array(["a", "a", "b", "b", "b"])
        pool = np.array([0, 1, 2, 3, 4])
        sampled = sample_k_support(pool, labels, ["a", "b"], k=1, seed=7)
        sampled_labels = labels[sampled]
        self.assertEqual(sorted(sampled_labels.tolist()), ["a", "b"])

    def test_calibrate_alpha_beta_returns_grid_values(self):
        logits = np.array([[2.0, 0.0], [0.0, 2.0]])
        cache_logits = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array(["a", "b"])
        alpha, beta, nll = calibrate_alpha_beta(
            zero_shot_logits=logits,
            affinity=np.array([[1.0, 0.0], [0.0, 1.0]]),
            support_onehot=np.eye(2),
            labels=labels,
            classes=["a", "b"],
        )

        self.assertIn(alpha, ALPHA_GRID)
        self.assertIn(beta, BETA_GRID)
        self.assertGreaterEqual(nll, 0.0)

    def test_run_writes_episode_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            npz = tmp / "features_toy_vitb16.npz"
            out = tmp / "tip.csv"
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
            self.assertEqual(rows[0]["baseline"], "TipAdapter-hard")


if __name__ == "__main__":
    unittest.main()
