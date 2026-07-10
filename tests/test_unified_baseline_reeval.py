import unittest

import numpy as np

from scripts.unified_baseline_reeval import episode_logits


class TestUnifiedBaselineReeval(unittest.TestCase):
    def test_proto_adapter_uses_zero_shot_fusion_not_bare_prototypes(self):
        feats = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=float,
        )
        labels = np.array(
            ["ripe", "turning", "unripe", "ripe", "turning", "unripe", "ripe", "turning", "unripe"]
        )
        query_mask = np.array([False, False, False, False, False, False, True, True, True])
        text_protos = np.array([[0.2, 0.0], [0.0, 0.2], [-0.2, 0.0]], dtype=float)
        ep = {
            "support": {"ripe": [0], "turning": [1], "unripe": [2]},
            "calibration": {"ripe": [3], "turning": [4], "unripe": [5]},
        }

        logits_map = episode_logits(feats, labels, query_mask, text_protos, ep)
        proto_cal, proto_test, _temp = logits_map["Proto-Adapter"]

        bare_proto_test = feats[query_mask] @ feats[:3].T
        self.assertFalse(np.allclose(proto_test, bare_proto_test))
        self.assertGreater(proto_cal[0, 0], bare_proto_test[0, 0])


if __name__ == "__main__":
    unittest.main()
