import unittest
import numpy as np

from scripts.clip_selective_baselines import (
    class_prototypes,
    entropy,
    evaluate_actions,
    map_class_predictions_to_actions,
    make_calibration_selective_decisions,
    max_probability,
    one_hot,
    softmax,
    tip_adapter_cache_logits,
    threshold_by_coverage,
    top2_margin,
)


class TestClipSelectiveBaselines(unittest.TestCase):
    def test_maps_maturity_classes_to_actions(self):
        pred = np.array(["mature", "immature", "turning", "ripe", "unripe"])
        actions = map_class_predictions_to_actions(pred)
        self.assertEqual(actions.tolist(), ["pick", "wait", "revisit", "pick", "wait"])

    def test_selective_signals(self):
        probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]])
        self.assertTrue(np.allclose(max_probability(probs), [0.8, 0.4]))
        self.assertTrue(np.allclose(top2_margin(probs), [0.7, 0.05]))
        self.assertLess(entropy(probs)[0], entropy(probs)[1])

    def test_softmax_temperature(self):
        logits = np.array([[2.0, 1.0, 0.0]])
        probs = softmax(logits, temperature=1.0)
        self.assertTrue(np.allclose(probs.sum(axis=1), [1.0]))
        self.assertEqual(probs.argmax(axis=1).tolist(), [0])

    def test_class_prototypes_are_normalized(self):
        feats = np.array([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ])
        labels = np.array(["mature", "mature", "immature", "immature"])
        classes = ["mature", "immature"]
        protos = class_prototypes(feats, labels, classes)

        self.assertEqual(protos.shape, (2, 2))
        self.assertTrue(np.allclose(np.linalg.norm(protos, axis=1), [1.0, 1.0]))

    def test_one_hot(self):
        labels = np.array(["b", "a", "b"])
        classes = ["a", "b"]
        encoded = one_hot(labels, classes)
        expected = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.allclose(encoded, expected))

    def test_tip_adapter_cache_logits_favor_nearest_support_class(self):
        query_feats = np.array([[1.0, 0.0], [0.0, 1.0]])
        support_feats = np.array([[1.0, 0.0], [0.0, 1.0]])
        support_onehot = np.array([[1.0, 0.0], [0.0, 1.0]])

        logits = tip_adapter_cache_logits(
            query_feats,
            support_feats,
            support_onehot,
            beta=5.0,
            alpha=1.0,
        )

        self.assertEqual(logits.shape, (2, 2))
        self.assertGreater(logits[0, 0], logits[0, 1])
        self.assertGreater(logits[1, 1], logits[1, 0])

    def test_threshold_by_coverage(self):
        scores = np.array([0.9, 0.8, 0.5, 0.1])
        threshold = threshold_by_coverage(scores, target_coverage=0.5, higher_is_confident=True)
        accepted = scores >= threshold
        self.assertEqual(int(accepted.sum()), 2)

    def test_calibration_selective_decisions_threshold_on_calibration_scores(self):
        cal_logits = np.array([[5.0, 0.0], [4.0, 0.0]])
        query_logits = np.array([[1.0, 0.0], [0.9, 0.0]])

        actions, info = make_calibration_selective_decisions(
            calibration_logits=cal_logits,
            query_logits=query_logits,
            classes=["mature", "immature"],
            selector="msp",
            target_coverage=0.5,
        )

        expected_threshold = softmax(cal_logits)[0].max()
        self.assertAlmostEqual(info["threshold"], expected_threshold)
        self.assertEqual(actions.tolist(), ["revisit", "revisit"])
        self.assertEqual(info["actual_accept_rate"], 0.0)

    def test_evaluate_actions(self):
        actions = np.array(["pick", "pick", "wait", "revisit"])
        labels = np.array(["mature", "immature", "immature", "turning"])
        metrics = evaluate_actions(actions, labels)

        self.assertAlmostEqual(metrics["false_pick_rate"], 0.5)
        self.assertAlmostEqual(metrics["pick_precision"], 0.5)
        self.assertAlmostEqual(metrics["pick_recall"], 1.0)
        self.assertAlmostEqual(metrics["revisit_burden"], 0.25)
        self.assertAlmostEqual(metrics["coverage"], 0.75)


if __name__ == "__main__":
    unittest.main()
