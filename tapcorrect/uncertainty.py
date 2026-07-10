"""Uncertainty candidate functions for revisit prioritization."""

from __future__ import annotations

import numpy as np

from tapcorrect.harvestability import (
    _softmax_probs,
    build_visual_prototypes,
    score_expectation,
)


CLASSES = ("ripe", "turning", "unripe")


def unc_entropy(features, visual_prototypes, temperature=1.0):
    """Return normalized predictive entropy."""
    probabilities = _softmax_probs(features, visual_prototypes, temperature)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    return entropy / np.log(len(CLASSES))


def unc_top2_margin(features, visual_prototypes, temperature=1.0):
    """Return one minus the top-two probability margin."""
    probabilities = _softmax_probs(features, visual_prototypes, temperature)
    sorted_probabilities = np.sort(probabilities, axis=1)
    return 1.0 - (sorted_probabilities[:, -1] - sorted_probabilities[:, -2])


def unc_distance_to_threshold(base_scores, base_labels):
    """Return negative distance to class-mean interval boundaries."""
    means = {
        class_name: base_scores[base_labels == class_name].mean()
        for class_name in CLASSES
    }
    high = (means["ripe"] + means["turning"]) / 2.0
    low = (means["turning"] + means["unripe"]) / 2.0
    distance = np.minimum(np.abs(base_scores - high), np.abs(base_scores - low))
    return -distance


def unc_bootstrap_variance(
    calibration_features,
    support_features,
    support_labels,
    B=20,
    seed=0,
):
    """Estimate endpoint variation under support resampling."""
    indices_by_class = {
        class_name: np.where(support_labels == class_name)[0]
        for class_name in CLASSES
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(B):
        bootstrap_indices = np.concatenate(
            [
                rng.choice(indices_by_class[class_name],
                           size=len(indices_by_class[class_name]),
                           replace=True)
                for class_name in CLASSES
            ]
        )
        prototypes = build_visual_prototypes(
            support_features[bootstrap_indices],
            support_labels[bootstrap_indices],
        )
        samples.append(score_expectation(calibration_features, prototypes, 1.0))
    return np.stack(samples, axis=0).std(axis=0)


def unc_prototype_disagreement(calibration_features, support_features, support_labels):
    """Return disagreement among one-shot prototype classifiers."""
    indices_by_class = {
        class_name: np.sort(np.where(support_labels == class_name)[0])
        for class_name in CLASSES
    }
    support_size = min(len(indices_by_class[class_name]) for class_name in CLASSES)
    if support_size <= 1:
        return np.zeros(calibration_features.shape[0])

    predictions = []
    for index in range(support_size):
        prototypes = {}
        for class_name in CLASSES:
            vector = support_features[indices_by_class[class_name][index]]
            prototypes[class_name] = vector / (np.linalg.norm(vector) + 1e-12)
        similarities = np.stack(
            [calibration_features @ prototypes[class_name] for class_name in CLASSES],
            axis=1,
        )
        predictions.append(similarities.argmax(axis=1))

    predictions = np.stack(predictions, axis=0)
    disagreement = np.empty(calibration_features.shape[0])
    for index in range(calibration_features.shape[0]):
        counts = np.bincount(predictions[:, index], minlength=len(CLASSES))
        disagreement[index] = 1.0 - counts.max() / support_size
    return disagreement


def all_uncertainty_scores(
    calibration_features,
    calibration_labels,
    support_features,
    support_labels,
    visual_prototypes,
    seed=0,
    B=20,
):
    """Compute all registered uncertainty candidates."""
    base_scores = score_expectation(calibration_features, visual_prototypes, 1.0)
    return {
        "entropy": unc_entropy(calibration_features, visual_prototypes),
        "top2_margin": unc_top2_margin(calibration_features, visual_prototypes),
        "dist_to_threshold": unc_distance_to_threshold(base_scores, calibration_labels),
        "bootstrap_variance": unc_bootstrap_variance(
            calibration_features, support_features, support_labels, B, seed
        ),
        "prototype_disagreement": unc_prototype_disagreement(
            calibration_features, support_features, support_labels
        ),
    }
