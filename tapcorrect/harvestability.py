"""Prototype-based maturity endpoints used by the delivery pipeline."""

from __future__ import annotations

import numpy as np


CLASSES = ("ripe", "turning", "unripe")
EXPECT_ANCHOR = {"unripe": 0.0, "turning": 0.5, "ripe": 1.0}


def build_visual_prototypes(support_features, support_labels):
    """Build normalized class prototypes from support features."""
    prototypes = {}
    for class_name in CLASSES:
        mask = support_labels == class_name
        if not mask.any():
            raise ValueError(f"Support contains no samples for {class_name!r}.")
        mean_vector = support_features[mask].mean(axis=0)
        prototypes[class_name] = mean_vector / (np.linalg.norm(mean_vector) + 1e-12)
    return prototypes


def load_text_prototypes(npz_path):
    """Load normalized text prototypes from a frozen feature cache."""
    data = np.load(npz_path, allow_pickle=True)
    prototypes = {}
    for class_name in CLASSES:
        mean_vector = data[f"text_{class_name}"].mean(axis=0)
        prototypes[class_name] = mean_vector / (np.linalg.norm(mean_vector) + 1e-12)
    return prototypes


def score_axis(features, text_prototypes):
    """Return a zero-shot projection between ripe and unripe text prototypes."""
    axis = text_prototypes["ripe"] - text_prototypes["unripe"]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return features @ axis


def score_margin(features, visual_prototypes):
    """Return the ripe-versus-unripe prototype margin."""
    return (
        features @ visual_prototypes["ripe"]
        - features @ visual_prototypes["unripe"]
    )


def _softmax_probs(features, visual_prototypes, temperature=1.0):
    similarities = np.stack(
        [features @ visual_prototypes[class_name] for class_name in CLASSES],
        axis=1,
    )
    logits = similarities / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def score_expectation(features, visual_prototypes, temperature=1.0):
    """Return the ordered prototype-expectation endpoint in [0, 1]."""
    probabilities = _softmax_probs(features, visual_prototypes, temperature)
    anchors = np.array([EXPECT_ANCHOR[class_name] for class_name in CLASSES])
    return probabilities @ anchors


def score_calibrated_softmax(features, visual_prototypes, temperature):
    """Return the expectation endpoint using a calibrated temperature."""
    return score_expectation(features, visual_prototypes, temperature=temperature)


def all_harvestability_scores(
    calibration_features,
    visual_prototypes,
    text_prototypes,
    temperature_for_calibration=1.0,
):
    """Compute the registered endpoint candidates for calibration data."""
    return {
        "axis": score_axis(calibration_features, text_prototypes),
        "expectation": score_expectation(calibration_features, visual_prototypes),
        "margin": score_margin(calibration_features, visual_prototypes),
        "calibrated_softmax": score_calibrated_softmax(
            calibration_features,
            visual_prototypes,
            temperature_for_calibration,
        ),
    }
