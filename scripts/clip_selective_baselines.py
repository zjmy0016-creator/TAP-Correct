# -*- coding: utf-8 -*-
"""Shared utilities for CLIP/selective baselines.

This module intentionally contains only reusable primitives:
- class/action mapping
- probability/logit utilities
- prototype logits
- selective uncertainty scores
- coverage thresholding
- pick/wait/revisit metrics

Dataset-specific runners should import these functions instead of duplicating
logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PICK_CLASSES = {"ripe", "mature"}
WAIT_CLASSES = {"unripe", "immature"}


@dataclass(frozen=True)
class MetricResult:
    false_pick_rate: float
    pick_precision: float
    pick_recall: float
    revisit_burden: float
    coverage: float
    n_samples: int
    n_pick: int
    n_wait: int
    n_revisit: int
    n_false_pick: int
    n_true_pick: int


def as_str_array(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = np.asarray(logits, dtype=float) / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def one_hot(labels, classes: Iterable[str]) -> np.ndarray:
    labels = as_str_array(labels)
    classes = [str(c) for c in classes]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    encoded = np.zeros((len(labels), len(classes)), dtype=float)
    for row, label in enumerate(labels):
        if label not in class_to_idx:
            raise ValueError(f"label {label!r} not found in classes={classes}")
        encoded[row, class_to_idx[label]] = 1.0
    return encoded


def class_prototypes(
    support_feats: np.ndarray,
    support_labels,
    classes: Iterable[str],
) -> np.ndarray:
    support_feats = np.asarray(support_feats, dtype=float)
    support_labels = as_str_array(support_labels)
    classes = [str(c) for c in classes]

    protos = []
    for cls in classes:
        mask = support_labels == cls
        if not np.any(mask):
            raise ValueError(f"no support samples for class {cls!r}")
        mean = support_feats[mask].mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm <= 1e-12:
            raise ValueError(f"zero prototype norm for class {cls!r}")
        protos.append(mean / norm)

    return np.stack(protos, axis=0)


def prototype_logits(query_feats: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    query_feats = np.asarray(query_feats, dtype=float)
    prototypes = np.asarray(prototypes, dtype=float)
    return query_feats @ prototypes.T


def tip_adapter_cache_logits(
    query_feats: np.ndarray,
    support_feats: np.ndarray,
    support_onehot: np.ndarray,
    beta: float = 5.5,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute Tip-Adapter cache logits from frozen support features.

    The form follows the training-free cache model:
    exp(beta * (query @ support.T - 1)) @ one_hot_labels, scaled by alpha.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    query_feats = np.asarray(query_feats, dtype=float)
    support_feats = np.asarray(support_feats, dtype=float)
    support_onehot = np.asarray(support_onehot, dtype=float)

    if support_feats.shape[0] != support_onehot.shape[0]:
        raise ValueError("support_feats and support_onehot must have matching rows")

    affinity = query_feats @ support_feats.T
    cache_logits = np.exp(beta * (affinity - 1.0)) @ support_onehot
    return float(alpha) * cache_logits


def predict_classes_from_logits(logits: np.ndarray, classes: Iterable[str]) -> np.ndarray:
    classes = np.asarray([str(c) for c in classes])
    pred_idx = np.asarray(logits).argmax(axis=1)
    return classes[pred_idx]


def map_class_predictions_to_actions(pred_classes) -> np.ndarray:
    pred_classes = as_str_array(pred_classes)
    actions = np.full(pred_classes.shape, "revisit", dtype=object)

    actions[np.isin(pred_classes, list(PICK_CLASSES))] = "pick"
    actions[np.isin(pred_classes, list(WAIT_CLASSES))] = "wait"

    return actions


def apply_reject(actions, accept_mask) -> np.ndarray:
    actions = as_str_array(actions).astype(object)
    accept_mask = np.asarray(accept_mask, dtype=bool)

    if len(actions) != len(accept_mask):
        raise ValueError("actions and accept_mask must have the same length")

    out = actions.copy()
    out[~accept_mask] = "revisit"
    return out


def max_probability(probs: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(probs, dtype=float), axis=1)


def top2_margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(np.asarray(probs, dtype=float), axis=1)[:, ::-1]
    return sorted_probs[:, 0] - sorted_probs[:, 1]


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    return -np.sum(probs * np.log(probs + eps), axis=1)


def threshold_by_coverage(
    confidence_scores: np.ndarray,
    target_coverage: float,
    higher_is_confident: bool = True,
) -> float:
    """Return the score threshold that accepts the top target_coverage fraction."""
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")

    scores = np.asarray(confidence_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("confidence_scores must be one-dimensional")
    if len(scores) == 0:
        raise ValueError("confidence_scores must not be empty")

    n_accept = int(np.ceil(target_coverage * len(scores)))
    n_accept = min(max(n_accept, 1), len(scores))

    if higher_is_confident:
        ordered = np.sort(scores)[::-1]
    else:
        ordered = np.sort(scores)

    return float(ordered[n_accept - 1])


def accept_by_threshold(
    confidence_scores: np.ndarray,
    threshold: float,
    higher_is_confident: bool = True,
) -> np.ndarray:
    scores = np.asarray(confidence_scores, dtype=float)
    if higher_is_confident:
        return scores >= threshold
    return scores <= threshold


def evaluate_actions(actions, true_labels) -> dict:
    actions = as_str_array(actions)
    true_labels = as_str_array(true_labels)

    if len(actions) != len(true_labels):
        raise ValueError("actions and true_labels must have the same length")

    pick_mask = actions == "pick"
    wait_mask = actions == "wait"
    revisit_mask = actions == "revisit"

    should_pick = np.isin(true_labels, list(PICK_CLASSES))
    should_wait = np.isin(true_labels, list(WAIT_CLASSES))

    n_samples = len(actions)
    n_pick = int(pick_mask.sum())
    n_wait = int(wait_mask.sum())
    n_revisit = int(revisit_mask.sum())

    false_pick_mask = pick_mask & should_wait
    true_pick_mask = pick_mask & should_pick

    n_false_pick = int(false_pick_mask.sum())
    n_true_pick = int(true_pick_mask.sum())
    n_should_pick = int(should_pick.sum())

    false_pick_rate = n_false_pick / n_pick if n_pick > 0 else 0.0
    pick_precision = 1.0 - false_pick_rate if n_pick > 0 else 0.0
    pick_recall = n_true_pick / n_should_pick if n_should_pick > 0 else 0.0
    revisit_burden = n_revisit / n_samples if n_samples > 0 else 0.0
    coverage = 1.0 - revisit_burden

    result = MetricResult(
        false_pick_rate=float(false_pick_rate),
        pick_precision=float(pick_precision),
        pick_recall=float(pick_recall),
        revisit_burden=float(revisit_burden),
        coverage=float(coverage),
        n_samples=n_samples,
        n_pick=n_pick,
        n_wait=n_wait,
        n_revisit=n_revisit,
        n_false_pick=n_false_pick,
        n_true_pick=n_true_pick,
    )
    return result.__dict__.copy()


def make_selective_decisions(
    logits: np.ndarray,
    classes: Iterable[str],
    selector: str,
    target_coverage: float,
    temperature: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Convert logits to pick/wait/revisit actions with a coverage selector.

    selector:
      - "msp": accept high max probability
      - "margin": accept high top-2 margin
      - "entropy": accept low entropy
    """
    probs = softmax(logits, temperature=temperature)
    pred_classes = predict_classes_from_logits(logits, classes)
    base_actions = map_class_predictions_to_actions(pred_classes)

    if selector == "msp":
        scores = max_probability(probs)
        threshold = threshold_by_coverage(scores, target_coverage, True)
        accept = accept_by_threshold(scores, threshold, True)
    elif selector == "margin":
        scores = top2_margin(probs)
        threshold = threshold_by_coverage(scores, target_coverage, True)
        accept = accept_by_threshold(scores, threshold, True)
    elif selector == "entropy":
        scores = entropy(probs)
        threshold = threshold_by_coverage(scores, target_coverage, False)
        accept = accept_by_threshold(scores, threshold, False)
    else:
        raise ValueError(f"unknown selector: {selector}")

    actions = apply_reject(base_actions, accept)
    info = {
        "selector": selector,
        "target_coverage": float(target_coverage),
        "threshold": float(threshold),
        "actual_accept_rate": float(np.mean(accept)),
    }
    return actions, info
