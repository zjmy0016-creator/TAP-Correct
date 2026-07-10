from __future__ import annotations

import math

import numpy as np


CLASSES = ("ripe", "turning", "unripe")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def binary_auroc(scores, labels, positive_label: str, negative_label: str) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    keep = (labels == positive_label) | (labels == negative_label)
    kept_scores = scores[keep]
    kept_labels = labels[keep]
    pos = kept_labels == positive_label
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan

    ranks = _average_ranks(kept_scores)
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def turning_in_band_rate(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    med = {c: float(np.median(scores[labels == c])) for c in CLASSES}
    if not (med["ripe"] > med["turning"] > med["unripe"]):
        return math.nan
    turning_scores = scores[labels == "turning"]
    in_band = (turning_scores >= med["unripe"]) & (turning_scores <= med["ripe"])
    return float(in_band.mean())


def risk_coverage_auc(uncertainty, errors) -> float:
    uncertainty = np.asarray(uncertainty, dtype=float)
    errors = np.asarray(errors, dtype=bool)
    if len(uncertainty) == 0:
        return math.nan
    if len(uncertainty) != len(errors):
        raise ValueError("uncertainty and errors must have the same length")

    order = np.argsort(uncertainty, kind="mergesort")
    sorted_errors = errors[order].astype(float)
    coverage_count = np.arange(1, len(sorted_errors) + 1, dtype=float)
    cumulative_risk = np.cumsum(sorted_errors) / coverage_count
    return float(cumulative_risk.mean())


def top_fraction_enrichment(uncertainty, labels, target_label: str, fraction=0.2) -> float:
    uncertainty = np.asarray(uncertainty, dtype=float)
    labels = np.asarray(labels)
    if len(uncertainty) == 0:
        return math.nan
    if len(uncertainty) != len(labels):
        raise ValueError("uncertainty and labels must have the same length")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    n_top = max(1, int(math.ceil(len(uncertainty) * fraction)))
    order = np.argsort(-uncertainty, kind="mergesort")[:n_top]
    return float((labels[order] == target_label).mean())


def score_correlation_matrix(score_by_name: dict[str, np.ndarray]):
    names = list(score_by_name)
    arrays = [np.asarray(score_by_name[name], dtype=float) for name in names]
    if not arrays:
        return [], np.empty((0, 0), dtype=float)
    lengths = {len(arr) for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("all score arrays must have the same length")

    corr = np.empty((len(arrays), len(arrays)), dtype=float)
    for i, left in enumerate(arrays):
        for j, right in enumerate(arrays):
            corr[i, j] = _pearson(left, right)
    return names, corr


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return math.nan
    return float((left @ right) / denom)
