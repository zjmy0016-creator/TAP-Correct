"""Leakage-safe access to support, calibration, and query data."""

from __future__ import annotations

import numpy as np


CLASSES = ("ripe", "turning", "unripe")


class EpisodeView:
    """Expose one episode while enforcing split separation."""

    def __init__(self, episode, features, labels, query_indices):
        self._features = features
        self._labels = labels
        self._support_indices = np.concatenate(
            [np.asarray(episode["support"][class_name], dtype=int) for class_name in CLASSES]
        )
        self._calibration_indices = np.concatenate(
            [
                np.asarray(episode["calibration"][class_name], dtype=int)
                for class_name in CLASSES
            ]
        )
        self._query_indices = np.asarray(query_indices, dtype=int)
        self._assert_leakproof()

    def _assert_leakproof(self):
        support = set(self._support_indices.tolist())
        calibration = set(self._calibration_indices.tolist())
        query = set(self._query_indices.tolist())

        support_calibration_overlap = support & calibration
        if support_calibration_overlap:
            raise ValueError(
                "Support and calibration overlap at rows "
                f"{sorted(support_calibration_overlap)[:5]}."
            )

        training_query_overlap = (support | calibration) & query
        if training_query_overlap:
            raise ValueError(
                "Support or calibration overlaps query data at rows "
                f"{sorted(training_query_overlap)[:5]}."
            )

        calibration_counts = [
            len(rows) for rows in self._calibration_by_class().values()
        ]
        if len(set(calibration_counts)) != 1:
            raise ValueError(
                "Calibration counts must be balanced across classes: "
                f"{dict(zip(CLASSES, calibration_counts))}."
            )

    def _calibration_by_class(self):
        return {
            class_name: self._calibration_indices[
                np.isin(
                    self._calibration_indices,
                    np.where(self._labels == class_name)[0],
                )
            ]
            for class_name in CLASSES
        }

    def support(self):
        """Return support features and class labels."""
        return self._features[self._support_indices], self._labels[self._support_indices]

    def calibration(self):
        """Return calibration features and class labels."""
        return (
            self._features[self._calibration_indices],
            self._labels[self._calibration_indices],
        )

    def query_features(self):
        """Return query features without exposing query labels."""
        return self._features[self._query_indices]
