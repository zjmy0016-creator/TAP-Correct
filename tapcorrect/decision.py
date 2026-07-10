"""Canonical pick, wait, and revisit decision logic."""

from __future__ import annotations

import numpy as np


def compute_uncertainty(scores, high_threshold, low_threshold):
    """Return negative distance from the confident decision interval.

    Values closer to zero are less certain. Values farther from the interval
    are more certain and therefore more negative.
    """
    scores = np.asarray(scores, dtype=float)
    below = scores <= low_threshold
    above = scores >= high_threshold
    distance = np.where(
        below,
        low_threshold - scores,
        np.where(
            above,
            scores - high_threshold,
            np.minimum(scores - low_threshold, high_threshold - scores),
        ),
    )
    return -distance


def decide_episode_v1_endpoint(scores, high_threshold, low_threshold, uncertainty_cut):
    """Apply the frozen ordered endpoint decision rule to one episode.

    The endpoint maps ``unripe`` to 0, ``turning`` to 0.5, and ``ripe`` to 1.
    Samples outside the confident interval are marked for revisit.
    """
    scores = np.asarray(scores, dtype=float)
    uncertainty = compute_uncertainty(scores, high_threshold, low_threshold)

    decisions = np.full(len(scores), "revisit", dtype=object)
    revisit_source = np.full(len(scores), "none", dtype=object)

    confident = uncertainty < uncertainty_cut
    uncertain = ~confident
    pick = confident & (scores >= high_threshold)
    wait = confident & (scores <= low_threshold)
    boundary = confident & (scores > low_threshold) & (scores < high_threshold)

    decisions[pick] = "pick"
    decisions[wait] = "wait"
    revisit_source[uncertain] = "uncertainty"
    revisit_source[boundary] = "boundary_band"
    return decisions, revisit_source
