"""Reproducible support and calibration episode construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CLASSES = ("ripe", "turning", "unripe")


def load_pools(npz_path):
    """Load frozen features and return support/query row indices."""
    data = np.load(npz_path, allow_pickle=True)
    features = data["image_feats"]
    labels = data["classes"]
    splits = data["splits"]

    support_indices = {
        class_name: np.where(
            (splits == "support_pool") & (labels == class_name)
        )[0]
        for class_name in CLASSES
    }
    query_indices = np.where(splits == "query_test")[0]
    return features, labels, support_indices, query_indices


def sample_one_episode(support_indices_by_class, k, calibration_per_class=200, seed=0):
    """Sample disjoint support and calibration rows for one episode."""
    rng = np.random.default_rng(seed)
    support = {}
    calibration = {}

    for class_name in CLASSES:
        pool = support_indices_by_class[class_name].copy()
        required = k + calibration_per_class
        if len(pool) < required:
            raise ValueError(
                f"Class {class_name!r} has {len(pool)} rows; "
                f"{required} are required for support and calibration."
            )
        shuffled = rng.permutation(pool)
        support[class_name] = np.sort(shuffled[:k])
        calibration[class_name] = np.sort(
            shuffled[k : k + calibration_per_class]
        )

    return {"support": support, "calibration": calibration}


def build_manifest(
    support_indices_by_class,
    k_list=(1, 2, 4, 8, 16),
    episodes_per_k=100,
    calibration_per_class=200,
    base_seed=42,
):
    """Build a deterministic episode manifest for all requested support sizes."""
    episodes = []
    counter = 0
    for k in k_list:
        for episode_index in range(episodes_per_k):
            seed = base_seed + counter
            sampled = sample_one_episode(
                support_indices_by_class,
                k=k,
                calibration_per_class=calibration_per_class,
                seed=seed,
            )
            episodes.append(
                {
                    "k": k,
                    "episode_idx": episode_index,
                    "seed": seed,
                    "support": {
                        class_name: sampled["support"][class_name].tolist()
                        for class_name in CLASSES
                    },
                    "calibration": {
                        class_name: sampled["calibration"][class_name].tolist()
                        for class_name in CLASSES
                    },
                }
            )
            counter += 1

    metadata = {
        "k_list": list(k_list),
        "n_ep_per_k": episodes_per_k,
        "m_calib": calibration_per_class,
        "base_seed": base_seed,
        "total_episodes": len(episodes),
        "note": "The query_test split is fixed globally and is evaluated only at the final stage.",
    }
    return {"meta": metadata, "episodes": episodes}


def save_manifest(manifest, out_path):
    """Write a manifest as human-readable JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
