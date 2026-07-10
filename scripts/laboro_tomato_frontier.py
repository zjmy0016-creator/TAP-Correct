"""Run the frozen TAP-Correct frontier on Laboro Tomato crops."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tapcorrect.decision import compute_uncertainty
CLASSES = ("mature", "turning", "immature")
ANCHORS = np.array([1.0, 0.5, 0.0])
ALPHA, GAMMA, REVISIT_FRACTION, N_CANDIDATES = 0.05, 0.10, 0.20, 200
DEFERS = np.round(np.arange(0.0, 0.601, 0.05), 3)
M_CALIB, BASE_SEED = 200, 42


def prototypes_from_support(support_features, support_labels):
    prototypes = {}
    for class_name in CLASSES:
        mean_vector = support_features[support_labels == class_name].mean(axis=0)
        prototypes[class_name] = mean_vector / (np.linalg.norm(mean_vector) + 1e-12)
    return prototypes


protos_from_support = prototypes_from_support


def score_E_and_sims(features, prototypes):
    similarities = np.stack(
        [features @ prototypes[class_name] for class_name in CLASSES], axis=1
    )
    logits = similarities - similarities.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities @ ANCHORS, similarities


def calib_T_high(mature, turning, immature, alpha):
    candidates = np.linspace(immature.min(), mature.max(), N_CANDIDATES)
    valid = [threshold for threshold in candidates if np.mean(immature >= threshold) <= alpha]
    return min(valid) if valid else immature.max() + 0.01


def calib_T_low(mature, turning, immature, gamma):
    candidates = np.linspace(immature.min(), mature.max(), N_CANDIDATES)
    valid = [threshold for threshold in candidates if np.mean(mature <= threshold) <= gamma]
    return max(valid) if valid else mature.min() - 0.01


def order_thr(first, second):
    return min(first, second), max(first, second)


def top2_margin(similarities):
    ordered = np.sort(similarities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def build_frontier(episodes, ground_truth_pick, score_key, high_key, low_key):
    rows = []
    for defer in DEFERS:
        totals = dict(pick=0, false_pick=0, wait=0, total=0, true_pick=0, ground_truth=0)
        for episode in episodes:
            uncertainty = episode["u"]
            score = episode[score_key]
            n_deferred = int(len(uncertainty) * defer)
            keep = np.argsort(uncertainty)[::-1][n_deferred:]
            decisions = np.full(len(score), "revisit", dtype=object)
            decisions[keep[score[keep] >= episode[high_key]]] = "pick"
            decisions[keep[score[keep] <= episode[low_key]]] = "wait"
            pick = decisions == "pick"
            wait = decisions == "wait"
            totals["pick"] += int(pick.sum())
            totals["false_pick"] += int((pick & ~ground_truth_pick).sum())
            totals["wait"] += int(wait.sum())
            totals["total"] += len(score)
            totals["true_pick"] += int((pick & ground_truth_pick).sum())
            totals["ground_truth"] += int(ground_truth_pick.sum())
        rows.append(
            {
                "defer": defer,
                "false_pick_rate": totals["false_pick"] / totals["pick"] if totals["pick"] else 0.0,
                "actual_coverage": (totals["pick"] + totals["wait"]) / totals["total"],
                "pick_recall": totals["true_pick"] / totals["ground_truth"] if totals["ground_truth"] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_b5_family_frontier(episodes, ground_truth_pick):
    rows = []
    for defer in DEFERS:
        totals = dict(pick=0, false_pick=0, wait=0, total=0, true_pick=0, ground_truth=0)
        for episode in episodes:
            margin = episode["margin_q"]
            predicted = episode["pred_q"]
            n_deferred = int(len(predicted) * defer)
            keep = np.argsort(margin)[::-1][: len(predicted) - n_deferred]
            decisions = np.full(len(predicted), "revisit", dtype=object)
            decisions[keep[predicted[keep] == "mature"]] = "pick"
            decisions[keep[predicted[keep] == "immature"]] = "wait"
            pick = decisions == "pick"
            wait = decisions == "wait"
            totals["pick"] += int(pick.sum())
            totals["false_pick"] += int((pick & ~ground_truth_pick).sum())
            totals["wait"] += int(wait.sum())
            totals["total"] += len(predicted)
            totals["true_pick"] += int((pick & ground_truth_pick).sum())
            totals["ground_truth"] += int(ground_truth_pick.sum())
        rows.append(
            {
                "defer": defer,
                "false_pick_rate": totals["false_pick"] / totals["pick"] if totals["pick"] else 0.0,
                "actual_coverage": (totals["pick"] + totals["wait"]) / totals["total"],
                "pick_recall": totals["true_pick"] / totals["ground_truth"] if totals["ground_truth"] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def b5_operating_point(episodes, ground_truth_pick):
    totals = dict(pick=0, false_pick=0, wait=0, total=0, true_pick=0, ground_truth=0)
    for episode in episodes:
        threshold = np.percentile(episode["margin_cal"], REVISIT_FRACTION * 100)
        margin = episode["margin_q"]
        predicted = episode["pred_q"]
        decisions = np.full(len(predicted), "revisit", dtype=object)
        keep = margin >= threshold
        decisions[keep & (predicted == "mature")] = "pick"
        decisions[keep & (predicted == "immature")] = "wait"
        pick = decisions == "pick"
        wait = decisions == "wait"
        totals["pick"] += int(pick.sum())
        totals["false_pick"] += int((pick & ~ground_truth_pick).sum())
        totals["wait"] += int(wait.sum())
        totals["total"] += len(predicted)
        totals["true_pick"] += int((pick & ground_truth_pick).sum())
        totals["ground_truth"] += int(ground_truth_pick.sum())
    return {
        "false_pick_rate": totals["false_pick"] / totals["pick"] if totals["pick"] else 0.0,
        "actual_coverage": (totals["pick"] + totals["wait"]) / totals["total"],
        "pick_recall": totals["true_pick"] / totals["ground_truth"] if totals["ground_truth"] else 0.0,
    }


def sample_episode(labels, splits, k, calibration_per_class, seed):
    rng = np.random.default_rng(seed)
    support, calibration = [], []
    for class_name in CLASSES:
        pool = np.where((splits == "train") & (labels == class_name))[0]
        shuffled = rng.permutation(pool)
        support.append(shuffled[:k])
        calibration.append(shuffled[k : k + calibration_per_class])
    return np.concatenate(support), np.concatenate(calibration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n_episodes", type=int, default=100)
    args = parser.parse_args()

    tag = args.npz.stem.replace("features_", "")
    out_dir = ROOT / "outputs/probe_512d_endpoint" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.npz, allow_pickle=True)
    features = data["image_feats"]
    labels = data["classes"].astype(str)
    splits = data["splits"].astype(str)
    query_mask = splits == "test"
    query_features, query_labels = features[query_mask], labels[query_mask]
    ground_truth_pick = query_labels == "mature"

    episodes = []
    for episode_index in range(args.n_episodes):
        support_indices, calibration_indices = sample_episode(
            labels, splits, args.k, M_CALIB, BASE_SEED + episode_index
        )
        support_features, support_labels = features[support_indices], labels[support_indices]
        calibration_features, calibration_labels = features[calibration_indices], labels[calibration_indices]
        prototypes = prototypes_from_support(support_features, support_labels)
        calibration_endpoint, calibration_similarities = score_E_and_sims(calibration_features, prototypes)
        pick_threshold = calib_T_high(
            calibration_endpoint[calibration_labels == "mature"],
            calibration_endpoint[calibration_labels == "turning"],
            calibration_endpoint[calibration_labels == "immature"],
            ALPHA,
        )
        wait_threshold = calib_T_low(
            calibration_endpoint[calibration_labels == "mature"],
            calibration_endpoint[calibration_labels == "turning"],
            calibration_endpoint[calibration_labels == "immature"],
            GAMMA,
        )
        low_threshold, high_threshold = order_thr(pick_threshold, wait_threshold)
        query_endpoint, query_similarities = score_E_and_sims(query_features, prototypes)
        episodes.append(
            {
                "E": query_endpoint,
                "u": compute_uncertainty(query_endpoint, high_threshold, low_threshold),
                "T_high_E": high_threshold,
                "T_low_E": low_threshold,
                "margin_cal": top2_margin(calibration_similarities),
                "margin_q": top2_margin(query_similarities),
                "pred_q": np.array([CLASSES[index] for index in query_similarities.argmax(1)]),
            }
        )

    v1 = build_frontier(episodes, ground_truth_pick, "E", "T_high_E", "T_low_E")
    v1["variant"] = "V1_E_512D_endpoint"
    b5_family = build_b5_family_frontier(episodes, ground_truth_pick)
    b5_family["variant"] = "B5family_argmax_margin"
    b5 = b5_operating_point(episodes, ground_truth_pick)
    b5_point = pd.DataFrame([{**{"defer": REVISIT_FRACTION}, **b5}])
    b5_point["variant"] = "B5_calibrated_argmax_margin"
    frontier = pd.concat([v1, b5_family, b5_point], ignore_index=True)
    frontier.to_csv(out_dir / f"frontier_{tag}_K{args.k}.csv", index=False, float_format="%.6f")

    plt.figure(figsize=(8, 6))
    for frame, color, label in [
        (b5_family, "#7f8c8d", "B5 family"),
        (v1, "#2980b9", "TAP-Correct V1"),
    ]:
        ordered = frame.sort_values("actual_coverage")
        plt.plot(ordered["actual_coverage"], ordered["false_pick_rate"], "o-", color=color, label=label)
    plt.xlabel("Actual coverage")
    plt.ylabel("False-pick rate")
    plt.title(f"Laboro Tomato risk-coverage frontier: {tag} (K={args.k})")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"frontier_{tag}_K{args.k}.png", dpi=200, bbox_inches="tight")
    print(f"Saved frontier artifacts to {out_dir}.")


if __name__ == "__main__":
    main()
