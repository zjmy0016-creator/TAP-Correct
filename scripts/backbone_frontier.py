"""Build a frozen risk-coverage frontier for one CLIP backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tapcorrect.contract import EpisodeView
from tapcorrect.decision import compute_uncertainty
from tapcorrect.episodes import load_pools


CLASSES = ("ripe", "turning", "unripe")
ALPHA = 0.05
GAMMA = 0.10
REVISIT_FRACTION = 0.20
N_CANDIDATES = 200
DEFERS = np.round(np.arange(0.0, 0.601, 0.05), 3)


def prototypes_from_support(support_features, support_labels):
    prototypes = {}
    for class_name in CLASSES:
        mean_vector = support_features[support_labels == class_name].mean(axis=0)
        prototypes[class_name] = mean_vector / (np.linalg.norm(mean_vector) + 1e-12)
    return prototypes


def endpoint_and_similarities(features, prototypes):
    similarities = np.stack(
        [features @ prototypes[class_name] for class_name in CLASSES], axis=1
    )
    logits = similarities - similarities.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    endpoint = probabilities @ np.array([1.0, 0.5, 0.0])
    return endpoint, similarities


def calibration_high_threshold(ripe, turning, unripe, alpha):
    candidates = np.linspace(unripe.min(), ripe.max(), N_CANDIDATES)
    valid = [threshold for threshold in candidates if np.mean(unripe >= threshold) <= alpha]
    return min(valid) if valid else unripe.max() + 0.01


def calibration_low_threshold(ripe, turning, unripe, gamma):
    candidates = np.linspace(unripe.min(), ripe.max(), N_CANDIDATES)
    valid = [threshold for threshold in candidates if np.mean(ripe <= threshold) <= gamma]
    return max(valid) if valid else ripe.min() - 0.01


def order_thresholds(first, second):
    return min(first, second), max(first, second)


def top2_margin(similarities):
    ordered = np.sort(similarities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def build_frontier(episodes, ground_truth_pick):
    rows = []
    for defer in DEFERS:
        totals = dict(pick=0, false_pick=0, wait=0, total=0, true_pick=0, ground_truth=0)
        for episode in episodes:
            uncertainty = episode["uncertainty"]
            endpoint = episode["endpoint"]
            n_deferred = int(len(uncertainty) * defer)
            keep = np.argsort(uncertainty)[::-1][n_deferred:]
            decisions = np.full(len(endpoint), "revisit", dtype=object)
            decisions[keep[endpoint[keep] >= episode["high_threshold"]]] = "pick"
            decisions[keep[endpoint[keep] <= episode["low_threshold"]]] = "wait"
            pick = decisions == "pick"
            wait = decisions == "wait"
            totals["pick"] += int(pick.sum())
            totals["false_pick"] += int((pick & ~ground_truth_pick).sum())
            totals["wait"] += int(wait.sum())
            totals["total"] += len(endpoint)
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
            margin = episode["margin_query"]
            predicted = episode["predicted_class"]
            n_deferred = int(len(predicted) * defer)
            keep = np.argsort(margin)[::-1][: len(predicted) - n_deferred]
            decisions = np.full(len(predicted), "revisit", dtype=object)
            decisions[keep[predicted[keep] == "ripe"]] = "pick"
            decisions[keep[predicted[keep] == "unripe"]] = "wait"
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
        threshold = np.percentile(episode["margin_calibration"], REVISIT_FRACTION * 100)
        margin = episode["margin_query"]
        predicted = episode["predicted_class"]
        decisions = np.full(len(predicted), "revisit", dtype=object)
        keep = margin >= threshold
        decisions[keep & (predicted == "ripe")] = "pick"
        decisions[keep & (predicted == "unripe")] = "wait"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/episodes/manifest_K1-16_ep100.json",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=ROOT / "outputs/decision_gold/turning_decision_dataset/labels/test_decision_ground_truth_clean.csv",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    tag = args.npz.stem.replace("features_", "")
    out_dir = args.out_dir or ROOT / "outputs/probe_512d_endpoint" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    features, labels, support_indices, query_indices = load_pools(args.npz)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["episodes"]
    selected_episodes = [(index, episode) for index, episode in enumerate(manifest) if episode["k"] == args.k]
    ground_truth = pd.read_csv(args.gt, encoding="utf-8-sig")
    ground_truth_decisions = ground_truth["decision_label"].replace("borderline", "revisit").to_numpy()
    ground_truth_pick = ground_truth_decisions == "pick"
    query_features = features[query_indices]
    if len(query_features) != len(ground_truth_pick):
        raise ValueError("Query feature count does not match decision ground truth.")

    episodes = []
    for _, episode in selected_episodes:
        view = EpisodeView(episode, features, labels, query_indices)
        support_features, support_labels = view.support()
        calibration_features, calibration_labels = view.calibration()
        prototypes = prototypes_from_support(support_features, support_labels)
        calibration_endpoint, calibration_similarities = endpoint_and_similarities(
            calibration_features, prototypes
        )
        pick_threshold = calibration_high_threshold(
            calibration_endpoint[calibration_labels == "ripe"],
            calibration_endpoint[calibration_labels == "turning"],
            calibration_endpoint[calibration_labels == "unripe"],
            ALPHA,
        )
        wait_threshold = calibration_low_threshold(
            calibration_endpoint[calibration_labels == "ripe"],
            calibration_endpoint[calibration_labels == "turning"],
            calibration_endpoint[calibration_labels == "unripe"],
            GAMMA,
        )
        low_threshold, high_threshold = order_thresholds(pick_threshold, wait_threshold)
        query_endpoint, query_similarities = endpoint_and_similarities(query_features, prototypes)
        episodes.append(
            {
                "endpoint": query_endpoint,
                "uncertainty": compute_uncertainty(query_endpoint, high_threshold, low_threshold),
                "high_threshold": high_threshold,
                "low_threshold": low_threshold,
                "margin_calibration": top2_margin(calibration_similarities),
                "margin_query": top2_margin(query_similarities),
                "predicted_class": np.array([CLASSES[index] for index in query_similarities.argmax(1)]),
            }
        )

    v1 = build_frontier(episodes, ground_truth_pick)
    v1["variant"] = "V1_E_512D_endpoint"
    b5_family = build_b5_family_frontier(episodes, ground_truth_pick)
    b5_family["variant"] = "B5family_argmax_margin"
    b5 = b5_operating_point(episodes, ground_truth_pick)
    b5_calibrated = pd.DataFrame([{**{"defer": REVISIT_FRACTION}, **b5}])
    b5_calibrated["variant"] = "B5_calibrated_argmax_margin"
    frontier = pd.concat([v1, b5_family, b5_calibrated], ignore_index=True)
    frontier_path = out_dir / f"frontier_{tag}_K{args.k}.csv"
    frontier.to_csv(frontier_path, index=False, float_format="%.6f")

    def interpolate(frame, coverage, metric):
        ordered = frame.sort_values("actual_coverage")
        return float(np.interp(coverage, ordered["actual_coverage"], ordered[metric]))

    b5_coverage = b5["actual_coverage"]
    print(f"B5 operating point: coverage={b5_coverage:.3f}, false-pick={b5['false_pick_rate']:.3f}")
    print(f"V1 at B5 coverage: false-pick={interpolate(v1, b5_coverage, 'false_pick_rate'):.3f}")
    common_low = max(v1["actual_coverage"].min(), b5_family["actual_coverage"].min())
    common_high = min(v1["actual_coverage"].max(), b5_family["actual_coverage"].max())
    common_grid = np.linspace(common_low, common_high, 25)
    dominance_share = np.mean(
        [
            interpolate(v1, coverage, "false_pick_rate")
            <= interpolate(b5_family, coverage, "false_pick_rate")
            for coverage in common_grid
        ]
    )
    print(f"V1 false-pick <= B5 family over common coverage: {dominance_share:.3f}")

    plt.figure(figsize=(8, 6))
    for frame, color, label in [
        (b5_family, "#7f8c8d", "B5 family"),
        (v1, "#2980b9", "TAP-Correct V1"),
    ]:
        ordered = frame.sort_values("actual_coverage")
        plt.plot(ordered["actual_coverage"], ordered["false_pick_rate"], "o-", color=color, label=label)
    plt.scatter([b5_coverage], [b5["false_pick_rate"]], marker="*", s=280, color="#c0392b", label="B5 operating point")
    plt.xlabel("Actual coverage")
    plt.ylabel("False-pick rate")
    plt.title(f"Risk-coverage frontier: {tag} (K={args.k})")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"frontier_{tag}_K{args.k}.png", dpi=200, bbox_inches="tight")

    plt.figure(figsize=(8, 6))
    for frame, color, label in [
        (b5_family, "#7f8c8d", "B5 family"),
        (v1, "#2980b9", "TAP-Correct V1"),
    ]:
        ordered = frame.sort_values("actual_coverage")
        plt.plot(ordered["actual_coverage"], ordered["pick_recall"], "o-", color=color, label=label)
    plt.scatter([b5_coverage], [b5["pick_recall"]], marker="*", s=280, color="#c0392b", label="B5 operating point")
    plt.xlabel("Actual coverage")
    plt.ylabel("Pick recall")
    plt.title(f"Recall-coverage frontier: {tag} (K={args.k})")
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"recall_{tag}_K{args.k}.png", dpi=200, bbox_inches="tight")
    print(f"Saved frontier artifacts to {out_dir}.")


if __name__ == "__main__":
    main()
