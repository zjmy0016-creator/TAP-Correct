# -*- coding: utf-8 -*-
"""Run false-pick-controlled CLIP/selective baselines.

For each score family, choose an action confidence threshold on calibration data
so calibration false-pick rate among predicted pick actions is <= alpha, then
evaluate the selected threshold on query/test.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clip_selective_baselines import (
    class_prototypes,
    evaluate_actions,
    map_class_predictions_to_actions,
    one_hot,
    predict_classes_from_logits,
    prototype_logits,
    softmax,
    tip_adapter_cache_logits,
)
from scripts.run_tip_adapter_selective_baseline import (
    calibrate_alpha_beta,
    sample_k_support,
)
from scripts.run_proto_adapter_selective_baseline import (
    calibrate_proto_weight,
    fused_proto_logits,
)
from scripts.run_zs_temp_selective_baseline import (
    calibrate_temperature,
    infer_backbone,
    infer_dataset_name,
    load_classes,
    split_masks,
    text_prototypes,
    zero_shot_logits,
)


FALSE_PICK_ALPHAS = (0.05, 0.10)
FAMILIES = ("ZS-temp", "TipAdapter", "ProtoAdapter")


def pick_confidence(logits: np.ndarray, classes: list[str], temperature: float = 1.0) -> np.ndarray:
    probs = softmax(logits, temperature=temperature)
    classes_arr = np.asarray(classes).astype(str)
    pick_mask = np.isin(classes_arr, ["ripe", "mature"])
    if not np.any(pick_mask):
        raise ValueError("No pick class found in classes")
    return probs[:, pick_mask].max(axis=1)


def base_actions(logits: np.ndarray, classes: list[str]) -> np.ndarray:
    pred = predict_classes_from_logits(logits, classes)
    return map_class_predictions_to_actions(pred)


def apply_pick_threshold(actions: np.ndarray, pick_scores: np.ndarray, threshold: float) -> np.ndarray:
    actions = np.asarray(actions).astype(object).copy()
    pick_scores = np.asarray(pick_scores, dtype=float)
    keep_pick = (actions == "pick") & (pick_scores >= threshold)
    actions[(actions == "pick") & ~keep_pick] = "revisit"
    return actions


def calibrate_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    false_pick_alpha: float,
    temperature: float = 1.0,
) -> tuple[float, dict]:
    scores = pick_confidence(logits, classes, temperature=temperature)
    actions = base_actions(logits, classes)

    candidates = sorted(set(scores[actions == "pick"].tolist()), reverse=True)
    if not candidates:
        return float("inf"), evaluate_actions(np.full(len(labels), "revisit", dtype=object), labels)

    best_threshold = None
    best_metrics = None

    for threshold in candidates:
        tuned_actions = apply_pick_threshold(actions, scores, threshold)
        metrics = evaluate_actions(tuned_actions, labels)
        if metrics["false_pick_rate"] <= false_pick_alpha:
            if best_metrics is None or metrics["coverage"] > best_metrics["coverage"]:
                best_threshold = float(threshold)
                best_metrics = metrics

    if best_threshold is None:
        threshold = float("inf")
        tuned_actions = apply_pick_threshold(actions, scores, threshold)
        return threshold, evaluate_actions(tuned_actions, labels)

    return best_threshold, best_metrics


def make_result_row(
    dataset: str,
    backbone: str,
    family: str,
    episode: str,
    alpha_limit: float,
    threshold: float,
    calibration_metrics: dict,
    query_logits: np.ndarray,
    query_labels: np.ndarray,
    classes: list[str],
    temperature: float = 1.0,
    extra: dict | None = None,
) -> dict:
    scores = pick_confidence(query_logits, classes, temperature=temperature)
    actions = apply_pick_threshold(base_actions(query_logits, classes), scores, threshold)
    metrics = evaluate_actions(actions, query_labels)

    row = {
        "dataset": dataset,
        "backbone": backbone,
        "family": family,
        "episode": episode,
        "false_pick_alpha": alpha_limit,
        "pick_score_threshold": threshold if np.isfinite(threshold) else "inf",
        "calibration_false_pick_rate": calibration_metrics["false_pick_rate"],
        "calibration_coverage": calibration_metrics["coverage"],
        **metrics,
    }
    if extra:
        row.update(extra)
    return row


def run(
    npz_path: Path,
    out_csv: Path,
    dataset: str | None = None,
    backbone: str | None = None,
    k: int = 16,
    n_episodes: int = 20,
    base_seed: int = 20260707,
) -> list[dict]:
    data = np.load(npz_path, allow_pickle=True)
    classes = load_classes(data)
    dataset = dataset or infer_dataset_name(npz_path)
    backbone = backbone or infer_backbone(npz_path)

    image_feats = np.asarray(data["image_feats"], dtype=float)
    labels = data["classes"].astype(str)
    train_mask, calibration_mask, query_mask = split_masks(data["splits"])

    train_indices = np.where(train_mask)[0]
    cal_feats = image_feats[calibration_mask]
    cal_labels = labels[calibration_mask]
    query_feats = image_feats[query_mask]
    query_labels = labels[query_mask]

    text_protos = text_prototypes(data, classes)
    all_zs_logits = zero_shot_logits(image_feats, text_protos)
    cal_zs_logits = all_zs_logits[calibration_mask]
    query_zs_logits = all_zs_logits[query_mask]

    rows = []

    zs_temp, zs_nll = calibrate_temperature(cal_zs_logits, cal_labels, classes)
    for fp_alpha in FALSE_PICK_ALPHAS:
        threshold, cal_metrics = calibrate_threshold(
            cal_zs_logits, cal_labels, classes, fp_alpha, temperature=zs_temp
        )
        rows.append(
            make_result_row(
                dataset,
                backbone,
                "ZS-temp",
                "",
                fp_alpha,
                threshold,
                cal_metrics,
                query_zs_logits,
                query_labels,
                classes,
                temperature=zs_temp,
                extra={"temperature": zs_temp, "calibration_nll": zs_nll},
            )
        )

    for episode in range(n_episodes):
        support_indices = sample_k_support(train_indices, labels, classes, k=k, seed=base_seed + episode)
        support_feats = image_feats[support_indices]
        support_labels = labels[support_indices]
        support_onehot = one_hot(support_labels, classes)

        cal_affinity = cal_feats @ support_feats.T
        tip_alpha, tip_beta, tip_nll = calibrate_alpha_beta(
            zero_shot_logits=cal_zs_logits,
            affinity=cal_affinity,
            support_onehot=support_onehot,
            labels=cal_labels,
            classes=classes,
        )
        cal_tip_logits = cal_zs_logits + tip_adapter_cache_logits(
            cal_feats, support_feats, support_onehot, beta=tip_beta, alpha=tip_alpha
        )
        query_tip_logits = query_zs_logits + tip_adapter_cache_logits(
            query_feats, support_feats, support_onehot, beta=tip_beta, alpha=tip_alpha
        )

        protos = class_prototypes(support_feats, support_labels, classes)
        cal_proto_logits = prototype_logits(cal_feats, protos)
        proto_weight, proto_nll = calibrate_proto_weight(
            zero_shot_logits=cal_zs_logits,
            proto_logits=cal_proto_logits,
            labels=cal_labels,
            classes=classes,
        )
        query_proto_logits = fused_proto_logits(
            query_zs_logits, prototype_logits(query_feats, protos), proto_weight
        )
        cal_proto_fused = fused_proto_logits(cal_zs_logits, cal_proto_logits, proto_weight)

        for fp_alpha in FALSE_PICK_ALPHAS:
            threshold, cal_metrics = calibrate_threshold(cal_tip_logits, cal_labels, classes, fp_alpha)
            rows.append(
                make_result_row(
                    dataset,
                    backbone,
                    "TipAdapter",
                    str(episode),
                    fp_alpha,
                    threshold,
                    cal_metrics,
                    query_tip_logits,
                    query_labels,
                    classes,
                    extra={"alpha": tip_alpha, "beta": tip_beta, "calibration_nll": tip_nll},
                )
            )

            threshold, cal_metrics = calibrate_threshold(cal_proto_fused, cal_labels, classes, fp_alpha)
            rows.append(
                make_result_row(
                    dataset,
                    backbone,
                    "ProtoAdapter",
                    str(episode),
                    fp_alpha,
                    threshold,
                    cal_metrics,
                    query_proto_logits,
                    query_labels,
                    classes,
                    extra={"proto_weight": proto_weight, "calibration_nll": proto_nll},
                )
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "backbone", "family", "episode", "false_pick_alpha",
        "pick_score_threshold", "calibration_false_pick_rate", "calibration_coverage",
        "false_pick_rate", "pick_precision", "pick_recall", "revisit_burden", "coverage",
        "n_samples", "n_pick", "n_wait", "n_revisit", "n_false_pick", "n_true_pick",
        "temperature", "alpha", "beta", "proto_weight", "calibration_nll",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"dataset={dataset} backbone={backbone} wrote {len(rows)} rows -> {out_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=20260707)
    args = parser.parse_args()

    run(
        npz_path=args.npz,
        out_csv=args.out,
        dataset=args.dataset,
        backbone=args.backbone,
        k=args.k,
        n_episodes=args.n_episodes,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
