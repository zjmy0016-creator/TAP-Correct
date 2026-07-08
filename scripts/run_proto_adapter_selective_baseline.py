# -*- coding: utf-8 -*-
"""Run Proto-Adapter prototype-logit selective baselines.

This runner uses frozen CLIP image/text features only. For each episode, it
samples K support examples per class from the non-calibration support pool,
builds normalized class prototypes, calibrates the prototype fusion weight on
the calibration split, and evaluates hard/selective actions on query/test.
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

from scripts.clip_selective_baselines import (  # noqa: E402
    class_prototypes,
    evaluate_actions,
    make_selective_decisions,
    map_class_predictions_to_actions,
    predict_classes_from_logits,
    prototype_logits,
    softmax,
)
from scripts.run_tip_adapter_selective_baseline import sample_k_support  # noqa: E402
from scripts.run_zs_temp_selective_baseline import (  # noqa: E402
    COVERAGE_GRID,
    SELECTORS,
    infer_backbone,
    infer_dataset_name,
    labels_to_indices,
    load_classes,
    split_masks,
    text_prototypes,
    zero_shot_logits,
)


PROTO_WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def fused_proto_logits(
    zero_shot_logits: np.ndarray,
    proto_logits: np.ndarray,
    proto_weight: float,
) -> np.ndarray:
    if proto_weight < 0:
        raise ValueError("proto_weight must be non-negative")

    zero_shot_logits = np.asarray(zero_shot_logits, dtype=float)
    proto_logits = np.asarray(proto_logits, dtype=float)
    if zero_shot_logits.shape != proto_logits.shape:
        raise ValueError("zero_shot_logits and proto_logits must have matching shapes")

    return zero_shot_logits + float(proto_weight) * proto_logits


def nll_from_logits(logits: np.ndarray, labels: np.ndarray, classes: list[str]) -> float:
    probs = softmax(logits, temperature=1.0)
    y = labels_to_indices(labels, classes)
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def calibrate_proto_weight(
    zero_shot_logits: np.ndarray,
    proto_logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
) -> tuple[float, float]:
    best = None
    for weight in PROTO_WEIGHT_GRID:
        logits = fused_proto_logits(zero_shot_logits, proto_logits, weight)
        nll = nll_from_logits(logits, labels, classes)
        item = (float(weight), nll)
        if best is None or item[1] < best[1]:
            best = item
    if best is None:
        raise RuntimeError("prototype weight grid is empty")
    return best


def hard_decision_row(
    dataset: str,
    backbone: str,
    episode: int,
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    proto_weight: float,
    calibration_nll: float,
) -> dict:
    pred = predict_classes_from_logits(logits, classes)
    actions = map_class_predictions_to_actions(pred)
    metrics = evaluate_actions(actions, labels)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "episode": episode,
        "baseline": "ProtoAdapter-hard",
        "selector": "none",
        "proto_weight": proto_weight,
        "calibration_nll": calibration_nll,
        "target_coverage": 1.0,
        "selector_threshold": "",
        **metrics,
    }


def selective_rows(
    dataset: str,
    backbone: str,
    episode: int,
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    proto_weight: float,
    calibration_nll: float,
) -> list[dict]:
    rows = []
    for selector in SELECTORS:
        for target_coverage in COVERAGE_GRID:
            actions, info = make_selective_decisions(
                logits=logits,
                classes=classes,
                selector=selector,
                target_coverage=target_coverage,
                temperature=1.0,
            )
            metrics = evaluate_actions(actions, labels)
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "episode": episode,
                    "baseline": "ProtoAdapter-selective",
                    "selector": selector,
                    "proto_weight": proto_weight,
                    "calibration_nll": calibration_nll,
                    "target_coverage": target_coverage,
                    "selector_threshold": info["threshold"],
                    **metrics,
                }
            )
    return rows


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
    text_protos = text_prototypes(data, classes)
    all_zs_logits = zero_shot_logits(image_feats, text_protos)

    cal_feats = image_feats[calibration_mask]
    cal_labels = labels[calibration_mask]
    cal_zs_logits = all_zs_logits[calibration_mask]
    query_feats = image_feats[query_mask]
    query_labels = labels[query_mask]
    query_zs_logits = all_zs_logits[query_mask]

    rows = []
    for episode in range(n_episodes):
        support_indices = sample_k_support(
            pool_indices=train_indices,
            labels=labels,
            classes=classes,
            k=k,
            seed=base_seed + episode,
        )
        support_feats = image_feats[support_indices]
        support_labels = labels[support_indices]

        protos = class_prototypes(support_feats, support_labels, classes)
        cal_proto_logits = prototype_logits(cal_feats, protos)
        proto_weight, calibration_nll = calibrate_proto_weight(
            zero_shot_logits=cal_zs_logits,
            proto_logits=cal_proto_logits,
            labels=cal_labels,
            classes=classes,
        )

        query_proto_logits = prototype_logits(query_feats, protos)
        query_logits = fused_proto_logits(query_zs_logits, query_proto_logits, proto_weight)

        rows.append(
            hard_decision_row(
                dataset,
                backbone,
                episode,
                query_logits,
                query_labels,
                classes,
                proto_weight,
                calibration_nll,
            )
        )
        rows.extend(
            selective_rows(
                dataset,
                backbone,
                episode,
                query_logits,
                query_labels,
                classes,
                proto_weight,
                calibration_nll,
            )
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "backbone",
        "episode",
        "baseline",
        "selector",
        "proto_weight",
        "calibration_nll",
        "target_coverage",
        "selector_threshold",
        "false_pick_rate",
        "pick_precision",
        "pick_recall",
        "revisit_burden",
        "coverage",
        "n_samples",
        "n_pick",
        "n_wait",
        "n_revisit",
        "n_false_pick",
        "n_true_pick",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"dataset={dataset} backbone={backbone} classes={classes}")
    print(f"k={k} episodes={n_episodes} calibration samples={int(calibration_mask.sum())} query samples={int(query_mask.sum())}")
    print(f"wrote {len(rows)} rows -> {out_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/clip_selective_baselines/proto_adapter_selective_summary.csv"),
    )
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
