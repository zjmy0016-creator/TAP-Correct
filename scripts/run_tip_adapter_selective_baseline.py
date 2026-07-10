# -*- coding: utf-8 -*-
"""Run Tip-Adapter cache-logit selective baselines.

This runner uses frozen CLIP image/text features only. For each episode, it
samples K support examples per class from the non-calibration support pool,
calibrates alpha/beta on the calibration split, and evaluates on query/test.
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
    evaluate_actions,
    make_calibration_selective_decisions,
    map_class_predictions_to_actions,
    one_hot,
    predict_classes_from_logits,
    softmax,
    tip_adapter_cache_logits,
)
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


ALPHA_GRID = (0.1, 0.5, 1.0, 2.0)
BETA_GRID = (1.0, 3.0, 5.5, 10.0)


def nll_from_logits(logits: np.ndarray, labels: np.ndarray, classes: list[str]) -> float:
    probs = softmax(logits, temperature=1.0)
    y = labels_to_indices(labels, classes)
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def sample_k_support(
    pool_indices: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    k: int,
    seed: int,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")

    labels = np.asarray(labels).astype(str)
    pool_indices = np.asarray(pool_indices, dtype=int)
    rng = np.random.default_rng(seed)

    selected = []
    for cls in classes:
        cls_pool = pool_indices[labels[pool_indices] == cls]
        if len(cls_pool) < k:
            raise ValueError(f"not enough support samples for class {cls!r}: need {k}, got {len(cls_pool)}")
        selected.append(rng.choice(cls_pool, size=k, replace=False))
    return np.concatenate(selected)


def cache_logits_from_affinity(
    affinity: np.ndarray,
    support_onehot: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    return float(alpha) * (np.exp(float(beta) * (affinity - 1.0)) @ support_onehot)


def calibrate_alpha_beta(
    zero_shot_logits: np.ndarray,
    affinity: np.ndarray,
    support_onehot: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
) -> tuple[float, float, float]:
    best = None
    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            cache_logits = cache_logits_from_affinity(affinity, support_onehot, alpha, beta)
            fused_logits = zero_shot_logits + cache_logits
            nll = nll_from_logits(fused_logits, labels, classes)
            item = (float(alpha), float(beta), nll)
            if best is None or item[2] < best[2]:
                best = item
    if best is None:
        raise RuntimeError("alpha/beta grid is empty")
    return best


def hard_decision_row(
    dataset: str,
    backbone: str,
    episode: int,
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    alpha: float,
    beta: float,
    calibration_nll: float,
) -> dict:
    pred = predict_classes_from_logits(logits, classes)
    actions = map_class_predictions_to_actions(pred)
    metrics = evaluate_actions(actions, labels)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "episode": episode,
        "baseline": "TipAdapter-hard",
        "selector": "none",
        "alpha": alpha,
        "beta": beta,
        "calibration_nll": calibration_nll,
        "target_coverage": 1.0,
        "selector_threshold": "",
        **metrics,
    }


def selective_rows(
    dataset: str,
    backbone: str,
    episode: int,
    calibration_logits: np.ndarray,
    query_logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    alpha: float,
    beta: float,
    calibration_nll: float,
) -> list[dict]:
    rows = []
    for selector in SELECTORS:
        for target_coverage in COVERAGE_GRID:
            actions, info = make_calibration_selective_decisions(
                calibration_logits=calibration_logits,
                query_logits=query_logits,
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
                    "baseline": "TipAdapter-selective",
                    "selector": selector,
                    "alpha": alpha,
                    "beta": beta,
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
        support_onehot = one_hot(support_labels, classes)

        cal_affinity = cal_feats @ support_feats.T
        alpha, beta, calibration_nll = calibrate_alpha_beta(
            zero_shot_logits=cal_zs_logits,
            affinity=cal_affinity,
            support_onehot=support_onehot,
            labels=cal_labels,
            classes=classes,
        )

        cal_cache_logits = tip_adapter_cache_logits(
            query_feats=cal_feats,
            support_feats=support_feats,
            support_onehot=support_onehot,
            beta=beta,
            alpha=alpha,
        )
        cal_logits = cal_zs_logits + cal_cache_logits
        query_cache_logits = tip_adapter_cache_logits(
            query_feats=query_feats,
            support_feats=support_feats,
            support_onehot=support_onehot,
            beta=beta,
            alpha=alpha,
        )
        query_logits = query_zs_logits + query_cache_logits

        rows.append(
            hard_decision_row(
                dataset,
                backbone,
                episode,
                query_logits,
                query_labels,
                classes,
                alpha,
                beta,
                calibration_nll,
            )
        )
        rows.extend(
            selective_rows(
                dataset,
                backbone,
                episode,
                cal_logits,
                query_logits,
                query_labels,
                classes,
                alpha,
                beta,
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
        "alpha",
        "beta",
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
        default=Path("outputs/clip_selective_baselines/tip_adapter_selective_summary.csv"),
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
