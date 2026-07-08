# -*- coding: utf-8 -*-
"""Strict unsafe-pick diagnostic for FP-controlled baselines.

This keeps the existing false_pick_rate definition unchanged and adds a stricter
diagnostic:

unsafe_pick_rate = picked non-target fruit / all predicted pick actions

Target pick classes:
- strawberry: ripe
- tomato: mature

Therefore, picking turning is counted as unsafe here.
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

from scripts.run_fp_controlled_baseline import (
    FALSE_PICK_ALPHAS,
    apply_pick_threshold,
    base_actions,
    calibrate_threshold,
    pick_confidence,
)
from scripts.run_fp_controlled_baseline import run as run_fp_controlled
from scripts.run_zs_temp_selective_baseline import (
    calibrate_temperature,
    infer_backbone,
    infer_dataset_name,
    load_classes,
    split_masks,
    text_prototypes,
    zero_shot_logits,
)
from scripts.run_tip_adapter_selective_baseline import calibrate_alpha_beta, sample_k_support
from scripts.run_proto_adapter_selective_baseline import calibrate_proto_weight, fused_proto_logits
from scripts.clip_selective_baselines import (
    class_prototypes,
    one_hot,
    prototype_logits,
    tip_adapter_cache_logits,
)


PICK_CLASSES = {"ripe", "mature"}


def unsafe_pick_metrics(actions, labels) -> dict:
    actions = np.asarray(actions).astype(str)
    labels = np.asarray(labels).astype(str)

    pick_mask = actions == "pick"
    target_pick = np.isin(labels, list(PICK_CLASSES))
    unsafe_pick = pick_mask & ~target_pick
    true_pick = pick_mask & target_pick

    n_pick = int(pick_mask.sum())
    n_unsafe_pick = int(unsafe_pick.sum())
    n_true_pick = int(true_pick.sum())

    return {
        "n_pick": n_pick,
        "n_true_pick_strict": n_true_pick,
        "n_unsafe_pick": n_unsafe_pick,
        "unsafe_pick_rate": n_unsafe_pick / n_pick if n_pick else 0.0,
        "strict_pick_precision": n_true_pick / n_pick if n_pick else 0.0,
    }


def eval_strict(logits, labels, classes, threshold, temperature=1.0) -> dict:
    scores = pick_confidence(logits, classes, temperature=temperature)
    actions = apply_pick_threshold(base_actions(logits, classes), scores, threshold)
    return unsafe_pick_metrics(actions, labels)


def row(dataset, backbone, family, episode, alpha, threshold, cal_strict, query_strict, extra=None):
    out = {
        "dataset": dataset,
        "backbone": backbone,
        "family": family,
        "episode": episode,
        "false_pick_alpha": alpha,
        "pick_score_threshold": threshold if np.isfinite(threshold) else "inf",
        "calibration_unsafe_pick_rate": cal_strict["unsafe_pick_rate"],
        "calibration_strict_pick_precision": cal_strict["strict_pick_precision"],
        "calibration_n_pick": cal_strict["n_pick"],
        "calibration_n_unsafe_pick": cal_strict["n_unsafe_pick"],
        "query_unsafe_pick_rate": query_strict["unsafe_pick_rate"],
        "query_strict_pick_precision": query_strict["strict_pick_precision"],
        "query_n_pick": query_strict["n_pick"],
        "query_n_unsafe_pick": query_strict["n_unsafe_pick"],
    }
    if extra:
        out.update(extra)
    return out


def run(npz_path, out_csv, dataset=None, backbone=None, k=16, n_episodes=20, base_seed=20260707):
    data = np.load(npz_path, allow_pickle=True)
    classes = load_classes(data)
    dataset = dataset or infer_dataset_name(npz_path)
    backbone = backbone or infer_backbone(npz_path)

    image_feats = np.asarray(data["image_feats"], dtype=float)
    labels = data["classes"].astype(str)
    train_mask, cal_mask, query_mask = split_masks(data["splits"])

    train_indices = np.where(train_mask)[0]
    cal_feats = image_feats[cal_mask]
    cal_labels = labels[cal_mask]
    query_feats = image_feats[query_mask]
    query_labels = labels[query_mask]

    text_protos = text_prototypes(data, classes)
    all_zs_logits = zero_shot_logits(image_feats, text_protos)
    cal_zs_logits = all_zs_logits[cal_mask]
    query_zs_logits = all_zs_logits[query_mask]

    rows = []

    zs_temp, zs_nll = calibrate_temperature(cal_zs_logits, cal_labels, classes)
    for fp_alpha in FALSE_PICK_ALPHAS:
        threshold, _ = calibrate_threshold(cal_zs_logits, cal_labels, classes, fp_alpha, temperature=zs_temp)
        cal_strict = eval_strict(cal_zs_logits, cal_labels, classes, threshold, temperature=zs_temp)
        query_strict = eval_strict(query_zs_logits, query_labels, classes, threshold, temperature=zs_temp)
        rows.append(
            row(
                dataset,
                backbone,
                "ZS-temp",
                "",
                fp_alpha,
                threshold,
                cal_strict,
                query_strict,
                {"temperature": zs_temp, "calibration_nll": zs_nll},
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
        cal_proto_fused = fused_proto_logits(cal_zs_logits, cal_proto_logits, proto_weight)
        query_proto_fused = fused_proto_logits(query_zs_logits, prototype_logits(query_feats, protos), proto_weight)

        for fp_alpha in FALSE_PICK_ALPHAS:
            threshold, _ = calibrate_threshold(cal_tip_logits, cal_labels, classes, fp_alpha)
            rows.append(
                row(
                    dataset,
                    backbone,
                    "TipAdapter",
                    str(episode),
                    fp_alpha,
                    threshold,
                    eval_strict(cal_tip_logits, cal_labels, classes, threshold),
                    eval_strict(query_tip_logits, query_labels, classes, threshold),
                    {"alpha": tip_alpha, "beta": tip_beta, "calibration_nll": tip_nll},
                )
            )

            threshold, _ = calibrate_threshold(cal_proto_fused, cal_labels, classes, fp_alpha)
            rows.append(
                row(
                    dataset,
                    backbone,
                    "ProtoAdapter",
                    str(episode),
                    fp_alpha,
                    threshold,
                    eval_strict(cal_proto_fused, cal_labels, classes, threshold),
                    eval_strict(query_proto_fused, query_labels, classes, threshold),
                    {"proto_weight": proto_weight, "calibration_nll": proto_nll},
                )
            )

    fieldnames = [
        "dataset", "backbone", "family", "episode", "false_pick_alpha",
        "pick_score_threshold",
        "calibration_unsafe_pick_rate", "calibration_strict_pick_precision",
        "calibration_n_pick", "calibration_n_unsafe_pick",
        "query_unsafe_pick_rate", "query_strict_pick_precision",
        "query_n_pick", "query_n_unsafe_pick",
        "temperature", "alpha", "beta", "proto_weight", "calibration_nll",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"dataset={dataset} backbone={backbone} wrote {len(rows)} rows -> {out_csv}")
    return rows


def main():
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
