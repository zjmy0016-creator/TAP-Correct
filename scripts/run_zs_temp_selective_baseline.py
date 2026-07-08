# -*- coding: utf-8 -*-
"""Run zero-shot CLIP temperature-scaled selective baselines.

This script is intentionally limited to zero-shot text-prototype logits plus
selector-based abstention. It is the first runner in the CLIP/selective
baseline suite.
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
    entropy,
    evaluate_actions,
    make_selective_decisions,
    map_class_predictions_to_actions,
    max_probability,
    predict_classes_from_logits,
    softmax,
    top2_margin,
)


TEMPERATURE_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
COVERAGE_GRID = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
SELECTORS = ("msp", "margin", "entropy")


def normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / (np.linalg.norm(x) + 1e-12)


def infer_dataset_name(npz_path: Path) -> str:
    name = npz_path.stem.lower()
    if "laboro" in name or "tomato" in name:
        return "laboro_tomato"
    if "strawberry" in name or "vitb" in name or "vitl" in name:
        return "strawberry"
    return npz_path.stem


def infer_backbone(npz_path: Path) -> str:
    name = npz_path.stem.lower()
    for tag in ("vitb32", "vitb16", "vitl14"):
        if tag in name:
            return tag
    if "laboro_tomato" in name:
        return "vitb16"
    return "unknown"


def split_masks(splits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = np.asarray(splits).astype(str)
    support_mask = np.isin(splits, ["support_pool", "train"])
    query_mask = np.isin(splits, ["query_test", "test"])

    if not np.any(support_mask):
        raise ValueError("No support/train split found")
    if not np.any(query_mask):
        raise ValueError("No query_test/test split found")

    support_indices = np.where(support_mask)[0]
    rng = np.random.default_rng(20260707)
    shuffled = rng.permutation(support_indices)

    n_cal = max(1, int(round(0.20 * len(shuffled))))
    calibration_indices = shuffled[:n_cal]
    train_indices = shuffled[n_cal:]
    query_indices = np.where(query_mask)[0]

    train_mask = np.zeros(len(splits), dtype=bool)
    calibration_mask = np.zeros(len(splits), dtype=bool)
    query_mask_out = np.zeros(len(splits), dtype=bool)

    train_mask[train_indices] = True
    calibration_mask[calibration_indices] = True
    query_mask_out[query_indices] = True

    return train_mask, calibration_mask, query_mask_out


def load_classes(data: np.lib.npyio.NpzFile) -> list[str]:
    classes = sorted(str(x) for x in np.unique(data["classes"].astype(str)))
    for cls in classes:
        if f"text_{cls}" not in data.files:
            raise ValueError(f"Missing text prototype key text_{cls}")
    return classes


def text_prototypes(data: np.lib.npyio.NpzFile, classes: list[str]) -> np.ndarray:
    protos = []
    for cls in classes:
        text_feats = np.asarray(data[f"text_{cls}"], dtype=float)
        protos.append(normalize_vector(text_feats.mean(axis=0)))
    return np.stack(protos, axis=0)


def zero_shot_logits(image_feats: np.ndarray, text_protos: np.ndarray) -> np.ndarray:
    return np.asarray(image_feats, dtype=float) @ np.asarray(text_protos, dtype=float).T


def labels_to_indices(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    out = []
    for label in labels.astype(str):
        if label not in class_to_idx:
            raise ValueError(f"Unknown label {label!r}")
        out.append(class_to_idx[label])
    return np.asarray(out, dtype=int)


def negative_log_likelihood(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    temperature: float,
) -> float:
    probs = softmax(logits, temperature=temperature)
    y = labels_to_indices(labels, classes)
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def calibrate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
) -> tuple[float, float]:
    rows = []
    for temp in TEMPERATURE_GRID:
        nll = negative_log_likelihood(logits, labels, classes, temp)
        rows.append((temp, nll))
    return min(rows, key=lambda item: item[1])


def hard_decision_row(
    dataset: str,
    backbone: str,
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    temperature: float,
) -> dict:
    pred = predict_classes_from_logits(logits, classes)
    actions = map_class_predictions_to_actions(pred)
    metrics = evaluate_actions(actions, labels)
    return {
        "dataset": dataset,
        "backbone": backbone,
        "baseline": "ZS-hard",
        "selector": "none",
        "temperature": temperature,
        "target_coverage": 1.0,
        "selector_threshold": "",
        **metrics,
    }


def selective_rows(
    dataset: str,
    backbone: str,
    logits: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    temperature: float,
) -> list[dict]:
    rows = []
    for selector in SELECTORS:
        for target_coverage in COVERAGE_GRID:
            actions, info = make_selective_decisions(
                logits=logits,
                classes=classes,
                selector=selector,
                target_coverage=target_coverage,
                temperature=temperature,
            )
            metrics = evaluate_actions(actions, labels)
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "baseline": "ZS-temp-selective",
                    "selector": selector,
                    "temperature": temperature,
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
) -> list[dict]:
    data = np.load(npz_path, allow_pickle=True)
    classes = load_classes(data)
    dataset = dataset or infer_dataset_name(npz_path)
    backbone = backbone or infer_backbone(npz_path)

    image_feats = np.asarray(data["image_feats"], dtype=float)
    labels = data["classes"].astype(str)
    _, calibration_mask, query_mask = split_masks(data["splits"])

    text_protos = text_prototypes(data, classes)
    all_logits = zero_shot_logits(image_feats, text_protos)

    cal_logits = all_logits[calibration_mask]
    cal_labels = labels[calibration_mask]
    query_logits = all_logits[query_mask]
    query_labels = labels[query_mask]

    best_temp, best_nll = calibrate_temperature(cal_logits, cal_labels, classes)

    rows = [
        {
            **hard_decision_row(dataset, backbone, query_logits, query_labels, classes, temperature=1.0),
            "calibration_nll": "",
        }
    ]

    for row in selective_rows(dataset, backbone, query_logits, query_labels, classes, best_temp):
        row["calibration_nll"] = best_nll
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "backbone",
        "baseline",
        "selector",
        "temperature",
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
    print(f"calibration samples={int(calibration_mask.sum())} query samples={int(query_mask.sum())}")
    print(f"best temperature={best_temp} calibration_nll={best_nll:.6f}")
    print(f"wrote {len(rows)} rows -> {out_csv}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/clip_selective_baselines/zs_temp_selective_summary.csv"),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--backbone", default=None)
    args = parser.parse_args()

    run(args.npz, args.out, args.dataset, args.backbone)


if __name__ == "__main__":
    main()