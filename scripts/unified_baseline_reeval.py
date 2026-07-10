# -*- coding: utf-8 -*-
"""Unified re-evaluation of CLIP selective baselines for release comparison.

Re-evaluates ZS-Temp / Tip-Adapter / Proto-Adapter under the SAME protocol as
the main table (Table 2):
  - ground truth  : human decision labels (borderline -> revisit), 593 samples
  - episodes      : the frozen calibration/evaluation manifest, K=16 block, 100 episodes
  - hyperparams   : temperature / alpha,beta calibrated on the episode
                    calibration split (NLL over class labels)
  - reject rule   : selector threshold set on CALIBRATION scores at target
                    coverage, then applied to test (test never used for tuning)
  - metrics       : scripts.pooled_metrics.compute_pooled_metrics

This replaces the old Table 3 numbers, which used class-label GT (picking a
turning sample was not counted as a false pick) and test-side thresholds.

Run from project root:
    python scripts/unified_baseline_reeval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clip_selective_baselines import (
    accept_by_threshold,
    apply_reject,
    entropy,
    map_class_predictions_to_actions,
    max_probability,
    predict_classes_from_logits,
    softmax,
    threshold_by_coverage,
    top2_margin,
)
from scripts.pooled_metrics import compute_pooled_metrics
from scripts.run_proto_adapter_selective_baseline import calibrate_proto_weight, fused_proto_logits
from scripts.official_evaluation import load_ground_truth

CLASSES = ["ripe", "turning", "unripe"]  # sorted() order, same as runners
TEMPERATURE_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
ALPHA_GRID = (0.1, 0.5, 1.0, 2.0)
BETA_GRID = (1.0, 3.0, 5.5, 10.0)
COVERAGE_GRID = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0)
SELECTORS = ("msp", "margin", "entropy")
K = 16


def normalize_vector(x):
    x = np.asarray(x, dtype=float)
    return x / (np.linalg.norm(x) + 1e-12)


def nll(probs, label_idx):
    p_true = np.clip(probs[np.arange(len(label_idx)), label_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def selector_scores(probs, selector):
    if selector == "msp":
        return max_probability(probs), True
    if selector == "margin":
        return top2_margin(probs), True
    if selector == "entropy":
        return entropy(probs), False
    raise ValueError(selector)


def load_everything(root: Path):
    data = np.load(root / "outputs" / "features_vitb32.npz", allow_pickle=True)
    feats = np.asarray(data["image_feats"], dtype=float)
    splits = data["splits"].astype(str)
    class_labels = data["classes"].astype(str)
    paths = data["image_paths"].astype(str)

    query_mask = splits == "query_test"
    q_names = np.array([p.split("\\")[-1] for p in paths[query_mask]])
    gt_df = load_ground_truth(root)
    if not (q_names == gt_df["filename"].astype(str).to_numpy()).all():
        raise ValueError("query_test order does not match decision GT csv")
    gt_decisions = gt_df["decision_label"].to_numpy()

    text_protos = np.stack(
        [normalize_vector(np.asarray(data[f"text_{c}"], dtype=float).mean(axis=0)) for c in CLASSES]
    )

    manifest = json.loads((root / "outputs" / "episodes" / "manifest_K1-16_ep100.json").read_text(encoding="utf-8"))
    episodes = [e for e in manifest["episodes"] if e["k"] == K]
    if len(episodes) != 100:
        raise ValueError(f"expected 100 K={K} episodes, got {len(episodes)}")
    return feats, class_labels, query_mask, gt_decisions, text_protos, episodes


def episode_logits(feats, class_labels, query_mask, text_protos, ep):
    """Return dict of {baseline: (cal_logits, test_logits)} plus cal label idx."""
    sup_idx = np.concatenate([np.asarray(ep["support"][c], dtype=int) for c in CLASSES])
    cal_idx = np.concatenate([np.asarray(ep["calibration"][c], dtype=int) for c in CLASSES])

    sup_feats, sup_labels = feats[sup_idx], class_labels[sup_idx]
    cal_feats, cal_labels = feats[cal_idx], class_labels[cal_idx]
    test_feats = feats[query_mask]

    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    cal_y = np.array([cls_to_idx[c] for c in cal_labels])
    sup_onehot = np.zeros((len(sup_labels), len(CLASSES)))
    for i, c in enumerate(sup_labels):
        sup_onehot[i, cls_to_idx[c]] = 1.0

    zs_cal = cal_feats @ text_protos.T
    zs_test = test_feats @ text_protos.T

    out = {}

    # --- ZS-Temp: temperature by calibration NLL ---
    best_t = min(TEMPERATURE_GRID, key=lambda t: nll(softmax(zs_cal, t), cal_y))
    out["ZS-Temp"] = (zs_cal, zs_test, best_t)

    # --- Tip-Adapter: alpha/beta by calibration NLL, fused with zero-shot ---
    aff_cal = cal_feats @ sup_feats.T
    aff_test = test_feats @ sup_feats.T
    best = None
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            fused = zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)
            score = nll(softmax(fused, 1.0), cal_y)
            if best is None or score < best[0]:
                best = (score, a, b)
    _, a, b = best
    tip_cal = zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)
    tip_test = zs_test + a * (np.exp(b * (aff_test - 1.0)) @ sup_onehot)
    out["Tip-Adapter"] = (tip_cal, tip_test, 1.0)

    # --- Proto-Adapter: support prototypes, temperature 1.0 (as original) ---
    protos = np.stack(
        [normalize_vector(sup_feats[sup_labels == c].mean(axis=0)) for c in CLASSES]
    )
    cal_proto = cal_feats @ protos.T
    test_proto = test_feats @ protos.T
    proto_weight, _proto_nll = calibrate_proto_weight(
        zero_shot_logits=zs_cal,
        proto_logits=cal_proto,
        labels=cal_labels,
        classes=CLASSES,
    )
    out["Proto-Adapter"] = (
        fused_proto_logits(zs_cal, cal_proto, proto_weight),
        fused_proto_logits(zs_test, test_proto, proto_weight),
        1.0,
    )

    return out


def run(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feats, class_labels, query_mask, gt_decisions, text_protos, episodes = load_everything(root)

    # pooled[(baseline, selector, coverage)] -> list of per-episode action arrays
    pooled: dict[tuple, list] = {}
    for ep in episodes:
        logits_map = episode_logits(feats, class_labels, query_mask, text_protos, ep)
        for baseline, (cal_logits, test_logits, temp) in logits_map.items():
            cal_probs = softmax(cal_logits, temp)
            test_probs = softmax(test_logits, temp)
            pred = predict_classes_from_logits(test_logits, CLASSES)
            base_actions = map_class_predictions_to_actions(pred)
            for selector in SELECTORS:
                cal_s, hi = selector_scores(cal_probs, selector)
                test_s, _ = selector_scores(test_probs, selector)
                for cov in COVERAGE_GRID:
                    thr = threshold_by_coverage(cal_s, cov, hi)
                    accept = accept_by_threshold(test_s, thr, hi)
                    actions = apply_reject(base_actions, accept).astype(str)
                    pooled.setdefault((baseline, selector, cov), []).append(actions)

    rows = []
    for (baseline, selector, cov), action_list in pooled.items():
        all_preds = np.concatenate(action_list)
        all_gts = np.tile(gt_decisions, len(action_list))
        m = compute_pooled_metrics(all_preds, all_gts)
        rows.append(
            {
                "baseline": baseline,
                "selector": selector,
                "target_coverage": cov,
                "false_pick_rate": m["false_pick_rate"],
                "pick_precision": m["pick_precision"],
                "pick_recall": m["pick_recall"],
                "revisit_burden": m["revisit_burden"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["baseline", "selector", "target_coverage"]).reset_index(drop=True)
    csv_path = out_dir / "unified_baseline_reeval_K16.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    # --- markdown: msp selector, all coverages, plus TAP/B5 reference rows ---
    lines = [
        "# Unified baseline re-evaluation (decision GT, calibration-only thresholds, K=16)",
        "",
        "Old Table 3 used class-label GT (picking a turning sample was NOT a false",
        "pick) and test-side coverage thresholds. This table uses the exact main-",
        "table protocol. Selector shown: msp (margin/entropy in CSV, similar).",
        "",
        "| Method | Target cov. | False-pick % | Precision % | Recall % | Revisit % |",
        "|--------|-------------|-------------|-------------|----------|-----------|",
    ]
    sub = df[df["selector"] == "msp"]
    for baseline in ("ZS-Temp", "Tip-Adapter", "Proto-Adapter"):
        for _, r in sub[sub["baseline"] == baseline].iterrows():
            lines.append(
                f"| {baseline} | {r['target_coverage']:.0%} | {r['false_pick_rate']*100:.1f} "
                f"| {r['pick_precision']*100:.1f} | {r['pick_recall']*100:.1f} "
                f"| {r['revisit_burden']*100:.1f} |"
            )
    lines += [
        "| FS-Proto-Reject (ref) | - | 4.2 | 95.8 | 65.7 | 39.9 |",
        "| TAP-Correct (ref) | - | 9.1 | 90.9 | 80.6 | 19.2 |",
    ]
    (out_dir / "UNIFIED_BASELINE_REEVAL_K16.md").write_text("\n".join(lines), encoding="utf-8")

    print(df[df["selector"] == "msp"].to_string(index=False))
    print(f"\ncsv:   {csv_path}")
    print(f"table: {out_dir / 'UNIFIED_BASELINE_REEVAL_K16.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out if args.out is not None else args.root / "outputs" / "unified_baseline_reeval"
    run(args.root, out_dir)


if __name__ == "__main__":
    main()
