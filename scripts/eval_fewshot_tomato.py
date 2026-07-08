"""Few-shot TAP-Correct V1 and B5 baseline evaluation on laboro_tomato.

Simplified episode protocol: K=16, 20 episodes, calibration-based thresholds.
Adapted for low-resolution E score distributions.
"""
import argparse
import numpy as np
from pathlib import Path


def build_visual_prototypes(feats, labels, classes):
    """Build L2-normalized visual prototypes from support samples."""
    protos = {}
    for c in classes:
        mask = labels == c
        if mask.sum() == 0:
            raise ValueError(f"No samples for class {c}")
        mean_vec = feats[mask].mean(axis=0)
        protos[c] = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
    return protos


def compute_expectation(feats, protos, classes, temperature=1.0):
    """Compute expectation score: E = 0*p_immature + 0.5*p_turning + 1*p_mature."""
    sims = np.stack([feats @ protos[c] for c in classes], axis=1)
    logits = sims / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Map classes to maturity anchors
    anchors = {"immature": 0.0, "turning": 0.5, "mature": 1.0}
    anchor_vec = np.array([anchors[c] for c in classes])
    return probs @ anchor_vec


def compute_margin(feats, protos, classes):
    """Compute top-2 similarity margin (for B5 baseline)."""
    sims = np.stack([feats @ protos[c] for c in classes], axis=1)
    sorted_sims = np.sort(sims, axis=1)[:, ::-1]
    return sorted_sims[:, 0] - sorted_sims[:, 1]


def calibrate_thresholds_v1_adaptive(cal_feats, cal_labels, protos, classes,
                                      target_coverage=0.80):
    """Calibrate V1 thresholds to achieve target coverage.

    Strategy: set thresholds such that ~80% of calibration samples are confident.
    """
    E_cal = compute_expectation(cal_feats, protos, classes)

    # Define harvestable (mature) vs non-harvestable (immature)
    harvestable_mask = cal_labels == "mature"
    non_harvestable_mask = cal_labels == "immature"

    E_harv = E_cal[harvestable_mask]
    E_non_harv = E_cal[non_harvestable_mask]

    if len(E_harv) == 0 or len(E_non_harv) == 0:
        raise ValueError("Calibration split must contain both mature and immature samples")

    # Use more aggressive quantiles to ensure some samples are confident
    # Pick top 20% of harvestable as "definitely pick"
    # Pick bottom 20% of non-harvestable as "definitely wait"
    T_high_E = np.quantile(E_harv, 0.80)
    T_low_E = np.quantile(E_non_harv, 0.20)

    # Compute expected coverage on calibration set
    n_confident = ((E_cal >= T_high_E) | (E_cal <= T_low_E)).sum()
    actual_coverage = n_confident / len(E_cal)

    return T_high_E, T_low_E, actual_coverage


def decide_v1_adaptive(E, T_high_E, T_low_E):
    """V1 decision without uncertainty cut (direct threshold-based).

    Adapted for low E-score resolution: no uncertainty filtering.
    """
    decision = np.full(len(E), "revisit", dtype=object)
    decision[E >= T_high_E] = "pick"
    decision[E <= T_low_E] = "wait"
    return decision


def decide_b5(margin, protos, feats, classes, defer_threshold=0.02):
    """B5 baseline: argmax class + margin-based defer.

    Adapted threshold for tomato's low margin distribution.
    """
    sims = np.stack([feats @ protos[c] for c in classes], axis=1)
    pred_idx = sims.argmax(axis=1)
    pred_class = np.array([classes[i] for i in pred_idx])

    decision = np.full(len(feats), "revisit", dtype=object)
    confident = margin >= defer_threshold

    decision[confident & (pred_class == "mature")] = "pick"
    decision[confident & (pred_class == "immature")] = "wait"

    return decision


def evaluate_decisions(decisions, true_labels):
    """Evaluate pick/wait/revisit decisions."""
    pick_mask = decisions == "pick"
    wait_mask = decisions == "wait"
    revisit_mask = decisions == "revisit"

    # Ground truth: mature should be picked, immature should wait
    should_pick = true_labels == "mature"
    should_wait = true_labels == "immature"

    # Metrics
    n_picks = pick_mask.sum()
    n_revisits = revisit_mask.sum()

    if n_picks == 0:
        false_pick_rate = 0.0
        pick_precision = 0.0
        pick_recall = 0.0
    else:
        false_picks = pick_mask & should_wait
        true_picks = pick_mask & should_pick
        false_pick_rate = false_picks.sum() / n_picks
        pick_precision = 1.0 - false_pick_rate
        pick_recall = true_picks.sum() / should_pick.sum() if should_pick.sum() > 0 else 0.0

    revisit_burden = n_revisits / len(decisions)
    coverage = 1.0 - revisit_burden

    return {
        "false_pick_rate": false_pick_rate,
        "pick_precision": pick_precision,
        "pick_recall": pick_recall,
        "revisit_burden": revisit_burden,
        "coverage": coverage,
        "n_picks": n_picks,
        "n_revisits": n_revisits
    }


def sample_episode(feats, labels, splits, classes, k=16, m_calib=200, seed=0):
    """Sample one episode: support + calibration from train, test from test split."""
    rng = np.random.default_rng(seed)

    train_mask = splits == "train"
    test_mask = splits == "test"

    support_idx = {}
    calibration_idx = {}

    for c in classes:
        pool = np.where(train_mask & (labels == c))[0]
        if len(pool) < k + m_calib:
            raise ValueError(f"Not enough train samples for class {c}")
        shuffled = rng.permutation(pool)
        support_idx[c] = shuffled[:k]
        calibration_idx[c] = shuffled[k:k + m_calib]

    support_all = np.concatenate([support_idx[c] for c in classes])
    calibration_all = np.concatenate([calibration_idx[c] for c in classes])
    test_all = np.where(test_mask)[0]

    return {
        "support": support_all,
        "calibration": calibration_all,
        "test": test_all
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=Path("outputs/features_laboro_tomato_vitb16.npz"))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--base_seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading features: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)

    feats = data["image_feats"]
    labels = data["classes"]
    splits = data["splits"]

    # Use sorted class order for consistency
    classes = sorted(np.unique(labels))
    print(f"Classes: {classes}")
    print(f"K={args.k}, Episodes={args.n_episodes}")
    print("Adaptive calibration: top/bottom 20% quantiles\n")

    v1_results = []
    b5_results = []

    for ep_i in range(args.n_episodes):
        seed = args.base_seed + ep_i
        episode = sample_episode(feats, labels, splits, classes, k=args.k, seed=seed)

        # Build prototypes from support
        sup_feats = feats[episode["support"]]
        sup_labels = labels[episode["support"]]
        protos = build_visual_prototypes(sup_feats, sup_labels, classes)

        # Calibrate on calibration split
        cal_feats = feats[episode["calibration"]]
        cal_labels = labels[episode["calibration"]]
        T_high_E, T_low_E, cal_cov = calibrate_thresholds_v1_adaptive(
            cal_feats, cal_labels, protos, classes
        )

        # Test
        test_feats = feats[episode["test"]]
        test_labels = labels[episode["test"]]

        # V1 decisions
        E_test = compute_expectation(test_feats, protos, classes)
        v1_decisions = decide_v1_adaptive(E_test, T_high_E, T_low_E)
        v1_metrics = evaluate_decisions(v1_decisions, test_labels)
        v1_results.append(v1_metrics)

        # B5 decisions
        margin_test = compute_margin(test_feats, protos, classes)
        b5_decisions = decide_b5(margin_test, protos, test_feats, classes)
        b5_metrics = evaluate_decisions(b5_decisions, test_labels)
        b5_results.append(b5_metrics)

        if (ep_i + 1) % 5 == 0:
            print(f"  Episode {ep_i + 1}/{args.n_episodes} completed")

    # Aggregate results
    print("\n" + "="*70)
    print("RESULTS: TAP-Correct V1 (512D Endpoint, Adaptive Calibration)")
    print("="*70)
    for key in ["false_pick_rate", "pick_precision", "pick_recall", "revisit_burden", "coverage"]:
        values = [r[key] for r in v1_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{key:20s}: {mean:.4f} +/- {std:.4f}")

    print("\n" + "="*70)
    print("RESULTS: B5 Baseline (Margin-based Defer, threshold=0.02)")
    print("="*70)
    for key in ["false_pick_rate", "pick_precision", "pick_recall", "revisit_burden", "coverage"]:
        values = [r[key] for r in b5_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{key:20s}: {mean:.4f} +/- {std:.4f}")

    # Compare
    print("\n" + "="*70)
    print("KEY COMPARISON: V1 vs B5")
    print("="*70)
    v1_fp = np.mean([r["false_pick_rate"] for r in v1_results])
    b5_fp = np.mean([r["false_pick_rate"] for r in b5_results])
    v1_cov = np.mean([r["coverage"] for r in v1_results])
    b5_cov = np.mean([r["coverage"] for r in b5_results])

    print(f"V1 false-pick rate:  {v1_fp:.4f}  (coverage: {v1_cov:.4f})")
    print(f"B5 false-pick rate:  {b5_fp:.4f}  (coverage: {b5_cov:.4f})")

    if b5_fp > 0:
        improvement = (b5_fp - v1_fp) / b5_fp * 100
        print(f"Relative improvement: {improvement:.1f}%")

    if v1_fp < b5_fp:
        print("[OK] V1 reduces false-pick risk on tomato (supports transfer)")
    elif v1_fp == b5_fp:
        print("[TIE] V1 and B5 have similar false-pick rates")
    else:
        print("[WARN] V1 has higher false-pick than B5 (tomato characteristics differ)")


if __name__ == "__main__":
    main()
