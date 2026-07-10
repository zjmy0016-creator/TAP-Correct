# -*- coding: utf-8 -*-
"""Rebuild of the uncertainty evidence chain (T3 + T2), source-parameterized.

The original scripts/t2_borderline_auroc.py and t3_aurc_main_evidence.py were
lost after producing the numbers in the work log (2026-07-08). This script
rebuilds both permanently and adds the endpoint-source dimension:

  T3 (primary): uncertainty-ordered deferral vs random deferral on all 593
      query samples x 100 K=16 episodes; pooled false-pick per defer budget;
      AURC = mean pooled risk over budgets; bootstrap CI on the difference.
  T2 (secondary): per-fruit mean uncertainty over 100 episodes on the 129
      turning samples; AUROC borderline(69) vs non-borderline(60);
      one-sided Mann-Whitney U.

Thresholds per episode use the calibration recipe on the calibration split only.
Regression anchors (source=visual, from work log): T3 risk 12.23%->3.41%,
AURC diff -4.72pp [-5.06,-4.38]; T2 AUROC 0.6215 [0.519,0.718], p=0.0088.

Run from project root:
    python scripts/uncertainty_evidence.py            # all three sources
    python scripts/uncertainty_evidence.py --source tip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.endpoint_source_ablation import (
    ALPHA,
    GAMMA,
    calib_T_high,
    calib_T_low,
    episode_E_scores,
    load_everything,
)
from tapcorrect.decision import compute_uncertainty

DEFERS = np.round(np.arange(0.0, 0.601, 0.05), 3)
N_RANDOM_SEEDS = 5
CLASSES = ("ripe", "turning", "unripe")


def episode_thresholds(E_cal, cal_labels):
    sr = E_cal[cal_labels == "ripe"]
    su = E_cal[cal_labels == "unripe"]
    pick_min = calib_T_high(sr, su, ALPHA)
    wait_max = calib_T_low(sr, su, GAMMA)
    return min(pick_min, wait_max), max(pick_min, wait_max)


def defer_counts(E, u_order, t_low, t_high, gt_is_pick, budget):
    """Apply banded endpoint decisions after deferring top-budget by given order."""
    n = len(E)
    nd = int(n * budget)
    keep = u_order[nd:]
    picks = keep[E[keep] >= t_high]
    n_pick = len(picks)
    n_fp = int((~gt_is_pick[picks]).sum())
    return n_pick, n_fp


def run_source(source, feats, class_labels, query_mask, gt_decisions, text_protos,
               episodes, turning_df, gt_filenames, seed):
    gt_is_pick = gt_decisions == "pick"
    n_ep = len(episodes)
    # per-episode, per-budget counts: ordered and random
    ord_counts = np.zeros((n_ep, len(DEFERS), 2))
    rnd_counts = np.zeros((n_ep, len(DEFERS), 2))
    u_sum = np.zeros(int(query_mask.sum()))

    rng = np.random.RandomState(seed)
    for i, ep in enumerate(episodes):
        scores, cal_labels = episode_E_scores(feats, class_labels, query_mask, text_protos, ep)
        E_cal, E_test = scores[source]
        t_low, t_high = episode_thresholds(E_cal, cal_labels)
        u = compute_uncertainty(E_test, t_high, t_low)
        u_sum += u

        order_u = np.argsort(u)[::-1]  # most uncertain first
        for j, dr in enumerate(DEFERS):
            ord_counts[i, j] = defer_counts(E_test, order_u, t_low, t_high, gt_is_pick, dr)
        for _ in range(N_RANDOM_SEEDS):
            order_r = rng.permutation(len(u))
            for j, dr in enumerate(DEFERS):
                rnd_counts[i, j] += defer_counts(E_test, order_r, t_low, t_high, gt_is_pick, dr)

    # --- T3: risk curves + AURC ---
    def pooled_risk(counts, idx):
        pick = counts[idx, :, 0].sum(axis=0)
        fp = counts[idx, :, 1].sum(axis=0)
        return np.where(pick > 0, fp / pick, 0.0)

    all_idx = np.arange(n_ep)
    risk_ord = pooled_risk(ord_counts, all_idx)
    risk_rnd = pooled_risk(rnd_counts, all_idx)
    aurc_ord, aurc_rnd = risk_ord.mean(), risk_rnd.mean()

    boot_diffs = []
    for _ in range(1000):
        idx = rng.choice(n_ep, n_ep, replace=True)
        boot_diffs.append(pooled_risk(ord_counts, idx).mean() - pooled_risk(rnd_counts, idx).mean())
    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])

    # --- T2: borderline AUROC on per-fruit mean uncertainty ---
    u_mean = u_sum / n_ep
    name_to_idx = {n: i for i, n in enumerate(gt_filenames)}
    t_idx = np.array([name_to_idx[f] for f in turning_df["filename"].astype(str)])
    is_border = (turning_df["decision_label"] == "borderline").to_numpy()
    u_t = u_mean[t_idx]

    pos, neg = u_t[is_border], u_t[~is_border]
    auroc = float(np.mean([(pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()]))
    boot_auroc = []
    for _ in range(1000):
        p = pos[rng.choice(len(pos), len(pos), replace=True)]
        nn = neg[rng.choice(len(neg), len(neg), replace=True)]
        boot_auroc.append((p[:, None] > nn[None, :]).mean() + 0.5 * (p[:, None] == nn[None, :]).mean())
    a_lo, a_hi = np.percentile(boot_auroc, [2.5, 97.5])
    stat, pval = mannwhitneyu(pos, neg, alternative="greater")

    return {
        "risk_curve": pd.DataFrame({"defer": DEFERS, "risk_ordered": risk_ord, "risk_random": risk_rnd}),
        "t3": dict(aurc_ordered=aurc_ord, aurc_random=aurc_rnd,
                   diff=aurc_ord - aurc_rnd, ci_lower=lo, ci_upper=hi),
        "t2": dict(auroc=auroc, ci_lower=a_lo, ci_upper=a_hi, U=float(stat), p=float(pval),
                   n_border=int(is_border.sum()), n_non=int((~is_border).sum())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--source", choices=["visual", "tip", "text", "all"], default="all")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.root / "outputs" / "uncertainty_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    feats, class_labels, query_mask, gt_decisions, text_protos, episodes = load_everything(
        args.root, None, args.k
    )
    gt_df = pd.read_csv(
        args.root / "outputs" / "decision_gold" / "turning_decision_dataset" / "labels"
        / "test_decision_ground_truth_clean.csv", encoding="utf-8-sig",
    )
    gt_filenames = gt_df["filename"].astype(str).to_numpy()
    turning_df = pd.read_csv(
        args.root / "outputs" / "decision_gold" / "turning_decision_dataset" / "labels"
        / "turning_decision_labels_clean.csv", encoding="utf-8-sig",
    )

    sources = ["visual", "tip", "text"] if args.source == "all" else [args.source]
    summary_rows = []
    for source in sources:
        print(f"\n===== source: {source} (K={args.k}) =====")
        res = run_source(source, feats, class_labels, query_mask, gt_decisions,
                         text_protos, episodes, turning_df, gt_filenames, args.seed)
        rc = res["risk_curve"]
        rc.to_csv(out_dir / f"t3_risk_curve_{source}_K{args.k}.csv", index=False, float_format="%.6f")
        t3, t2 = res["t3"], res["t2"]
        print("T3 risk (ordered vs random):")
        for dr in (0.0, 0.2, 0.4, 0.6):
            row = rc[rc["defer"] == dr].iloc[0]
            print(f"  defer {dr:.1f}: {row['risk_ordered']*100:5.2f}% vs {row['risk_random']*100:5.2f}%")
        print(f"T3 AURC: {t3['aurc_ordered']*100:.2f}% vs {t3['aurc_random']*100:.2f}% | "
              f"diff {t3['diff']*100:+.2f}pp [{t3['ci_lower']*100:+.2f}, {t3['ci_upper']*100:+.2f}]")
        print(f"T2 AUROC: {t2['auroc']:.4f} [{t2['ci_lower']:.4f}, {t2['ci_upper']:.4f}] | "
              f"U={t2['U']:.0f} p={t2['p']:.4f} (border n={t2['n_border']} vs non n={t2['n_non']})")
        summary_rows.append({"source": source, **{f"t3_{k}": v for k, v in t3.items()},
                             **{f"t2_{k}": v for k, v in t2.items()}})

    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"uncertainty_evidence_summary_K{args.k}.csv", index=False, float_format="%.6f"
    )
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
