# -*- coding: utf-8 -*-
"""Calibration-label budget sweep: TAP E(tip) vs linear probe (post-B2 question).

Both methods consume the SAME m labeled calibration samples per class
(nested subsampling, paired within episode):
  - TAP-E(tip): alpha/beta by NLL + calibration-recipe thresholds + U_cut(q=80),
                all on the m-per-class subsample; zero training.
  - LP        : logistic regression trained on 48 support + 3*m calibration
                samples; hard argmax readout (turning -> revisit).

Question: where (if anywhere) does the training-free method overtake the
trained probe as calibration labels shrink? Regression anchors at m=200:
E(tip) ~ 7.6/92.4/82.8/18.5, LP ~ 3.1/96.9/90.3/21.8.

Run from project root:
    python scripts/calibration_budget_sweep.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pooled_metrics import compute_pooled_metrics
from scripts.endpoint_source_ablation import (
    ALPHA,
    ALPHA_GRID,
    BETA_GRID,
    GAMMA,
    calib_T_high,
    calib_T_low,
    load_everything,
    nll,
    softmax_rows,
)
from scripts.official_evaluation import METRICS
from tapcorrect.decision import compute_uncertainty, decide_episode_v1_endpoint

CLASSES = ("ripe", "turning", "unripe")
BUDGETS = (200, 100, 50, 30, 20, 10, 5)
Q_FROZEN = 80
K = 16
ANCHORS = np.array([1.0, 0.5, 0.0])


def tap_tip_actions(zs_cal, zs_test, aff_cal, exp_b_test, sup_onehot, cal_y, cal_labels):
    """Full TAP-E(tip) pipeline on one calibration subsample."""
    best = None
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            fused = zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)
            s = nll(softmax_rows(fused), cal_y)
            if best is None or s < best[0]:
                best = (s, a, b)
    _, a, b = best
    E_cal = softmax_rows(zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)) @ ANCHORS
    E_test = softmax_rows(zs_test + a * (exp_b_test[b] @ sup_onehot)) @ ANCHORS

    sr = E_cal[cal_labels == "ripe"]
    su = E_cal[cal_labels == "unripe"]
    pick_min = calib_T_high(sr, su, ALPHA)
    wait_max = calib_T_low(sr, su, GAMMA)
    t_low, t_high = min(pick_min, wait_max), max(pick_min, wait_max)
    u_cal = compute_uncertainty(E_cal, t_high, t_low)
    u_cut = float(np.percentile(u_cal, Q_FROZEN))
    dec, _ = decide_episode_v1_endpoint(E_test, t_high, t_low, u_cut)
    return dec.astype(str)


def lp_actions(X_train, y_train, test_feats, seed):
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X_train, y_train)
    order = [list(clf.classes_).index(c) for c in CLASSES]
    pred = np.array(CLASSES)[clf.predict_proba(test_feats)[:, order].argmax(axis=1)]
    actions = np.full(len(pred), "revisit", dtype=object)
    actions[pred == "ripe"] = "pick"
    actions[pred == "unripe"] = "wait"
    return actions.astype(str)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    out_dir = args.root / "outputs" / "calibration_budget_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    feats, class_labels, query_mask, gt_decisions, text_protos, episodes = load_everything(
        args.root, None, args.k
    )
    test_feats = feats[query_mask]
    zs_test = test_feats @ text_protos.T
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

    coll = {("TAP-E(tip)", m): [] for m in BUDGETS}
    coll.update({("LP", m): [] for m in BUDGETS})

    rng = np.random.RandomState(args.seed)
    for i, ep in enumerate(episodes):
        sup_idx = np.concatenate([np.asarray(ep["support"][c], dtype=int) for c in CLASSES])
        sup_feats, sup_labels = feats[sup_idx], class_labels[sup_idx]
        sup_onehot = np.zeros((len(sup_labels), len(CLASSES)))
        for j, c in enumerate(sup_labels):
            sup_onehot[j, cls_to_idx[c]] = 1.0

        aff_test = test_feats @ sup_feats.T
        exp_b_test = {b: np.exp(b * (aff_test - 1.0)) for b in BETA_GRID}

        # one permutation per class -> nested subsamples across budgets
        perms = {c: rng.permutation(np.asarray(ep["calibration"][c], dtype=int)) for c in CLASSES}

        for m in BUDGETS:
            cal_idx = np.concatenate([perms[c][:m] for c in CLASSES])
            cal_feats, cal_labels = feats[cal_idx], class_labels[cal_idx]
            cal_y = np.array([cls_to_idx[c] for c in cal_labels])

            zs_cal = cal_feats @ text_protos.T
            aff_cal = cal_feats @ sup_feats.T
            coll[("TAP-E(tip)", m)].append(
                tap_tip_actions(zs_cal, zs_test, aff_cal, exp_b_test, sup_onehot, cal_y, cal_labels)
            )
            coll[("LP", m)].append(
                lp_actions(
                    np.concatenate([sup_feats, cal_feats]),
                    np.concatenate([sup_labels, cal_labels]),
                    test_feats,
                    args.seed + i,
                )
            )
        if (i + 1) % 20 == 0:
            print(f"  episode {i+1}/100 done", flush=True)

    rows = []
    for (method, m), action_list in coll.items():
        preds = np.concatenate(action_list)
        gts = np.tile(gt_decisions, len(action_list))
        metrics = compute_pooled_metrics(preds, gts)
        rows.append({"method": method, "m_per_class": m, **{k: float(metrics[k]) for k in METRICS}})
    df = pd.DataFrame(rows).sort_values(["method", "m_per_class"], ascending=[True, False])
    df.to_csv(out_dir / f"calibration_budget_sweep_K{args.k}.csv", index=False, float_format="%.6f")

    if args.k == 16:
        print("\n=== regression anchors (K=16, m=200): E(tip)~7.6/92.4/82.8/18.5, LP~3.1/96.9/90.3/21.8 ===")
    else:
        print(f"\n=== K={args.k}: compare TAP m=200 row with endpoint_source_ablation vitb32_K{args.k} tip q80 ===")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    styles = {"TAP-E(tip)": ("#1f77b4", "o-"), "LP": ("#d62728", "s--")}
    panels = [("false_pick_rate", "False-pick proportion (%)"),
              ("pick_recall", "Pick recall (%)"),
              ("revisit_burden", "Revisit burden (%)")]
    for ax, (metric, ylab) in zip(axes, panels):
        for method, (c, fmt) in styles.items():
            sub = df[df["method"] == method].sort_values("m_per_class")
            ax.plot(sub["m_per_class"], sub[metric] * 100, fmt, color=c, label=method)
        ax.set_xscale("log")
        ax.set_xticks(list(BUDGETS))
        ax.set_xticklabels([str(m) for m in BUDGETS])
        ax.set_xlabel("Calibration labels per class")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Calibration-label budget: training-free TAP-E(tip) vs trained linear probe (K={args.k})")
    fig.tight_layout()
    fig.savefig(out_dir / f"calibration_budget_sweep_K{args.k}.png", dpi=200)
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
