# -*- coding: utf-8 -*-
"""Linear-probe trained reference for supplementary release evaluation.

Two training budgets, evaluated under the exact main-table protocol
(decision GT, K=16 manifest episodes, pooled over 100 episodes):
  - LP-48 : logistic regression on the 48 support features only
            (same budget as the training-free methods)
  - LP-648: support + calibration (648 labeled samples) -- data upper bound
            (note: calibration is reused for training AND thresholding, so
             this reference is optimistic by construction)

Each probe is read out two ways:
  - Hard  : argmax class -> action map (turning -> revisit), no rejection
  - E(LP) : softmax probs -> ordinal expectation E -> calibration-recipe thresholds
            + U_cut gate (the probe as a fourth endpoint source)

Run from project root:
    python scripts/linear_probe_reference.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pooled_metrics import compute_pooled_metrics
from scripts.endpoint_source_ablation import (
    ALPHA,
    GAMMA,
    calib_T_high,
    calib_T_low,
    load_everything,
)
from scripts.official_evaluation import METRICS
from tapcorrect.decision import compute_uncertainty, decide_episode_v1_endpoint

CLASSES = ("ripe", "turning", "unripe")
Q_LIST = (50, 60, 70, 75, 80, 85, 90, 95)
Q_FROZEN = 80
K = 16


def fit_probe(X, y, seed):
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X, y)
    return clf


def probs_in_class_order(clf, X):
    p = clf.predict_proba(X)
    order = [list(clf.classes_).index(c) for c in CLASSES]
    return p[:, order]


def hard_actions(probs):
    pred = np.array(CLASSES)[probs.argmax(axis=1)]
    actions = np.full(len(pred), "revisit", dtype=object)
    actions[pred == "ripe"] = "pick"
    actions[pred == "unripe"] = "wait"
    return actions.astype(str)


def e_actions(E_cal, cal_labels, E_test, q):
    sr = E_cal[cal_labels == "ripe"]
    su = E_cal[cal_labels == "unripe"]
    pick_min = calib_T_high(sr, su, ALPHA)
    wait_max = calib_T_low(sr, su, GAMMA)
    t_low, t_high = min(pick_min, wait_max), max(pick_min, wait_max)
    u_cal = compute_uncertainty(E_cal, t_high, t_low)
    u_cut = float(np.percentile(u_cal, q))
    dec, _ = decide_episode_v1_endpoint(E_test, t_high, t_low, u_cut)
    return dec.astype(str)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.root / "outputs" / "linear_probe_reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    feats, class_labels, query_mask, gt_decisions, _text_protos, episodes = load_everything(
        args.root, None, K
    )
    test_feats = feats[query_mask]
    anchors = np.array([1.0, 0.5, 0.0])

    # collected[variant] -> list of per-episode action arrays;
    # variants: LP48-Hard, LP648-Hard, and (LP48-E, q) / (LP648-E, q)
    hard_coll = {"LP48": [], "LP648": []}
    e_coll = {("LP48", q): [] for q in Q_LIST}
    e_coll.update({("LP648", q): [] for q in Q_LIST})

    for i, ep in enumerate(episodes):
        sup_idx = np.concatenate([np.asarray(ep["support"][c], dtype=int) for c in CLASSES])
        cal_idx = np.concatenate([np.asarray(ep["calibration"][c], dtype=int) for c in CLASSES])
        Xs, ys = feats[sup_idx], class_labels[sup_idx]
        Xc, yc = feats[cal_idx], class_labels[cal_idx]

        for name, (X, y) in {
            "LP48": (Xs, ys),
            "LP648": (np.concatenate([Xs, Xc]), np.concatenate([ys, yc])),
        }.items():
            clf = fit_probe(X, y, args.seed + i)
            p_cal = probs_in_class_order(clf, Xc)
            p_test = probs_in_class_order(clf, test_feats)
            hard_coll[name].append(hard_actions(p_test))
            E_cal, E_test = p_cal @ anchors, p_test @ anchors
            for q in Q_LIST:
                e_coll[(name, q)].append(e_actions(E_cal, yc, E_test, q))
        if (i + 1) % 20 == 0:
            print(f"  episode {i+1}/100 done", flush=True)

    rows = []
    def add_row(label, action_list):
        preds = np.concatenate(action_list)
        gts = np.tile(gt_decisions, len(action_list))
        m = compute_pooled_metrics(preds, gts)
        rows.append({"variant": label, **{k: float(m[k]) for k in METRICS}})

    add_row("LP48-Hard", hard_coll["LP48"])
    add_row("LP648-Hard", hard_coll["LP648"])
    for name in ("LP48", "LP648"):
        for q in Q_LIST:
            add_row(f"{name}-E-q{q}", e_coll[(name, q)])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "linear_probe_reference_K16.csv", index=False, float_format="%.6f")

    print("\n=== headline comparison (decision GT, K=16, pooled 100 episodes) ===")
    print("references: TAP-Correct E(tip) q80 = fp 7.6 / prec 92.4 / rec 82.8 / rev 18.5")
    print("            TAP-Correct E(visual) q80 = fp 9.1 / prec 90.9 / rec 80.6 / rev 19.2")
    show = ["LP48-Hard", f"LP48-E-q{Q_FROZEN}", "LP648-Hard", f"LP648-E-q{Q_FROZEN}"]
    for label in show:
        r = df[df["variant"] == label].iloc[0]
        print(
            f"{label:14s}: fp {r['false_pick_rate']*100:5.1f} | prec {r['pick_precision']*100:5.1f} "
            f"| rec {r['pick_recall']*100:5.1f} | rev {r['revisit_burden']*100:5.1f}"
        )
    print(f"\nfull q-sweep csv -> {out_dir / 'linear_probe_reference_K16.csv'}")


if __name__ == "__main__":
    main()
