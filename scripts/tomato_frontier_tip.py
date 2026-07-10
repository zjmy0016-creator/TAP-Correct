# -*- coding: utf-8 -*-
"""Laboro Tomato frontier with endpoint-source ablation (text / visual / tip).

Reuses the frozen laboro_tomato_frontier.py protocol (strict GT: picking immature OR
turning is a false pick; calibration-recipe calibration on the episode calibration
split; deferral sweep on endpoint uncertainty). Adds two E sources on top of
the existing visual-prototype endpoint:
  - text : zero-shot text-prototype logits -> E
  - tip  : Tip-Adapter fused logits (alpha/beta by calibration NLL) -> E

Same episode seeds as laboro_tomato_frontier.py, so the visual frontier must
reproduce the existing frontier CSV (regression anchor).

Run from project root:
    python scripts/tomato_frontier_tip.py --npz outputs/features_laboro_tomato_vitb16.npz
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.laboro_tomato_frontier import (
    ALPHA,
    ANCHORS,
    BASE_SEED,
    CLASSES,
    GAMMA,
    M_CALIB,
    b5_operating_point,
    build_b5_family_frontier,
    build_frontier,
    calib_T_high,
    calib_T_low,
    order_thr,
    protos_from_support,
    sample_episode,
    score_E_and_sims,
    top2_margin,
)
from tapcorrect.decision import compute_uncertainty

ALPHA_GRID = (0.1, 0.5, 1.0, 2.0)
BETA_GRID = (1.0, 3.0, 5.5, 10.0)


def softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def nll(probs, label_idx):
    p_true = np.clip(probs[np.arange(len(label_idx)), label_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def text_prototypes(npz):
    protos = []
    for c in CLASSES:
        v = np.asarray(npz[f"text_{c}"], dtype=float).mean(axis=0)
        protos.append(v / (np.linalg.norm(v) + 1e-12))
    return np.stack(protos)


def calibrate_and_pack(E_cal, E_q, cal_labels, prefix):
    """calibration-recipe thresholds on calibration; return dict entries for one source."""
    pm = calib_T_high(E_cal[cal_labels == "mature"], E_cal[cal_labels == "turning"],
                      E_cal[cal_labels == "immature"], ALPHA)
    wm = calib_T_low(E_cal[cal_labels == "mature"], E_cal[cal_labels == "turning"],
                     E_cal[cal_labels == "immature"], GAMMA)
    t_low, t_high = order_thr(pm, wm)
    u = compute_uncertainty(E_q, t_high, t_low)
    return {
        f"E_{prefix}": E_q,
        f"u_{prefix}": u,
        f"T_high_{prefix}": t_high,
        f"T_low_{prefix}": t_low,
    }


def build_frontier_src(EPS, is_gt_pick, prefix):
    """build_frontier but with per-source uncertainty key."""
    eps_view = [
        dict(u=d[f"u_{prefix}"], sc=d[f"E_{prefix}"], Th=d[f"T_high_{prefix}"], Tl=d[f"T_low_{prefix}"])
        for d in EPS
    ]
    packed = [dict(u=e["u"], E=e["sc"], T_high_E=e["Th"], T_low_E=e["Tl"]) for e in eps_view]
    return build_frontier(packed, is_gt_pick, "E", "T_high_E", "T_low_E")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=ROOT / "outputs" / "features_laboro_tomato_vitb16.npz")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n_episodes", type=int, default=100)
    args = ap.parse_args()

    tag = args.npz.stem.replace("features_", "")
    out_dir = ROOT / "outputs" / "tomato_endpoint_source"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz, allow_pickle=True)
    feats = np.asarray(npz["image_feats"], dtype=float)
    labels = npz["classes"].astype(str)
    splits = npz["splits"].astype(str)
    t_protos = text_prototypes(npz)

    q_mask = splits == "test"
    qfeats, q_labels = feats[q_mask], labels[q_mask]
    is_gt_pick = q_labels == "mature"
    print(f"=== tomato endpoint-source frontier: {tag} K={args.k} eps={args.n_episodes} ===")
    print(f"query(test)={len(qfeats)} mature(pickGT)={int(is_gt_pick.sum())}")

    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    zs_q = qfeats @ t_protos.T

    EPS = []
    for ep in range(args.n_episodes):
        sup_idx, cal_idx = sample_episode(labels, splits, args.k, M_CALIB, BASE_SEED + ep)
        sf, sl = feats[sup_idx], labels[sup_idx]
        cf, cl = feats[cal_idx], labels[cal_idx]
        cal_y = np.array([cls_to_idx[c] for c in cl])
        sup_onehot = np.zeros((len(sl), len(CLASSES)))
        for i, c in enumerate(sl):
            sup_onehot[i, cls_to_idx[c]] = 1.0

        entry = {}

        # visual (regression anchor, identical to laboro_tomato_frontier.py)
        protos = protos_from_support(sf, sl)
        E_cal, _ = score_E_and_sims(cf, protos)
        E_q, sims_q = score_E_and_sims(qfeats, protos)
        entry.update(calibrate_and_pack(E_cal, E_q, cl, "vis"))

        # text
        zs_cal = cf @ t_protos.T
        entry.update(calibrate_and_pack(softmax_rows(zs_cal) @ ANCHORS,
                                        softmax_rows(zs_q) @ ANCHORS, cl, "text"))

        # tip
        aff_cal = cf @ sf.T
        best = None
        for a in ALPHA_GRID:
            for b in BETA_GRID:
                fused = zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)
                s = nll(softmax_rows(fused), cal_y)
                if best is None or s < best[0]:
                    best = (s, a, b)
        _, a, b = best
        tip_cal = zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)
        tip_q = zs_q + a * (np.exp(b * ((qfeats @ sf.T) - 1.0)) @ sup_onehot)
        entry.update(calibrate_and_pack(softmax_rows(tip_cal) @ ANCHORS,
                                        softmax_rows(tip_q) @ ANCHORS, cl, "tip"))

        # B5 reference (visual argmax + margin), same as laboro_tomato_frontier.py
        entry["margin_cal"] = top2_margin(score_E_and_sims(cf, protos)[1])
        entry["margin_q"] = top2_margin(sims_q)
        entry["pred_q"] = np.array([CLASSES[i] for i in sims_q.argmax(1)])
        EPS.append(entry)

    frontiers = {}
    for prefix in ("vis", "text", "tip"):
        frontiers[prefix] = build_frontier_src(EPS, is_gt_pick, prefix)
        frontiers[prefix]["variant"] = f"E_{prefix}"
    b5f = build_b5_family_frontier(EPS, is_gt_pick)
    b5f["variant"] = "B5family"
    b5 = b5_operating_point(EPS, is_gt_pick)
    b5cal = pd.DataFrame([{**{"defer": 0.20}, **b5}])
    b5cal["variant"] = "B5_calibrated_argmax_margin"

    all_df = pd.concat([*frontiers.values(), b5f, b5cal], ignore_index=True)
    all_df.to_csv(out_dir / f"tomato_source_frontier_{tag}_K{args.k}.csv", index=False, float_format="%.6f")

    def interp(df, x, c):
        dd = df.sort_values("actual_coverage")
        return float(np.interp(x, dd["actual_coverage"], dd[c]))

    b5c = b5["actual_coverage"]
    print(f"\nB5 point: coverage={b5c*100:.1f}% fp={b5['false_pick_rate']*100:.2f}% rec={b5['pick_recall']*100:.1f}%")
    print("=== regression anchor: E_vis at B5 coverage (~52.8% false-pick; V1 max coverage ~49.8%) ===")
    for prefix in ("vis", "tip", "text"):
        f = frontiers[prefix]
        print(
            f"E_{prefix}: fp@B5cov={interp(f, b5c, 'false_pick_rate')*100:.2f}% "
            f"rec@B5cov={interp(f, b5c, 'pick_recall')*100:.1f}% "
            f"max_cov={f['actual_coverage'].max()*100:.1f}%"
        )

    plt.figure(figsize=(8, 6))
    for prefix, c, lab in [("text", "#888888", "E(text)"), ("vis", "#d62728", "E(visual)"),
                           ("tip", "#1f77b4", "E(tip)")]:
        dd = frontiers[prefix].sort_values("actual_coverage")
        plt.plot(dd["actual_coverage"], dd["false_pick_rate"], "o-", color=c, lw=2, ms=4, label=lab)
    dd = b5f.sort_values("actual_coverage")
    plt.plot(dd["actual_coverage"], dd["false_pick_rate"], "s--", color="#7f8c8d", lw=1.5, ms=4, label="B5 family")
    plt.scatter([b5c], [b5["false_pick_rate"]], marker="*", s=300, color="#c0392b", zorder=5, label="B5 point")
    plt.xlabel("Actual coverage")
    plt.ylabel("Pooled false-pick rate")
    plt.title(f"Tomato stress test: endpoint sources ({tag}, K={args.k})")
    plt.grid(alpha=0.3, ls="--")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / f"tomato_source_frontier_{tag}_K{args.k}.png", dpi=200, bbox_inches="tight")
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
