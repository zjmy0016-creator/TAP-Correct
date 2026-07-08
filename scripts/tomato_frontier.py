# -*- coding: utf-8 -*-
"""Run the Laboro Tomato frozen-protocol risk-coverage frontier.

This script mirrors the strawberry V1 frontier protocol while changing only the
dataset adapter:
- classes: mature / turning / immature
- support/calibration are sampled from the train split
- the test split is used as query
- ground-truth pick is mature only
- picking immature or turning is counted as a false pick

Example:
python scripts/tomato_frontier.py --npz outputs/features_laboro_tomato_vitb16.npz --k 16
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tapcorrect.d2_decision import compute_uncertainty

CLASSES = ("mature", "turning", "immature")
ANCHORS = np.array([1.0, 0.5, 0.0])
ALPHA, GAMMA, REVISIT_FRACTION, N_CANDIDATES = 0.05, 0.10, 0.20, 200
DEFERS = np.round(np.arange(0.0, 0.601, 0.05), 3)
M_CALIB, BASE_SEED = 200, 42


def make_axis(npz):
    tr = npz["text_mature"].mean(0); tu = npz["text_immature"].mean(0)
    tr = tr / np.linalg.norm(tr); tu = tu / np.linalg.norm(tu)
    axis = tr - tu
    return axis / (np.linalg.norm(axis) + 1e-12)

def protos_from_support(sf, sl):
    return {c: (sf[sl == c].mean(0) / np.linalg.norm(sf[sl == c].mean(0))) for c in CLASSES}

def score_H(feats, axis):
    return feats @ axis

def score_E_and_sims(feats, protos):
    sims = np.stack([feats @ protos["mature"], feats @ protos["turning"],
                     feats @ protos["immature"]], axis=1)
    z = sims - sims.max(1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(1, keepdims=True)
    return probs @ ANCHORS, sims

def calib_T_high(sr, st, su, alpha):
    cand = np.linspace(su.min(), sr.max(), N_CANDIDATES)
    valid = [t for t in cand if np.mean(su >= t) <= alpha]
    return min(valid) if valid else su.max() + 0.01

def calib_T_low(sr, st, su, gamma):
    cand = np.linspace(su.min(), sr.max(), N_CANDIDATES)
    valid = [t for t in cand if np.mean(sr <= t) <= gamma]
    return max(valid) if valid else sr.min() - 0.01

def order_thr(a, b):
    return min(a, b), max(a, b)

def top2_margin(sims):
    s = np.sort(sims, axis=1)
    return s[:, -1] - s[:, -2]

def build_frontier(EPS, is_gt_pick, score_key, thr_hi, thr_lo):
    rows = []
    for dr in DEFERS:
        S = dict(pick=0, fp=0, wait=0, tot=0, tp=0, gtp=0)
        for d in EPS:
            u = d["u"]; sc = d[score_key]; Th = d[thr_hi]; Tl = d[thr_lo]
            n = len(u); nd = int(n * dr)
            keep = np.argsort(u)[::-1][nd:]
            dec = np.full(n, "revisit", dtype=object)
            dec[keep[sc[keep] >= Th]] = "pick"
            dec[keep[sc[keep] <= Tl]] = "wait"
            pm = dec == "pick"; wm = dec == "wait"
            S["pick"] += int(pm.sum()); S["fp"] += int((pm & ~is_gt_pick).sum())
            S["wait"] += int(wm.sum()); S["tot"] += n
            S["tp"] += int((pm & is_gt_pick).sum()); S["gtp"] += int(is_gt_pick.sum())
        rows.append(dict(defer=dr,
                         false_pick_rate=S["fp"]/S["pick"] if S["pick"] else 0.0,
                         actual_coverage=(S["pick"]+S["wait"])/S["tot"],
                         pick_recall=S["tp"]/S["gtp"] if S["gtp"] else 0.0))
    return pd.DataFrame(rows)

def build_b5_family_frontier(EPS, is_gt_pick):
    rows = []
    for dr in DEFERS:
        S = dict(pick=0, fp=0, wait=0, tot=0, tp=0, gtp=0)
        for d in EPS:
            margin = d["margin_q"]; pred = d["pred_q"]; n = len(pred)
            nd = int(n * dr)
            keep = np.argsort(margin)[::-1][:n - nd] if nd > 0 else np.arange(n)
            dec = np.full(n, "revisit", dtype=object)
            dec[keep[pred[keep] == "mature"]] = "pick"
            dec[keep[pred[keep] == "immature"]] = "wait"
            pm = dec == "pick"; wm = dec == "wait"
            S["pick"] += int(pm.sum()); S["fp"] += int((pm & ~is_gt_pick).sum())
            S["wait"] += int(wm.sum()); S["tot"] += n
            S["tp"] += int((pm & is_gt_pick).sum()); S["gtp"] += int(is_gt_pick.sum())
        rows.append(dict(defer=dr,
                         false_pick_rate=S["fp"]/S["pick"] if S["pick"] else 0.0,
                         actual_coverage=(S["pick"]+S["wait"])/S["tot"],
                         pick_recall=S["tp"]/S["gtp"] if S["gtp"] else 0.0))
    return pd.DataFrame(rows)

def b5_operating_point(EPS, is_gt_pick):
    S = dict(pick=0, fp=0, wait=0, tot=0, tp=0, gtp=0)
    for d in EPS:
        thr = np.percentile(d["margin_cal"], REVISIT_FRACTION * 100)
        margin = d["margin_q"]; pred = d["pred_q"]; n = len(pred)
        dec = np.full(n, "revisit", dtype=object)
        keep = margin >= thr
        dec[keep & (pred == "mature")] = "pick"
        dec[keep & (pred == "immature")] = "wait"
        pm = dec == "pick"; wm = dec == "wait"
        S["pick"] += int(pm.sum()); S["fp"] += int((pm & ~is_gt_pick).sum())
        S["wait"] += int(wm.sum()); S["tot"] += n
        S["tp"] += int((pm & is_gt_pick).sum()); S["gtp"] += int(is_gt_pick.sum())
    return dict(false_pick_rate=S["fp"]/S["pick"] if S["pick"] else 0.0,
                actual_coverage=(S["pick"]+S["wait"])/S["tot"],
                pick_recall=S["tp"]/S["gtp"] if S["gtp"] else 0.0)

def sample_episode(labels, splits, k, m, seed):
    rng = np.random.default_rng(seed)
    sup, cal = [], []
    for c in CLASSES:
        pool = np.where((splits == "train") & (labels == c))[0]
        sh = rng.permutation(pool)
        sup.append(sh[:k]); cal.append(sh[k:k+m])
    return np.concatenate(sup), np.concatenate(cal)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n_episodes", type=int, default=100)
    args = ap.parse_args()

    tag = args.npz.stem.replace("features_", "")
    out_dir = ROOT/"outputs/probe_512d_endpoint"/tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Laboro Tomato frozen frontier: {tag}  K={args.k}  episodes={args.n_episodes} ===")
    npz = np.load(args.npz, allow_pickle=True)
    feats, labels, splits = npz["image_feats"], npz["classes"], npz["splits"]
    axis = make_axis(npz)

    q_mask = (splits == "test")
    qfeats, q_labels = feats[q_mask], labels[q_mask]
    is_gt_pick = (q_labels == "mature")
    print(f"  query(test)={len(qfeats)}  mature(pickGT)={int(is_gt_pick.sum())}  dim={feats.shape[1]}")

    EPS = []
    for ep in range(args.n_episodes):
        sup_idx, cal_idx = sample_episode(labels, splits, args.k, M_CALIB, BASE_SEED + ep)
        sf, sl = feats[sup_idx], labels[sup_idx]
        cf, cl = feats[cal_idx], labels[cal_idx]
        protos = protos_from_support(sf, sl)
        Hc = score_H(cf, axis); Ec, _ = score_E_and_sims(cf, protos)
        Tpm = calib_T_high(Hc[cl=="mature"], Hc[cl=="turning"], Hc[cl=="immature"], ALPHA)
        Twm = calib_T_low (Hc[cl=="mature"], Hc[cl=="turning"], Hc[cl=="immature"], GAMMA)
        T_low, T_high = order_thr(Tpm, Twm)
        Epm = calib_T_high(Ec[cl=="mature"], Ec[cl=="turning"], Ec[cl=="immature"], ALPHA)
        Ewm = calib_T_low (Ec[cl=="mature"], Ec[cl=="turning"], Ec[cl=="immature"], GAMMA)
        T_low_E, T_high_E = order_thr(Epm, Ewm)
        Hq = score_H(qfeats, axis); Eq, sims_q = score_E_and_sims(qfeats, protos)
        u = compute_uncertainty(Eq, T_high_E, T_low_E)
        EPS.append(dict(H=Hq, E=Eq, u=u, T_high=T_high, T_low=T_low,
                        T_high_E=T_high_E, T_low_E=T_low_E,
                        margin_cal=top2_margin(score_E_and_sims(cf, protos)[1]),
                        margin_q=top2_margin(sims_q),
                        pred_q=np.array([CLASSES[i] for i in sims_q.argmax(1)])))

    v0 = build_frontier(EPS, is_gt_pick, "H", "T_high", "T_low")
    v1 = build_frontier(EPS, is_gt_pick, "E", "T_high_E", "T_low_E")
    b5f = build_b5_family_frontier(EPS, is_gt_pick)
    b5 = b5_operating_point(EPS, is_gt_pick)

    v0["variant"] = "V0_H_axis_endpoint"; v1["variant"] = "V1_E_512D_endpoint"
    b5f["variant"] = "B5family_argmax_margin"
    pd.concat([v0, v1, b5f], ignore_index=True).to_csv(
        out_dir/f"frontier_{tag}_K{args.k}.csv", index=False, float_format="%.6f")

    def interp(df, x, c):
        dd = df.sort_values("actual_coverage")
        return float(np.interp(x, dd["actual_coverage"], dd[c]))
    b5c = b5["actual_coverage"]
    print(f"\nB5 operating point: coverage={b5c*100:.1f}%  false-pick={b5['false_pick_rate']*100:.2f}%  recall={b5['pick_recall']*100:.1f}%")
    print(f"V0(H)  @B5cov : false-pick={interp(v0,b5c,'false_pick_rate')*100:.2f}%")
    print(f"V1(E)  @B5cov : false-pick={interp(v1,b5c,'false_pick_rate')*100:.2f}%  recall={interp(v1,b5c,'pick_recall')*100:.1f}%")
    print(f"B5 family @B5cov: false-pick={interp(b5f,b5c,'false_pick_rate')*100:.2f}%  recall={interp(b5f,b5c,'pick_recall')*100:.1f}%")
    lo = max(v1["actual_coverage"].min(), b5f["actual_coverage"].min())
    hi = min(v1["actual_coverage"].max(), b5f["actual_coverage"].max())
    grid = np.linspace(lo, hi, 25)
    v1_below = np.mean([interp(v1, x, "false_pick_rate") <= interp(b5f, x, "false_pick_rate") for x in grid])
    print(f"V1 frontier fp<=B5 family over common coverage: {v1_below*100:.0f}%")
    print(f"V1 max coverage: {v1['actual_coverage'].max()*100:.1f}%  | B5 family max coverage: {b5f['actual_coverage'].max()*100:.1f}%")

    plt.figure(figsize=(8, 6))
    for df, c, lab in [(v0, "#e67e22", "V0: H (1D axis)"),
                       (b5f, "#7f8c8d", "B5-family: argmax+margin (swept)"),
                       (v1, "#2980b9", "V1: E (512D prototype)")]:
        dd = df.sort_values("actual_coverage")
        plt.plot(dd["actual_coverage"], dd["false_pick_rate"], "o-", color=c, lw=2, ms=5, label=lab)
    plt.scatter([b5c], [b5["false_pick_rate"]], marker="*", s=340, color="#c0392b", zorder=5,
                label=f"B5 pt ({b5c*100:.0f}%, {b5['false_pick_rate']*100:.1f}%)")
    plt.xlabel("Actual coverage"); plt.ylabel("Pooled false-pick rate")
    plt.title(f"Risk-Coverage on {tag} (K={args.k}) - lower-left better")
    plt.grid(alpha=.3, ls="--"); plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(out_dir/f"frontier_{tag}_K{args.k}.png", dpi=200, bbox_inches="tight")
    print(f"\nsaved -> {out_dir}\\frontier_{tag}_K{args.k}.(csv|png)")

if __name__ == "__main__":
    main()
