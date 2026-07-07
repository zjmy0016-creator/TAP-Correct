# -*- coding: utf-8 -*-
"""
跨 backbone 前沿复现（V0:H-endpoint vs V1:E-512D-endpoint vs B5家族前沿/B5标定点）
方法学纪律：episode 分堆复用主 manifest（与 backbone 无关）；换 backbone 后在 calibration 上
【重新标定】阈值（镜像 d2_step2，ALPHA=0.05,GAMMA=0.10）；query_test 只做最终评估；全程 frozen、无训练。
对比：V0=H(1D轴)endpoint | V1=E(512D原型)endpoint | B5家族=argmax+margin扫比例 | B5标定点(红星,20%分位)。
用法：python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb16.npz --k 16
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tapcorrect.episodes import load_pools
from tapcorrect.contract import EpisodeView
from tapcorrect.d2_decision import compute_uncertainty

CLASSES = ("ripe", "turning", "unripe")
ALPHA = 0.05
GAMMA = 0.10
REVISIT_FRACTION = 0.20
N_CANDIDATES = 200
DEFERS = np.round(np.arange(0.0, 0.601, 0.05), 3)


def make_axis(npz):
    tr = npz["text_ripe"].mean(0); tu = npz["text_unripe"].mean(0)
    tr = tr / np.linalg.norm(tr); tu = tu / np.linalg.norm(tu)
    axis = tr - tu
    return axis / (np.linalg.norm(axis) + 1e-12)


def protos_from_support(sf, sl):
    p = {}
    for c in CLASSES:
        v = sf[sl == c].mean(0)
        p[c] = v / np.linalg.norm(v)
    return p


def score_H(feats, axis):
    return feats @ axis


def score_E_and_sims(feats, protos):
    sims = np.stack([feats @ protos["ripe"], feats @ protos["turning"],
                     feats @ protos["unripe"]], axis=1)
    z = sims - sims.max(1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(1, keepdims=True)
    E = probs @ np.array([1.0, 0.5, 0.0])
    return E, sims


def calib_T_high(sr, st, su, alpha):
    cand = np.linspace(su.min(), sr.max(), N_CANDIDATES)
    valid = [t for t in cand if np.mean(su >= t) <= alpha]
    return min(valid) if valid else su.max() + 0.01


def calib_T_low(sr, st, su, gamma):
    cand = np.linspace(su.min(), sr.max(), N_CANDIDATES)
    valid = [t for t in cand if np.mean(sr <= t) <= gamma]
    return max(valid) if valid else sr.min() - 0.01


def order_thr(pick_min, wait_max):
    return min(pick_min, wait_max), max(pick_min, wait_max)


def top2_margin(sims):
    s = np.sort(sims, axis=1)
    return s[:, -1] - s[:, -2]


def build_frontier(EPS, is_gt_pick, score_key, thr_hi_key, thr_lo_key):
    rows = []
    for dr in DEFERS:
        S = dict(pick=0, fp=0, wait=0, tot=0, tp=0, gtp=0)
        for d in EPS:
            u = d["u"]; sc = d[score_key]; Th = d[thr_hi_key]; Tl = d[thr_lo_key]
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
            dec[keep[pred[keep] == "ripe"]] = "pick"
            dec[keep[pred[keep] == "unripe"]] = "wait"
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
        dec[keep & (pred == "ripe")] = "pick"
        dec[keep & (pred == "unripe")] = "wait"
        pm = dec == "pick"; wm = dec == "wait"
        S["pick"] += int(pm.sum()); S["fp"] += int((pm & ~is_gt_pick).sum())
        S["wait"] += int(wm.sum()); S["tot"] += n
        S["tp"] += int((pm & is_gt_pick).sum()); S["gtp"] += int(is_gt_pick.sum())
    return dict(false_pick_rate=S["fp"]/S["pick"] if S["pick"] else 0.0,
                actual_coverage=(S["pick"]+S["wait"])/S["tot"],
                pick_recall=S["tp"]/S["gtp"] if S["gtp"] else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--manifest", type=Path,
                    default=ROOT/"outputs/episodes/manifest_K1-16_ep100.json")
    ap.add_argument("--gt", type=Path,
                    default=ROOT/"outputs/decision_gold/turning_decision_dataset/labels/test_decision_ground_truth_clean.csv")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    tag = args.npz.stem.replace("features_", "")
    out_dir = args.out_dir or (ROOT/"outputs/probe_512d_endpoint"/tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 跨 backbone 前沿：{tag}  K={args.k} ===")
    feats, labels, sp_idx_by_cls, query_idx = load_pools(args.npz)
    npz = np.load(args.npz, allow_pickle=True)
    axis = make_axis(npz)
    manifest = json.load(open(args.manifest, encoding="utf-8"))["episodes"]
    eps = [(i, e) for i, e in enumerate(manifest) if e["k"] == args.k]
    print(f"  episodes (K={args.k}): {len(eps)}  | feat dim = {feats.shape[1]}")

    gt = pd.read_csv(args.gt, encoding="utf-8-sig")
    gt_dec = gt["decision_label"].replace("borderline", "revisit").values
    is_gt_pick = (gt_dec == "pick")
    qfeats = feats[query_idx]
    assert len(qfeats) == len(gt_dec), f"query({len(qfeats)}) != GT({len(gt_dec)})"

    EPS = []
    for ep_idx, e in eps:
        view = EpisodeView(e, feats, labels, query_idx)
        sf, sl = view.support()
        cf, cl = view.calibration()
        protos = protos_from_support(sf, sl)
        Hc = score_H(cf, axis); Ec, sims_c = score_E_and_sims(cf, protos)
        T_pick_min = calib_T_high(Hc[cl=="ripe"], Hc[cl=="turning"], Hc[cl=="unripe"], ALPHA)
        T_wait_max = calib_T_low (Hc[cl=="ripe"], Hc[cl=="turning"], Hc[cl=="unripe"], GAMMA)
        T_low, T_high = order_thr(T_pick_min, T_wait_max)
        E_pick_min = calib_T_high(Ec[cl=="ripe"], Ec[cl=="turning"], Ec[cl=="unripe"], ALPHA)
        E_wait_max = calib_T_low (Ec[cl=="ripe"], Ec[cl=="turning"], Ec[cl=="unripe"], GAMMA)
        T_low_E, T_high_E = order_thr(E_pick_min, E_wait_max)
        Hq = score_H(qfeats, axis); Eq, sims_q = score_E_and_sims(qfeats, protos)
        u = compute_uncertainty(Eq, T_high_E, T_low_E)
        margin_cal = top2_margin(sims_c)
        margin_q = top2_margin(sims_q)
        pred_q = np.array([CLASSES[i] for i in sims_q.argmax(1)])
        EPS.append(dict(H=Hq, E=Eq, u=u, T_high=T_high, T_low=T_low,
                        T_high_E=T_high_E, T_low_E=T_low_E,
                        margin_cal=margin_cal, margin_q=margin_q, pred_q=pred_q))

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
    print(f"\nB5 标定点     : coverage={b5c*100:.1f}%  false-pick={b5['false_pick_rate']*100:.2f}%  recall={b5['pick_recall']*100:.1f}%")
    print(f"V0(H)  @B5cov : false-pick={interp(v0,b5c,'false_pick_rate')*100:.2f}%")
    print(f"V1(E)  @B5cov : false-pick={interp(v1,b5c,'false_pick_rate')*100:.2f}%  recall={interp(v1,b5c,'pick_recall')*100:.1f}%")
    print(f"B5家族 @B5cov : false-pick={interp(b5f,b5c,'false_pick_rate')*100:.2f}%  recall={interp(b5f,b5c,'pick_recall')*100:.1f}%")
    lo = max(v1["actual_coverage"].min(), b5f["actual_coverage"].min())
    hi = min(v1["actual_coverage"].max(), b5f["actual_coverage"].max())
    grid = np.linspace(lo, hi, 25)
    v1_below = np.mean([interp(v1, x, "false_pick_rate") <= interp(b5f, x, "false_pick_rate") for x in grid])
    print(f"V1 前沿在公共区间内 fp<=B5家族 的占比: {v1_below*100:.0f}%  (100%=整条压过)")
    print(f"V1 最高覆盖: {v1['actual_coverage'].max()*100:.1f}%  | B5家族最高覆盖: {b5f['actual_coverage'].max()*100:.1f}%")

    plt.figure(figsize=(8, 6))
    for df, c, lab in [(v0, "#e67e22", "V0: H (1D axis) endpoint"),
                       (b5f, "#7f8c8d", "B5-family: argmax + margin (swept)"),
                       (v1, "#2980b9", "V1: E (512D prototype) endpoint")]:
        dd = df.sort_values("actual_coverage")
        plt.plot(dd["actual_coverage"], dd["false_pick_rate"], "o-", color=c, lw=2, ms=5, label=lab)
    plt.scatter([b5c], [b5["false_pick_rate"]], marker="*", s=340, color="#c0392b", zorder=5,
                label=f"B5 calibrated pt ({b5c*100:.0f}%, {b5['false_pick_rate']*100:.1f}%)")
    plt.xlabel("Actual coverage"); plt.ylabel("Pooled false-pick rate")
    plt.title(f"Risk-Coverage on {tag} (K={args.k}) - lower-left better")
    plt.grid(alpha=.3, ls="--"); plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(out_dir/f"frontier_{tag}_K{args.k}.png", dpi=200, bbox_inches="tight")

    plt.figure(figsize=(8, 6))
    for df, c, lab in [(b5f, "#7f8c8d", "B5-family: argmax + margin (swept)"),
                       (v1, "#2980b9", "V1: E (512D prototype) endpoint")]:
        dd = df.sort_values("actual_coverage")
        plt.plot(dd["actual_coverage"], dd["pick_recall"], "o-", color=c, lw=2, ms=5, label=lab)
    plt.scatter([b5c], [b5["pick_recall"]], marker="*", s=340, color="#c0392b", zorder=5,
                label="B5 calibrated pt")
    plt.xlabel("Actual coverage"); plt.ylabel("Pooled pick recall")
    plt.title(f"Recall-Coverage on {tag} (K={args.k}) - upper-right better")
    plt.grid(alpha=.3, ls="--"); plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(out_dir/f"recall_{tag}_K{args.k}.png", dpi=200, bbox_inches="tight")
    print(f"\nsaved -> {out_dir}/frontier_{tag}_K{args.k}.(csv|png) + recall_{tag}_K{args.k}.png")


if __name__ == "__main__":
    main()
