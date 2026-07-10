# -*- coding: utf-8 -*-
"""Endpoint-source ablation for the TAP-Correct decision layer (route 2).

The decision layer (ordinal expectation E + calibrated thresholds + U_cut
uncertainty gate) is orthogonal to the probability source. This script plugs
three training-free sources into the SAME frozen decision layer:
  - text   : zero-shot text-prototype logits
  - visual : 512D visual prototypes (current TAP-Correct / V1)
  - tip    : Tip-Adapter fused logits (zero-shot + cache, alpha/beta by
             calibration NLL)

Per episode (K=16 manifest block), thresholds are recalibrated on the 600
calibration samples with the exact calibration recipe (ALPHA=0.05, GAMMA=0.10,
200-candidate grid, U_cut = 80th percentile of calibration uncertainty).
Test split is used for final evaluation only. Regression anchor: the visual
source must reproduce the frozen V1 headline (~9.1/90.9/80.6/19.2).

Run from project root:
    python scripts/endpoint_source_ablation.py
"""
from __future__ import annotations

import argparse
import json
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

from scripts.pooled_metrics import compute_pooled_metrics
from scripts.paired_difference import compute_episode_metrics
from scripts.official_evaluation import METRICS, load_ground_truth
from tapcorrect.decision import compute_uncertainty, decide_episode_v1_endpoint

CLASSES = ("ripe", "turning", "unripe")
ALPHA = 0.05
GAMMA = 0.10
N_CANDIDATES = 200
Q_LIST = (50, 60, 70, 75, 80, 85, 90, 95)
Q_FROZEN = 80
ALPHA_GRID = (0.1, 0.5, 1.0, 2.0)
BETA_GRID = (1.0, 3.0, 5.5, 10.0)
K = 16
SOURCES = ("text", "visual", "tip")


def normalize_vector(x):
    x = np.asarray(x, dtype=float)
    return x / (np.linalg.norm(x) + 1e-12)


def softmax_rows(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def expectation(logits):
    return softmax_rows(logits) @ np.array([1.0, 0.5, 0.0])


def calib_T_high(sr, su, alpha):
    cand = np.linspace(su.min(), sr.max(), N_CANDIDATES)
    valid = [t for t in cand if np.mean(su >= t) <= alpha]
    return min(valid) if valid else su.max() + 0.01

def calib_T_low(sr, su, gamma):
    cand = np.linspace(su.min(), sr.max(), N_CANDIDATES)
    valid = [t for t in cand if np.mean(sr <= t) <= gamma]
    return max(valid) if valid else sr.min() - 0.01


def nll(probs, label_idx):
    p_true = np.clip(probs[np.arange(len(label_idx)), label_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))


def load_everything(root: Path, npz_path: Path | None = None, k: int = 16):
    npz_path = npz_path or (root / "outputs" / "features_vitb32.npz")
    data = np.load(npz_path, allow_pickle=True)
    tag = npz_path.stem.replace("features_", "")
    feats = np.asarray(data["image_feats"], dtype=float)
    splits = data["splits"].astype(str)
    class_labels = data["classes"].astype(str)
    paths = data["image_paths"].astype(str)

    query_mask = splits == "query_test"
    q_names = np.array([p.split("\\")[-1] for p in paths[query_mask]])
    gt_df = load_ground_truth(root)
    if not (q_names == gt_df["filename"].astype(str).to_numpy()).all():
        raise ValueError("query_test order does not match decision GT csv")

    text_protos = np.stack(
        [normalize_vector(np.asarray(data[f"text_{c}"], dtype=float).mean(axis=0)) for c in CLASSES]
    )
    manifest = json.loads(
        (root / "outputs" / "episodes" / "manifest_K1-16_ep100.json").read_text(encoding="utf-8")
    )
    episodes = [e for e in manifest["episodes"] if e["k"] == k]
    if len(episodes) != 100:
        raise ValueError(f"expected 100 K={k} episodes, got {len(episodes)}")
    return feats, class_labels, query_mask, gt_df["decision_label"].to_numpy(), text_protos, episodes


def episode_E_scores(feats, class_labels, query_mask, text_protos, ep):
    """Return {source: (E_cal, E_test)} plus calibration class labels."""
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

    out = {}

    zs_cal = cal_feats @ text_protos.T
    zs_test = test_feats @ text_protos.T
    out["text"] = (expectation(zs_cal), expectation(zs_test))

    protos = np.stack([normalize_vector(sup_feats[sup_labels == c].mean(axis=0)) for c in CLASSES])
    out["visual"] = (expectation(cal_feats @ protos.T), expectation(test_feats @ protos.T))

    aff_cal = cal_feats @ sup_feats.T
    aff_test = test_feats @ sup_feats.T
    best = None
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            fused = zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)
            score = nll(softmax_rows(fused), cal_y)
            if best is None or score < best[0]:
                best = (score, a, b)
    _, a, b = best
    tip_cal = zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)
    tip_test = zs_test + a * (np.exp(b * (aff_test - 1.0)) @ sup_onehot)
    out["tip"] = (expectation(tip_cal), expectation(tip_test))

    return out, cal_labels


def run(root: Path, out_dir: Path, n_bootstrap: int, seed: int, npz_path: Path | None = None, k: int = 16) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (npz_path.stem.replace("features_", "") if npz_path else "vitb32")
    feats, class_labels, query_mask, gt_decisions, text_protos, episodes = load_everything(root, npz_path, k)

    # decisions[source][q] -> list of per-episode action arrays
    decisions = {s: {q: [] for q in Q_LIST} for s in SOURCES}
    for ep in episodes:
        scores, cal_labels = episode_E_scores(feats, class_labels, query_mask, text_protos, ep)
        for source, (E_cal, E_test) in scores.items():
            sr = E_cal[cal_labels == "ripe"]
            su = E_cal[cal_labels == "unripe"]
            pick_min = calib_T_high(sr, su, ALPHA)
            wait_max = calib_T_low(sr, su, GAMMA)
            t_low, t_high = min(pick_min, wait_max), max(pick_min, wait_max)
            u_cal = compute_uncertainty(E_cal, t_high, t_low)
            for q in Q_LIST:
                u_cut = float(np.percentile(u_cal, q))
                dec, _ = decide_episode_v1_endpoint(E_test, t_high, t_low, u_cut)
                decisions[source][q].append(dec.astype(str))

    rows = []
    for source in SOURCES:
        for q in Q_LIST:
            preds = np.concatenate(decisions[source][q])
            gts = np.tile(gt_decisions, len(decisions[source][q]))
            m = compute_pooled_metrics(preds, gts)
            rows.append({"source": source, "q": q, **{k: float(m[k]) for k in METRICS}})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"endpoint_source_ablation_{tag}_K{k}.csv", index=False, float_format="%.6f")

    # --- regression anchor: visual @ q=80 ---
    v = df[(df["source"] == "visual") & (df["q"] == Q_FROZEN)].iloc[0]
    EXPECTED_VITB32 = {1: "16.7/83.3/38.9/52.4", 2: "14.4/85.6/49.0/44.3",
                       4: "11.5/88.5/64.2/30.1", 8: "10.2/89.8/74.5/23.2",
                       16: "9.1/90.9/80.6/19.2"}
    exp = EXPECTED_VITB32.get(k, "n/a") if tag == "vitb32" else "other backbone: check direction only"
    print(f"=== regression anchor [{tag} K={k}] (expect ~ {exp}) ===")
    print(
        f"visual q=80: fp {v['false_pick_rate']*100:.1f} | prec {v['pick_precision']*100:.1f} "
        f"| rec {v['pick_recall']*100:.1f} | rev {v['revisit_burden']*100:.1f}"
    )

    # --- paired bootstrap: tip vs visual at q=80 ---
    rng = np.random.RandomState(seed)
    vis_m = [compute_episode_metrics(d, gt_decisions) for d in decisions["visual"][Q_FROZEN]]
    tip_m = [compute_episode_metrics(d, gt_decisions) for d in decisions["tip"][Q_FROZEN]]
    paired_rows = []
    print("\n=== paired diff (tip - visual) at q=80 ===")
    for metric in METRICS:
        diffs = np.array([t[metric] - v_[metric] for t, v_ in zip(tip_m, vis_m)])
        boot = [
            float(diffs[rng.choice(len(diffs), len(diffs), replace=True)].mean())
            for _ in range(n_bootstrap)
        ]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = not (lo <= 0 <= hi)
        paired_rows.append(
            {"metric": metric, "mean_diff": diffs.mean(), "ci_lower": lo, "ci_upper": hi, "significant": sig}
        )
        print(f"  {metric}: {diffs.mean()*100:+.2f}pp [{lo*100:+.2f}, {hi*100:+.2f}] {'SIG' if sig else 'n.s.'}")
    pd.DataFrame(paired_rows).to_csv(
        out_dir / f"paired_tip_vs_visual_q80_{tag}_K{k}.csv", index=False, float_format="%.6f"
    )


    # --- figure: frontiers (fp vs revisit, recall vs revisit) ---
    styles = {"text": ("#888888", "s--"), "visual": ("#d62728", "o-"), "tip": ("#1f77b4", "^-")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for source in SOURCES:
        sub = df[df["source"] == source].sort_values("q")
        c, fmt = styles[source]
        axes[0].plot(sub["revisit_burden"] * 100, sub["false_pick_rate"] * 100, fmt, color=c,
                     label=f"E({source})", ms=5)
        axes[1].plot(sub["revisit_burden"] * 100, sub["pick_recall"] * 100, fmt, color=c,
                     label=f"E({source})", ms=5)
    # reference points
    refs = [("Tip-Adapter no-reject", 21.1, 9.4, 81.0), ("FS-Proto-Reject", 39.9, 4.2, 65.7)]
    for name, rev, fp, rec in refs:
        axes[0].scatter([rev], [fp], marker="*", s=140, zorder=5, label=name)
        axes[1].scatter([rev], [rec], marker="*", s=140, zorder=5, label=name)
    for ax, ylab in zip(axes, ("False-pick proportion (%)", "Pick recall (%)")):
        ax.set_xlabel("Revisit burden (%)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Endpoint-source ablation: same decision layer, three training-free sources ({tag}, K={k})")
    fig.tight_layout()
    fig.savefig(out_dir / f"endpoint_source_frontiers_{tag}_K{k}.png", dpi=200)
    plt.close(fig)

    # --- markdown ---
    lines = [
        f"# Endpoint-source ablation ({tag}, K={k}, decision GT)",
        "",
        "Same decision layer (E + calibration-recipe thresholds + U_cut@q80); only the",
        "probability source changes. All calibration on calibration split only.",
        "",
        "| E source | q | False-pick % | Precision % | Recall % | Revisit % |",
        "|----------|---|-------------|-------------|----------|-----------|",
    ]
    for source in SOURCES:
        for _, r in df[df["source"] == source].iterrows():
            mark = " (frozen)" if r["q"] == Q_FROZEN else ""
            lines.append(
                f"| {source} | {r['q']:.0f}{mark} | {r['false_pick_rate']*100:.1f} "
                f"| {r['pick_precision']*100:.1f} | {r['pick_recall']*100:.1f} "
                f"| {r['revisit_burden']*100:.1f} |"
            )
    (out_dir / f"ENDPOINT_SOURCE_ABLATION_{tag}_K{k}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=None, help="features npz (default: features_vitb32.npz)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()
    out_dir = args.out if args.out is not None else args.root / "outputs" / "endpoint_source_ablation"
    run(args.root, out_dir, args.n_bootstrap, args.seed, args.npz, args.k)


if __name__ == "__main__":
    main()
