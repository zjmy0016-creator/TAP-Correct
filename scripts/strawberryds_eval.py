"""Stage: Evaluate TAP + baselines on Strawberry-DS (K=16, 20 episodes, calibration-only thresholds).

Three modes:
  maincal   : support + calibration from main domain, test on all external (default)
  recalib   : support from main domain, calibration from external domain, test on external remainder
  indomain  : support + calibration both from external domain, main domain not used at runtime

Baselines: Axis-Endpoint / FS-Proto-Reject / TAP E(tip).

Usage:
  python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz
  python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode recalib
  python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode indomain
  python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode indomain --m_ext 60
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.endpoint_source_ablation import (
    ALPHA, GAMMA, ALPHA_GRID, BETA_GRID, calib_T_high, calib_T_low, nll, softmax_rows, normalize_vector
)

CLASSES = ("ripe", "turning", "unripe")
ANCHORS = np.array([1.0, 0.5, 0.0])

def protos_from_support(sf, sl):
    return {c: normalize_vector(sf[sl == c].mean(0)) for c in CLASSES}

def score_E_and_sims(feats, protos):
    sims = np.stack([feats @ protos["ripe"], feats @ protos["turning"],
                     feats @ protos["unripe"]], axis=1)
    z = sims - sims.max(1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(1, keepdims=True)
    return probs @ ANCHORS, sims

def top2_margin(sims):
    s = np.sort(sims, axis=1)
    return s[:, -1] - s[:, -2]

from tapcorrect.decision import compute_uncertainty, decide_episode_v1_endpoint
from scripts.pooled_metrics import compute_pooled_metrics

K, M_CALIB, N_EP, BASE_SEED = 16, 200, 20, 42
M_EXT_RECALIB = 30  # per-class calibration samples drawn from external domain (recalib mode)

def sample_episode(labels, k, m, seed):
    rng = np.random.RandomState(seed)
    sup_idx = np.concatenate([rng.choice(np.where(labels == c)[0], k, False) for c in CLASSES])
    cal_idx = np.concatenate([rng.choice(np.where(labels == c)[0], m, False) for c in CLASSES])
    return sup_idx, cal_idx

def axis_endpoint(E_cal, cal_labels, E_test, gt_is_pick):
    sr, su = E_cal[cal_labels == "ripe"], E_cal[cal_labels == "unripe"]
    pm, wm = calib_T_high(sr, su, ALPHA), calib_T_low(sr, su, GAMMA)
    t_low, t_high = min(pm, wm), max(pm, wm)
    u = compute_uncertainty(E_test, t_high, t_low)
    u_cut = float(np.percentile(compute_uncertainty(E_cal, t_high, t_low), 80))
    dec, _ = decide_episode_v1_endpoint(E_test, t_high, t_low, u_cut)
    return compute_pooled_metrics(dec.astype(str), ["pick" if p else "wait" for p in gt_is_pick])

def b5_family(sf, sl, cf, cl, qf, gt_is_pick):
    protos = protos_from_support(sf, sl)
    _, sims_q = score_E_and_sims(qf, protos)
    margin = top2_margin(sims_q)
    thr = np.percentile(top2_margin(score_E_and_sims(cf, protos)[1]), 20)
    pred = np.array([CLASSES[i] for i in sims_q.argmax(1)])
    reject = margin < thr
    actions = np.full(len(pred), "revisit", dtype=object)
    actions[~reject & (pred == "ripe")] = "pick"
    actions[~reject & (pred == "unripe")] = "wait"
    return compute_pooled_metrics(actions.astype(str), ["pick" if p else "wait" for p in gt_is_pick])

def tap_tip(sf, sl, cf, cl, qf, text_protos, cls_to_idx, gt_is_pick):
    sup_onehot = np.zeros((len(sl), 3))
    for i, c in enumerate(sl):
        sup_onehot[i, cls_to_idx[c]] = 1.0
    zs_cal, zs_q = cf @ text_protos.T, qf @ text_protos.T
    aff_cal, aff_q = cf @ sf.T, qf @ sf.T
    cal_y = np.array([cls_to_idx[c] for c in cl])
    best = None
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            s = nll(softmax_rows(zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)), cal_y)
            if best is None or s < best[0]:
                best = (s, a, b)
    _, a, b = best
    E_cal = softmax_rows(zs_cal + a * (np.exp(b * (aff_cal - 1.0)) @ sup_onehot)) @ ANCHORS
    E_q = softmax_rows(zs_q + a * (np.exp(b * (aff_q - 1.0)) @ sup_onehot)) @ ANCHORS
    return axis_endpoint(E_cal, cl, E_q, gt_is_pick)

def eval_all_methods(sf, sl, cf, cl, qf, q_labels, text_protos, cls_to_idx):
    """Run the three methods on given support/calib/query; return dict of metrics."""
    gt_is_pick = q_labels == "ripe"
    protos = protos_from_support(sf, sl)
    E_cal, _ = score_E_and_sims(cf, protos)
    E_q, _ = score_E_and_sims(qf, protos)
    return {
        "Axis-Endpoint": axis_endpoint(E_cal, cl, E_q, gt_is_pick),
        "FS-Proto-Reject": b5_family(sf, sl, cf, cl, qf, gt_is_pick),
        "TAP-E(tip)": tap_tip(sf, sl, cf, cl, qf, text_protos, cls_to_idx, gt_is_pick),
    }

def run(npz_path: Path, main_feats: Path, out_dir: Path, mode: str, m_ext: int = M_EXT_RECALIB):
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = np.load(npz_path, allow_pickle=True)
    ext_feats, ext_labels = np.asarray(ext["image_feats"], float), ext["classes"].astype(str)

    main = np.load(main_feats, allow_pickle=True)
    pool_feats, pool_labels = np.asarray(main["image_feats"], float), main["classes"].astype(str)
    pool_mask = main["splits"].astype(str) == "support_pool"
    text_protos = np.stack([np.asarray(main[f"text_{c}"], float).mean(0) for c in CLASSES])
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

    results = []
    for ep in range(N_EP):
        rng = np.random.RandomState(BASE_SEED + ep)

        if mode == "indomain":
            # Support + calibration both from external domain; main domain not used
            sup_idx_list, cal_idx_list = [], []
            test_mask = np.ones(len(ext_labels), dtype=bool)
            for c in CLASSES:
                idx_c = np.where(ext_labels == c)[0].copy()
                rng.shuffle(idx_c)
                sup_pick = idx_c[:K]
                cal_pick = idx_c[K : K + m_ext]
                sup_idx_list.extend(sup_pick.tolist())
                cal_idx_list.extend(cal_pick.tolist())
                test_mask[sup_pick] = False
                test_mask[cal_pick] = False
            sf, sl = ext_feats[sup_idx_list], ext_labels[sup_idx_list]
            cf, cl = ext_feats[cal_idx_list], ext_labels[cal_idx_list]
            qf, ql = ext_feats[test_mask], ext_labels[test_mask]
        else:
            # maincal / recalib: support from main domain
            sup_idx, cal_idx = sample_episode(pool_labels[pool_mask], K, M_CALIB, BASE_SEED + ep)
            sup_abs = np.where(pool_mask)[0][sup_idx]
            cal_abs = np.where(pool_mask)[0][cal_idx]
            sf, sl = pool_feats[sup_abs], pool_labels[sup_abs]

            if mode == "recalib":
                # calibration from external domain; remaining external = test
                cal_ext, test_mask = [], np.ones(len(ext_labels), bool)
                for c in CLASSES:
                    idx_c = np.where(ext_labels == c)[0]
                    pick = rng.choice(idx_c, min(m_ext, len(idx_c) // 2), replace=False)
                    cal_ext.append(pick)
                    test_mask[pick] = False
                cal_ext = np.concatenate(cal_ext)
                cf, cl = ext_feats[cal_ext], ext_labels[cal_ext]
                qf, ql = ext_feats[test_mask], ext_labels[test_mask]
            else:
                # maincal: calibration from main domain
                cf, cl = pool_feats[cal_abs], pool_labels[cal_abs]
                qf, ql = ext_feats, ext_labels

        for method, metrics in eval_all_methods(sf, sl, cf, cl, qf, ql, text_protos, cls_to_idx).items():
            results.append({"method": method, "episode": ep, **metrics})
        if (ep + 1) % 5 == 0:
            print(f"  episode {ep+1}/{N_EP} done", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"strawberryds_eval_{mode}_K16.csv", index=False, float_format="%.6f")
    summary = df.groupby("method").mean(numeric_only=True).reset_index()
    print(f"\n=== Strawberry-DS ({mode}, K=16, m_ext={m_ext}, {N_EP} ep) ===")
    print(summary[["method", "false_pick_rate", "pick_precision", "pick_recall", "revisit_burden"]].to_string(index=False))
    summary.to_csv(out_dir / f"strawberryds_eval_{mode}_summary.csv", index=False, float_format="%.6f")
    print(f"\nSaved -> {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, required=True, help="Strawberry-DS features npz")
    ap.add_argument("--main", type=Path, default=ROOT / "outputs" / "features_vitb32.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "strawberryds_eval")
    ap.add_argument("--mode", choices=["maincal", "recalib", "indomain"], default=None,
                    help="Evaluation mode (default: maincal)")
    ap.add_argument("--recalib", action="store_true", help="Shorthand for --mode recalib (compatibility)")
    ap.add_argument("--m_ext", type=int, default=M_EXT_RECALIB,
                    help=f"Per-class calibration samples in recalib/indomain mode (default: {M_EXT_RECALIB})")
    args = ap.parse_args()

    mode = args.mode or ("recalib" if args.recalib else "maincal")
    run(args.npz, args.main, args.out, mode, args.m_ext)
