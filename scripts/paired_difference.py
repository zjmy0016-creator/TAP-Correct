# -*- coding: utf-8 -*-
"""
Paired bootstrap comparison:  bootstrap
==================================

 TAP-Correct vs B5  episode
 bootstrap 1000  95% CI

 CI  0


  outputs/baselines/paired_differences_TAP_vs_B5.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============  ============
TAP_DIR = ROOT / "outputs" / "query_evaluation"
B5_DIR = ROOT / "outputs" / "baselines" / "B5_kshot_hard_reject"
GROUND_TRUTH_PATH = ROOT / "outputs" / "decision_gold" / "turning_decision_dataset" / "labels" / "test_decision_ground_truth_clean.csv"
OUT_DIR = ROOT / "outputs" / "baselines"

K_LIST = [1, 2, 4, 8, 16]
N_EP_PER_K = 100
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def load_ground_truth():
    """"""
    df = pd.read_csv(GROUND_TRUTH_PATH, encoding='utf-8-sig')
    df['decision_label'] = df['decision_label'].replace('borderline', 'revisit')
    return df['decision_label'].values


def compute_episode_metrics(decisions, gt_decisions):
    """
     episode


      dict: false_pick_rate, pick_precision, pick_recall, revisit_burden
    """
    decisions = np.asarray(decisions)
    gt_decisions = np.asarray(gt_decisions)

    n_pred_pick = (decisions == 'pick').sum()
    n_pred_revisit = (decisions == 'revisit').sum()

    n_gt_pick = (gt_decisions == 'pick').sum()

    n_pick_pick = ((decisions == 'pick') & (gt_decisions == 'pick')).sum()
    n_false_pick = ((decisions == 'pick') & (gt_decisions != 'pick')).sum()

    false_pick_rate = n_false_pick / n_pred_pick if n_pred_pick > 0 else 0.0
    pick_precision = n_pick_pick / n_pred_pick if n_pred_pick > 0 else 0.0
    pick_recall = n_pick_pick / n_gt_pick if n_gt_pick > 0 else 0.0
    revisit_burden = n_pred_revisit / len(decisions)

    return {
        'false_pick_rate': false_pick_rate,
        'pick_precision': pick_precision,
        'pick_recall': pick_recall,
        'revisit_burden': revisit_burden,
    }


def load_episode_metrics(method_dir, k, gt_decisions):
    """
     K  episode


      (100,)  dict  4
    """
    if k == 1:
        ep_range = range(0, 100)
    elif k == 2:
        ep_range = range(100, 200)
    elif k == 4:
        ep_range = range(200, 300)
    elif k == 8:
        ep_range = range(300, 400)
    elif k == 16:
        ep_range = range(400, 500)
    else:
        raise ValueError(f" K : {k}")

    episode_metrics = []

    for ep_idx in ep_range:
        npz_path = method_dir / f"query_decisions_K{k}_ep{ep_idx}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f": {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        decisions = data['decisions']  # (593,)

        metrics = compute_episode_metrics(decisions, gt_decisions)
        episode_metrics.append(metrics)

    return episode_metrics


def bootstrap_paired_difference(tap_metrics, b5_metrics, metric_name):
    """
     bootstrap


      tap_metrics: list of dict (100  episode)
      b5_metrics: list of dict (100  episode)
      metric_name:  'pick_recall'


      dict: mean_diff, ci_lower, ci_upper
    """
    #  episode
    tap_values = np.array([m[metric_name] for m in tap_metrics])
    b5_values = np.array([m[metric_name] for m in b5_metrics])
    paired_diffs = tap_values - b5_values  # (100,)

    # Bootstrap
    np.random.seed(BOOTSTRAP_SEED)

    bootstrap_diffs = []

    for _ in range(N_BOOTSTRAP):
        #  100  episode
        sampled_indices = np.random.choice(N_EP_PER_K, size=N_EP_PER_K, replace=True)
        sampled_diffs = paired_diffs[sampled_indices]

        #
        bootstrap_diffs.append(sampled_diffs.mean())

    bootstrap_diffs = np.array(bootstrap_diffs)

    #  mean  95% CI
    mean_diff = paired_diffs.mean()
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'contains_zero': (ci_lower <= 0 <= ci_upper),
    }


def main():
    print("=== Paired bootstrap comparison:  bootstrap  ===\n")

    #
    print("1.  ...")
    gt_decisions = load_ground_truth()
    print(f"   : {len(gt_decisions)}\n")

    print(f"2. Bootstrap : {N_BOOTSTRAP} , seed={BOOTSTRAP_SEED}\n")

    results = []

    for k in K_LIST:
        print(f"3.  K={k} ...")

        #  TAP-Correct  episode
        print(f"    TAP-Correct episode  ...", end=" ")
        tap_metrics = load_episode_metrics(TAP_DIR, k, gt_decisions)
        print(f" ({len(tap_metrics)} episodes)")

        #  B5  episode
        print(f"    B5 episode  ...", end=" ")
        b5_metrics = load_episode_metrics(B5_DIR, k, gt_decisions)
        print(f" ({len(b5_metrics)} episodes)")

        #  bootstrap
        print(f"    ...")

        row = {'k': k}

        for metric in ['false_pick_rate', 'pick_precision', 'pick_recall', 'revisit_burden']:
            result = bootstrap_paired_difference(tap_metrics, b5_metrics, metric)

            row[f'{metric}_mean_diff'] = result['mean_diff']
            row[f'{metric}_ci_lower'] = result['ci_lower']
            row[f'{metric}_ci_upper'] = result['ci_upper']
            row[f'{metric}_contains_zero'] = result['contains_zero']

            sig_marker = "" if result['contains_zero'] else "  "
            print(f"     {metric}: {result['mean_diff']:+.4f} [{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]{sig_marker}")

        results.append(row)
        print()

    #
    df = pd.DataFrame(results)

    #
    cols = ['k']
    for metric in ['false_pick_rate', 'pick_precision', 'pick_recall', 'revisit_burden']:
        cols.extend([
            f'{metric}_mean_diff',
            f'{metric}_ci_lower',
            f'{metric}_ci_upper',
            f'{metric}_contains_zero',
        ])

    df = df[cols]

    out_path = OUT_DIR / "paired_differences_TAP_vs_B5.csv"
    df.to_csv(out_path, index=False, float_format='%.6f')

    print("="*70)
    print(f" {out_path}")
    print("="*70)

    #  K=16
    print("\nTAP-Correct vs B5 (K=16, )\n")

    row_k16 = df[df['k'] == 16].iloc[0]

    print("false_pick_rate:")
    print(f"  : {row_k16['false_pick_rate_mean_diff']:+.4f} "
          f"[{row_k16['false_pick_rate_ci_lower']:+.4f}, {row_k16['false_pick_rate_ci_upper']:+.4f}]")
    if row_k16['false_pick_rate_contains_zero']:
        print("  : CI  0")
    else:
        print("  : CI  0TAP  false_pick  B5")

    print("\npick_recall:")
    print(f"  : {row_k16['pick_recall_mean_diff']:+.4f} "
          f"[{row_k16['pick_recall_ci_lower']:+.4f}, {row_k16['pick_recall_ci_upper']:+.4f}]")
    if row_k16['pick_recall_contains_zero']:
        print("  : CI  0")
    else:
        print("  : CI  0TAP  recall  B5")

    print("\nrevisit_burden:")
    print(f"  : {row_k16['revisit_burden_mean_diff']:+.4f} "
          f"[{row_k16['revisit_burden_ci_lower']:+.4f}, {row_k16['revisit_burden_ci_upper']:+.4f}]")
    if row_k16['revisit_burden_contains_zero']:
        print("  : CI  0")
    else:
        print("  : CI  0TAP  revisit  B5")

    print("\n" + "="*70)
    print(" CI  0")
    print("'significantly higher recall'  'significantly lower revisit burden'")
    print("="*70)


if __name__ == "__main__":
    main()
