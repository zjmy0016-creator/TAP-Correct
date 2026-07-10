# -*- coding: utf-8 -*-
"""
Pooled evaluation metrics:  pooled
=============================

 K  100  episode pooled

  - false-pick rate ()
  - pick precision ()
  - pick recall / harvestable coverage ()
  - revisit burden ()

 pooled  mean-of-ratiosinitial  12


   - outputs/query_evaluation/pooled_metrics_by_k.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============  ============
QUERY_DIR = ROOT / "outputs" / "query_evaluation"
GROUND_TRUTH_PATH = ROOT / "outputs" / "decision_gold" / "turning_decision_dataset" / "labels" / "test_decision_ground_truth_clean.csv"
OUT_DIR = ROOT / "outputs" / "query_evaluation"

K_LIST = [1, 2, 4, 8, 16]
N_EP_PER_K = 100


def load_ground_truth():
    """
     (593  query_test )


      gt_decisions: (593,)  'pick' / 'wait' / 'revisit'
      gt_filenames: (593,)
      gt_classes: (593,)  'ripe' / 'turning' / 'unripe'
    """
    df = pd.read_csv(GROUND_TRUTH_PATH, encoding='utf-8-sig')

    #  decision_label  'borderline'  'revisit'
    df['decision_label'] = df['decision_label'].replace('borderline', 'revisit')

    return df['decision_label'].values, df['filename'].values, df['gt_class'].values


def compute_pooled_metrics(all_preds, all_gts):
    """
     pooled


      all_preds: (n_total,)
      all_gts: (n_total,)


      dict:
    """
    all_preds = np.asarray(all_preds)
    all_gts = np.asarray(all_gts)

    #  sklearn
    n_pred_pick = (all_preds == 'pick').sum()
    n_pred_wait = (all_preds == 'wait').sum()
    n_pred_revisit = (all_preds == 'revisit').sum()

    n_gt_pick = (all_gts == 'pick').sum()
    n_gt_wait = (all_gts == 'wait').sum()
    n_gt_revisit = (all_gts == 'revisit').sum()

    # pick-pick:  pick  pick
    n_pick_pick = ((all_preds == 'pick') & (all_gts == 'pick')).sum()

    # false-pick:  pick  pick
    n_false_pick = ((all_preds == 'pick') & (all_gts != 'pick')).sum()

    # false-wait:  wait  wait
    n_false_wait = ((all_preds == 'wait') & (all_gts != 'wait')).sum()

    #
    false_pick_rate = n_false_pick / n_pred_pick if n_pred_pick > 0 else 0.0
    pick_precision = n_pick_pick / n_pred_pick if n_pred_pick > 0 else 0.0
    pick_recall = n_pick_pick / n_gt_pick if n_gt_pick > 0 else 0.0
    revisit_burden = n_pred_revisit / len(all_preds)

    return {
        'n_total': len(all_preds),
        'n_pred_pick': n_pred_pick,
        'n_pred_wait': n_pred_wait,
        'n_pred_revisit': n_pred_revisit,
        'n_gt_pick': n_gt_pick,
        'n_gt_wait': n_gt_wait,
        'n_gt_revisit': n_gt_revisit,
        'n_pick_pick': n_pick_pick,
        'n_false_pick': n_false_pick,
        'n_false_wait': n_false_wait,
        'false_pick_rate': false_pick_rate,
        'pick_precision': pick_precision,
        'pick_recall': pick_recall,
        'revisit_burden': revisit_burden,
    }


def main():
    print("=== Pooled evaluation metrics:  pooled  ===\n")

    #
    print("1.  ...")
    gt_decisions, gt_filenames, gt_classes = load_ground_truth()
    print(f"   : {len(gt_decisions)}")
    print(f"   : pick={np.sum(gt_decisions=='pick')}, "
          f"wait={np.sum(gt_decisions=='wait')}, "
          f"revisit={np.sum(gt_decisions=='revisit')}\n")

    results = []

    for k in K_LIST:
        print(f"2.  K={k} ...")

        #  K  episode
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

        #  episode
        all_preds = []
        all_gts_repeated = []

        for ep_idx in ep_range:
            npz_path = QUERY_DIR / f"query_decisions_K{k}_ep{ep_idx}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f": {npz_path}")

            data = np.load(npz_path, allow_pickle=True)
            decisions = data['decisions']  # (593,)

            all_preds.extend(decisions)
            all_gts_repeated.extend(gt_decisions)

        #  numpy
        all_preds = np.array(all_preds)
        all_gts_repeated = np.array(all_gts_repeated)

        print(f"   : {len(all_preds)}  ( {N_EP_PER_K * 593})")

        #  pooled
        metrics = compute_pooled_metrics(all_preds, all_gts_repeated)
        metrics['k'] = k
        results.append(metrics)

        print(f"   false-pick rate: {metrics['false_pick_rate']:.4f}")
        print(f"   pick precision:  {metrics['pick_precision']:.4f}")
        print(f"   pick recall:     {metrics['pick_recall']:.4f}")
        print(f"   revisit burden:  {metrics['revisit_burden']:.4f}\n")

    #
    df = pd.DataFrame(results)
    #
    cols = ['k', 'n_total', 'false_pick_rate', 'pick_precision', 'pick_recall', 'revisit_burden',
            'n_pred_pick', 'n_pred_wait', 'n_pred_revisit',
            'n_gt_pick', 'n_gt_wait', 'n_gt_revisit',
            'n_pick_pick', 'n_false_pick', 'n_false_wait']
    df = df[cols]

    out_path = OUT_DIR / "pooled_metrics_by_k.csv"
    df.to_csv(out_path, index=False, float_format='%.6f')

    print(f" {out_path}")
    print("\n")
    print(df[['k', 'false_pick_rate', 'pick_precision', 'pick_recall', 'revisit_burden']].to_string(index=False))
    print("\nNext analysis: turning audit")


if __name__ == "__main__":
    main()
