# -*- coding: utf-8 -*-
"""
Turning audit: turning
===========================

 revisit  borderline

 turning 129
  - revisit enrichment:  revisit  turning  borderline
  - borderline defer rate:  borderline  revisit


  -  turning  593
  -  revisit  borderline/ambiguous


   - outputs/query_evaluation/turning_audit.csv
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
    """ DataFrame filename, gt_class, decision_label"""
    df = pd.read_csv(GROUND_TRUTH_PATH, encoding='utf-8-sig')
    #  borderline  revisit
    df['decision_label'] = df['decision_label'].replace('borderline', 'revisit')
    return df


def compute_turning_audit(all_preds, gt_df):
    """
     turning release


      all_preds: (n_episodes * 593,)
      gt_df:  DataFrame (593 )


      dict: turning  enrichment lift  defer rate gap
    """
    n_episodes = len(all_preds) // len(gt_df)

    #  turning
    turning_mask = (gt_df['gt_class'] == 'turning').values
    turning_indices = np.where(turning_mask)[0]
    n_turning = len(turning_indices)

    #  borderline  non-borderline
    human_borderline_mask = (gt_df['gt_class'] == 'turning') & (gt_df['decision_label'] == 'revisit')
    human_borderline_indices = np.where(human_borderline_mask)[0]
    n_human_borderline = len(human_borderline_indices)

    human_non_borderline_mask = (gt_df['gt_class'] == 'turning') & (gt_df['decision_label'] != 'revisit')
    human_non_borderline_indices = np.where(human_non_borderline_mask)[0]
    n_human_non_borderline = len(human_non_borderline_indices)

    #  revisit  turning
    n_model_revisit_turning = 0
    n_model_revisit_turning_borderline = 0
    n_human_borderline_deferred = 0
    n_human_non_borderline_deferred = 0

    for ep_i in range(n_episodes):
        ep_start = ep_i * len(gt_df)
        ep_preds = all_preds[ep_start : ep_start + len(gt_df)]

        #  episode
        # 1.  revisit  turning
        for idx in turning_indices:
            if ep_preds[idx] == 'revisit':
                n_model_revisit_turning += 1
                #  turning  borderline
                if gt_df.iloc[idx]['decision_label'] == 'revisit':
                    n_model_revisit_turning_borderline += 1

        # 2.  borderline  revisit
        for idx in human_borderline_indices:
            if ep_preds[idx] == 'revisit':
                n_human_borderline_deferred += 1

        # 3.  non-borderline  revisit
        for idx in human_non_borderline_indices:
            if ep_preds[idx] == 'revisit':
                n_human_non_borderline_deferred += 1

    #
    revisit_enrichment = (n_model_revisit_turning_borderline / n_model_revisit_turning
                          if n_model_revisit_turning > 0 else 0.0)
    borderline_defer_rate = (n_human_borderline_deferred / (n_human_borderline * n_episodes)
                              if n_human_borderline > 0 else 0.0)

    # release
    borderline_base_rate = n_human_borderline / n_turning if n_turning > 0 else 0.0
    enrichment_lift = revisit_enrichment - borderline_base_rate

    non_borderline_defer_rate = (n_human_non_borderline_deferred / (n_human_non_borderline * n_episodes)
                                  if n_human_non_borderline > 0 else 0.0)
    defer_rate_gap = borderline_defer_rate - non_borderline_defer_rate

    return {
        'n_turning': n_turning,
        'n_human_borderline': n_human_borderline,
        'n_human_non_borderline': n_human_non_borderline,
        'n_episodes': n_episodes,
        'n_model_revisit_turning': n_model_revisit_turning,
        'n_model_revisit_turning_borderline': n_model_revisit_turning_borderline,
        'n_human_borderline_deferred': n_human_borderline_deferred,
        'n_human_non_borderline_deferred': n_human_non_borderline_deferred,
        'revisit_enrichment': revisit_enrichment,
        'borderline_defer_rate': borderline_defer_rate,
        # release
        'borderline_base_rate': borderline_base_rate,
        'enrichment_lift': enrichment_lift,
        'non_borderline_defer_rate': non_borderline_defer_rate,
        'defer_rate_gap': defer_rate_gap,
    }


def main():
    print("=== Turning audit: turning  ===\n")

    #
    print("1.  ...")
    gt_df = load_ground_truth()
    print(f"   : {len(gt_df)}")

    #  turning
    turning_df = gt_df[gt_df['gt_class'] == 'turning']
    print(f"   turning : {len(turning_df)}")
    print(f"   turning :")
    print(f"     pick: {(turning_df['decision_label']=='pick').sum()}")
    print(f"     revisit (borderline): {(turning_df['decision_label']=='revisit').sum()}")
    print(f"     wait: {(turning_df['decision_label']=='wait').sum()}\n")

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

        for ep_idx in ep_range:
            npz_path = QUERY_DIR / f"query_decisions_K{k}_ep{ep_idx}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f": {npz_path}")

            data = np.load(npz_path, allow_pickle=True)
            decisions = data['decisions']  # (593,)
            all_preds.extend(decisions)

        #  numpy
        all_preds = np.array(all_preds)

        #  turning
        metrics = compute_turning_audit(all_preds, gt_df)
        metrics['k'] = k
        results.append(metrics)

        print(f"   revisit enrichment:      {metrics['revisit_enrichment']:.4f}")
        print(f"   borderline_base_rate:    {metrics['borderline_base_rate']:.4f}")
        print(f"   enrichment_lift:         {metrics['enrichment_lift']:.4f}")
        print(f"   borderline defer rate:   {metrics['borderline_defer_rate']:.4f}")
        print(f"   non_borderline defer:    {metrics['non_borderline_defer_rate']:.4f}")
        print(f"   defer_rate_gap:          {metrics['defer_rate_gap']:.4f}")
        print(f"   ( revisit  turning: {metrics['n_model_revisit_turning']}, "
              f" borderline: {metrics['n_model_revisit_turning_borderline']})")
        print(f"   ( borderline  defer: {metrics['n_human_borderline_deferred']} / "
              f"{metrics['n_human_borderline'] * metrics['n_episodes']})\n")

    #
    df = pd.DataFrame(results)
    cols = ['k', 'n_turning', 'n_human_borderline', 'n_human_non_borderline', 'n_episodes',
            'revisit_enrichment', 'borderline_base_rate', 'enrichment_lift',
            'borderline_defer_rate', 'non_borderline_defer_rate', 'defer_rate_gap',
            'n_model_revisit_turning', 'n_model_revisit_turning_borderline',
            'n_human_borderline_deferred', 'n_human_non_borderline_deferred']
    df = df[cols]

    out_path = OUT_DIR / "turning_audit.csv"
    df.to_csv(out_path, index=False, float_format='%.6f')

    print(f" {out_path}")
    print("\nturning release ")
    print(df[['k', 'revisit_enrichment', 'enrichment_lift',
              'borderline_defer_rate', 'defer_rate_gap']].to_string(index=False))
    print("\nNext analysis: risk-coverage frontier")


if __name__ == "__main__":
    main()
