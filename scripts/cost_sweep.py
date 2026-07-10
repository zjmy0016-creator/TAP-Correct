# -*- coding: utf-8 -*-
"""
Cost sensitivity sweep: cost robustness sweep
=================================

 TAP-Correct

2026-07-05
   TAP-Correct ** baseline **
  " /  / "
   baseline  baseline


  - C_fp: false-pick cost/borderline
  - C_fw: false-wait cost gt==pick  wait =
  - C_rev: revisit cost


  1. C_fp  [1,2,5,10,20], C_fw  [0.5,1,2], C_rev = 0.3
  2.  TAP-Correct
  3. heatmap

 C_rev C_rev(y)  future work


   - outputs/query_evaluation/cost_sweep_K{k}.csv
   - outputs/query_evaluation/cost_heatmap_K{k}.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============  ============
QUERY_DIR = ROOT / "outputs" / "query_evaluation"
GROUND_TRUTH_PATH = ROOT / "outputs" / "decision_gold" / "turning_decision_dataset" / "labels" / "test_decision_ground_truth_clean.csv"
OUT_DIR = ROOT / "outputs" / "query_evaluation"

K_LIST = [1, 2, 4, 8, 16]
N_EP_PER_K = 100

#
C_FP_RANGE = [1, 2, 5, 10, 20]      # false-pick cost
C_FW_RANGE = [0.5, 1.0, 2.0]        # false-wait cost
C_REV = 0.3                          # revisit cost


def load_ground_truth():
    """"""
    df = pd.read_csv(GROUND_TRUTH_PATH, encoding='utf-8-sig')
    df['decision_label'] = df['decision_label'].replace('borderline', 'revisit')
    return df['decision_label'].values


def compute_decision_cost(decisions, gt_decisions, C_fp, C_fw, C_rev):
    """


    2026-07-05  false-wait
      - false-pick (pred=pick, gtpick): C_fp
      - false-wait (pred=wait, gt==pick): C_fw wait =
         gt==pick  gt==revisit(borderline)
        wait  borderline
      - revisit (pred=revisit): C_rev
      - : 0



    """
    decisions = np.asarray(decisions)
    gt_decisions = np.asarray(gt_decisions)

    costs = np.zeros(len(decisions))

    # false-pick
    fp_mask = (decisions == 'pick') & (gt_decisions != 'pick')
    costs[fp_mask] = C_fp

    # false-waitgt==pick wait
    fw_mask = (decisions == 'wait') & (gt_decisions == 'pick')
    costs[fw_mask] = C_fw

    # revisit
    rev_mask = (decisions == 'revisit')
    costs[rev_mask] = C_rev

    return costs.mean()


def main():
    print("=== Cost sensitivity sweep: cost robustness sweep ===\n")

    #
    print("1.  ...")
    gt_decisions = load_ground_truth()

    print(f"2. :")
    print(f"   C_fp (false-pick): {C_FP_RANGE}")
    print(f"   C_fw (false-wait): {C_FW_RANGE}")
    print(f"   C_rev (revisit): {C_REV} ()\n")

    for k in K_LIST:
        print(f"3.  K={k} ...")

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
            data = np.load(npz_path, allow_pickle=True)
            decisions = data['decisions']
            all_preds.extend(decisions)

        all_preds = np.array(all_preds)
        all_gts = np.tile(gt_decisions, N_EP_PER_K)

        #
        results = []
        for C_fp in C_FP_RANGE:
            for C_fw in C_FW_RANGE:
                cost = compute_decision_cost(all_preds, all_gts, C_fp, C_fw, C_REV)
                # release:
                protocol_valid = (C_fp > C_fw) and (C_fw >= C_REV)
                results.append({
                    'k': k,
                    'C_fp': C_fp,
                    'C_fw': C_fw,
                    'C_rev': C_REV,
                    'mean_cost': cost,
                    'protocol_valid': protocol_valid,
                })

        #  CSV
        df = pd.DataFrame(results)
        out_csv = OUT_DIR / f"cost_sweep_K{k}.csv"
        df.to_csv(out_csv, index=False, float_format='%.6f')
        print(f"   : {out_csv.name}")

        #
        #  (C_fp  C_fw)
        cost_matrix = df.pivot(index='C_fp', columns='C_fw', values='mean_cost').values

        fig, ax = plt.subplots(figsize=(8, 6))

        #
        im = ax.imshow(cost_matrix, cmap='RdYlGn_r', aspect='auto')

        #
        ax.set_xticks(np.arange(len(C_FW_RANGE)))
        ax.set_yticks(np.arange(len(C_FP_RANGE)))
        ax.set_xticklabels(C_FW_RANGE)
        ax.set_yticklabels(C_FP_RANGE)

        #
        ax.set_xlabel('C_fw (false-wait cost)', fontsize=12)
        ax.set_ylabel('C_fp (false-pick cost)', fontsize=12)
        ax.set_title(f'Decision Cost Heatmap: TAP-Correct K={k}\n(C_rev={C_REV})',
                     fontsize=14, fontweight='bold')

        #
        for i in range(len(C_FP_RANGE)):
            for j in range(len(C_FW_RANGE)):
                text = ax.text(j, i, f'{cost_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)

        #
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean cost per sample', rotation=270, labelpad=20, fontsize=11)

        #
        out_img = OUT_DIR / f"cost_heatmap_K{k}.png"
        plt.tight_layout()
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   : {out_img.name}\n")

    print("")
    print(f"CSV files: cost_sweep_K{{1,2,4,8,16}}.csv")
    print(f"Image files: cost_heatmap_K{{1,2,4,8,16}}.png")
    print("\nNext analysis: paired bootstrap comparison")


if __name__ == "__main__":
    main()
