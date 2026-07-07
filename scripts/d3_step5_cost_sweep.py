# -*- coding: utf-8 -*-
"""
D3 Step 5: cost robustness sweep
=================================

扫描合理成本区域，计算 TAP-Correct 自身的期望决策成本结构。

范围说明（2026-07-05）：
  本步骤只刻画 TAP-Correct 在不同成本参数下的成本结构，**不含 baseline 对照**，
  因此不能据此下"成本有利 / 成本鲁棒 / 优势区域"等比较性结论。
  与 baseline 的成本对比留到 D4 之后再做。

成本矩阵参数：
  - C_fp: false-pick cost（误采未熟果/borderline，不可逆）
  - C_fw: false-wait cost（把 gt==pick 的成熟果判成 wait = 漏采）
  - C_rev: revisit cost（延迟重访，统一成本）

方法：
  1. 定义成本参数空间（C_fp ∈ [1,2,5,10,20], C_fw ∈ [0.5,1,2], C_rev = 0.3）
  2. 对每个成本组合，计算 TAP-Correct 的期望成本
  3. 输出成本热图（heatmap）

注意：当前使用统一 C_rev，状态相关 C_rev(y) 留作 future work。

输出：
  - outputs/d3_evaluate/step5_cost_sweep_K{k}.csv (每个 K 一个文件)
  - outputs/d3_evaluate/step5_cost_heatmap_K{k}.png (每个 K 一张图)
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

# ============ 配置 ============
QUERY_DIR = ROOT / "outputs" / "d3_evaluate"
GROUND_TRUTH_PATH = ROOT / "outputs" / "decision_gold" / "turning_decision_dataset" / "labels" / "test_decision_ground_truth_clean.csv"
OUT_DIR = ROOT / "outputs" / "d3_evaluate"

K_LIST = [1, 2, 4, 8, 16]
N_EP_PER_K = 100

# 成本参数空间
C_FP_RANGE = [1, 2, 5, 10, 20]      # false-pick cost（不可逆）
C_FW_RANGE = [0.5, 1.0, 2.0]        # false-wait cost（漏采）
C_REV = 0.3                          # revisit cost（统一延迟成本）


def load_ground_truth():
    """加载决策真值"""
    df = pd.read_csv(GROUND_TRUTH_PATH, encoding='utf-8-sig')
    df['decision_label'] = df['decision_label'].replace('borderline', 'revisit')
    return df['decision_label'].values


def compute_decision_cost(decisions, gt_decisions, C_fp, C_fw, C_rev):
    """
    计算给定成本参数下的期望决策成本。

    成本定义（2026-07-05 校正 false-wait 口径）：
      - false-pick (pred=pick, gt≠pick): C_fp（误采，不可逆）
      - false-wait (pred=wait, gt==pick): C_fw（把该采的成熟果判成 wait = 漏采）
        注意：只对 gt==pick 计漏采成本，不含 gt==revisit(borderline)。
        wait 一个 borderline 果未必是错误，故不计入漏采成本。
      - revisit (pred=revisit): C_rev（延迟重访，统一成本）
      - 正确决策: 0

    返回：
      平均每样本成本
    """
    decisions = np.asarray(decisions)
    gt_decisions = np.asarray(gt_decisions)

    costs = np.zeros(len(decisions))

    # false-pick
    fp_mask = (decisions == 'pick') & (gt_decisions != 'pick')
    costs[fp_mask] = C_fp

    # false-wait：只对本该采的成熟果（gt==pick）被判 wait 计漏采成本
    fw_mask = (decisions == 'wait') & (gt_decisions == 'pick')
    costs[fw_mask] = C_fw

    # revisit（统一成本，不管真值是什么）
    rev_mask = (decisions == 'revisit')
    costs[rev_mask] = C_rev

    return costs.mean()


def main():
    print("=== D3 Step 5: cost robustness sweep ===\n")
    
    # 加载真值
    print("1. 加载决策真值 ...")
    gt_decisions = load_ground_truth()
    
    print(f"2. 成本参数空间:")
    print(f"   C_fp (false-pick): {C_FP_RANGE}")
    print(f"   C_fw (false-wait): {C_FW_RANGE}")
    print(f"   C_rev (revisit): {C_REV} (统一)\n")
    
    for k in K_LIST:
        print(f"3. 处理 K={k} ...")
        
        # 找到该 K 的所有 episode
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
            raise ValueError(f"未知的 K 值: {k}")
        
        # 堆叠所有 episode 的预测
        all_preds = []
        for ep_idx in ep_range:
            npz_path = QUERY_DIR / f"query_decisions_K{k}_ep{ep_idx}.npz"
            data = np.load(npz_path, allow_pickle=True)
            decisions = data['decisions']
            all_preds.extend(decisions)
        
        all_preds = np.array(all_preds)
        all_gts = np.tile(gt_decisions, N_EP_PER_K)
        
        # 扫描成本空间
        results = []
        for C_fp in C_FP_RANGE:
            for C_fw in C_FW_RANGE:
                cost = compute_decision_cost(all_preds, all_gts, C_fp, C_fw, C_REV)
                # D3-FINAL: 判断是否符合冻结协议
                protocol_valid = (C_fp > C_fw) and (C_fw >= C_REV)
                results.append({
                    'k': k,
                    'C_fp': C_fp,
                    'C_fw': C_fw,
                    'C_rev': C_REV,
                    'mean_cost': cost,
                    'protocol_valid': protocol_valid,
                })
        
        # 保存 CSV
        df = pd.DataFrame(results)
        out_csv = OUT_DIR / f"step5_cost_sweep_K{k}.csv"
        df.to_csv(out_csv, index=False, float_format='%.6f')
        print(f"   已保存: {out_csv.name}")
        
        # 绘制热图
        # 重塑为矩阵 (C_fp × C_fw)
        cost_matrix = df.pivot(index='C_fp', columns='C_fw', values='mean_cost').values
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 使用颜色映射
        im = ax.imshow(cost_matrix, cmap='RdYlGn_r', aspect='auto')
        
        # 设置刻度
        ax.set_xticks(np.arange(len(C_FW_RANGE)))
        ax.set_yticks(np.arange(len(C_FP_RANGE)))
        ax.set_xticklabels(C_FW_RANGE)
        ax.set_yticklabels(C_FP_RANGE)
        
        # 标签
        ax.set_xlabel('C_fw (false-wait cost)', fontsize=12)
        ax.set_ylabel('C_fp (false-pick cost)', fontsize=12)
        ax.set_title(f'Decision Cost Heatmap: TAP-Correct K={k}\n(C_rev={C_REV})', 
                     fontsize=14, fontweight='bold')
        
        # 在每个格子里标注数值
        for i in range(len(C_FP_RANGE)):
            for j in range(len(C_FW_RANGE)):
                text = ax.text(j, i, f'{cost_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10)
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean cost per sample', rotation=270, labelpad=20, fontsize=11)
        
        # 保存图片
        out_img = OUT_DIR / f"step5_cost_heatmap_K{k}.png"
        plt.tight_layout()
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   已保存: {out_img.name}\n")
    
    print("完成！")
    print(f"CSV 文件: step5_cost_sweep_K{{1,2,4,8,16}}.csv")
    print(f"图片文件: step5_cost_heatmap_K{{1,2,4,8,16}}.png")
    print("\n下一步：D3 Step6 - bootstrap 置信区间")


if __name__ == "__main__":
    main()