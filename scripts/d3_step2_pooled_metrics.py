# -*- coding: utf-8 -*-
"""
D3 Step 2: 计算 pooled 主指标
=============================

对每个 K 值，将 100 个 episode 的预测堆进一个大混淆矩阵（pooled），
然后计算主指标：
  - false-pick rate (不该采却采，核心风险)
  - pick precision (采下来的果中有多少确实应采)
  - pick recall / harvestable coverage (不能把大量可采果错过)
  - revisit burden (延迟重访比例)

红线：必须用 pooled 混淆矩阵，不能用 mean-of-ratios（D0 已证实后者系统性高估约 12 点）。

输出：
  - outputs/d3_evaluate/step2_pooled_metrics_by_k.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============ 配置 ============
QUERY_DIR = ROOT / "outputs" / "d3_evaluate"
GROUND_TRUTH_PATH = ROOT / "outputs" / "decision_gold" / "turning_decision_dataset" / "labels" / "test_decision_ground_truth_clean.csv"
OUT_DIR = ROOT / "outputs" / "d3_evaluate"

K_LIST = [1, 2, 4, 8, 16]
N_EP_PER_K = 100


def load_ground_truth():
    """
    加载决策真值 (593 个 query_test 样本)。
    
    返回：
      gt_decisions: (593,) 数组，取值 'pick' / 'wait' / 'revisit'
      gt_filenames: (593,) 数组，文件名（用于对齐检查）
      gt_classes: (593,) 数组，原始类别 'ripe' / 'turning' / 'unripe'
    """
    df = pd.read_csv(GROUND_TRUTH_PATH, encoding='utf-8-sig')
    
    # 将 decision_label 中的 'borderline' 映射为 'revisit'（论文口径）
    df['decision_label'] = df['decision_label'].replace('borderline', 'revisit')
    
    return df['decision_label'].values, df['filename'].values, df['gt_class'].values


def compute_pooled_metrics(all_preds, all_gts):
    """
    从 pooled 预测和真值计算主指标。
    
    参数：
      all_preds: (n_total,) 预测决策数组
      all_gts: (n_total,) 真值决策数组
    
    返回：
      dict: 包含所有主指标
    """
    all_preds = np.asarray(all_preds)
    all_gts = np.asarray(all_gts)
    
    # 构建混淆矩阵（手动统计，避免 sklearn 依赖）
    n_pred_pick = (all_preds == 'pick').sum()
    n_pred_wait = (all_preds == 'wait').sum()
    n_pred_revisit = (all_preds == 'revisit').sum()
    
    n_gt_pick = (all_gts == 'pick').sum()
    n_gt_wait = (all_gts == 'wait').sum()
    n_gt_revisit = (all_gts == 'revisit').sum()
    
    # pick-pick: 预测 pick 且真值 pick
    n_pick_pick = ((all_preds == 'pick') & (all_gts == 'pick')).sum()
    
    # false-pick: 预测 pick 但真值不是 pick
    n_false_pick = ((all_preds == 'pick') & (all_gts != 'pick')).sum()
    
    # false-wait: 预测 wait 但真值不是 wait
    n_false_wait = ((all_preds == 'wait') & (all_gts != 'wait')).sum()
    
    # 核心指标
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
    print("=== D3 Step 2: 计算 pooled 主指标 ===\n")
    
    # 加载真值
    print("1. 加载决策真值 ...")
    gt_decisions, gt_filenames, gt_classes = load_ground_truth()
    print(f"   真值样本数: {len(gt_decisions)}")
    print(f"   真值分布: pick={np.sum(gt_decisions=='pick')}, "
          f"wait={np.sum(gt_decisions=='wait')}, "
          f"revisit={np.sum(gt_decisions=='revisit')}\n")
    
    results = []
    
    for k in K_LIST:
        print(f"2. 处理 K={k} ...")
        
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
        all_gts_repeated = []
        
        for ep_idx in ep_range:
            npz_path = QUERY_DIR / f"query_decisions_K{k}_ep{ep_idx}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"缺少文件: {npz_path}")
            
            data = np.load(npz_path, allow_pickle=True)
            decisions = data['decisions']  # (593,)
            
            all_preds.extend(decisions)
            all_gts_repeated.extend(gt_decisions)
        
        # 转为 numpy 数组
        all_preds = np.array(all_preds)
        all_gts_repeated = np.array(all_gts_repeated)
        
        print(f"   堆叠完成: {len(all_preds)} 个样本 (应为 {N_EP_PER_K * 593})")
        
        # 计算 pooled 指标
        metrics = compute_pooled_metrics(all_preds, all_gts_repeated)
        metrics['k'] = k
        results.append(metrics)
        
        print(f"   false-pick rate: {metrics['false_pick_rate']:.4f}")
        print(f"   pick precision:  {metrics['pick_precision']:.4f}")
        print(f"   pick recall:     {metrics['pick_recall']:.4f}")
        print(f"   revisit burden:  {metrics['revisit_burden']:.4f}\n")
    
    # 保存结果
    df = pd.DataFrame(results)
    # 调整列顺序
    cols = ['k', 'n_total', 'false_pick_rate', 'pick_precision', 'pick_recall', 'revisit_burden',
            'n_pred_pick', 'n_pred_wait', 'n_pred_revisit',
            'n_gt_pick', 'n_gt_wait', 'n_gt_revisit',
            'n_pick_pick', 'n_false_pick', 'n_false_wait']
    df = df[cols]
    
    out_path = OUT_DIR / "step2_pooled_metrics_by_k.csv"
    df.to_csv(out_path, index=False, float_format='%.6f')
    
    print(f"完成！结果已保存到 {out_path}")
    print("\n主指标汇总：")
    print(df[['k', 'false_pick_rate', 'pick_precision', 'pick_recall', 'revisit_burden']].to_string(index=False))
    print("\n下一步：D3 Step3 - turning 子集审计")


if __name__ == "__main__":
    main()