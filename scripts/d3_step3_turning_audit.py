# -*- coding: utf-8 -*-
"""
D3 Step 3: turning 子集审计
===========================

检验模型是否把 revisit 集中到人工 borderline 样本。

只在 turning 129 子集内计算两个审计指标：
  - revisit enrichment: 模型 revisit 的 turning 中，有多少是人工 borderline
  - borderline defer rate: 人工 borderline 中，有多少被模型 revisit

红线：
  - 这两个指标只在 turning 子集内报告，不对全 593 做全局审计
  - 人工 revisit 标签解释为 borderline/ambiguous，不是第三类客观真值

输出：
  - outputs/d3_evaluate/step3_turning_audit.csv
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
    """加载决策真值，返回 DataFrame（带 filename, gt_class, decision_label）"""
    df = pd.read_csv(GROUND_TRUTH_PATH, encoding='utf-8-sig')
    # 将 borderline 映射为 revisit（论文口径）
    df['decision_label'] = df['decision_label'].replace('borderline', 'revisit')
    return df


def compute_turning_audit(all_preds, gt_df):
    """
    计算 turning 子集的审计指标（D3-FINAL：含对照量）。

    参数：
      all_preds: (n_episodes * 593,) 预测决策数组
      gt_df: 真值 DataFrame (593 行)

    返回：
      dict: turning 审计指标（含 enrichment lift 和 defer rate gap）
    """
    n_episodes = len(all_preds) // len(gt_df)

    # 找到 turning 样本的索引
    turning_mask = (gt_df['gt_class'] == 'turning').values
    turning_indices = np.where(turning_mask)[0]
    n_turning = len(turning_indices)

    # 找到人工 borderline 与 non-borderline 样本的索引
    human_borderline_mask = (gt_df['gt_class'] == 'turning') & (gt_df['decision_label'] == 'revisit')
    human_borderline_indices = np.where(human_borderline_mask)[0]
    n_human_borderline = len(human_borderline_indices)

    human_non_borderline_mask = (gt_df['gt_class'] == 'turning') & (gt_df['decision_label'] != 'revisit')
    human_non_borderline_indices = np.where(human_non_borderline_mask)[0]
    n_human_non_borderline = len(human_non_borderline_indices)

    # 统计：模型 revisit 的 turning 样本
    n_model_revisit_turning = 0
    n_model_revisit_turning_borderline = 0
    n_human_borderline_deferred = 0
    n_human_non_borderline_deferred = 0

    for ep_i in range(n_episodes):
        ep_start = ep_i * len(gt_df)
        ep_preds = all_preds[ep_start : ep_start + len(gt_df)]

        # 统计该 episode 中：
        # 1. 模型 revisit 的 turning 样本数
        for idx in turning_indices:
            if ep_preds[idx] == 'revisit':
                n_model_revisit_turning += 1
                # 如果该 turning 样本人工也标为 borderline
                if gt_df.iloc[idx]['decision_label'] == 'revisit':
                    n_model_revisit_turning_borderline += 1

        # 2. 人工 borderline 被模型 revisit 的数量
        for idx in human_borderline_indices:
            if ep_preds[idx] == 'revisit':
                n_human_borderline_deferred += 1

        # 3. 人工 non-borderline 被模型 revisit 的数量（对照组）
        for idx in human_non_borderline_indices:
            if ep_preds[idx] == 'revisit':
                n_human_non_borderline_deferred += 1

    # 原有指标
    revisit_enrichment = (n_model_revisit_turning_borderline / n_model_revisit_turning
                          if n_model_revisit_turning > 0 else 0.0)
    borderline_defer_rate = (n_human_borderline_deferred / (n_human_borderline * n_episodes)
                              if n_human_borderline > 0 else 0.0)

    # 新增对照量（D3-FINAL）
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
        # D3-FINAL 新增对照量
        'borderline_base_rate': borderline_base_rate,
        'enrichment_lift': enrichment_lift,
        'non_borderline_defer_rate': non_borderline_defer_rate,
        'defer_rate_gap': defer_rate_gap,
    }


def main():
    print("=== D3 Step 3: turning 子集审计 ===\n")
    
    # 加载真值
    print("1. 加载决策真值 ...")
    gt_df = load_ground_truth()
    print(f"   真值样本数: {len(gt_df)}")
    
    # 统计 turning 分布
    turning_df = gt_df[gt_df['gt_class'] == 'turning']
    print(f"   turning 样本数: {len(turning_df)}")
    print(f"   turning 人工标注分布:")
    print(f"     pick: {(turning_df['decision_label']=='pick').sum()}")
    print(f"     revisit (borderline): {(turning_df['decision_label']=='revisit').sum()}")
    print(f"     wait: {(turning_df['decision_label']=='wait').sum()}\n")
    
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
        
        for ep_idx in ep_range:
            npz_path = QUERY_DIR / f"query_decisions_K{k}_ep{ep_idx}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"缺少文件: {npz_path}")
            
            data = np.load(npz_path, allow_pickle=True)
            decisions = data['decisions']  # (593,)
            all_preds.extend(decisions)
        
        # 转为 numpy 数组
        all_preds = np.array(all_preds)
        
        # 计算 turning 审计指标
        metrics = compute_turning_audit(all_preds, gt_df)
        metrics['k'] = k
        results.append(metrics)

        print(f"   revisit enrichment:      {metrics['revisit_enrichment']:.4f}")
        print(f"   borderline_base_rate:    {metrics['borderline_base_rate']:.4f}")
        print(f"   enrichment_lift:         {metrics['enrichment_lift']:.4f}")
        print(f"   borderline defer rate:   {metrics['borderline_defer_rate']:.4f}")
        print(f"   non_borderline defer:    {metrics['non_borderline_defer_rate']:.4f}")
        print(f"   defer_rate_gap:          {metrics['defer_rate_gap']:.4f}")
        print(f"   (模型 revisit 的 turning: {metrics['n_model_revisit_turning']}, "
              f"其中 borderline: {metrics['n_model_revisit_turning_borderline']})")
        print(f"   (人工 borderline 被 defer: {metrics['n_human_borderline_deferred']} / "
              f"{metrics['n_human_borderline'] * metrics['n_episodes']})\n")

    # 保存结果
    df = pd.DataFrame(results)
    cols = ['k', 'n_turning', 'n_human_borderline', 'n_human_non_borderline', 'n_episodes',
            'revisit_enrichment', 'borderline_base_rate', 'enrichment_lift',
            'borderline_defer_rate', 'non_borderline_defer_rate', 'defer_rate_gap',
            'n_model_revisit_turning', 'n_model_revisit_turning_borderline',
            'n_human_borderline_deferred', 'n_human_non_borderline_deferred']
    df = df[cols]

    out_path = OUT_DIR / "step3_turning_audit.csv"
    df.to_csv(out_path, index=False, float_format='%.6f')

    print(f"完成！结果已保存到 {out_path}")
    print("\nturning 审计指标汇总（D3-FINAL 含对照量）：")
    print(df[['k', 'revisit_enrichment', 'enrichment_lift',
              'borderline_defer_rate', 'defer_rate_gap']].to_string(index=False))
    print("\n下一步：D3 Step4 - risk-coverage curve")


if __name__ == "__main__":
    main()