"""
D1 第3步：uncertainty（不确定性）5 个候选公式
================================
每个候选把一张图映射成一个"越大越不确定"的连续分数（方向统一约定）。
本步只实现打分 + 方向自检（turning 平均不确定性应最高），
正式 AURC / turning 富集评选留到 Step4。

5 个预注册候选：
  - entropy               : 三类 softmax 概率的归一化熵
  - top2_margin           : 1 - (top1 概率 - top2 概率)，越接近越不确定
  - dist_to_threshold     : 到判分边界的距离（代理边界=calibration 三类均值中点）
  - bootstrap_variance    : support 有放回重采样后分数的标准差（估计稳定性）
  - prototype_disagreement: K 个 single-shot 分类器预测的不一致率

所有特征已在 D0 阶段 L2 归一化，内积即余弦相似度。
"""
from __future__ import annotations
import numpy as np

from tapcorrect.harvestability import (
    _softmax_probs, build_visual_prototypes, score_expectation,
)

CLASSES = ("ripe", "turning", "unripe")


# ── 候选1 · predictive entropy ──
def unc_entropy(feats, vis_protos, temperature=1.0):
    """三类 softmax 概率的熵，除以 log(3) 归一到 0-1。越大越不确定。"""
    p = _softmax_probs(feats, vis_protos, temperature)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    return ent / np.log(len(CLASSES))


# ── 候选2 · top1-top2 margin ──
def unc_top2_margin(feats, vis_protos, temperature=1.0):
    """1 - (最大概率 - 次大概率)。两者越接近越不确定，故取 1-margin，越大越不确定。"""
    p = _softmax_probs(feats, vis_protos, temperature)
    ps = np.sort(p, axis=1)              # 升序
    margin = ps[:, -1] - ps[:, -2]       # top1 - top2
    return 1.0 - margin


# ── 候选3 · distance-to-threshold ──
def unc_distance_to_threshold(base_scores, base_labels):
    """到判分边界的距离取负（越接近边界越不确定 → 越大）。

    代理边界（D1 可得口径，D2 才有真 T_high/T_low）：
      T_high = (mean_ripe + mean_turning)/2
      T_low  = (mean_turning + mean_unripe)/2
    base_scores 用 expectation(τ=1) 分数；均值/边界只从 calibration 标签算。
    """
    means = {c: base_scores[base_labels == c].mean() for c in CLASSES}
    t_high = (means["ripe"] + means["turning"]) / 2.0
    t_low = (means["turning"] + means["unripe"]) / 2.0
    min_dist = np.minimum(np.abs(base_scores - t_high),
                          np.abs(base_scores - t_low))
    return -min_dist                     # 越接近边界(min_dist小) → 值越大 → 越不确定


# ── 候选4 · bootstrap variance ──
def unc_bootstrap_variance(cal_feats, sup_feats, sup_labels, B=20, seed=0):
    """support 每类有放回重采样 B 次，各自建原型算 expectation 分数，取每样本的 std。

    衡量"估计稳定性"：分数在不同 support 抽样下波动大 = 不确定。
    K=1 时重采样只能取到那一张，std≈0（此候选在低 K 天然失效，属预期诊断）。
    """
    by_cls = {c: np.where(sup_labels == c)[0] for c in CLASSES}
    rng = np.random.default_rng(seed)
    stack = []
    for _ in range(B):
        boot_idx = []
        for c in CLASSES:
            idx_c = by_cls[c]
            boot_idx.append(rng.choice(idx_c, size=len(idx_c), replace=True))
        boot_idx = np.concatenate(boot_idx)
        protos = build_visual_prototypes(sup_feats[boot_idx], sup_labels[boot_idx])
        stack.append(score_expectation(cal_feats, protos, 1.0))
    return np.stack(stack, axis=0).std(axis=0)


# ── 候选5 · prototype disagreement ──
def unc_prototype_disagreement(cal_feats, sup_feats, sup_labels):
    """K 个 single-shot 分类器（第 i 个用每类第 i 张 support 建三类原型）的预测不一致率。

    对每个样本，K 个分类器各投一票，disagreement = 1 - 众数票占比。
    K=1 时只有 1 个分类器，disagreement 恒为 0（低 K 天然失效，属预期诊断）。
    """
    by_cls = {c: np.sort(np.where(sup_labels == c)[0]) for c in CLASSES}
    K = min(len(by_cls[c]) for c in CLASSES)
    N = cal_feats.shape[0]
    if K <= 1:
        return np.zeros(N)
    preds = []
    for i in range(K):
        protos = {}
        for c in CLASSES:
            v = sup_feats[by_cls[c][i]]
            protos[c] = v / (np.linalg.norm(v) + 1e-12)
        sims = np.stack([cal_feats @ protos[c] for c in CLASSES], axis=1)
        preds.append(sims.argmax(axis=1))
    preds = np.stack(preds, axis=0)      # (K, N)
    disagree = np.empty(N)
    for j in range(N):
        counts = np.bincount(preds[:, j], minlength=len(CLASSES))
        disagree[j] = 1.0 - counts.max() / K
    return disagree


# ── 统一入口 ──
def all_uncertainty_scores(cal_feats, cal_labels, sup_feats, sup_labels,
                           vis_protos, seed=0, B=20):
    """一次算出 5 个 uncertainty 候选分数。全部"越大越不确定"。

    返回: dict {候选名: (N,) 分数}
    """
    base = score_expectation(cal_feats, vis_protos, 1.0)
    return {
        "entropy":                unc_entropy(cal_feats, vis_protos),
        "top2_margin":            unc_top2_margin(cal_feats, vis_protos),
        "dist_to_threshold":      unc_distance_to_threshold(base, cal_labels),
        "bootstrap_variance":     unc_bootstrap_variance(cal_feats, sup_feats,
                                                         sup_labels, B, seed),
        "prototype_disagreement": unc_prototype_disagreement(cal_feats, sup_feats,
                                                             sup_labels),
    }
