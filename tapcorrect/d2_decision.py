# -*- coding: utf-8 -*-
"""
D2 决策逻辑（canonical）
========================
pick / wait / revisit 决策的唯一权威实现。
Step 3 脚本、D2 交棒契约测试、D3 评估都从这里 import，禁止另写副本。

约定：
  H = raw axis projection（D1 冻结的文本轴投影）
  E = expectation(τ=1)（few-shot 视觉原型期望）
  uncertainty u = -min_dist(E, T_low_E, T_high_E)，值越大越不确定。

决策（合并两个 revisit 来源）：
  1. u >= U_cut（低把握）            -> revisit（uncertainty 触发）
  2. u <  U_cut（高把握）:
       H >= T_high                   -> pick
       H <= T_low                    -> wait
       T_low < H < T_high（H-band）  -> revisit（boundary_band，H-space 端点模糊）

红线：boundary_band 是 H-space endpoint ambiguity，不是 turning proxy。

V1 冻结协议：
  pick / wait endpoint 从 H 切换为 512D 原型期望 E；uncertainty 仍是 E 到
  E 端点阈值的距离。V0 函数保留以复现 D0-D5，V1 使用
  decide_episode_v1_endpoint。
"""
import numpy as np


def compute_uncertainty(E_scores, T_high_E, T_low_E):
    """
    uncertainty = -min_distance(E, T_low_E, T_high_E)
    值越大（越接近 0）越不确定；离边界越远越负（越确定）。
    返回与 E_scores 同形状的数组。
    """
    E_scores = np.asarray(E_scores, dtype=float)
    below = E_scores <= T_low_E
    above = E_scores >= T_high_E
    dist = np.where(
        below, T_low_E - E_scores,
        np.where(
            above, E_scores - T_high_E,
            np.minimum(E_scores - T_low_E, T_high_E - E_scores),
        ),
    )
    return -dist


def decide_episode(H, E, T_high, T_low, T_high_E, T_low_E, U_cut):
    """
    对一个 episode 的样本做 pick/wait/revisit 决策。

    参数：
      H, E          : 同长度的分数数组
      T_high, T_low : H 空间的有序动作阈值（T_low <= T_high）
      T_high_E, T_low_E : E 空间的有序 uncertainty 参考边界
      U_cut         : uncertainty 阈值（负数）

    返回：
      decision    : object 数组，取值 pick / wait / revisit
      revisit_src : object 数组，取值 none / uncertainty / boundary_band
    """
    H = np.asarray(H, dtype=float)
    E = np.asarray(E, dtype=float)

    u = compute_uncertainty(E, T_high_E, T_low_E)

    n = len(H)
    decision = np.full(n, "revisit", dtype=object)
    revisit_src = np.full(n, "none", dtype=object)

    confident = u < U_cut          # 高把握
    uncertain = ~confident         # 低把握 -> uncertainty 触发 revisit

    pick_mask = confident & (H >= T_high)
    wait_mask = confident & (H <= T_low)
    band_mask = confident & (H > T_low) & (H < T_high)

    decision[pick_mask] = "pick"
    decision[wait_mask] = "wait"
    # band_mask 与 uncertain 保持 revisit

    revisit_src[uncertain] = "uncertainty"
    revisit_src[band_mask] = "boundary_band"

    return decision, revisit_src


def decide_episode_v1_endpoint(E, T_high_E, T_low_E, U_cut):
    """
    V1 冻结决策：pick / wait endpoint 使用 512D 原型期望 E。

    参数：
      E          : expectation(ripe=1, turning=0.5, unripe=0) 分数数组
      T_high_E   : E 空间 pick 阈值
      T_low_E    : E 空间 wait 阈值
      U_cut      : uncertainty 阈值（负数）

    返回：
      decision    : object 数组，取值 pick / wait / revisit
      revisit_src : object 数组，取值 none / uncertainty / boundary_band
    """
    E = np.asarray(E, dtype=float)
    u = compute_uncertainty(E, T_high_E, T_low_E)

    n = len(E)
    decision = np.full(n, "revisit", dtype=object)
    revisit_src = np.full(n, "none", dtype=object)

    confident = u < U_cut
    uncertain = ~confident

    pick_mask = confident & (E >= T_high_E)
    wait_mask = confident & (E <= T_low_E)
    band_mask = confident & (E > T_low_E) & (E < T_high_E)

    decision[pick_mask] = "pick"
    decision[wait_mask] = "wait"

    revisit_src[uncertain] = "uncertainty"
    revisit_src[band_mask] = "boundary_band"

    return decision, revisit_src
