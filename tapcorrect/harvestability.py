"""
D1 第2步：harvestability（可采性分数）4 个候选公式
================================
每个候选把一张图映射成一个"越可采越高"的连续分数。
本步只实现打分 + 方向自检（ripe > turning > unripe），
正式 AUROC / in-band 评选留到 Step4。

锚点消歧（避免 axis 与 margin 数学等价而退化成同一候选）：
  - axis score           : 用【文本】原型端点差向量（zero-shot，不依赖 support）
  - ripe-vs-unripe margin: 用【视觉】support 原型（few-shot）
  - 512D 期望            : 视觉 support 三类原型，τ=1 softmax 加权
  - calibrated softmax   : 视觉 support 三类原型 + 标定温度 τ

所有特征已在 D0 阶段 L2 归一化，内积即余弦相似度。
"""
from __future__ import annotations
import numpy as np

CLASSES = ("ripe", "turning", "unripe")
# 期望公式的成熟度锚值：unripe=0, turning=0.5, ripe=1
EXPECT_ANCHOR = {"unripe": 0.0, "turning": 0.5, "ripe": 1.0}


# ── 原型构建 ────────────────────
def build_visual_prototypes(sup_feats, sup_labels):
    """用 support 特征按类求均值再 L2 归一化，得到 3 个视觉原型。

    返回: dict {类: (512,) 归一化原型向量}
    """
    protos = {}
    for c in CLASSES:
        mask = (sup_labels == c)
        if mask.sum() == 0:
            raise ValueError(f"support 中类别 {c} 没有样本，无法建原型")
        mean_vec = sup_feats[mask].mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        protos[c] = mean_vec / (norm + 1e-12)
    return protos


def load_text_prototypes(npz_path):
    """读 D0 缓存里的三类文本原型（每类5条prompt取平均再归一化）。

    返回: dict {类: (512,) 归一化文本原型}
    """
    data = np.load(npz_path, allow_pickle=True)
    text_protos = {}
    for c in CLASSES:
        arr = data[f"text_{c}"]          # (5, 512)
        mean_vec = arr.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        text_protos[c] = mean_vec / (norm + 1e-12)
    return text_protos


# ── 4 个候选打分函数 ────────────────────────────
def score_axis(feats, text_protos):
    """候选1 · axis score：文本端点差向量上的投影（zero-shot）。

    v = normalize(text_ripe - text_unripe); s = <img, v>
    返回原始投影（未归一化到0-1，归一化在Step4按calibration分位数做）。
    """
    v = text_protos["ripe"] - text_protos["unripe"]
    v = v / (np.linalg.norm(v) + 1e-12)
    return feats @ v


def score_margin(feats, vis_protos):
    """候选3 · ripe-vs-unripe margin：视觉原型相似度差（few-shot）。

    m = <img, proto_ripe> - <img, proto_unripe>
    """
    return feats @ vis_protos["ripe"] - feats @ vis_protos["unripe"]


def _softmax_probs(feats, vis_protos, temperature=1.0):
    """三类视觉原型相似度 → 带温度 softmax 概率。

    返回: (N, 3) 概率，列顺序 = CLASSES
    """
    sims = np.stack([feats @ vis_protos[c] for c in CLASSES], axis=1)  # (N,3)
    logits = sims / temperature
    logits = logits - logits.max(axis=1, keepdims=True)               # 数值稳定
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def score_expectation(feats, vis_protos, temperature=1.0):
    """候选2 · 512D 全空间期望：softmax 概率对成熟度锚值加权求期望。

    E = 0*p_unripe + 0.5*p_turning + 1*p_ripe   （天然落在 0-1）
    temperature=1.0 时是"期望"候选；传入标定温度即"calibrated softmax"候选。
    """
    probs = _softmax_probs(feats, vis_protos, temperature)
    anchors = np.array([EXPECT_ANCHOR[c] for c in CLASSES])  # 按 CLASSES 顺序
    return probs @ anchors


def score_calibrated_softmax(feats, vis_protos, temperature):
    """候选4 · calibrated softmax expectation：带标定温度的期望。

    与候选2同式，但温度由 calibration 标定（NLL 最小化，Step4 实现）。
    """
    return score_expectation(feats, vis_protos, temperature=temperature)


# ── 统一入口：给一个 episode 的 calibration 算全部候选分数 ──
def all_harvestability_scores(cal_feats, vis_protos, text_protos,
                              temperature_for_calib=1.0):
    """对一堆 calibration 特征，一次算出 4 个候选的分数。

    参数:
      temperature_for_calib: calibrated softmax 用的温度。
                Step2 自检先传占位 1.0；Step4 会传标定后的 τ。
    返回: dict {候选名: (N,) 分数}
    """
    return {
        "axis":               score_axis(cal_feats, text_protos),
        "expectation":        score_expectation(cal_feats, vis_protos, 1.0),
        "margin":             score_margin(cal_feats, vis_protos),
        "calibrated_softmax": score_calibrated_softmax(
                                cal_feats, vis_protos, temperature_for_calib),
       }
