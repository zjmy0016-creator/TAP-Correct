"""
D0 第1步：分堆器（episode sampler）
=================================
作用：从"复习资料"(support_pool) 里，为一次实验(episode)分出两堆：
  - support    : 每类 K 张，用来建"草莓长什么样"的原型
  - calibration: 每类 M 张(默认200)，用来定判分线；与 support 不重叠
"考卷"(query_test) 全程不碰，留到最后一次性评估。

本文件只负责"分堆"，不算训练：不改 CLIP 权重、无梯度、无 loss。
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

# 三个类别，顺序固定，后面都按这个顺序
CLASSES = ("ripe", "turning", "unripe")


def load_pools(npz_path):
    """读缓存特征，按 split 和 class 整理出每一堆的"行号"。

    返回:
      feats         : (6985, 512) 全部特征，后面用行号去取
      labels        : (6985,)     每行的类别名
      sp_idx_by_cls : dict，如 {"ripe": array([...]), ...}，support_pool 各类行号
      query_idx     : array，     query_test 全部行号(考卷，固定不动)
    """
    data = np.load(npz_path, allow_pickle=True)
    feats  = data["image_feats"]
    labels = data["classes"]
    splits = data["splits"]

    sp_idx_by_cls = {}
    for c in CLASSES:
        # 同时满足"属于复习资料"且"是这个类别"的行号
        sp_idx_by_cls[c] = np.where((splits == "support_pool") & (labels == c))[0]

    query_idx = np.where(splits == "query_test")[0]
    return feats, labels, sp_idx_by_cls, query_idx


def sample_one_episode(sp_idx_by_cls, k, m_calib=200, seed=0):
    """分一次堆：每类抽 K 张 support，再从剩下的抽 M 张 calibration。

    参数:
      sp_idx_by_cls : load_pools 返回的复习资料行号(按类)
      k             : 每类 support 张数(K-shot)
      m_calib       : 每类 calibration 张数(默认200)
      seed          : 随机种子；同一个 seed 结果完全一样(可复现)

    返回:
      dict: {"support": {类:行号array}, "calibration": {类:行号array}}
    """
    rng = np.random.default_rng(seed)   # 带种子的随机器，保证可复现
    support = {}
    calibration = {}

    for c in CLASSES:
        pool = sp_idx_by_cls[c].copy()   # 这一类所有复习资料的行号
        # 安全检查：这一类的资料够不够抽 K + M 张
        if len(pool) < k + m_calib:
            raise ValueError(
                f"类别 {c} 资料不够: 有 {len(pool)} 张, "
                f"但要抽 support {k} + calibration {m_calib} = {k + m_calib} 张"
            )
        # 先把这一类的行号打乱(用带种子的随机器)
        shuffled = rng.permutation(pool)
        # 前 K 张给 support
        support[c] = np.sort(shuffled[:k])
        # 紧接着 M 张给 calibration —— 天然和 support 不重叠
        calibration[c] = np.sort(shuffled[k:k + m_calib])

    return {"support": support, "calibration": calibration}
    
def build_manifest(sp_idx_by_cls, k_list=(1, 2, 4, 8, 16), n_ep=100,
                   m_calib=200, base_seed=42):
    """一次性抽好所有 episode(默认 5种K × 100次 = 500个)。

    每个 episode 用不同种子(base_seed + 全局序号)，既随机又可复现。

    返回:
      dict: {"meta": {...说明...}, "episodes": [每个episode的堆, ...]}
    """
    episodes = []
    counter = 0   # 全局序号，保证每个 episode 种子都不同
    for k in k_list:
        for ep_i in range(n_ep):
            seed = base_seed + counter
            drawn = sample_one_episode(sp_idx_by_cls, k=k, m_calib=m_calib, seed=seed)
            episodes.append({
                "k": k,
                "episode_idx": ep_i,
                "seed": seed,
                # numpy 数组不能直接存 JSON，转成普通 list
                "support":     {c: drawn["support"][c].tolist()     for c in CLASSES},
                "calibration": {c: drawn["calibration"][c].tolist() for c in CLASSES},
            })
            counter += 1

    meta = {
        "k_list": list(k_list),
        "n_ep_per_k": n_ep,
        "m_calib": m_calib,
        "base_seed": base_seed,
        "total_episodes": len(episodes),
        "note": "考卷 query_test 全局固定593张，不逐 episode 存；下游统一另取。",
    }
    return {"meta": meta, "episodes": episodes}


def save_manifest(manifest, out_path):
    """把 manifest 存成 JSON 档案文件。"""
    import json
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)   # 目录不存在就建
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"已保存 manifest 到: {out_path}")

# ── 手动试跑区：直接运行本文件时，抽一个 episode 看效果 ──
# if __name__ == "__main__":
#     # 找到 outputs/features_vitb32.npz（不管从哪个目录运行都能找到）
#     npz_path = Path(__file__).resolve().parents[1] / "outputs" / "features_vitb32.npz"
#     feats, labels, sp_idx_by_cls, query_idx = load_pools(npz_path)

#     print("=== 复习资料(support_pool)每类张数 ===")
#     for c in CLASSES:
#         print(f"  {c}: {len(sp_idx_by_cls[c])}")
#     print(f"=== 考卷(query_test)总张数: {len(query_idx)} ===\n")

#     ep = sample_one_episode(sp_idx_by_cls, k=4, m_calib=200, seed=0)
#     print("=== 抽一个 episode (K=4, M=200, seed=0) ===")
#     for c in CLASSES:
#         print(f"  {c}: support {len(ep['support'][c])} 张, "
#               f"calibration {len(ep['calibration'][c])} 张")

#     # 自检：support 和 calibration 有没有重叠(应该全是 0)
#     print("\n=== 自检：两堆重叠张数(应全为 0) ===")
#     for c in CLASSES:
#         overlap = set(ep["support"][c]) & set(ep["calibration"][c])
#         print(f"  {c}: {len(overlap)}")
# ── 手动试跑区：生成并保存完整 manifest ──
if __name__ == "__main__":
    npz_path = Path(__file__).resolve().parents[1] / "outputs" / "features_vitb32.npz"
    feats, labels, sp_idx_by_cls, query_idx = load_pools(npz_path)

    # 生成全部 500 个 episode
    manifest = build_manifest(sp_idx_by_cls)

    # 存到 outputs/episodes/manifest_K1-16_ep100.json
    out_path = Path(__file__).resolve().parents[1] / "outputs" / "episodes" / "manifest_K1-16_ep100.json"
    save_manifest(manifest, out_path)

    # ── 存完自检 ──
    m = manifest["meta"]
    print(f"\n=== manifest 概况 ===")
    print(f"  K 种类: {m['k_list']}")
    print(f"  每种 K 抽: {m['n_ep_per_k']} 次")
    print(f"  总 episode 数: {m['total_episodes']}  (应为 500)")

    # 抽查第1个和最后1个 episode，确认种子不同、张数对
    first, last = manifest["episodes"][0], manifest["episodes"][-1]
    print(f"\n  第1个:   K={first['k']}, seed={first['seed']}, "
          f"turning support {len(first['support']['turning'])} 张")
    print(f"  最后1个: K={last['k']}, seed={last['seed']}, "
          f"turning support {len(last['support']['turning'])} 张")

    # 可复现自检：同参数再抽一次，应完全一样
    manifest2 = build_manifest(sp_idx_by_cls)
    same = manifest["episodes"] == manifest2["episodes"]
    print(f"\n  可复现自检(两次生成是否完全一致): {same}  (应为 True)")