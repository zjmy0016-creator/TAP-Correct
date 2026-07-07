"""
D0 第3步：安全锁 / 门卫(EpisodeView)
===================================
作用：下游(D1/D2/D4)想用某个 episode 的数据，必须经过这个门卫。
门卫在交数据前，自动检查三堆绝不重叠：
  1. support ∩ calibration = 空
  2. (support + calibration) ∩ 考卷(query_test) = 空
一旦发现重叠 → 直接报错停下(方案 A)，从代码层面杜绝"偷看考卷"。

用法示意：
    view = EpisodeView(episode_dict, feats, labels, query_idx)
    sup_feats, sup_labels     = view.support()       # 建原型用
    cal_feats, cal_labels     = view.calibration()   # 定判分线用(唯一可用于选择的带标签数据)
    qry_feats                 = view.query_features() # 最终评估用(标签不在这里给)
"""
from __future__ import annotations
import numpy as np

CLASSES = ("ripe", "turning", "unripe")


class EpisodeView:
    """把一个 episode(第2步 manifest 里的一条)绑到特征上，并强制防泄漏检查。"""

    def __init__(self, episode, feats, labels, query_idx):
        """
        参数:
          episode   : manifest["episodes"] 里的一条 dict，含 support / calibration
          feats     : (N, 512) 全部特征
          labels    : (N,)     每行类别名
          query_idx : 考卷(query_test)的全部行号
        """
        self._feats = feats
        self._labels = labels

        # 把 support / calibration 的行号(按类)摊平成一维数组，方便取数和检查
        self._support_idx = np.concatenate(
            [np.asarray(episode["support"][c], dtype=int) for c in CLASSES]
        )
        self._calib_idx = np.concatenate(
            [np.asarray(episode["calibration"][c], dtype=int) for c in CLASSES]
        )
        self._query_idx = np.asarray(query_idx, dtype=int)

        # ★ 门卫核心：一建视图就立刻查，查不过直接报错
        self._assert_leakproof()

    def _assert_leakproof(self):
        """三条防泄漏硬检查，任一不过直接 raise ValueError(方案A: 停下)。"""
        sup = set(self._support_idx.tolist())
        cal = set(self._calib_idx.tolist())
        qry = set(self._query_idx.tolist())

        # 检查1: support 和 calibration 不重叠
        overlap_sc = sup & cal
        if overlap_sc:
            raise ValueError(
                f"[防泄漏失败] support 和 calibration 重叠 {len(overlap_sc)} 个行号: "
                f"{sorted(overlap_sc)[:5]}..."
            )

        # 检查2: support/calibration 都不碰考卷
        overlap_train_query = (sup | cal) & qry
        if overlap_train_query:
            raise ValueError(
                f"[防泄漏失败] 复习资料(support/calibration)碰到了考卷(query_test) "
                f"{len(overlap_train_query)} 个行号: {sorted(overlap_train_query)[:5]}..."
            )

        # 检查3: calibration 每类张数一致(平衡)，防止哪一类被漏抽
        counts = [len(episode_cls) for episode_cls in self._calib_by_class().values()]
        if len(set(counts)) != 1:
            raise ValueError(
                f"[防泄漏失败] calibration 各类张数不平衡: "
                f"{dict(zip(CLASSES, counts))}"
            )

    def _calib_by_class(self):
        """按类返回 calibration 行号(供检查3用)。"""
        result = {}
        for c in CLASSES:
            # 从 labels 里筛出属于类 c 的 calibration 行号
            mask = np.isin(self._calib_idx, np.where(self._labels == c)[0])
            result[c] = self._calib_idx[mask]
        return result

    # ── 三段只读视图 ──
    def support(self):
        """建原型用: 返回 (特征, 类别标签)。"""
        return self._feats[self._support_idx], self._labels[self._support_idx]

    def calibration(self):
        """定判分线用: 返回 (特征, 类别标签)。这是唯一可用于'选择'的带标签数据。"""
        return self._feats[self._calib_idx], self._labels[self._calib_idx]

    def query_features(self):
        """最终评估用: 只返回特征，不返回标签。
        标签由评估器在最后一步单独持有，避免中途手滑用到考卷答案。"""
        return self._feats[self._query_idx]


# ── 手动试跑区: 用第2步的 manifest 验证门卫能正常放行、也能拦住作弊 ──
if __name__ == "__main__":
    import json
    from pathlib import Path
    from episodes import load_pools   # 复用第1步的读数据函数

    root = Path(__file__).resolve().parents[1]
    npz_path = root / "outputs" / "features_vitb32.npz"
    manifest_path = root / "outputs" / "episodes" / "manifest_K1-16_ep100.json"

    feats, labels, sp_idx_by_cls, query_idx = load_pools(npz_path)
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    # 测试1: 正常 episode 应顺利放行
    ep0 = manifest["episodes"][0]
    view = EpisodeView(ep0, feats, labels, query_idx)
    sup_f, sup_l = view.support()
    cal_f, cal_l = view.calibration()
    qry_f = view.query_features()
    print("== 测试1: 正常 episode 放行 ===")
    print(f"  support:     {sup_f.shape[0]} 张")
    print(f"  calibration: {cal_f.shape[0]} 张")
    print(f"  query:       {qry_f.shape[0]} 张 (应为 593)")
    print("  ✓ 正常放行\n")

    # 测试2: 故意制造作弊(把一张考卷塞进 support)，门卫应报错拦下
    print("=== 测试2: 故意塞一张考卷进 support，门卫应拦下 ===")
    import copy
    bad_ep = copy.deepcopy(ep0)
    cheat_idx = int(query_idx[0])   # 拿一张考卷的行号
    bad_ep["support"]["ripe"].append(cheat_idx)   # 塞进 support
    try:
        EpisodeView(bad_ep, feats, labels, query_idx)
        print("  ✗ 没拦住！(不应出现这行)")
    except ValueError as e:
        print(f"  ✓ 成功拦下: {e}")

    # 测试3: 跑遍全部 500 个 episode，确认门卫对真实 manifest 全部放行
    print("\n=== 测试3: 全部 500 个 episode 逐一过门卫 ===")
    ok = 0
    for ep in manifest["episodes"]:
        EpisodeView(ep, feats, labels, query_idx)
        ok += 1
    print(f"  ✓ {ok}/500 全部通过防泄漏检查")