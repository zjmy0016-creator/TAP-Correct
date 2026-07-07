# -*- coding: utf-8 -*-
"""Build official D3-style evaluation artifacts for frozen TAP-Correct V1.

V1 reuses the existing D3 query scores and frozen D2 calibration table, but
switches the pick/wait endpoint from the old H axis to the 512D prototype
expectation endpoint E. This reproduces the V1 headline point while keeping the
query/test split out of threshold selection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.d3_step2_pooled_metrics import compute_pooled_metrics
from scripts.d3_step3_turning_audit import compute_turning_audit
from scripts.d3_step5_cost_sweep import (
    C_FP_RANGE,
    C_FW_RANGE,
    C_REV,
    compute_decision_cost,
)
from scripts.d4_step6_paired_difference import compute_episode_metrics
from tapcorrect.d2_decision import decide_episode_v1_endpoint


METRICS = ("false_pick_rate", "pick_precision", "pick_recall", "revisit_burden")


def episode_range_for_k(k: int) -> range:
    starts = {1: 0, 2: 100, 4: 200, 8: 300, 16: 400}
    if k not in starts:
        raise ValueError(f"Unknown K: {k}")
    start = starts[k]
    return range(start, start + 100)


def load_ground_truth(root: Path) -> pd.DataFrame:
    gt_path = (
        root
        / "outputs"
        / "decision_gold"
        / "turning_decision_dataset"
        / "labels"
        / "test_decision_ground_truth_clean.csv"
    )
    gt = pd.read_csv(gt_path, encoding="utf-8-sig")
    gt["decision_label"] = gt["decision_label"].replace("borderline", "revisit")
    return gt


def _threshold_row(thresholds: pd.DataFrame, ep_idx: int) -> pd.Series:
    rows = thresholds[thresholds["episode_idx"] == ep_idx]
    if len(rows) != 1:
        raise ValueError(f"Expected one threshold row for episode {ep_idx}, got {len(rows)}")
    return rows.iloc[0]


def generate_v1_episode_decisions(root: Path, k: int) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Generate frozen V1 query decisions for all episodes in one K block."""
    d3_dir = root / "outputs" / "d3_evaluate"
    thresholds = pd.read_csv(root / "outputs" / "d2_calibrate" / "thresholds_all_episodes.csv")

    decisions_list: list[np.ndarray] = []
    revisit_src_list: list[np.ndarray] = []
    episode_indices: list[int] = []

    for ep_idx in episode_range_for_k(k):
        row = _threshold_row(thresholds, ep_idx)
        data = np.load(d3_dir / f"query_decisions_K{k}_ep{ep_idx}.npz", allow_pickle=True)
        decisions, revisit_src = decide_episode_v1_endpoint(
            data["E_scores"],
            row["T_high_E"],
            row["T_low_E"],
            row["U_cut"],
        )
        decisions_list.append(decisions.astype(str))
        revisit_src_list.append(revisit_src.astype(str))
        episode_indices.append(ep_idx)

    return decisions_list, revisit_src_list, episode_indices


def bootstrap_metric_ci(
    episode_decisions: list[np.ndarray],
    gt_decisions: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    values = {metric: [] for metric in METRICS}
    n_episodes = len(episode_decisions)

    for _ in range(n_bootstrap):
        sampled = rng.choice(n_episodes, size=n_episodes, replace=True)
        preds = np.concatenate([episode_decisions[i] for i in sampled])
        gts = np.tile(gt_decisions, len(sampled))
        metrics = compute_pooled_metrics(preds, gts)
        for metric in METRICS:
            values[metric].append(float(metrics[metric]))

    rows = []
    for metric in METRICS:
        arr = np.asarray(values[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "mean": float(arr.mean()),
                "ci_lower": float(np.percentile(arr, 2.5)),
                "ci_upper": float(np.percentile(arr, 97.5)),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def load_comparator_decisions(root: Path, method_dir: Path, k: int) -> list[np.ndarray]:
    decisions = []
    for ep_idx in episode_range_for_k(k):
        data = np.load(method_dir / f"query_decisions_K{k}_ep{ep_idx}.npz", allow_pickle=True)
        decisions.append(data["decisions"].astype(str))
    return decisions


def paired_difference_ci(
    v1_decisions: list[np.ndarray],
    comparator_decisions: list[np.ndarray],
    gt_decisions: np.ndarray,
    comparator_name: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    v1_metrics = [compute_episode_metrics(d, gt_decisions) for d in v1_decisions]
    cmp_metrics = [compute_episode_metrics(d, gt_decisions) for d in comparator_decisions]

    rows = []
    for metric in METRICS:
        diffs = np.asarray(
            [v1[metric] - cmp[metric] for v1, cmp in zip(v1_metrics, cmp_metrics)],
            dtype=float,
        )
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(diffs), size=len(diffs), replace=True)
            boot.append(float(diffs[idx].mean()))
        boot = np.asarray(boot, dtype=float)
        lower = float(np.percentile(boot, 2.5))
        upper = float(np.percentile(boot, 97.5))
        rows.append(
            {
                "comparator": comparator_name,
                "metric": metric,
                "mean_diff_v1_minus_comparator": float(diffs.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "contains_zero": bool(lower <= 0 <= upper),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def cost_sweep(v1_decisions: list[np.ndarray], gt_decisions: np.ndarray, k: int) -> pd.DataFrame:
    preds = np.concatenate(v1_decisions)
    gts = np.tile(gt_decisions, len(v1_decisions))
    rows = []
    for c_fp in C_FP_RANGE:
        for c_fw in C_FW_RANGE:
            rows.append(
                {
                    "k": k,
                    "C_fp": c_fp,
                    "C_fw": c_fw,
                    "C_rev": C_REV,
                    "mean_cost": compute_decision_cost(preds, gts, c_fp, c_fw, C_REV),
                    "protocol_valid": (c_fp > c_fw) and (c_fw >= C_REV),
                }
            )
    return pd.DataFrame(rows)


def render_summary(metrics: pd.Series, ci: pd.DataFrame, turning: pd.Series) -> str:
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    ci_lines = []
    for _, row in ci.iterrows():
        ci_lines.append(
            f"- {row['metric']}: {pct(row['mean'])} "
            f"[{pct(row['ci_lower'])}, {pct(row['ci_upper'])}]"
        )

    lines = [
        "# V1 Official Evaluation",
        "",
        "This report evaluates the frozen V1 endpoint on the existing D3 query scores.",
        "The only method change from V0 is that pick/wait boundaries use E instead of H.",
        "",
        "## Pooled K=16 Metrics",
        "",
        f"- false-pick rate: {pct(metrics['false_pick_rate'])}",
        f"- pick precision: {pct(metrics['pick_precision'])}",
        f"- pick recall: {pct(metrics['pick_recall'])}",
        f"- revisit burden: {pct(metrics['revisit_burden'])}",
        "",
        "## Bootstrap CI",
        "",
        *ci_lines,
        "",
        "## Turning Audit",
        "",
        f"- revisit enrichment: {turning['revisit_enrichment']:.4f}",
        f"- enrichment lift: {turning['enrichment_lift']:.4f}",
        f"- defer-rate gap: {turning['defer_rate_gap']:.4f}",
        "",
        "Interpretation: turning uncertainty enrichment remains a supplementary audit, not the main contribution.",
        "",
    ]
    return "\n".join(lines)


def build_v1_official_eval(
    root: str | Path,
    out_dir: str | Path | None = None,
    k: int = 16,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / "outputs" / "v1_official_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_df = load_ground_truth(root)
    gt_decisions = gt_df["decision_label"].to_numpy()

    decisions, revisit_src, episode_indices = generate_v1_episode_decisions(root, k)
    all_preds = np.concatenate(decisions)
    all_gts = np.tile(gt_decisions, len(decisions))

    pooled = compute_pooled_metrics(all_preds, all_gts)
    metrics = pd.DataFrame([{**{"method": "V1_TAP_512D_endpoint", "k": k}, **pooled}])

    turning = compute_turning_audit(all_preds, gt_df)
    turning_df = pd.DataFrame([{**{"method": "V1_TAP_512D_endpoint", "k": k}, **turning}])

    ci = bootstrap_metric_ci(decisions, gt_decisions, n_bootstrap, seed)
    ci.insert(0, "k", k)
    ci.insert(0, "method", "V1_TAP_512D_endpoint")

    v0 = load_comparator_decisions(root, root / "outputs" / "d3_evaluate", k)
    b5 = load_comparator_decisions(
        root, root / "outputs" / "d4_baselines" / "B5_kshot_hard_reject", k
    )
    paired_v0 = paired_difference_ci(decisions, v0, gt_decisions, "V0_H_endpoint", n_bootstrap, seed)
    paired_b5 = paired_difference_ci(decisions, b5, gt_decisions, "B5_kshot_hard_reject", n_bootstrap, seed)

    costs = cost_sweep(decisions, gt_decisions, k)

    decisions_npz = out_dir / f"v1_query_decisions_K{k}.npz"
    metrics_csv = out_dir / f"v1_pooled_metrics_K{k}.csv"
    ci_csv = out_dir / f"v1_bootstrap_ci_K{k}.csv"
    turning_csv = out_dir / f"v1_turning_audit_K{k}.csv"
    paired_v0_csv = out_dir / f"v1_paired_diff_vs_v0_K{k}.csv"
    paired_b5_csv = out_dir / f"v1_paired_diff_vs_b5_K{k}.csv"
    cost_csv = out_dir / f"v1_cost_sweep_K{k}.csv"
    summary_md = out_dir / "V1_OFFICIAL_EVAL_SUMMARY.md"

    np.savez_compressed(
        decisions_npz,
        decisions=np.stack(decisions),
        revisit_src=np.stack(revisit_src),
        episode_idx=np.asarray(episode_indices, dtype=int),
        gt_decisions=gt_decisions.astype(str),
    )
    metrics.to_csv(metrics_csv, index=False, float_format="%.6f")
    ci.to_csv(ci_csv, index=False, float_format="%.6f")
    turning_df.to_csv(turning_csv, index=False, float_format="%.6f")
    paired_v0.to_csv(paired_v0_csv, index=False, float_format="%.6f")
    paired_b5.to_csv(paired_b5_csv, index=False, float_format="%.6f")
    costs.to_csv(cost_csv, index=False, float_format="%.6f")
    summary_md.write_text(render_summary(metrics.iloc[0], ci, turning_df.iloc[0]), encoding="utf-8")

    return {
        "decisions_npz": decisions_npz,
        "metrics_csv": metrics_csv,
        "ci_csv": ci_csv,
        "turning_csv": turning_csv,
        "paired_v0_csv": paired_v0_csv,
        "paired_b5_csv": paired_b5_csv,
        "cost_csv": cost_csv,
        "summary_md": summary_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifacts = build_v1_official_eval(
        root=args.root,
        out_dir=args.out,
        k=args.k,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
