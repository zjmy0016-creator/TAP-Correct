"""Build official V1 evaluation artifacts from frozen query decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cost_sweep import C_FP_RANGE, C_FW_RANGE, C_REV, compute_decision_cost
from scripts.paired_difference import compute_episode_metrics
from scripts.pooled_metrics import compute_pooled_metrics
from scripts.turning_audit import compute_turning_audit
from tapcorrect.decision import decide_episode_v1_endpoint


METRICS = ("false_pick_rate", "pick_precision", "pick_recall", "revisit_burden")


def episode_range_for_k(k: int) -> range:
    starts = {1: 0, 2: 100, 4: 200, 8: 300, 16: 400}
    if k not in starts:
        raise ValueError(f"Unknown K: {k}")
    return range(starts[k], starts[k] + 100)


def load_ground_truth(root: Path) -> pd.DataFrame:
    path = root / "outputs/decision_gold/turning_decision_dataset/labels/test_decision_ground_truth_clean.csv"
    ground_truth = pd.read_csv(path, encoding="utf-8-sig")
    ground_truth["decision_label"] = ground_truth["decision_label"].replace("borderline", "revisit")
    return ground_truth


def _threshold_row(thresholds: pd.DataFrame, episode_index: int) -> pd.Series:
    rows = thresholds[thresholds["episode_idx"] == episode_index]
    if len(rows) != 1:
        raise ValueError(f"Expected one threshold row for episode {episode_index}, got {len(rows)}")
    return rows.iloc[0]


def generate_v1_episode_decisions(root: Path, k: int):
    """Generate V1 query decisions for all episodes in one support-size block."""
    query_dir = root / "outputs/query_evaluation"
    thresholds = pd.read_csv(root / "outputs/calibration/thresholds_all_episodes.csv")
    decisions, revisit_sources, episode_indices = [], [], []
    for episode_index in episode_range_for_k(k):
        threshold = _threshold_row(thresholds, episode_index)
        data = np.load(query_dir / f"query_decisions_K{k}_ep{episode_index}.npz", allow_pickle=True)
        episode_decisions, source = decide_episode_v1_endpoint(
            data["E_scores"], threshold["T_high_E"], threshold["T_low_E"], threshold["U_cut"]
        )
        decisions.append(episode_decisions.astype(str))
        revisit_sources.append(source.astype(str))
        episode_indices.append(episode_index)
    return decisions, revisit_sources, episode_indices


def bootstrap_metric_ci(episode_decisions, ground_truth_decisions, n_bootstrap, seed):
    rng = np.random.RandomState(seed)
    values = {metric: [] for metric in METRICS}
    for _ in range(n_bootstrap):
        sampled = rng.choice(len(episode_decisions), size=len(episode_decisions), replace=True)
        predictions = np.concatenate([episode_decisions[index] for index in sampled])
        ground_truth = np.tile(ground_truth_decisions, len(sampled))
        metrics = compute_pooled_metrics(predictions, ground_truth)
        for metric in METRICS:
            values[metric].append(float(metrics[metric]))
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": float(np.mean(values[metric])),
                "ci_lower": float(np.percentile(values[metric], 2.5)),
                "ci_upper": float(np.percentile(values[metric], 97.5)),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            }
            for metric in METRICS
        ]
    )


def load_comparator_decisions(root: Path, method_dir: Path, k: int):
    decisions = []
    for episode_index in episode_range_for_k(k):
        data = np.load(method_dir / f"query_decisions_K{k}_ep{episode_index}.npz", allow_pickle=True)
        decisions.append(data["decisions"].astype(str))
    return decisions


def paired_difference_ci(
    v1_decisions,
    comparator_decisions,
    ground_truth_decisions,
    comparator_name,
    n_bootstrap,
    seed,
):
    rng = np.random.RandomState(seed)
    v1_metrics = [compute_episode_metrics(decisions, ground_truth_decisions) for decisions in v1_decisions]
    comparator_metrics = [compute_episode_metrics(decisions, ground_truth_decisions) for decisions in comparator_decisions]
    rows = []
    for metric in METRICS:
        differences = np.asarray(
            [v1[metric] - comparator[metric] for v1, comparator in zip(v1_metrics, comparator_metrics)],
            dtype=float,
        )
        bootstrap = np.asarray(
            [differences[rng.choice(len(differences), size=len(differences), replace=True)].mean() for _ in range(n_bootstrap)],
            dtype=float,
        )
        lower, upper = np.percentile(bootstrap, [2.5, 97.5])
        rows.append(
            {
                "comparator": comparator_name,
                "metric": metric,
                "mean_diff_v1_minus_comparator": float(differences.mean()),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "contains_zero": bool(lower <= 0 <= upper),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def cost_sweep(v1_decisions, ground_truth_decisions, k):
    predictions = np.concatenate(v1_decisions)
    ground_truth = np.tile(ground_truth_decisions, len(v1_decisions))
    rows = []
    for false_pick_cost in C_FP_RANGE:
        for wait_cost in C_FW_RANGE:
            rows.append(
                {
                    "k": k,
                    "C_fp": false_pick_cost,
                    "C_fw": wait_cost,
                    "C_rev": C_REV,
                    "mean_cost": compute_decision_cost(
                        predictions, ground_truth, false_pick_cost, wait_cost, C_REV
                    ),
                    "protocol_valid": (false_pick_cost > wait_cost) and (wait_cost >= C_REV),
                }
            )
    return pd.DataFrame(rows)


def render_summary(metrics: pd.Series, ci: pd.DataFrame, turning: pd.Series) -> str:
    def pct(value):
        return f"{value * 100:.1f}%"

    lines = [
        "# V1 Official Evaluation",
        "",
        "This report evaluates the frozen 512D prototype-expectation endpoint with calibration-only thresholds.",
        "",
        "## Pooled K=16 metrics",
        "",
        f"- false-pick rate: {pct(metrics['false_pick_rate'])}",
        f"- pick precision: {pct(metrics['pick_precision'])}",
        f"- pick recall: {pct(metrics['pick_recall'])}",
        f"- revisit burden: {pct(metrics['revisit_burden'])}",
        "",
        "## Bootstrap confidence intervals",
        "",
        *[
            f"- {row['metric']}: {pct(row['mean'])} [{pct(row['ci_lower'])}, {pct(row['ci_upper'])}]"
            for _, row in ci.iterrows()
        ],
        "",
        "## Turning audit",
        "",
        f"- revisit enrichment: {turning['revisit_enrichment']:.4f}",
        f"- enrichment lift: {turning['enrichment_lift']:.4f}",
        f"- defer-rate gap: {turning['defer_rate_gap']:.4f}",
        "",
        "The turning audit is supplementary evidence; the primary contribution is the calibrated risk-coverage decision layer.",
    ]
    return "\n".join(lines)


def build_official_evaluation(
    root: str | Path,
    out_dir: str | Path | None = None,
    k: int = 16,
    n_bootstrap: int = 1000,
    seed: int = 42,
):
    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / "outputs/official_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_frame = load_ground_truth(root)
    ground_truth_decisions = ground_truth_frame["decision_label"].to_numpy()
    decisions, revisit_sources, episode_indices = generate_v1_episode_decisions(root, k)
    predictions = np.concatenate(decisions)
    tiled_ground_truth = np.tile(ground_truth_decisions, len(decisions))
    pooled = compute_pooled_metrics(predictions, tiled_ground_truth)
    metrics = pd.DataFrame([{**{"method": "V1_TAP_512D_endpoint", "k": k}, **pooled}])
    turning = compute_turning_audit(predictions, ground_truth_frame)
    turning_frame = pd.DataFrame([{**{"method": "V1_TAP_512D_endpoint", "k": k}, **turning}])
    confidence_intervals = bootstrap_metric_ci(decisions, ground_truth_decisions, n_bootstrap, seed)
    confidence_intervals.insert(0, "k", k)
    confidence_intervals.insert(0, "method", "V1_TAP_512D_endpoint")
    b5_decisions = load_comparator_decisions(root, root / "outputs/baselines/B5_kshot_hard_reject", k)
    paired_b5 = paired_difference_ci(
        decisions,
        b5_decisions,
        ground_truth_decisions,
        "B5_kshot_hard_reject",
        n_bootstrap,
        seed,
    )
    costs = cost_sweep(decisions, ground_truth_decisions, k)

    artifacts = {
        "decisions_npz": out_dir / f"v1_query_decisions_K{k}.npz",
        "metrics_csv": out_dir / f"v1_pooled_metrics_K{k}.csv",
        "ci_csv": out_dir / f"v1_bootstrap_ci_K{k}.csv",
        "turning_csv": out_dir / f"v1_turning_audit_K{k}.csv",
        "paired_b5_csv": out_dir / f"v1_paired_diff_vs_b5_K{k}.csv",
        "cost_csv": out_dir / f"v1_cost_sweep_K{k}.csv",
        "summary_md": out_dir / "OFFICIAL_EVALUATION_SUMMARY.md",
    }
    np.savez_compressed(
        artifacts["decisions_npz"],
        decisions=np.stack(decisions),
        revisit_src=np.stack(revisit_sources),
        episode_idx=np.asarray(episode_indices, dtype=int),
        gt_decisions=ground_truth_decisions.astype(str),
    )
    metrics.to_csv(artifacts["metrics_csv"], index=False, float_format="%.6f")
    confidence_intervals.to_csv(artifacts["ci_csv"], index=False, float_format="%.6f")
    turning_frame.to_csv(artifacts["turning_csv"], index=False, float_format="%.6f")
    paired_b5.to_csv(artifacts["paired_b5_csv"], index=False, float_format="%.6f")
    costs.to_csv(artifacts["cost_csv"], index=False, float_format="%.6f")
    artifacts["summary_md"].write_text(
        render_summary(metrics.iloc[0], confidence_intervals, turning_frame.iloc[0]),
        encoding="utf-8",
    )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name, path in build_official_evaluation(args.root, args.out, args.k, args.n_bootstrap, args.seed).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
