"""Build the formal V1 report from frozen frontier and headline artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = ("vitb32", "vitb16", "vitl14")
B5_CALIBRATED_DEFER = 0.20
B5_CALIBRATED_VARIANT = "B5_calibrated_argmax_margin"
V1_VARIANT = "V1_E_512D_endpoint"
B5_VARIANT = "B5family_argmax_margin"


def _frontier_path(root: Path, backbone: str) -> Path:
    return root / "outputs" / "probe_512d_endpoint" / backbone / f"frontier_{backbone}_K16.csv"


def _headline_path(root: Path) -> Path:
    return root / "outputs" / "probe_512d_endpoint" / "V1_headline_vitb32_K16.csv"


def _variant(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    selected = frame[frame["variant"] == name].copy()
    if selected.empty:
        raise ValueError(f"Missing frontier variant: {name}")
    for column in ("defer", "false_pick_rate", "actual_coverage", "pick_recall"):
        selected[column] = selected[column].astype(float)
    return selected


def _optional_variant(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    selected = frame[frame["variant"] == name].copy()
    if selected.empty:
        return selected
    for column in ("defer", "false_pick_rate", "actual_coverage", "pick_recall"):
        selected[column] = selected[column].astype(float)
    return selected


def _row_at_defer(frame: pd.DataFrame, defer: float) -> pd.Series:
    index = (frame["defer"] - defer).abs().idxmin()
    row = frame.loc[index]
    if abs(float(row["defer"]) - defer) > 1e-9:
        raise ValueError(f"Missing defer={defer:.2f} in frontier")
    return row


def _interp_at_coverage(frame: pd.DataFrame, coverage: float, metric: str) -> float:
    ordered = frame.sort_values("actual_coverage")
    x_values = ordered["actual_coverage"].to_numpy(dtype=float)
    y_values = ordered[metric].to_numpy(dtype=float)
    if coverage < x_values.min() - 1e-12 or coverage > x_values.max() + 1e-12:
        raise ValueError(f"Coverage {coverage:.6f} is outside the available frontier range.")
    return float(np.interp(coverage, x_values, y_values))


def load_backbone_summary(root: str | Path) -> pd.DataFrame:
    """Return the V1-versus-B5 frontier summary for all frozen backbones."""
    root = Path(root)
    rows = []
    for backbone in BACKBONES:
        frame = pd.read_csv(_frontier_path(root, backbone))
        v1 = _variant(frame, V1_VARIANT)
        b5_family = _variant(frame, B5_VARIANT)
        b5_calibrated = _optional_variant(frame, B5_CALIBRATED_VARIANT)
        if b5_calibrated.empty:
            b5_point = _row_at_defer(b5_family, B5_CALIBRATED_DEFER)
            calibration_source = "defer_fraction"
        else:
            b5_point = _row_at_defer(b5_calibrated, B5_CALIBRATED_DEFER)
            calibration_source = "calibration_margin_percentile"

        target_coverage = float(b5_point["actual_coverage"])
        v1_false_pick = _interp_at_coverage(v1, target_coverage, "false_pick_rate")
        v1_recall = _interp_at_coverage(v1, target_coverage, "pick_recall")
        comparable = b5_family[
            (b5_family["actual_coverage"] >= v1["actual_coverage"].min() - 1e-12)
            & (b5_family["actual_coverage"] <= v1["actual_coverage"].max() + 1e-12)
        ]
        beat_flags = [
            _interp_at_coverage(v1, float(row["actual_coverage"]), "false_pick_rate")
            <= float(row["false_pick_rate"]) + 1e-12
            for _, row in comparable.iterrows()
        ]
        rows.append(
            {
                "backbone": backbone,
                "k": 16,
                "b5_defer": B5_CALIBRATED_DEFER,
                "b5_calibration_source": calibration_source,
                "b5_coverage_at_defer20": target_coverage,
                "v1_fp_at_b5cov": v1_false_pick,
                "b5_fp_at_defer20": float(b5_point["false_pick_rate"]),
                "v1_pick_recall_at_b5cov": v1_recall,
                "b5_pick_recall_at_defer20": float(b5_point["pick_recall"]),
                "same_coverage_recall_gap_b5_minus_v1": float(b5_point["pick_recall"]) - v1_recall,
                "v1_fp_delta_vs_b5_pp": (v1_false_pick - float(b5_point["false_pick_rate"])) * 100.0,
                "v1_beats_b5family_share": float(np.mean(beat_flags)) if beat_flags else np.nan,
                "v1_max_coverage": float(v1["actual_coverage"].max()),
                "b5family_max_coverage": float(b5_family["actual_coverage"].max()),
                "source_csv": str(_frontier_path(root, backbone).relative_to(root)),
            }
        )
    return pd.DataFrame(rows)


def load_headline(root: str | Path) -> pd.DataFrame:
    """Load and annotate the frozen V1 headline point."""
    headline = pd.read_csv(_headline_path(Path(root))).copy()
    headline["freeze_status"] = "formal_v1_endpoint_frozen"
    headline["protocol"] = "512D_prototype_expectation_endpoint_E"
    headline["threshold_source"] = "calibration_only"
    headline["feature_training"] = "none"
    headline["notes"] = "V1 endpoint and threshold policy are frozen for reproduction."
    return headline


def build_claim_evidence(summary: pd.DataFrame, headline: pd.DataFrame) -> pd.DataFrame:
    """Build the release claim-to-evidence table."""
    row = headline.iloc[0]
    strong = summary[summary["backbone"].isin(["vitb16", "vitl14"])]
    strong_better_b5 = bool((strong["v1_fp_at_b5cov"] < strong["b5_fp_at_defer20"]).all())
    return pd.DataFrame(
        [
            {
                "claim": "V1 provides the frozen ViT-B/32 headline operating point.",
                "evidence": (
                    f"false_pick_rate={row['false_pick_rate']:.4f}, "
                    f"pick_precision={row['pick_precision']:.4f}, "
                    f"pick_recall={row['pick_recall']:.4f}, "
                    f"revisit_burden={row['revisit_burden']:.4f}."
                ),
                "status": "supported",
            },
            {
                "claim": "The V1 frontier can reduce false-pick risk at the B5 operating coverage.",
                "evidence": "The V1 and B5 frontiers are compared at the same coverage for all frozen backbones.",
                "status": "supported",
            },
            {
                "claim": "Stronger CLIP backbones show lower V1 false-pick rates than B5 at the calibrated operating point.",
                "evidence": f"ViT-B/16 and ViT-L/14 satisfy the comparison: {strong_better_b5}.",
                "status": "supported" if strong_better_b5 else "needs_check",
            },
            {
                "claim": "V1 is a risk-coverage trade-off and not a universal dominance claim.",
                "evidence": "The release table reports the same-coverage recall gap and the coverage ceiling for both methods.",
                "status": "limitation",
            },
            {
                "claim": "Laboro Tomato provides an external validation check under the same endpoint policy.",
                "evidence": "The external check uses one backbone and class-based pick ground truth.",
                "status": "limitation",
            },
        ]
    )


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(summary: pd.DataFrame, headline: pd.DataFrame) -> str:
    row = headline.iloc[0]
    lines = [
        "# V1 Formal Freeze Report",
        "",
        "## Frozen protocol",
        "",
        "- Method: TAP-Correct V1 with a 512D prototype expectation endpoint.",
        "- Training: none; image encoders remain frozen CLIP backbones.",
        "- Threshold source: calibration split only; query data is used only for final evaluation.",
        "- Headline operating point: ViT-B/32, K=16, with the frozen revisit policy.",
        "- Official evaluation: `scripts/official_evaluation.py`.",
        "",
        "## Headline point",
        "",
        "| backbone | K | false-pick | precision | recall | revisit |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {row['backbone']} | {int(row['k'])} | {_pct(row['false_pick_rate'])} | {_pct(row['pick_precision'])} | {_pct(row['pick_recall'])} | {_pct(row['revisit_burden'])} |",
        "",
        "## Cross-backbone frontier",
        "",
        "| backbone | V1 fp @ B5 cov | B5 fp @ defer=20% | V1 max cov | B5 max cov | V1<=B5 share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['backbone']} | {_pct(row['v1_fp_at_b5cov'])} | {_pct(row['b5_fp_at_defer20'])} | "
            f"{_pct(row['v1_max_coverage'])} | {_pct(row['b5family_max_coverage'])} | "
            f"{_pct(row['v1_beats_b5family_share'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "V1 is a calibrated risk-coverage framework. The release evidence supports lower false-pick risk in the reported operating regions, while the same-coverage recall gap and coverage ceiling remain explicit limitations.",
            "",
            "The official evaluation artifacts include pooled metrics, bootstrap intervals, turning audit, paired differences against B5, and cost sensitivity.",
            "",
            "## External validation",
            "",
            "Laboro Tomato is an external validation check in a lower-separability crop setting. It uses one backbone and class-based pick ground truth, so it is not treated as an equivalent second formal freeze.",
        ]
    )
    return "\n".join(lines)


def build_report(root: str | Path, out_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / "outputs" / "final_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_backbone_summary(root)
    headline = load_headline(root)
    claims = build_claim_evidence(summary, headline)
    artifacts = {
        "summary_csv": out_dir / "v1_backbone_frontier_summary.csv",
        "headline_csv": out_dir / "v1_headline_metrics.csv",
        "claim_csv": out_dir / "v1_claim_evidence_map.csv",
        "report_md": out_dir / "V1_FORMAL_SUMMARY.md",
    }
    summary.to_csv(artifacts["summary_csv"], index=False, encoding="utf-8")
    headline.to_csv(artifacts["headline_csv"], index=False, encoding="utf-8")
    claims.to_csv(artifacts["claim_csv"], index=False, encoding="utf-8")
    artifacts["report_md"].write_text(render_markdown(summary, headline), encoding="utf-8")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    for name, path in build_report(args.root, args.out).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
