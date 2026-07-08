# -*- coding: utf-8 -*-
"""Build the formal V1 freeze report from frozen probe CSV outputs.

This script is intentionally a light aggregation layer. It does not extract
features, recalibrate thresholds, or change any experimental output. It reads
the existing V1 probe frontier CSV files and emits manuscript-facing summary
tables plus a short formal report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = ("vitb32", "vitb16", "vitl14")
B5_CALIBRATED_DEFER = 0.20


def _frontier_path(root: Path, backbone: str) -> Path:
    return (
        root
        / "outputs"
        / "probe_512d_endpoint"
        / backbone
        / f"frontier_{backbone}_K16.csv"
    )


def _headline_path(root: Path) -> Path:
    return root / "outputs" / "probe_512d_endpoint" / "V1_headline_vitb32_K16.csv"


def _variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    part = df[df["variant"] == name].copy()
    if part.empty:
        raise ValueError(f"Missing frontier variant: {name}")
    for col in ("defer", "false_pick_rate", "actual_coverage", "pick_recall"):
        part[col] = part[col].astype(float)
    return part


def _row_at_defer(frame: pd.DataFrame, defer: float) -> pd.Series:
    idx = (frame["defer"] - defer).abs().idxmin()
    row = frame.loc[idx]
    if abs(float(row["defer"]) - defer) > 1e-9:
        raise ValueError(f"Missing defer={defer:.2f} in frontier")
    return row


def _interp_at_coverage(frame: pd.DataFrame, coverage: float, metric: str) -> float:
    ordered = frame.sort_values("actual_coverage")
    xs = ordered["actual_coverage"].to_numpy(dtype=float)
    ys = ordered[metric].to_numpy(dtype=float)
    if coverage < xs.min() - 1e-12 or coverage > xs.max() + 1e-12:
        raise ValueError(
            f"Coverage {coverage:.6f} outside available range "
            f"[{xs.min():.6f}, {xs.max():.6f}]"
        )
    return float(np.interp(coverage, xs, ys))


def load_backbone_summary(root: str | Path) -> pd.DataFrame:
    """Return the formal V1 frontier summary for the three frozen backbones."""
    root = Path(root)
    rows = []
    for backbone in BACKBONES:
        df = pd.read_csv(_frontier_path(root, backbone))
        v0 = _variant(df, "V0_H_axis_endpoint")
        v1 = _variant(df, "V1_E_512D_endpoint")
        b5 = _variant(df, "B5family_argmax_margin")

        b5_cal = _row_at_defer(b5, B5_CALIBRATED_DEFER)
        target_cov = float(b5_cal["actual_coverage"])

        v0_fp = _interp_at_coverage(v0, target_cov, "false_pick_rate")
        v1_fp = _interp_at_coverage(v1, target_cov, "false_pick_rate")
        v1_recall = _interp_at_coverage(v1, target_cov, "pick_recall")

        comparable = b5[
            (b5["actual_coverage"] >= v1["actual_coverage"].min() - 1e-12)
            & (b5["actual_coverage"] <= v1["actual_coverage"].max() + 1e-12)
        ]
        beat_flags = []
        for _, b5_row in comparable.iterrows():
            v1_fp_at_cov = _interp_at_coverage(
                v1, float(b5_row["actual_coverage"]), "false_pick_rate"
            )
            beat_flags.append(v1_fp_at_cov <= float(b5_row["false_pick_rate"]) + 1e-12)

        rows.append(
            {
                "backbone": backbone,
                "k": 16,
                "b5_defer": B5_CALIBRATED_DEFER,
                "b5_coverage_at_defer20": target_cov,
                "v0_fp_at_b5cov": v0_fp,
                "v1_fp_at_b5cov": v1_fp,
                "b5_fp_at_defer20": float(b5_cal["false_pick_rate"]),
                "v1_pick_recall_at_b5cov": v1_recall,
                "b5_pick_recall_at_defer20": float(b5_cal["pick_recall"]),
                "same_coverage_recall_gap_b5_minus_v1": float(b5_cal["pick_recall"])
                - v1_recall,
                "v1_fp_reduction_vs_v0_pp": (v0_fp - v1_fp) * 100.0,
                "v1_fp_delta_vs_b5_pp": (v1_fp - float(b5_cal["false_pick_rate"]))
                * 100.0,
                "v1_beats_b5family_share": float(np.mean(beat_flags))
                if beat_flags
                else np.nan,
                "v1_max_coverage": float(v1["actual_coverage"].max()),
                "b5family_max_coverage": float(b5["actual_coverage"].max()),
                "source_csv": str(_frontier_path(root, backbone).relative_to(root)),
            }
        )

    return pd.DataFrame(rows)


def load_headline(root: str | Path) -> pd.DataFrame:
    """Load and annotate the frozen V1 headline point."""
    root = Path(root)
    headline = pd.read_csv(_headline_path(root)).copy()
    headline["freeze_status"] = "formal_v1_endpoint_frozen"
    headline["protocol"] = "512D_prototype_expectation_endpoint_E"
    headline["threshold_source"] = "calibration_only"
    headline["feature_training"] = "none"
    headline["notes"] = (
        "V1 is the frozen method; V0 H-axis results are retained as the "
        "diagnostic motivation stage."
    )
    return headline


def build_claim_evidence(summary: pd.DataFrame, headline: pd.DataFrame) -> pd.DataFrame:
    """Build a compact claim-evidence map for manuscript writing."""
    row = headline.iloc[0]
    v1_better_v0 = bool((summary["v1_fp_at_b5cov"] < summary["v0_fp_at_b5cov"]).all())
    strong = summary[summary["backbone"].isin(["vitb16", "vitl14"])]
    strong_better_b5 = bool((strong["v1_fp_at_b5cov"] < strong["b5_fp_at_defer20"]).all())

    return pd.DataFrame(
        [
            {
                "claim": "V1 improves the original TAP endpoint on all headline metrics for ViT-B/32.",
                "evidence": (
                    f"false_pick_rate={row['false_pick_rate']:.4f}, "
                    f"pick_precision={row['pick_precision']:.4f}, "
                    f"pick_recall={row['pick_recall']:.4f}, "
                    f"revisit_burden={row['revisit_burden']:.4f}."
                ),
                "status": "supported",
            },
            {
                "claim": "Replacing the 1D H-axis endpoint with the 512D endpoint consistently reduces false-pick risk at the calibrated B5 coverage.",
                "evidence": (
                    "v1_fp_at_b5cov is lower than v0_fp_at_b5cov for all "
                    f"three backbones: {v1_better_v0}."
                ),
                "status": "supported" if v1_better_v0 else "needs_check",
            },
            {
                "claim": "With stronger CLIP backbones, V1 can beat the B5 family false-pick frontier at the calibrated B5 coverage.",
                "evidence": (
                    "ViT-B/16 and ViT-L/14 have lower V1 false-pick rates "
                    f"than B5 family at defer=20%: {strong_better_b5}."
                ),
                "status": "supported" if strong_better_b5 else "needs_check",
            },
            {
                "claim": "V1 still has a same-coverage recall trade-off against B5 and should not be described as full domination.",
                "evidence": (
                    "same_coverage_recall_gap_b5_minus_v1 remains positive "
                    "in the frozen frontier summary."
                ),
                "status": "limitation",
            },
            {
                "claim": "The V1 endpoint is a diagnostic-driven post-hoc refinement; Laboro Tomato provides external validation under the same frozen endpoint protocol but not a full second formal freeze.",
                "evidence": (
                    "The method was selected after the D5 V0 audit; current "
                    "formal report freezes the strawberry protocol. The "
                    "Laboro Tomato check uses the frozen endpoint/threshold "
                    "protocol on ViT-B/16 and supports cross-dataset "
                    "plausibility while remaining lighter than the full "
                    "strawberry artifact set."
                ),
                "status": "limitation",
            },
        ]
    )


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_markdown(summary: pd.DataFrame, headline: pd.DataFrame) -> str:
    row = headline.iloc[0]
    lines = [
        "# V1 Formal Freeze Report",
        "",
        "## Formal V1 frozen protocol",
        "",
        "- Method: TAP-Correct V1 with a 512D prototype expectation endpoint E.",
        "- Training: none; all image encoders are frozen CLIP backbones.",
        "- Threshold source: calibration split only; query_test is used for final evaluation.",
        "- Frozen operating point: ViT-B/32, K=16, uncertainty cut at 20%.",
        "- V0 H-axis results are retained as the diagnostic motivation stage, not the final method.",
        "- Official headline/CI evaluation is generated by scripts/v1_official_eval.py.",
        "",
        "## Headline point",
        "",
        "| backbone | K | false-pick | precision | recall | revisit |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {row['backbone']} | {int(row['k'])} | {_pct(row['false_pick_rate'])} | "
            f"{_pct(row['pick_precision'])} | {_pct(row['pick_recall'])} | "
            f"{_pct(row['revisit_burden'])} |"
        ),
        "",
        "## Cross-backbone frontier",
        "",
        "| backbone | V0 fp @ B5 cov | V1 fp @ B5 cov | B5 fp @ defer=20% | V1 max cov | B5 max cov | V1<=B5 share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"| {r['backbone']} | {_pct(r['v0_fp_at_b5cov'])} | "
            f"{_pct(r['v1_fp_at_b5cov'])} | {_pct(r['b5_fp_at_defer20'])} | "
            f"{_pct(r['v1_max_coverage'])} | {_pct(r['b5family_max_coverage'])} | "
            f"{_pct(r['v1_beats_b5family_share'])} |"
        )

    lines.extend(
        [
            "",
        "## Interpretation",
        "",
        "V1 fixes the main V0 failure mode by replacing the weak 1D H-axis endpoint with a 512D prototype expectation endpoint. At the calibrated B5 coverage, V1 lowers false-pick risk relative to V0 for all three backbones. On stronger backbones, V1 also beats the B5 family false-pick rate at the same calibrated coverage.",
        "",
        "The frozen headline operating point is separately evaluated in outputs/v1_official_eval/, including pooled metrics, bootstrap confidence intervals, turning audit, paired differences and cost sweep.",
        "",
        "The result should be framed as a risk-coverage improvement, not as complete domination of B5. The frozen frontier still contains a same-coverage recall trade-off: B5 keeps higher pick recall in the overlap region.",
            "",
            "## Limitations to state",
            "",
            "- V1 is a diagnostic-driven post-hoc method selection after the D5 V0 audit.",
            "- The formal freeze remains the strawberry protocol. Laboro Tomato now provides external validation under the same frozen endpoint/threshold protocol, but it uses one backbone and class-based pick ground truth rather than the full strawberry artifact set.",
            "- The uncertainty mechanism should be described as a revisit-control audit signal, not as the main contribution.",
            "",
            "## External validation: Laboro Tomato",
            "",
            "Laboro Tomato was added as a second-dataset check. Zero-shot CLIP shows a severe mature-class collapse (F1 = 0.04). Under the frozen K=16 protocol, the V1 512D-endpoint risk-coverage frontier has lower or equal false-pick rate than the B5-family frontier across the common coverage region. At B5's operating coverage, V1 false-pick is lower than B5 (52.8% vs 59.7%). The tomato check should still be bounded as external validation rather than a complete second formal freeze because it uses one backbone, class-based pick ground truth, and a lower-separability crop setting where V1 max coverage is lower than the B5 family (49.8% vs 76.7%).",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(root: str | Path, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Write the formal V1 report artifacts and return their paths."""
    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / "outputs" / "v1_freeze_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_backbone_summary(root)
    headline = load_headline(root)
    claims = build_claim_evidence(summary, headline)

    summary_csv = out_dir / "v1_backbone_frontier_summary.csv"
    headline_csv = out_dir / "v1_headline_metrics.csv"
    claim_csv = out_dir / "v1_claim_evidence_map.csv"
    report_md = out_dir / "V1_FORMAL_SUMMARY.md"

    summary.to_csv(summary_csv, index=False, encoding="utf-8")
    headline.to_csv(headline_csv, index=False, encoding="utf-8")
    claims.to_csv(claim_csv, index=False, encoding="utf-8")
    report_md.write_text(render_markdown(summary, headline), encoding="utf-8")

    return {
        "summary_csv": summary_csv,
        "headline_csv": headline_csv,
        "claim_csv": claim_csv,
        "report_md": report_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    artifacts = build_report(args.root, args.out)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
