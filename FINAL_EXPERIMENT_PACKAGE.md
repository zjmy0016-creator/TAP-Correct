# Final Experiment Package

This file inventories the public experimental package for TAP-Correct. It is
aligned with `README.md`, `REPRODUCIBILITY.md`, and `V1_PROTOCOL.md`.

## Status

The experimental supplement is complete enough for manuscript revision and
reviewer-facing reproduction. The manuscript draft itself is not part of this
package.

## Evidence Lines

1. Frozen strawberry V1 protocol and formal report.
2. Laboro Tomato external validation under the frozen endpoint/threshold
   protocol.
3. CLIP/selective baselines: ZS-temp, TipAdapter, and ProtoAdapter.
4. FP-controlled risk-control baselines.
5. Strict unsafe-pick diagnostic for non-target pick risk.

## Core Artifacts

| Artifact | Purpose |
|---|---|
| `README.md` | Repository overview and main reproduction entry points |
| `REPRODUCIBILITY.md` | Command-level reproduction guide |
| `V1_PROTOCOL.md` | Frozen V1 method, metrics, and claim boundaries |
| `outputs/features_vitb32.npz` | Strawberry ViT-B/32 frozen feature cache |
| `outputs/features_vitb16.npz` | Strawberry ViT-B/16 frozen feature cache |
| `outputs/features_vitl14.npz` | Strawberry ViT-L/14 frozen feature cache |
| `outputs/features_laboro_tomato_vitb16.npz` | Laboro Tomato ViT-B/16 frozen feature cache |
| `outputs/v1_freeze_report/V1_FORMAL_SUMMARY.md` | Formal V1 report |
| `outputs/v1_freeze_report/v1_claim_evidence_map.csv` | Claim-to-evidence map |
| `outputs/probe_512d_endpoint/laboro_tomato_vitb16/frontier_laboro_tomato_vitb16_K16.csv` | Laboro Tomato frozen-protocol frontier |
| `outputs/probe_512d_endpoint/laboro_tomato_vitb16/frontier_laboro_tomato_vitb16_K16.png` | Laboro Tomato frontier figure |
| `outputs/clip_selective_baselines/BASELINE_RESULTS_SUMMARY.md` | CLIP/selective baseline interpretation |
| `outputs/clip_selective_baselines/main_table.csv` | Operating-point baseline table |
| `outputs/clip_selective_baselines/fp_controlled_main_table.csv` | FP-controlled risk-control table |
| `outputs/clip_selective_baselines/unsafe_pick_main_table.csv` | Strict unsafe-pick table |

## Public Scripts

| Script | Purpose |
|---|---|
| `scripts/v1_freeze_report.py` | Regenerate the formal V1 summary artifacts |
| `scripts/v1_official_eval.py` | Regenerate official V1 evaluation artifacts |
| `scripts/xbackbone_step2_frontier.py` | Regenerate strawberry cross-backbone frontiers |
| `scripts/crop_laboro_tomato.py` | Prepare Laboro Tomato crops from raw COCO data |
| `scripts/prepare_laboro_index.py` | Build Laboro Tomato crop index |
| `scripts/eval_zeroshot_tomato.py` | Zero-shot tomato collapsed-class diagnosis |
| `scripts/tomato_frontier.py` | Laboro Tomato frozen-protocol frontier |
| `scripts/clip_selective_baselines.py` | Shared CLIP/selective baseline utilities |
| `scripts/run_zs_temp_selective_baseline.py` | ZS-temp selective baseline |
| `scripts/run_tip_adapter_selective_baseline.py` | TipAdapter selective baseline |
| `scripts/run_proto_adapter_selective_baseline.py` | ProtoAdapter selective baseline |
| `scripts/run_fp_controlled_baseline.py` | FP-controlled baseline |
| `scripts/run_unsafe_pick_diagnostic.py` | Strict unsafe-pick diagnostic |
| `scripts/summarize_clip_selective_baselines.py` | Main CLIP/selective baseline table aggregation |

## CSV Row Counts

| CSV | Rows |
|---|---:|
| `outputs/clip_selective_baselines/main_table.csv` | 12 |
| `outputs/clip_selective_baselines/fp_controlled_main_table.csv` | 12 |
| `outputs/clip_selective_baselines/unsafe_pick_main_table.csv` | 12 |
| `outputs/clip_selective_baselines/zs_temp_strawberry_vitb32.csv` | 19 |
| `outputs/clip_selective_baselines/zs_temp_laboro_tomato_vitb16.csv` | 19 |
| `outputs/clip_selective_baselines/tip_adapter_strawberry_vitb32.csv` | 380 |
| `outputs/clip_selective_baselines/tip_adapter_laboro_tomato_vitb16.csv` | 380 |
| `outputs/clip_selective_baselines/proto_adapter_strawberry_vitb32.csv` | 380 |
| `outputs/clip_selective_baselines/proto_adapter_laboro_tomato_vitb16.csv` | 380 |
| `outputs/clip_selective_baselines/fp_controlled_strawberry_vitb32.csv` | 82 |
| `outputs/clip_selective_baselines/fp_controlled_laboro_tomato_vitb16.csv` | 82 |
| `outputs/clip_selective_baselines/unsafe_pick_strawberry_vitb32.csv` | 82 |
| `outputs/clip_selective_baselines/unsafe_pick_laboro_tomato_vitb16.csv` | 82 |

## Paper-Ready Takeaway

The current evidence supports a conservative risk-aware narrative:

- Strawberry remains the formal frozen V1 result.
- Laboro Tomato provides external validation under the same frozen endpoint and
  threshold protocol, but not a full second formal freeze equivalent to the
  strawberry artifact set.
- On Laboro Tomato, V1 has lower or equal false-pick rate than the B5-family
  frontier over the common coverage region, and lower false-pick at B5's
  operating coverage (52.8% vs 59.7%).
- Laboro Tomato is lower-separability: V1 max coverage is lower than the B5
  family (49.8% vs 76.7%), so the result should be framed as conservative
  risk-control behavior rather than full metric domination.
- CLIP/selective baselines broaden the comparison set, and the FP-controlled
  and strict unsafe-pick diagnostics expose calibration-to-query risk transfer
  and non-target pick risk.

## Claim Boundary

Allowed:

- TAP-Correct V1 is a training-free risk-aware selective harvesting decision
  framework based on frozen CLIP features.
- The 512D endpoint reduces false-pick risk relative to the original H-axis
  endpoint and improves risk-coverage behavior under the frozen protocol.
- Laboro Tomato supports cross-dataset plausibility under a lower-separability
  crop setting.

Avoid:

- Claiming universal cross-crop generalization.
- Claiming V1 dominates every baseline on every metric.
- Claiming calibration guarantees external false-pick control.
- Treating the Laboro Tomato check as a complete second formal freeze.
