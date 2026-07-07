# Reproducibility

This document lists the public commands for reproducing the frozen TAP-Correct
V1 artifacts.

## Environment

Recommended Python: 3.10 or newer.

```bash
pip install -r requirements.txt
```

The included frozen feature caches are sufficient for the formal V1 report and
evaluation. Regenerating feature caches from raw crop images additionally needs
a local PyTorch/OpenCLIP setup and local access to the crop image dataset.

## Frozen Inputs

The public V1 reproduction path reads:

- `outputs/features_vitb32.npz`
- `outputs/features_vitb16.npz`
- `outputs/features_vitl14.npz`
- `outputs/episodes/manifest_K1-16_ep100.json`
- `outputs/decision_gold/turning_decision_dataset/labels/test_decision_ground_truth_clean.csv`
- `outputs/d2_calibrate/thresholds_all_episodes.csv`
- `outputs/d3_evaluate/query_decisions_K16_ep4*.npz`
- `outputs/d4_baselines/B5_kshot_hard_reject/query_decisions_K16_ep4*.npz`
- `outputs/probe_512d_endpoint/V1_headline_vitb32_K16.csv`
- `outputs/probe_512d_endpoint/<backbone>/frontier_<backbone>_K16.csv`

The frozen method and claim boundaries are defined in `V1_PROTOCOL.md`.

## Formal V1 Report

```bash
python scripts/v1_freeze_report.py
```

Expected outputs:

- `outputs/v1_freeze_report/V1_FORMAL_SUMMARY.md`
- `outputs/v1_freeze_report/v1_headline_metrics.csv`
- `outputs/v1_freeze_report/v1_backbone_frontier_summary.csv`
- `outputs/v1_freeze_report/v1_claim_evidence_map.csv`

## Official V1 Evaluation

```bash
python scripts/v1_official_eval.py
```

Expected outputs:

- `outputs/v1_official_eval/v1_query_decisions_K16.npz`
- `outputs/v1_official_eval/v1_pooled_metrics_K16.csv`
- `outputs/v1_official_eval/v1_bootstrap_ci_K16.csv`
- `outputs/v1_official_eval/v1_turning_audit_K16.csv`
- `outputs/v1_official_eval/v1_paired_diff_vs_v0_K16.csv`
- `outputs/v1_official_eval/v1_paired_diff_vs_b5_K16.csv`
- `outputs/v1_official_eval/v1_cost_sweep_K16.csv`
- `outputs/v1_official_eval/V1_OFFICIAL_EVAL_SUMMARY.md`

## Cross-Backbone Frontiers

```bash
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb32.npz --k 16
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb16.npz --k 16
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitl14.npz --k 16
```

These commands recompute the V0, V1, and B5-family risk-coverage frontiers from
the frozen feature caches. Thresholds are recalibrated on calibration data only.

## Feature Caches

Feature extraction is optional for the public reproduction path and requires raw
crop images to be available locally.

```bash
python scripts/extract_features.py --model ViT-B-32 --pretrained openai --out outputs/features_vitb32.npz
python scripts/extract_features.py --model ViT-B-16 --pretrained openai --out outputs/features_vitb16.npz
python scripts/extract_features.py --model ViT-L-14 --pretrained openai --out outputs/features_vitl14.npz
```

## Tests

```bash
python -m unittest discover -v
```

The public tests verify the V1 report and official evaluation contracts.
