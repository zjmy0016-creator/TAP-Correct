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

## Laboro Tomato External Check

The Laboro Tomato run is a second-dataset external validation check under the
same frozen endpoint/threshold protocol used for the strawberry V1 result. It
requires the raw Laboro Tomato COCO data under `data/laboro_tomato/raw/`.

Prepare crops and the crop index:

```bash
python scripts/crop_laboro_tomato.py --data_root data/laboro_tomato
python scripts/prepare_laboro_index.py
```

Extract ViT-B/16 features:

```bash
python scripts/extract_features.py --data_dir data/laboro_tomato --out outputs/features_laboro_tomato_vitb16.npz --model ViT-B-16 --pretrained openai --batch_size 64
```

Run the zero-shot collapsed-class diagnosis:

```bash
python scripts/eval_zeroshot_tomato.py --npz outputs/features_laboro_tomato_vitb16.npz
```

Run the frozen-protocol K=16 tomato frontier:

```bash
python scripts/tomato_frontier.py --npz outputs/features_laboro_tomato_vitb16.npz --k 16 --n_episodes 100
```

Expected summary from the current local run:

- 9430 crops total: 1617 mature, 1616 turning, 6197 immature.
- Zero-shot weakest class: mature, F1 = 0.04.
- Under the frozen protocol, the V1 512D-endpoint frontier has lower or equal
  false-pick rate than the B5-family frontier over the common coverage region.
- At B5's operating coverage, V1 false-pick is lower than B5 (52.8% vs 59.7%).
- V1 max coverage is lower than the B5 family (49.8% vs 76.7%), reflecting the
  conservative behavior required by low tomato class separability.

## CLIP/Selective Baselines

Generate the zero-shot, Tip-Adapter, and Proto-Adapter selective baselines:

```bash
python scripts/run_zs_temp_selective_baseline.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/zs_temp_strawberry_vitb32.csv --dataset strawberry --backbone vitb32
python scripts/run_zs_temp_selective_baseline.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/zs_temp_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16
python scripts/run_tip_adapter_selective_baseline.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/tip_adapter_strawberry_vitb32.csv --dataset strawberry --backbone vitb32 --k 16 --n_episodes 20
python scripts/run_tip_adapter_selective_baseline.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/tip_adapter_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16 --k 16 --n_episodes 20
python scripts/run_proto_adapter_selective_baseline.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/proto_adapter_strawberry_vitb32.csv --dataset strawberry --backbone vitb32 --k 16 --n_episodes 20
python scripts/run_proto_adapter_selective_baseline.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/proto_adapter_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16 --k 16 --n_episodes 20
python scripts/summarize_clip_selective_baselines.py --input_dir outputs/clip_selective_baselines --out outputs/clip_selective_baselines/main_table.csv --false_pick_alphas 0.05 0.10
```

Generate the false-pick-controlled and strict unsafe-pick diagnostics:

```bash
python scripts/run_fp_controlled_baseline.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/fp_controlled_strawberry_vitb32.csv --dataset strawberry --backbone vitb32 --k 16 --n_episodes 20
python scripts/run_fp_controlled_baseline.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/fp_controlled_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16 --k 16 --n_episodes 20
python scripts/run_unsafe_pick_diagnostic.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/unsafe_pick_strawberry_vitb32.csv --dataset strawberry --backbone vitb32 --k 16 --n_episodes 20
python scripts/run_unsafe_pick_diagnostic.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/unsafe_pick_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16 --k 16 --n_episodes 20
```

The generated interpretation tables are:

- `outputs/clip_selective_baselines/main_table.csv`
- `outputs/clip_selective_baselines/fp_controlled_main_table.csv`
- `outputs/clip_selective_baselines/unsafe_pick_main_table.csv`
- `outputs/clip_selective_baselines/BASELINE_RESULTS_SUMMARY.md`

## Tests

```bash
python -m unittest discover -v
```

The public tests verify the V1 report and official evaluation contracts.
