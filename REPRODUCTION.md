# Reproduction Guide

This guide describes the public inputs, execution order, and generated
artifacts for the TAP-Correct V1 release, including both external-validation
datasets.

## Environment

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`
- A CPU or CUDA PyTorch installation for feature extraction
- OpenCLIP for regenerating feature caches

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Public inputs

The formal commands use the following versioned inputs:

| Input | Purpose |
|---|---|
| `outputs/features_vitb32.npz` | Strawberry ViT-B/32 features |
| `outputs/features_vitb16.npz` | Strawberry ViT-B/16 features |
| `outputs/features_vitl14.npz` | Strawberry ViT-L/14 features |
| `outputs/features_laboro_tomato_vitb16.npz` | Laboro Tomato features |
| `outputs/features_strawberryds_vitb32.npz` | Strawberry-DS features for transfer and reference evaluation |
| `outputs/episodes/manifest_K1-16_ep100.json` | Deterministic support/calibration episodes |
| `outputs/decision_gold/turning_decision_dataset/labels/test_decision_ground_truth_clean.csv` | Query decision ground truth |
| `outputs/calibration/thresholds_all_episodes.csv` | Calibration thresholds |
| `outputs/query_evaluation/` | Frozen query score inputs |
| `outputs/baselines/B5_kshot_hard_reject/` | B5 comparator decisions |

## Formal release sequence

Generate the cross-backbone frontiers:

```bash
python scripts/backbone_frontier.py --npz outputs/features_vitb32.npz --k 16
python scripts/backbone_frontier.py --npz outputs/features_vitb16.npz --k 16
python scripts/backbone_frontier.py --npz outputs/features_vitl14.npz --k 16
```

Generate the Strawberry-DS external-validation summaries:

```bash
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode maincal
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode recalib
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode indomain
```

`maincal` uses main-domain support and calibration for direct transfer to
Strawberry-DS. `recalib` keeps main-domain support and recalibrates on the
external domain. `indomain` constructs support and calibration from
Strawberry-DS for an in-domain reference. Each command writes a per-episode
CSV and a corresponding summary CSV; only the three summary CSVs are included
in the public output allowlist.

Generate the official evaluation:

```bash
python scripts/official_evaluation.py --k 16
```

Generate the formal report:

```bash
python scripts/build_freeze_report.py
```

Run the test suite:

```bash
python -m unittest discover -v
```

## Generated artifacts

| Directory | Contents |
|---|---|
| `outputs/probe_512d_endpoint/` | Backbone frontiers and recall frontiers |
| `outputs/official_evaluation/` | Pooled metrics, confidence intervals, audits, B5 differences, and cost sensitivity |
| `outputs/final_report/` | Release summary, headline metrics, and claim evidence |
| `outputs/strawberryds_eval/` | Strawberry-DS summary metrics for all three evaluation modes |

## External validation utilities

The following commands require their corresponding feature cache or source
dataset and are not prerequisites for the formal release:

```bash
python scripts/laboro_tomato_frontier.py --npz outputs/features_laboro_tomato_vitb16.npz --k 16 --n_episodes 100
python scripts/tomato_frontier_tip.py --npz outputs/features_laboro_tomato_vitb16.npz --k 16 --n_episodes 100
python scripts/endpoint_source_ablation.py --k 16
python scripts/uncertainty_evidence.py --source all --k 16
python scripts/linear_probe_reference.py
python scripts/calibration_budget_sweep.py --k 16
python scripts/unified_baseline_reeval.py
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode maincal
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode recalib
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode indomain
```

## Reproduction controls

- Do not place raw datasets or local exports under the public output paths.
- Keep query labels separate from threshold selection.
- Preserve the manifest and random seeds when comparing runs.
- Record any regenerated result in `CHANGELOG.md` with the command and output
  path.
