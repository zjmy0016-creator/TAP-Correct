# TAP-Correct

TAP-Correct is a training-free selective harvesting decision framework built
on frozen CLIP features. It converts transitional maturity recognition into a
calibrated `pick` / `wait` / `revisit` decision policy.

The formal V1 release uses a 512-dimensional prototype-expectation endpoint:

```text
E = P(ripe) * 1.0 + P(turning) * 0.5 + P(unripe) * 0.0
```

Thresholds are selected on the calibration split only. The query split is
reserved for final evaluation, and every episode is checked for support,
calibration, and query separation.

## Release summary

At the frozen Strawberry ViT-B/32, K=16 operating point, the release reports:

| Metric | Value |
|---|---:|
| False-pick rate | 9.1% |
| Pick precision | 90.9% |
| Pick recall | 80.6% |
| Revisit burden | 19.2% |

The release also contains cross-backbone frontiers, official bootstrap
evaluation, a B5 comparison, and bounded external checks on Laboro Tomato and
Strawberry-DS.

## External validation coverage

The public delivery records both external datasets separately from the
headline Strawberry V1 evaluation:

| Dataset | Public feature cache | Evaluation entry point | Released output | Scope |
|---|---|---|---|---|
| Laboro Tomato | `outputs/features_laboro_tomato_vitb16.npz` | `scripts/laboro_tomato_frontier.py` | `outputs/probe_512d_endpoint/laboro_tomato_vitb16/frontier_laboro_tomato_vitb16_K16.csv` | 9,430 tomato crops with mature, turning, and immature labels; one-backbone frontier check |
| Strawberry-DS | `outputs/features_strawberryds_vitb32.npz` | `scripts/strawberryds_eval.py` | `outputs/strawberryds_eval/*_summary.csv` | 1,083 strawberry crops with 225 ripe, 132 turning, and 726 unripe labels; direct-transfer, recalibration, and in-domain reference modes |

The Strawberry-DS summaries use `K=16` and 20 deterministic evaluation
episodes. Values below are episode averages for the reported method and
reference baselines.

| Mode | Method | False-pick rate | Pick precision | Pick recall | Revisit burden |
|---|---|---:|---:|---:|---:|
| `maincal` | Axis-Endpoint | 17.3% | 82.7% | 66.5% | 12.9% |
| `maincal` | FS-Proto-Reject | 9.0% | 91.0% | 16.2% | 38.1% |
| `maincal` | TAP-E(tip) | 10.8% | 89.2% | 75.7% | 24.9% |
| `recalib` | Axis-Endpoint | 17.7% | 82.3% | 77.9% | 16.6% |
| `recalib` | FS-Proto-Reject | 7.9% | 92.1% | 17.3% | 35.9% |
| `recalib` | TAP-E(tip) | 16.9% | 83.1% | 75.0% | 15.3% |
| `indomain` | Axis-Endpoint | 16.4% | 83.6% | 80.4% | 14.6% |
| `indomain` | FS-Proto-Reject | 8.4% | 91.6% | 15.8% | 37.9% |
| `indomain` | TAP-E(tip) | 12.9% | 87.1% | 80.4% | 23.9% |

Here, `maincal` is the direct-transfer setting, `recalib` measures the
sensitivity to external-domain calibration, and `indomain` is an in-domain
reference. The complete summary CSVs are versioned under
`outputs/strawberryds_eval/`; per-episode CSVs and raw images are not part of
the public delivery.

## Repository layout

```text
tapcorrect/                         Core episode, prototype, decision, and metric logic
scripts/                            Reproduction and evaluation entry points
tests/                              Contract and reproduction tests
outputs/features_*.npz              Frozen CLIP feature caches
outputs/features_strawberryds_vitb32.npz Strawberry-DS external-validation features
outputs/episodes/                   Frozen episode manifest
outputs/decision_gold/              Decision ground truth
outputs/calibration/                Calibration thresholds
outputs/query_evaluation/           Frozen query score inputs
outputs/baselines/                  B5 comparator decisions
outputs/probe_512d_endpoint/       Cross-backbone frontier artifacts
outputs/final_report/               Formal release summary artifacts
outputs/official_evaluation/        Official metrics, intervals, and audits
outputs/strawberryds_eval/          Strawberry-DS summary metrics by evaluation mode
PROTOCOL.md                         Frozen method and claim boundary
REPRODUCTION.md                     Reproduction commands and outputs
DELIVERY.md                         Public release inventory
CHANGELOG.md                        Formal release trace
```

The public tree excludes raw crop images, source datasets, unpublished
research documents, local exports, and temporary files.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

The feature caches in `outputs/` are sufficient for the formal evaluation
commands. Feature extraction requires a working PyTorch and OpenCLIP setup.

## Reproduction

Regenerate the formal release report:

```bash
python scripts/build_freeze_report.py
```

Regenerate official V1 metrics, bootstrap intervals, audits, B5 paired
differences, and cost sensitivity:

```bash
python scripts/official_evaluation.py
```

Regenerate the frozen cross-backbone frontiers:

```bash
python scripts/backbone_frontier.py --npz outputs/features_vitb32.npz --k 16
python scripts/backbone_frontier.py --npz outputs/features_vitb16.npz --k 16
python scripts/backbone_frontier.py --npz outputs/features_vitl14.npz --k 16
```

Evaluate the public Strawberry-DS transfer and reference modes:

```bash
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode maincal
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode recalib
python scripts/strawberryds_eval.py --npz outputs/features_strawberryds_vitb32.npz --mode indomain
```

Run the test suite:

```bash
python -m unittest discover -v
```

See [REPRODUCTION.md](REPRODUCTION.md) for the complete asset map, optional
external validation commands, and expected output locations.

## Public evaluation utilities

The following entry points are included for reproducible comparison and
external validation:

- `scripts/run_zs_temp_selective_baseline.py`
- `scripts/run_tip_adapter_selective_baseline.py`
- `scripts/run_proto_adapter_selective_baseline.py`
- `scripts/run_fp_controlled_baseline.py`
- `scripts/run_unsafe_pick_diagnostic.py`
- `scripts/summarize_clip_selective_baselines.py`
- `scripts/endpoint_source_ablation.py`
- `scripts/uncertainty_evidence.py`
- `scripts/linear_probe_reference.py`
- `scripts/calibration_budget_sweep.py`
- `scripts/strawberryds_eval.py`
- `scripts/laboro_tomato_frontier.py`
- `scripts/tomato_frontier_tip.py`

Formal evaluation utilities use the release decision ground truth and
calibration-only discipline. External-validation utilities use the
dataset-specific ground truth and the calibration rules documented for each
evaluation mode.

The Laboro Tomato utility produces the external frontier artifact. The
Strawberry-DS utility produces mode-specific summary CSVs for direct transfer,
external recalibration sensitivity, and in-domain reference evaluation.

## Claim boundary

The release supports the following statements:

- TAP-Correct V1 is a training-free, risk-aware selective harvesting framework.
- The 512-dimensional prototype-expectation endpoint supports a calibrated
  risk-coverage trade-off under the frozen protocol.
- The Laboro Tomato frontier provides a bounded external validation check using
  one backbone and class-based ground truth.
- The Strawberry-DS summaries provide a separate cross-domain transfer check,
  recalibration sensitivity check, and in-domain reference under the stated
  evaluation modes.

The release does not support universal cross-crop generalization, universal
dominance over every comparator, or calibration guarantees on unseen domains.
The revisit signal is a decision-control mechanism and should not be presented
as a standalone uncertainty model.

## Citation

If you use this repository, cite the repository URL and identify the formal
protocol version in [PROTOCOL.md](PROTOCOL.md).
