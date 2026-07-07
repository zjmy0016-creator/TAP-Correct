# TAP-Correct

TAP-Correct is a training-free selective harvesting decision framework built on
frozen CLIP features. It converts the ambiguity of transitional strawberry
maturity into a calibrated `pick / wait / revisit` decision signal.

The frozen V1 method uses a 512D prototype-expectation endpoint rather than the
earlier 1D text-axis endpoint. The repository contains the code and frozen
artifacts needed to reproduce the V1 summary, official evaluation, and
cross-backbone risk-coverage frontiers.

## Highlights

- Training-free protocol: frozen CLIP features, few-shot prototypes, no
  gradient updates.
- Decision endpoint: prototype expectation over `ripe`, `turning`, and
  `unripe` class prototypes.
- Calibrated operating point: thresholds are selected on calibration data only.
- Formal outputs: headline metrics, bootstrap confidence intervals, paired
  differences, cost sweep, and risk-coverage frontiers.
- Claim control: V1 reduces false-pick risk relative to the original endpoint;
  it should not be described as dominating every baseline metric.

## Repository Layout

```text
tapcorrect/                    Core package: episodes, contracts, decisions
scripts/                       Public V1 reproduction scripts and helpers
tests/                         V1 contract tests
outputs/features_*.npz         Frozen CLIP feature caches
outputs/episodes/              Frozen episode manifest
outputs/decision_gold/         Public decision-label CSV used for evaluation
outputs/probe_512d_endpoint/   Cross-backbone frontier evidence
outputs/v1_freeze_report/      Formal V1 summary artifacts
outputs/v1_official_eval/      Official V1 evaluation artifacts
V1_PROTOCOL.md                 Frozen protocol and allowed claims
REPRODUCIBILITY.md             Reproduction commands
```

Raw crop images, manuscript drafts, planning notes, and local experiment logs
are not part of the public repository.

## Install

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

Feature extraction additionally requires a working PyTorch and OpenCLIP setup
for the local CPU/CUDA environment. The included frozen feature caches are
enough for the formal V1 reproduction commands below.

## Reproduce V1

Generate the formal V1 report:

```bash
python scripts/v1_freeze_report.py
```

Generate official V1 evaluation artifacts:

```bash
python scripts/v1_official_eval.py
```

Regenerate the cross-backbone frontiers from frozen feature caches:

```bash
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb32.npz --k 16
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb16.npz --k 16
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitl14.npz --k 16
```

Run the public tests:

```bash
python -m unittest discover -v
```

## Frozen V1 Result

At the frozen ViT-B/32, K=16 operating point, V1 reports:

| metric | value |
|---|---:|
| false-pick rate | 9.1% |
| pick precision | 90.9% |
| pick recall | 80.6% |
| revisit burden | 19.2% |

The cross-backbone frontier summary is written to
`outputs/v1_freeze_report/v1_backbone_frontier_summary.csv`. The claim-to-
evidence map is written to
`outputs/v1_freeze_report/v1_claim_evidence_map.csv`.

## Limitations

- The current public freeze reports one dataset family and should be validated
  on an independent held-out dataset before making strong external-validity
  claims.
- V1 improves false-pick risk and coverage behavior, but it keeps a
  same-coverage recall trade-off against the B5 baseline family.
- The `revisit` signal is a decision-control mechanism, not a standalone
  calibrated uncertainty model.

## Citation

No paper citation is available yet. If you use this repository, cite the
repository URL and the frozen protocol version in `V1_PROTOCOL.md`.
