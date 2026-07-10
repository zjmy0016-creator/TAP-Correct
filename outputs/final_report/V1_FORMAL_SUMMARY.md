# V1 Formal Freeze Report

## Frozen protocol

- Method: TAP-Correct V1 with a 512D prototype expectation endpoint.
- Training: none; image encoders remain frozen CLIP backbones.
- Threshold source: calibration split only; query data is used only for final evaluation.
- Headline operating point: ViT-B/32, K=16, with the frozen revisit policy.
- Official evaluation: `scripts/official_evaluation.py`.

## Headline point

| backbone | K | false-pick | precision | recall | revisit |
|---|---:|---:|---:|---:|---:|
| vitb32 | 16 | 9.1% | 90.9% | 80.6% | 19.2% |

## Cross-backbone frontier

| backbone | V1 fp @ B5 cov | B5 fp @ defer=20% | V1 max cov | B5 max cov | V1<=B5 share |
|---|---:|---:|---:|---:|---:|
| vitb32 | 6.0% | 4.2% | 92.5% | 72.8% | 0.0% |
| vitb16 | 4.8% | 5.9% | 93.5% | 75.8% | 81.8% |
| vitl14 | 5.9% | 10.5% | 92.2% | 72.9% | 100.0% |

## Interpretation

V1 is a calibrated risk-coverage framework. The release evidence supports lower false-pick risk in the reported operating regions, while the same-coverage recall gap and coverage ceiling remain explicit limitations.

The official evaluation artifacts include pooled metrics, bootstrap intervals, turning audit, paired differences against B5, and cost sensitivity.

## External validation

Laboro Tomato is an external validation check in a lower-separability crop setting. It uses one backbone and class-based pick ground truth, so it is not treated as an equivalent second formal freeze.
