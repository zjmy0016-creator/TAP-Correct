# V1 Official Evaluation

This report evaluates the frozen 512D prototype-expectation endpoint with calibration-only thresholds.

## Pooled K=16 metrics

- false-pick rate: 9.1%
- pick precision: 90.9%
- pick recall: 80.6%
- revisit burden: 19.2%

## Bootstrap confidence intervals

- false_pick_rate: 9.1% [8.5%, 9.8%]
- pick_precision: 90.9% [90.2%, 91.5%]
- pick_recall: 80.6% [79.4%, 81.7%]
- revisit_burden: 19.2% [18.5%, 19.8%]

## Turning audit

- revisit enrichment: 0.6086
- enrichment lift: 0.0737
- defer-rate gap: 0.1022

The turning audit is supplementary evidence; the primary contribution is the calibrated risk-coverage decision layer.