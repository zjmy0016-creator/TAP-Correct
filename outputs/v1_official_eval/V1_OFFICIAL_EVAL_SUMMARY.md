# V1 Official Evaluation

This report evaluates the frozen V1 endpoint on the existing D3 query scores.
The only method change from V0 is that pick/wait boundaries use E instead of H.

## Pooled K=16 Metrics

- false-pick rate: 9.1%
- pick precision: 90.9%
- pick recall: 80.6%
- revisit burden: 19.2%

## Bootstrap CI

- false_pick_rate: 9.1% [8.5%, 9.8%]
- pick_precision: 90.9% [90.2%, 91.5%]
- pick_recall: 80.6% [79.4%, 81.7%]
- revisit_burden: 19.2% [18.5%, 19.8%]

## Turning Audit

- revisit enrichment: 0.6086
- enrichment lift: 0.0737
- defer-rate gap: 0.1022

Interpretation: turning uncertainty enrichment remains a supplementary audit, not the main contribution.
