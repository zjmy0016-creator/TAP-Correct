# TAP-Correct V1 Frozen Protocol

Version: 2026-07-07 formal freeze

This document freezes the TAP-Correct V1 method and separates it from the
earlier V0 diagnostic stage. It is intended to be the manuscript-facing
protocol used for reproduction, reporting, and claim control.

## 1. Scope

V1 is the final method reported by this repository. V0 results from D0-D5 are
kept as a diagnostic motivation stage showing that the old 1D H-axis endpoint
was not sufficient against the B5 baseline family.

The V1 claim is:

> A training-free 512D prototype expectation endpoint reduces irreversible
> false-pick risk relative to the original TAP H-axis endpoint, and improves
> the false-pick vs coverage frontier on stronger frozen CLIP backbones.

The V1 claim is not:

> V1 fully dominates B5 on every metric.

Addendum, 2026-07-07:

> A Laboro Tomato external validation check has been completed with the same
> frozen endpoint and threshold protocol as the strawberry V1 result. It
> supports the transfer of the collapsed-class failure pattern and the
> 512D-endpoint false-pick reduction under a lower-separability crop setting,
> but it remains lighter than the strawberry freeze because it uses one
> backbone and class-based pick ground truth.

## 2. Frozen Inputs

- Dataset split and episode manifest: `outputs/episodes/manifest_K1-16_ep100.json`
- Decision ground truth: `outputs/decision_gold/turning_decision_dataset/labels/test_decision_ground_truth_clean.csv`
- Cached frozen CLIP features:
  - `outputs/features_vitb32.npz`
  - `outputs/features_vitb16.npz`
  - `outputs/features_vitl14.npz`
- K used for the formal V1 frontier: `K=16`
- Number of episodes: 100 for each K block, inherited from the frozen manifest

The query/test split must not be used for threshold selection, method tuning, or
operating-point selection.

## 3. Frozen Method

For each episode:

1. Build class prototypes from the support set for `ripe`, `turning`, and
   `unripe`.
2. Compute class similarities between each image feature and the three class
   prototypes.
3. Convert similarities to a three-class softmax probability vector.
4. Compute the V1 endpoint:

```text
E = P(ripe) * 1.0 + P(turning) * 0.5 + P(unripe) * 0.0
```

5. Calibrate pick and wait thresholds on the calibration split only:
   - `ALPHA = 0.05`
   - `GAMMA = 0.10`
   - `N_CANDIDATES = 200`
6. Predict:
   - `pick` when `E >= T_high_E`
   - `wait` when `E <= T_low_E`
   - `revisit` otherwise
7. Rank revisit candidates by endpoint-distance uncertainty:
   `compute_uncertainty(E, T_high_E, T_low_E)`.
8. Sweep defer/revisit fractions from 0.00 to 0.60 in steps of 0.05 for the
   risk-coverage frontier.

The canonical V1 decision function is `decide_episode_v1_endpoint` in
`tapcorrect/d2_decision.py`. The reference frontier implementation is
`scripts/xbackbone_step2_frontier.py`; the official headline/CI evaluator is
`scripts/v1_official_eval.py`; the report aggregator is
`scripts/v1_freeze_report.py`.

## 4. Frozen Baselines

V0:

- Same episode manifest and calibration protocol.
- Endpoint is the old 1D text-axis score `H`.
- Used only as the diagnostic predecessor.

B5 family:

- Class decision is prototype argmax.
- Defer/revisit is controlled by top-2 similarity margin.
- The calibrated comparison point uses a 20% margin defer/revisit operating
  point.
- The swept B5 family frontier is reported alongside V1.

## 5. Frozen Metrics

Primary:

- false-pick rate among predicted picks
- pick precision
- pick recall
- revisit burden
- actual coverage

Manuscript-facing frontier comparisons:

- V1 false-pick rate at the calibrated B5 coverage
- V0 false-pick rate at the calibrated B5 coverage
- B5 family false-pick rate at defer=20%
- V1 maximum coverage
- B5 family maximum coverage
- V1 false-pick dominance share over the B5 family frontier
- V1-vs-V0 and V1-vs-B5 paired bootstrap differences at the frozen headline
  operating point

Recall must be reported with the frontier because V1 has a same-coverage recall
trade-off against B5.

## 6. Reproduction Commands

Generate the formal V1 summary from the frozen CSV outputs:

```bash
python scripts/v1_freeze_report.py
```

Generate the official V1 headline evaluation, bootstrap CI, turning audit,
paired differences, and cost sweep:

```bash
python scripts/v1_official_eval.py
```

Re-run cross-backbone frontier outputs from cached feature files:

```bash
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb32.npz --k 16
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb16.npz --k 16
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitl14.npz --k 16
```

Re-run the test suite:

```bash
python -m unittest discover -v
```

## 7. Allowed Claims

- V1 improves all four frozen headline metrics over V0 on ViT-B/32 at K=16 and
  20% uncertainty cut.
- V1 consistently reduces false-pick risk relative to V0 at the calibrated B5
  coverage across ViT-B/32, ViT-B/16, and ViT-L/14.
- On ViT-B/16 and ViT-L/14, V1 beats the B5 family false-pick rate at the
  calibrated B5 coverage.
- V1 extends usable high-coverage regions beyond the B5 family frontier.
- The method is training-free with frozen CLIP features and calibration-only
  thresholds.

## 8. Claims Not Allowed Without More Evidence

- V1 fully dominates B5 on all metrics.
- The uncertainty signal significantly enriches human borderline samples.
- The Laboro Tomato check is a complete second frozen protocol equivalent to
  the strawberry V1 freeze.
- V1 has strong external validity across many datasets or crop families.
- V1 is a trained or learned maturity model.

## 9. Required Limitations

- V1 is a diagnostic-driven post-hoc method selection after the D5 V0 audit.
- The current repository formally freezes the strawberry protocol and reports
  multi-backbone robustness. Laboro Tomato provides external validation under
  the same frozen endpoint/threshold protocol, but it uses one backbone and
  class-based pick ground truth rather than the full strawberry artifact set.
- Same-coverage pick recall remains lower than B5 in the overlap region.
- The uncertainty mechanism should be described as revisit control and
  supplementary audit, not as the main contribution.

## 10. Preliminary External Validation: Laboro Tomato

- Dataset: Laboro Tomato crops generated from COCO boxes, 9430 patches across
  mature, turning, and immature levels.
- Backbone: frozen CLIP ViT-B/16.
- Zero-shot result: the weakest class is mature, with F1 = 0.04; this differs
  from the strawberry turning collapse but preserves the broader
  collapsed-class failure pattern.
- Frozen-protocol check: K=16, 100 episodes, ALPHA=0.05, GAMMA=0.10,
  N_CANDIDATES=200, defer sweep 0..0.6, and B5 margin 20th-percentile.
- Result: over the common risk-coverage region, the V1 512D-endpoint frontier
  has lower or equal false-pick rate than the B5-family frontier at every
  shared coverage point. At B5's operating coverage, V1 false-pick is lower
  than B5 (52.8% vs 59.7%).
- Limitation: tomato remains a hard, low-separability external check. V1 max
  coverage is lower than the B5 family (49.8% vs 76.7%), so the tomato result
  should be framed as conservative risk-control behavior rather than full
  metric domination.
- Claim boundary: use this evidence to argue cross-dataset plausibility under
  a frozen protocol, not as a complete second formal freeze equivalent to the
  strawberry protocol.
