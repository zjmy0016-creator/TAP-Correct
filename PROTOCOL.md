# TAP-Correct V1 Protocol

Status: formal release protocol, frozen 2026-07-07.

## 1. Scope

TAP-Correct V1 is a training-free selective harvesting policy built from
frozen CLIP image features and few-shot visual prototypes. It produces three
operational actions: `pick`, `wait`, and `revisit`.

The formal endpoint is the ordered prototype expectation:

```text
E = P(ripe) * 1.0 + P(turning) * 0.5 + P(unripe) * 0.0
```

The release claim is limited to calibrated risk-coverage behavior under the
specified data split, threshold policy, support sizes, and evaluation metrics.

## 2. Frozen inputs

- Episode manifest: `outputs/episodes/manifest_K1-16_ep100.json`
- Decision ground truth:
  `outputs/decision_gold/turning_decision_dataset/labels/test_decision_ground_truth_clean.csv`
- Feature caches:
  - `outputs/features_vitb32.npz`
  - `outputs/features_vitb16.npz`
  - `outputs/features_vitl14.npz`
  - `outputs/features_laboro_tomato_vitb16.npz`
  - `outputs/features_strawberryds_vitb32.npz`
- Formal support size: `K=16`
- Calibration size: 200 samples per class and episode
- Episodes: 100 per support-size block

The query split is never used for threshold selection, method tuning, or
operating-point selection.

## 3. Decision procedure

For each episode:

1. Build normalized visual prototypes for `ripe`, `turning`, and `unripe`.
2. Compute prototype similarities for each sample.
3. Convert the similarities to a three-class softmax probability vector.
4. Compute the ordered endpoint `E` using the anchors 1.0, 0.5, and 0.0.
5. Select high and low thresholds on the calibration split with:
   - `ALPHA = 0.05`
   - `GAMMA = 0.10`
   - `N_CANDIDATES = 200`
6. Assign actions:
   - `pick` when `E >= T_high_E`
   - `wait` when `E <= T_low_E`
   - `revisit` otherwise
7. Rank revisit candidates by distance to the confident endpoint interval.
8. Sweep revisit fractions from 0.00 through 0.60 in increments of 0.05.

The canonical implementation is
`tapcorrect.decision.decide_episode_v1_endpoint`.

## 4. Comparator

The B5 comparator uses visual prototype argmax decisions and a top-two
similarity margin to control the revisit fraction. It is reported as a
reference risk-coverage frontier and at the calibrated 20% revisit point.

## 5. Metrics

Primary metrics:

- false-pick rate among predicted picks;
- pick precision;
- pick recall;
- revisit burden;
- actual coverage.

The official artifacts also include bootstrap confidence intervals, turning
audits, paired V1-versus-B5 differences, and cost sensitivity.

## 6. Reproduction commands

```bash
python scripts/build_freeze_report.py
python scripts/official_evaluation.py
python scripts/backbone_frontier.py --npz outputs/features_vitb32.npz --k 16
python scripts/backbone_frontier.py --npz outputs/features_vitb16.npz --k 16
python scripts/backbone_frontier.py --npz outputs/features_vitl14.npz --k 16
python -m unittest discover -v
```

## 7. Allowed statements

- V1 is training-free and uses frozen CLIP features.
- Thresholds are selected on calibration data only.
- The 512-dimensional endpoint supports the reported risk-coverage behavior
  under the frozen protocol.
- Laboro Tomato is a bounded external check using one backbone and
  class-based pick ground truth.
- Strawberry-DS is reported as three explicitly labeled evaluations: direct
  transfer (`maincal`), external recalibration sensitivity (`recalib`), and an
  in-domain reference (`indomain`). These results use the public feature cache,
  `K=16`, and the stated episode construction.

## 8. Required limitations

- Results are protocol-specific and do not establish universal cross-crop
  generalization.
- Laboro Tomato and Strawberry-DS use separate external-validation protocols;
  neither dataset establishes universal deployment performance.
- V1 should not be described as dominating every comparator on every metric.
- Same-coverage recall and maximum coverage trade-offs remain visible in the
  frontier artifacts.
- Revisit is a decision-control action, not a standalone uncertainty model.
