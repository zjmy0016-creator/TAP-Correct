# TAP-Correct

TAP-Correct is a training-free selective harvesting decision framework built on
frozen CLIP features. It converts transitional fruit maturity from a hard
three-class recognition problem into a calibrated `pick / wait / revisit`
decision signal for selective harvesting.

The frozen V1 protocol uses a 512D prototype-expectation endpoint over visual
class prototypes instead of the earlier 1D text-axis endpoint. The repository
contains the code, frozen feature caches, frozen episode manifest, decision
labels, tests, and report scripts needed to reproduce the V1 summary, official
evaluation, and cross-backbone risk-coverage evidence.

## Highlights

- **Training-free protocol:** frozen CLIP image/text features, few-shot visual
  prototypes, and no gradient updates.
- **Frozen V1 endpoint:** prototype expectation over `ripe`, `turning`, and
  `unripe` class prototypes in the 512D CLIP feature space.
- **Calibration-only thresholding:** operating thresholds are selected on held-
  out calibration data and are not tuned on the query/test split.
- **Decision-oriented outputs:** false-pick risk, pick precision, pick recall,
  revisit burden, bootstrap confidence intervals, paired differences, cost
  sweep, and risk-coverage frontiers.
- **Claim control:** V1 should be described as a calibrated risk-coverage
  trade-off framework. It should not be described as dominating every baseline
  on every metric.

## Repository layout

```text
tapcorrect/                    Core package: episodes, contracts, decisions
scripts/                       Public V1 reproduction scripts and helpers
tests/                         Contract and reproduction tests
outputs/features_*.npz         Frozen CLIP feature caches
outputs/episodes/              Frozen episode manifest
outputs/decision_gold/         Decision-label files used for evaluation
outputs/probe_512d_endpoint/   Cross-backbone frontier evidence
outputs/v1_freeze_report/      Formal V1 summary artifacts
outputs/v1_official_eval/      Official V1 evaluation artifacts
outputs/clip_selective_baselines/  CLIP/selective baseline outputs
V1_PROTOCOL.md                 Frozen V1 protocol and allowed claims
REPRODUCIBILITY.md             Reproduction commands
```

Raw crop images, manuscript drafts, planning notes, and local temporary files
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

Generate the formal V1 freeze report:

```bash
python scripts/v1_freeze_report.py
```

Generate the official V1 evaluation artifacts:

```bash
python scripts/v1_official_eval.py
```

Regenerate cross-backbone frontiers from frozen feature caches:

```bash
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb32.npz --k 16
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitb16.npz --k 16
python scripts/xbackbone_step2_frontier.py --npz outputs/features_vitl14.npz --k 16
```

Run the public tests:

```bash
python -m unittest discover -v
```

## Frozen V1 result

At the frozen strawberry ViT-B/32, K=16 operating point, TAP-Correct V1 reports:

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

## CLIP/selective baselines

The CLIP/selective baseline suite is available under
`outputs/clip_selective_baselines/`. It includes zero-shot temperature-scaled,
Tip-Adapter selective, Proto-Adapter selective, false-pick-controlled, and
strict unsafe-pick diagnostic baselines.

### Zero-shot temperature-scaled selective baseline

```bash
python scripts/run_zs_temp_selective_baseline.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/zs_temp_strawberry_vitb32.csv --dataset strawberry --backbone vitb32
python scripts/run_zs_temp_selective_baseline.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/zs_temp_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16
```

Each CSV contains one hard zero-shot row plus MSP, margin, and entropy
selective rows over six target coverage points. Thresholds and temperatures are
selected without using the query/test split.

### Tip-Adapter selective baseline

```bash
python scripts/run_tip_adapter_selective_baseline.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/tip_adapter_strawberry_vitb32.csv --dataset strawberry --backbone vitb32 --k 16 --n_episodes 20
python scripts/run_tip_adapter_selective_baseline.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/tip_adapter_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16 --k 16 --n_episodes 20
```

The Tip-Adapter runner samples K support examples per class, calibrates
`alpha/beta` on the calibration split, and reports hard plus MSP/margin/entropy
selective rows for every episode.

### Proto-Adapter selective baseline

```bash
python scripts/run_proto_adapter_selective_baseline.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/proto_adapter_strawberry_vitb32.csv --dataset strawberry --backbone vitb32 --k 16 --n_episodes 20
python scripts/run_proto_adapter_selective_baseline.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/proto_adapter_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16 --k 16 --n_episodes 20
```

The Proto-Adapter runner samples K support examples per class, builds
normalized visual class prototypes, calibrates `proto_weight` on the
calibration split, and reports hard plus MSP/margin/entropy selective rows for
every episode.

### Baseline summary table

```bash
python scripts/summarize_clip_selective_baselines.py --input_dir outputs/clip_selective_baselines --out outputs/clip_selective_baselines/main_table.csv --false_pick_alphas 0.05 0.10
```

`main_table.csv` first averages episode-based rows by
`baseline/selector/target_coverage`, then selects the highest mean coverage
operating point satisfying each false-pick constraint. Rows with no feasible
operating point are retained as `no_feasible_row`.

### False-pick-controlled baseline

```bash
python scripts/run_fp_controlled_baseline.py --npz outputs/features_vitb32.npz --out outputs/clip_selective_baselines/fp_controlled_strawberry_vitb32.csv --dataset strawberry --backbone vitb32 --k 16 --n_episodes 20
python scripts/run_fp_controlled_baseline.py --npz outputs/features_laboro_tomato_vitb16.npz --out outputs/clip_selective_baselines/fp_controlled_laboro_tomato_vitb16.csv --dataset laboro_tomato --backbone vitb16 --k 16 --n_episodes 20
```

The FP-controlled table is written to
`outputs/clip_selective_baselines/fp_controlled_main_table.csv`.

### Strict unsafe-pick diagnostic

A stricter diagnostic is available at
`outputs/clip_selective_baselines/unsafe_pick_main_table.csv`. It treats
transitional fruit picked as `pick` as non-target pick risk, complementing the
original severe false-pick metric.

## External tomato check (frozen protocol)

The repository includes a second-dataset check on Laboro Tomato (9430 crops,
mature/turning/immature, ViT-B/16), run under the same frozen protocol as the
strawberry V1 result (ALPHA=0.05, GAMMA=0.10, N_CANDIDATES=200, defer sweep
0..0.6, B5 margin 20th-percentile). The V1-vs-B5 comparison is therefore
same-coverage and not tuned per dataset.

Key findings:

- Zero-shot CLIP shows a severe **mature-class collapse** on tomato
  (F1 = 0.04), while strawberry shows a severe turning-class collapse. The
  failure mode transfers across crops, although the collapsed class changes.
- Under the frozen protocol, the V1 512D-endpoint risk-coverage frontier
  dominates the B5-family frontier over the entire common coverage region
  (V1 false-pick <= B5 at every shared coverage point); at B5's operating
  coverage, V1 false-pick is lower (52.8% vs 59.7%).
- Absolute false-pick is high for every method on tomato (about 50-60%) because
  the tomato maturity classes are severely overlapped in CLIP space. This hard,
  low-separability regime is itself evidence for why calibrated selective
  decisions are needed.
- Honest limitation (opposite of strawberry): tomato V1 max coverage (49.8%) is
  lower than the B5 family (76.7%); low class separability limits confident
  picks, so here V1 sits at the conservative end rather than the high-coverage
  end.

The tomato check is a second-dataset external validation of the endpoint
mechanism under the frozen protocol, though with a single backbone (ViT-B/16)
and class-based pick ground truth rather than the full strawberry artifact set
and human decision labels.

## Protocol discipline

The official V1 protocol, endpoint, threshold source, evaluation metrics,
allowed claims, and forbidden claims are defined in `V1_PROTOCOL.md`.
Earlier H-endpoint experiments are retained only in local diagnostic logs and
baseline-audit history. They do not redefine the frozen V1 endpoint and should
not be used to claim that V1 was selected from the query/test results.

## Limitations

- The strawberry V1 protocol is the formal frozen result.
- The tomato result now uses the same frozen protocol as strawberry, but with a
  single backbone (ViT-B/16) and class-based pick ground truth rather than the
  full strawberry artifact set and human decision labels.
- V1 improves false-pick risk and coverage behavior on high-separability data
  such as strawberry, but it exhibits a strong coverage trade-off on lower-
  separability data such as tomato.
- V1 should not be described as fully dominating B5 on every metric.
- The `revisit` signal is a decision-control mechanism, not a standalone
  calibrated uncertainty model.

## Citation

No paper citation is available yet. If you use this repository, cite the
repository URL and the frozen protocol version in `V1_PROTOCOL.md`.
