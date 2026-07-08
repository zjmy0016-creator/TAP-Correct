# CLIP/Selective Baseline Results Summary

This note summarizes reviewer-facing CLIP/selective baselines for TAP-Correct.
It is derived from:

- `outputs/clip_selective_baselines/main_table.csv`
- `outputs/clip_selective_baselines/fp_controlled_main_table.csv`

The manuscript draft is not modified by this summary.

## Main Operating-Point Table

Rows are selected from ZS-temp, TipAdapter, and ProtoAdapter sweeps. For
episode-based methods, rows are first averaged by
`dataset/backbone/family/baseline/selector/target_coverage`; the table then
selects the highest mean coverage operating point satisfying
`false_pick_rate <= alpha`.

| Dataset | Family | Alpha | Status | Baseline | Selector | False-pick | Coverage |
|---|---|---:|---|---|---|---:|---:|
| laboro_tomato | ZS-temp | 0.05 | selected | ZS-temp-selective | entropy | 0.0000 | 0.6703 |
| laboro_tomato | ZS-temp | 0.10 | selected | ZS-temp-selective | entropy | 0.0000 | 0.6703 |
| laboro_tomato | TipAdapter | 0.05 | no_feasible_row | - | - | - | - |
| laboro_tomato | TipAdapter | 0.10 | selected | TipAdapter-selective | entropy | 0.0997 | 0.3722 |
| laboro_tomato | ProtoAdapter | 0.05 | no_feasible_row | - | - | - | - |
| laboro_tomato | ProtoAdapter | 0.10 | no_feasible_row | - | - | - | - |
| strawberry | ZS-temp | 0.05 | selected | ZS-hard | none | 0.0117 | 0.9157 |
| strawberry | ZS-temp | 0.10 | selected | ZS-hard | none | 0.0117 | 0.9157 |
| strawberry | TipAdapter | 0.05 | selected | TipAdapter-hard | none | 0.0255 | 0.7542 |
| strawberry | TipAdapter | 0.10 | selected | TipAdapter-hard | none | 0.0255 | 0.7542 |
| strawberry | ProtoAdapter | 0.05 | selected | ProtoAdapter-hard | none | 0.0108 | 0.7304 |
| strawberry | ProtoAdapter | 0.10 | selected | ProtoAdapter-hard | none | 0.0108 | 0.7304 |

## FP-Controlled Table

FP-controlled baselines choose pick-confidence thresholds on the calibration
split under false-pick constraints, then evaluate the resulting thresholds on
query/test.

| Dataset | Family | Alpha | Status | Calibration FP | Test FP | Test Coverage |
|---|---|---:|---|---:|---:|---:|
| laboro_tomato | ProtoAdapter | 0.05 | query_exceeds_alpha | 0.0466 | 0.1153 | 0.6302 |
| laboro_tomato | ProtoAdapter | 0.1 | query_exceeds_alpha | 0.0975 | 0.1700 | 0.6618 |
| laboro_tomato | TipAdapter | 0.05 | query_exceeds_alpha | 0.0471 | 0.1049 | 0.6726 |
| laboro_tomato | TipAdapter | 0.1 | query_exceeds_alpha | 0.0956 | 0.1577 | 0.6980 |
| laboro_tomato | ZS-temp | 0.05 | query_exceeds_alpha | 0.0000 | 0.4444 | 0.7527 |
| laboro_tomato | ZS-temp | 0.1 | query_exceeds_alpha | 0.0000 | 0.4444 | 0.7527 |
| strawberry | ProtoAdapter | 0.05 | query_within_alpha | 0.0136 | 0.0099 | 0.7290 |
| strawberry | ProtoAdapter | 0.1 | query_within_alpha | 0.0147 | 0.0105 | 0.7298 |
| strawberry | TipAdapter | 0.05 | query_within_alpha | 0.0180 | 0.0125 | 0.7374 |
| strawberry | TipAdapter | 0.1 | query_within_alpha | 0.0244 | 0.0180 | 0.7468 |
| strawberry | ZS-temp | 0.05 | query_within_alpha | 0.0104 | 0.0117 | 0.9157 |
| strawberry | ZS-temp | 0.1 | query_within_alpha | 0.0104 | 0.0117 | 0.9157 |

## Key Findings

1. On strawberry, all three baseline families can operate under low query/test
   false-pick rates. ZS-temp retains the highest coverage, while TipAdapter and
   ProtoAdapter remain conservative but stable.

2. The strawberry result should not be overread as a universal CLIP-adapter win:
   the high ZS-temp coverage indicates that this split is relatively forgiving
   for zero-shot text-prototype decisions.

3. On Laboro Tomato, the main sweep table exposes missing feasible operating
   points for adapter methods under stricter false-pick constraints after
   episode averaging.

4. The FP-controlled table is the strongest risk-control diagnostic. Calibration
   false-pick can be kept within the requested threshold on tomato, but
   query/test false-pick exceeds the threshold for all families.

5. The most severe transfer failure is Laboro Tomato ZS-temp: calibration
   false-pick is 0.0000, while query/test false-pick is 0.4444. This shows that
   apparent calibration safety may not transfer across the tomato query split.

6. TipAdapter and ProtoAdapter reduce the tomato FP-controlled failure relative
   to ZS-temp, but they still exceed the requested query/test false-pick
   constraints. Therefore, the tomato evidence should be framed as external
   validation difficulty and calibration-transfer stress testing, not adapter
   dominance.

7. For the paper narrative, the safest claim is: TAP-Correct addresses
   risk-aware harvesting decisions through calibrated selective action, and the
   second dataset demonstrates why calibration and abstention are necessary
   rather than proving universal superiority.

## Recommended Paper Use

Use `main_table.csv` as the baseline table skeleton and
`fp_controlled_main_table.csv` as the risk-control analysis table.

Suggested wording:

> CLIP adaptation baselines improve the breadth of comparison, but the external
> tomato data reveal substantial calibration-to-query risk transfer failures,
> especially for zero-shot CLIP. These results motivate risk-aware selective
> harvesting decisions rather than simple hard-label adaptation.

Avoid wording such as:

- "The adapter baselines are dominated on every dataset."
- "The method is fully validated across crops."
- "Calibration guarantees false-pick control on external data."
- "The tomato experiment proves universal generalization."

## Strict Unsafe-Pick Diagnostic

A strict unsafe-pick diagnostic is included because the original
`false_pick_rate` only counts immature/unripe fruit picked as `pick`. In
selective harvesting, picking transitional fruit can also be considered unsafe.

```text
unsafe_pick_rate = picked non-target fruit / all predicted pick actions
```

Target pick classes are `ripe` for strawberry and `mature` for tomato.

| Dataset | Family | Alpha | Query unsafe-pick | Strict pick precision |
|---|---|---:|---:|---:|
| laboro_tomato | ProtoAdapter | 0.05 | 0.3674 | 0.6326 |
| laboro_tomato | ProtoAdapter | 0.1 | 0.4307 | 0.5693 |
| laboro_tomato | TipAdapter | 0.05 | 0.3627 | 0.6373 |
| laboro_tomato | TipAdapter | 0.1 | 0.4184 | 0.5816 |
| laboro_tomato | ZS-temp | 0.05 | 0.6667 | 0.3333 |
| laboro_tomato | ZS-temp | 0.1 | 0.6667 | 0.3333 |
| strawberry | ProtoAdapter | 0.05 | 0.0903 | 0.9097 |
| strawberry | ProtoAdapter | 0.1 | 0.0906 | 0.9094 |
| strawberry | TipAdapter | 0.05 | 0.0997 | 0.9003 |
| strawberry | TipAdapter | 0.1 | 0.1085 | 0.8915 |
| strawberry | ZS-temp | 0.05 | 0.1907 | 0.8093 |
| strawberry | ZS-temp | 0.1 | 0.1907 | 0.8093 |

Interpretation: the original false-pick metric captures severe immature/unripe
mis-picks, while strict unsafe-pick also penalizes picking transitional fruit.
Under this stricter view, ZS-temp is substantially less safe, especially on
tomato. TipAdapter and ProtoAdapter reduce unsafe picks relative to ZS-temp, but
they still do not eliminate tomato risk-transfer failure.
