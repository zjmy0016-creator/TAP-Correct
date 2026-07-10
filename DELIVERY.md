# Public Delivery Inventory

This document defines the contents of the TAP-Correct V1 public release.

## Public components

| Component | Public path | Purpose |
|---|---|---|
| Core package | `tapcorrect/` | Episode construction, leakage checks, prototypes, decisions, uncertainty, and metrics |
| Reproduction scripts | `scripts/` | Formal evaluation, comparator evaluation, and bounded external validation |
| Tests | `tests/` | Unit, contract, dependency, and reproduction checks |
| Formal protocol | `PROTOCOL.md` | Frozen inputs, procedure, metrics, and claim boundary |
| Reproduction guide | `REPRODUCTION.md` | Commands, asset map, outputs, and reproduction controls |
| Dataset attribution | `DATASETS.md` | Public dataset sources, citations, and license notes |
| Release trace | `CHANGELOG.md` | Formal change and verification record |
| Frozen data | `outputs/` allowlist | Features, manifests, labels, thresholds, decisions, external-validation summaries, and final release summaries |

## Output allowlist

The public output tree contains only:

- frozen CLIP feature caches;
- Strawberry-DS external-validation feature cache and mode-specific summary CSVs;
- the deterministic episode manifest;
- the decision ground-truth CSV;
- calibration thresholds;
- frozen query and B5 comparator decisions;
- cross-backbone frontier artifacts;
- official V1 evaluation artifacts;
- formal release summary artifacts.

## Excluded content

The public release excludes raw source datasets, crop images, local exports,
unpublished research documents, temporary files, per-episode Strawberry-DS
CSVs, and intermediate analysis folders that are not required by the formal
reproduction sequence.

## Acceptance criteria

The release is considered deliverable only when:

1. all public filenames and user-facing text are in English;
2. the public tree contains no process-stage or superseded method identifiers;
3. the README commands resolve to existing public paths;
4. both Laboro Tomato and Strawberry-DS external-validation entries resolve to
   existing public inputs, scripts, and summary outputs;
5. dataset attribution and license notes are present in `DATASETS.md`;
6. the full test suite completes successfully;
7. the final Git tree contains no ignored research materials;
8. the release commit is pushed to the configured remote branch.
