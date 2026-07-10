# Changelog

## 2026-07-10 — Public delivery release

### Scope

- Rebuilt the public repository tree around the formal TAP-Correct V1 delivery
  workflow.
- Standardized public documentation, modules, scripts, output directories, and
  test names in English.
- Kept the frozen feature caches, episode manifest, decision labels,
  calibration thresholds, comparator decisions, frontier artifacts, and
  official evaluation outputs required for reproduction.
- Consolidated formal traceability in this file and in the Git commit history.

### Removed from the public tree

- Historical process-stage modules and output names that are not required by
  the formal release workflow.
- Unpublished research documents, local exports, temporary work products, and
  source datasets.
- Superseded comparator artifacts that could be mistaken for current release
  evidence.

### Verification record

The release audit covers:

- public path references in `README.md`, `PROTOCOL.md`, `REPRODUCTION.md`, and
  `DELIVERY.md`;
- English-only public text and filenames;
- absence of superseded method or process-stage identifiers in tracked text;
- Python syntax compilation;
- full `python -m unittest discover -v` execution;
- final Git status, staged tree, commit, and remote push verification.

The release identifier is `2026-07-10-public-delivery`. Exact file history,
commit ancestry, and the remote publication state remain available through
Git and the configured remote repository.
