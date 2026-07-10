# -*- coding: utf-8 -*-
"""Summarize CLIP/selective baseline CSVs into release-facing main table."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FAMILIES = ("ZS-temp", "TipAdapter", "ProtoAdapter")
FALSE_PICK_ALPHAS = (0.05, 0.10)
INPUT_GLOBS = (
    "zs_temp_*.csv",
    "tip_adapter_*.csv",
    "proto_adapter_*.csv",
)
OUTPUT_FIELDNAMES = [
    "dataset",
    "backbone",
    "family",
    "false_pick_alpha",
    "selection_status",
    "source_file",
    "n_source_rows",
    "n_episodes",
    "episode",
    "baseline",
    "selector",
    "target_coverage",
    "coverage",
    "false_pick_rate",
    "pick_precision",
    "pick_recall",
    "revisit_burden",
    "n_samples",
    "n_pick",
    "n_wait",
    "n_revisit",
    "n_false_pick",
    "n_true_pick",
    "temperature",
    "alpha",
    "beta",
    "proto_weight",
    "calibration_nll",
    "selector_threshold",
]
GROUP_FIELDS = (
    "dataset",
    "backbone",
    "family",
    "baseline",
    "selector",
    "target_coverage",
)
MEAN_FIELDS = (
    "false_pick_rate",
    "pick_precision",
    "pick_recall",
    "revisit_burden",
    "coverage",
    "n_samples",
    "n_pick",
    "n_wait",
    "n_revisit",
    "n_false_pick",
    "n_true_pick",
)
COMPACT_FIELDS = (
    "source_file",
    "temperature",
    "alpha",
    "beta",
    "proto_weight",
    "calibration_nll",
    "selector_threshold",
)


def family_from_filename(filename: str) -> str:
    name = Path(filename).name.lower()
    if name.startswith("zs_temp_"):
        return "ZS-temp"
    if name.startswith("tip_adapter_"):
        return "TipAdapter"
    if name.startswith("proto_adapter_"):
        return "ProtoAdapter"
    raise ValueError(f"cannot infer baseline family from {filename!r}")


def fmt_alpha(alpha: float) -> str:
    return f"{float(alpha):.2f}"


def float_value(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def compact_values(rows: list[dict], key: str) -> str:
    values = sorted({row.get(key, "") for row in rows if row.get(key, "") != ""})
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "varies" if key != "source_file" else ";".join(values)


def load_baseline_rows(input_dir: Path) -> list[dict]:
    rows = []
    seen_paths = set()
    for pattern in INPUT_GLOBS:
        for path in sorted(input_dir.glob(pattern)):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            family = family_from_filename(path.name)
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    row = dict(row)
                    row["family"] = family
                    row["source_file"] = path.name
                    rows.append(row)
    return rows


def aggregate_operating_points(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)

    aggregated = []
    for key, group_rows in sorted(groups.items()):
        row = {field: "" for field in OUTPUT_FIELDNAMES}
        for field, value in zip(GROUP_FIELDS, key):
            row[field] = value

        for field in COMPACT_FIELDS:
            row[field] = compact_values(group_rows, field)

        for field in MEAN_FIELDS:
            values = [float_value(item, field) for item in group_rows if item.get(field, "") != ""]
            if values:
                row[field] = format_float(sum(values) / len(values))

        episodes = sorted({item.get("episode", "") for item in group_rows if item.get("episode", "") != ""})
        row["episode"] = "" if not episodes else "mean"
        row["n_source_rows"] = str(len(group_rows))
        row["n_episodes"] = str(len(episodes)) if episodes else ""
        aggregated.append(row)

    return aggregated


def select_best_row(rows: list[dict], false_pick_alpha: float) -> dict | None:
    feasible = [
        row
        for row in rows
        if row.get("false_pick_rate", "") != ""
        and float_value(row, "false_pick_rate") <= float(false_pick_alpha)
    ]
    if not feasible:
        return None

    def sort_key(row: dict) -> tuple:
        return (
            -float_value(row, "coverage"),
            float_value(row, "false_pick_rate"),
            row.get("baseline", ""),
            row.get("selector", ""),
            row.get("episode", ""),
            row.get("source_file", ""),
        )

    return sorted(feasible, key=sort_key)[0]


def empty_result_row(
    dataset: str,
    backbone: str,
    family: str,
    false_pick_alpha: float,
    status: str,
) -> dict:
    row = {field: "" for field in OUTPUT_FIELDNAMES}
    row.update(
        {
            "dataset": dataset,
            "backbone": backbone,
            "family": family,
            "false_pick_alpha": fmt_alpha(false_pick_alpha),
            "selection_status": status,
        }
    )
    return row


def output_row_from_candidate(candidate: dict, false_pick_alpha: float) -> dict:
    row = {field: "" for field in OUTPUT_FIELDNAMES}
    for field in OUTPUT_FIELDNAMES:
        if field in candidate:
            row[field] = candidate.get(field, "")
    row["false_pick_alpha"] = fmt_alpha(false_pick_alpha)
    row["selection_status"] = "selected"
    return row


def build_main_table(
    rows: list[dict],
    false_pick_alphas: tuple[float, ...] = FALSE_PICK_ALPHAS,
    families: tuple[str, ...] = FAMILIES,
) -> list[dict]:
    rows = aggregate_operating_points(rows)
    dataset_backbones = sorted(
        {
            (row.get("dataset", ""), row.get("backbone", ""))
            for row in rows
            if row.get("dataset", "")
        }
    )
    table = []

    for dataset, backbone in dataset_backbones:
        for family in families:
            group = [
                row
                for row in rows
                if row.get("dataset", "") == dataset
                and row.get("backbone", "") == backbone
                and row.get("family", "") == family
            ]
            for alpha in false_pick_alphas:
                if not group:
                    table.append(empty_result_row(dataset, backbone, family, alpha, "missing_family"))
                    continue

                best = select_best_row(group, alpha)
                if best is None:
                    table.append(empty_result_row(dataset, backbone, family, alpha, "no_feasible_row"))
                else:
                    table.append(output_row_from_candidate(best, alpha))

    return table


def write_main_table(
    input_dir: Path,
    out_csv: Path,
    false_pick_alphas: tuple[float, ...] = FALSE_PICK_ALPHAS,
) -> list[dict]:
    rows = load_baseline_rows(input_dir)
    table = build_main_table(rows, false_pick_alphas=false_pick_alphas)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(table)

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/clip_selective_baselines"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/clip_selective_baselines/main_table.csv"),
    )
    parser.add_argument(
        "--false_pick_alphas",
        type=float,
        nargs="+",
        default=list(FALSE_PICK_ALPHAS),
    )
    args = parser.parse_args()

    table = write_main_table(
        input_dir=args.input_dir,
        out_csv=args.out,
        false_pick_alphas=tuple(args.false_pick_alphas),
    )
    selected = sum(1 for row in table if row["selection_status"] == "selected")
    print(f"read from {args.input_dir}")
    print(f"wrote {len(table)} rows ({selected} selected) -> {args.out}")


if __name__ == "__main__":
    main()
