import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_clip_selective_baselines import (
    build_main_table,
    family_from_filename,
    select_best_row,
    write_main_table,
)


class TestSummarizeClipSelectiveBaselines(unittest.TestCase):
    def test_family_from_filename(self):
        self.assertEqual(family_from_filename("zs_temp_strawberry_vitb32.csv"), "ZS-temp")
        self.assertEqual(family_from_filename("tip_adapter_strawberry_vitb32.csv"), "TipAdapter")
        self.assertEqual(family_from_filename("proto_adapter_strawberry_vitb32.csv"), "ProtoAdapter")

    def test_selects_highest_coverage_under_false_pick_alpha(self):
        rows = [
            {"false_pick_rate": "0.02", "coverage": "0.40", "selector": "msp", "baseline": "A"},
            {"false_pick_rate": "0.04", "coverage": "0.70", "selector": "margin", "baseline": "A"},
            {"false_pick_rate": "0.08", "coverage": "0.90", "selector": "entropy", "baseline": "A"},
        ]

        best = select_best_row(rows, false_pick_alpha=0.05)

        self.assertEqual(best["selector"], "margin")
        self.assertEqual(best["coverage"], "0.70")

    def test_build_main_table_keeps_no_feasible_rows(self):
        rows = [
            {
                "dataset": "toy",
                "backbone": "toy",
                "family": "ZS-temp",
                "source_file": "zs.csv",
                "baseline": "ZS-hard",
                "selector": "none",
                "false_pick_rate": "0.20",
                "coverage": "1.0",
            }
        ]

        table = build_main_table(rows, false_pick_alphas=(0.05,), families=("ZS-temp",))

        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]["selection_status"], "no_feasible_row")
        self.assertEqual(table[0]["dataset"], "toy")
        self.assertEqual(table[0]["family"], "ZS-temp")

    def test_write_main_table_reads_three_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "main_table.csv"
            self._write_baseline_csv(
                tmp / "zs_temp_strawberry_vitb32.csv",
                [
                    ("ZS-hard", "none", "", "0.02", "0.90"),
                    ("ZS-temp-selective", "msp", "", "0.00", "0.40"),
                ],
            )
            self._write_baseline_csv(
                tmp / "tip_adapter_strawberry_vitb32.csv",
                [
                    ("TipAdapter-hard", "none", "0", "0.20", "0.95"),
                    ("TipAdapter-selective", "margin", "0", "0.03", "0.75"),
                ],
            )
            self._write_baseline_csv(
                tmp / "proto_adapter_strawberry_vitb32.csv",
                [
                    ("ProtoAdapter-hard", "none", "0", "0.04", "0.80"),
                    ("ProtoAdapter-selective", "entropy", "0", "0.00", "0.50"),
                ],
            )

            table = write_main_table(input_dir=tmp, out_csv=out, false_pick_alphas=(0.05, 0.10))

            self.assertTrue(out.exists())
            self.assertEqual(len(table), 6)
            by_family = {(row["family"], row["false_pick_alpha"]): row for row in table}
            self.assertEqual(by_family[("ZS-temp", "0.05")]["baseline"], "ZS-hard")
            self.assertEqual(by_family[("TipAdapter", "0.05")]["selector"], "margin")
            self.assertEqual(by_family[("ProtoAdapter", "0.10")]["baseline"], "ProtoAdapter-hard")
            self.assertEqual(by_family[("TipAdapter", "0.05")]["n_source_rows"], "1")

            with out.open(encoding="utf-8", newline="") as f:
                written = list(csv.DictReader(f))
            self.assertEqual(len(written), 6)

    def test_write_main_table_aggregates_episode_rows_before_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "main_table.csv"
            self._write_baseline_csv(
                tmp / "tip_adapter_strawberry_vitb32.csv",
                [
                    ("TipAdapter-hard", "none", "0", "0.00", "0.90"),
                    ("TipAdapter-hard", "none", "1", "0.20", "0.90"),
                    ("TipAdapter-selective", "margin", "0", "0.04", "0.70"),
                    ("TipAdapter-selective", "margin", "1", "0.04", "0.70"),
                ],
            )

            table = write_main_table(input_dir=tmp, out_csv=out, false_pick_alphas=(0.05,))
            tip_row = next(row for row in table if row["family"] == "TipAdapter")

            self.assertEqual(tip_row["baseline"], "TipAdapter-selective")
            self.assertEqual(tip_row["selector"], "margin")
            self.assertEqual(tip_row["n_source_rows"], "2")
            self.assertAlmostEqual(float(tip_row["false_pick_rate"]), 0.04)

    def _write_baseline_csv(self, path: Path, rows):
        fieldnames = [
            "dataset",
            "backbone",
            "episode",
            "baseline",
            "selector",
            "temperature",
            "alpha",
            "beta",
            "proto_weight",
            "calibration_nll",
            "target_coverage",
            "selector_threshold",
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
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for baseline, selector, episode, fp, coverage in rows:
                writer.writerow(
                    {
                        "dataset": "strawberry",
                        "backbone": "vitb32",
                        "episode": episode,
                        "baseline": baseline,
                        "selector": selector,
                        "target_coverage": "1.0" if selector == "none" else "0.8",
                        "false_pick_rate": fp,
                        "pick_precision": "1.0",
                        "pick_recall": "0.5",
                        "revisit_burden": str(1.0 - float(coverage)),
                        "coverage": coverage,
                        "n_samples": "10",
                        "n_pick": "5",
                        "n_wait": "3",
                        "n_revisit": "2",
                        "n_false_pick": "0",
                        "n_true_pick": "5",
                    }
                )


if __name__ == "__main__":
    unittest.main()
