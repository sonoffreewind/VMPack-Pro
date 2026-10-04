"""Focused checks for time-budget experiment summaries (no solver required)."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import plot_results
from gen_gap_summary import (SOLVERS, generate_time_budget_report,
                             summarize_time_budgets)


class TimeBudgetSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scale").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def row(self, instance="i1", budget=10, **changes):
        row = {
            "instance": instance, "instance_hash": instance + "-hash",
            "data_file": "synthetic.csv", "funcase": "mixalgos", "seed": "7",
            "timelimit": str(budget), "heuristic_npms": "12", "heuristic_time": "0.1",
        }
        for solver in SOLVERS:
            row.update({solver["ub"]: "-1", solver["lb"]: "-1",
                        solver["time"]: "0.5", solver["status"]: "NoSolution"})
        row.update(changes)
        return row

    def write_rows(self, budget, rows, name="S1"):
        path = self.root / "scale" / f"{name}_tl{budget:g}.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def summaries(self, budgets=(10,)):
        return summarize_time_budgets(self.root, list(budgets), ["S1"])

    def by_solver(self, summaries, name):
        return next(item for item in summaries if item["solver"] == name)

    def test_reads_exact_budget_file_without_fallback(self):
        # A legacy/unbudgeted CSV must not stand in for the requested run.
        with (self.root / "scale" / "S1.csv").open("w", encoding="utf-8") as stream:
            stream.write("instance\ni1\n")
        with self.assertRaises(FileNotFoundError):
            self.summaries((10,))

    def test_budget_files_must_contain_same_instance_identity(self):
        first = self.row(budget=10)
        second = self.row(budget=20, instance_hash="different-hash")
        self.write_rows(10, [first])
        self.write_rows(20, [second])
        with self.assertRaisesRegex(ValueError, "Instance sets differ"):
            self.summaries((10, 20))

    def test_csv_budget_must_match_its_filename(self):
        self.write_rows(10, [self.row(budget=5)])
        with self.assertRaisesRegex(ValueError, "Budget does not match"):
            self.summaries((10,))

    def test_inconsistent_bounds_are_rejected(self):
        self.write_rows(10, [self.row(mip_ub="10", mip_lb="11")])
        with self.assertRaisesRegex(ValueError, "LB exceeds UB"):
            self.summaries()

    def test_no_solution_sentinel_is_not_counted_as_feasible(self):
        self.write_rows(10, [self.row()])
        result = self.by_solver(self.summaries(), "MIP")
        self.assertEqual(result["n_feasible"], 0)
        self.assertEqual(result["avg_ub"], "-")
        self.assertEqual(result["n_gap"], 0)

    def test_over_budget_equal_bounds_are_not_certified_in_budget(self):
        self.write_rows(10, [self.row(mip_ub="8", mip_lb="8", mip_time="10.1")])
        result = self.by_solver(self.summaries(), "MIP")
        self.assertEqual(result["n_certified"], 0)
        self.assertEqual(result["n_over_budget"], 1)

    def test_error_fallback_bounds_are_not_certified(self):
        self.write_rows(10, [self.row(mip_ub="8", mip_lb="8", mip_time="0.2",
                                       mip_status="Error")])
        result = self.by_solver(self.summaries(), "MIP")
        self.assertEqual(result["n_certified"], 0)
        self.assertEqual(result["n_error"], 1)

    def test_initialization_gain_uses_only_paired_feasible_instances(self):
        rows = [
            self.row(instance="i1", mip_ub="12", mip_mix_ub="10"),
            self.row(instance="i2", mip_ub="8", mip_mix_ub="-1"),
        ]
        self.write_rows(10, rows)
        result = self.by_solver(self.summaries(), "MIP")
        self.assertEqual(result["paired_n"], 1)
        self.assertEqual(result["paired_pm_gain"], "2.00")

    def test_missing_lower_bound_does_not_become_zero_gap(self):
        self.write_rows(10, [self.row(mip_ub="5", mip_lb="")])
        result = self.by_solver(self.summaries(), "MIP")
        self.assertEqual(result["n_gap"], 0)
        self.assertEqual(result["avg_rel_gap_percent"], "-")

    def test_report_displays_error_count_column_and_value(self):
        self.write_rows(10, [self.row(mip_status="Error")])
        tex_dir = self.root / "tables"
        generate_time_budget_report(self.root, [10], ["S1"], tex_dir=tex_dir)
        report = (tex_dir / "time_budget_comparison.tex").read_text(encoding="utf-8")
        self.assertIn("Errors", report)
        mip_line = next(line for line in report.splitlines() if "S1 / 10s & MIP &" in line)
        self.assertRegex(mip_line, r"& 0 & 1 &")

    def test_overrun_budget_tick_is_starred_and_explained(self):
        self.write_rows(10, [self.row(mip_ub="8", mip_lb="7", mip_time="10.1")])
        observed = {}
        fig = None

        def inspect_figure(_path):
            nonlocal fig
            fig = plot_results.plt.gcf()
            observed["ticks"] = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
            observed["xlabel"] = fig.axes[0].get_xlabel()

        try:
            with patch.object(plot_results, "save_figure", side_effect=inspect_figure):
                plot_results.plot_time_budget_comparison(self.root, [10], ["S1"])
            self.assertIn("10*", observed["ticks"])
            self.assertIn("* includes runtime overruns", observed["xlabel"])
        finally:
            if fig is not None:
                plot_results.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
