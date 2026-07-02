"""
gen_gap_summary.py

Generate exact-verifiable subset gap summary tables from scale experiment CSVs.

The table focuses on small/medium scales where exact solvers are more likely
to provide meaningful lower bounds or certified optimality.

Example:
    python gen_gap_summary.py --output_dir ./result/ --tag tl10
    python gen_gap_summary.py --output_dir ./result/ --tag tl10
    python gen_gap_summary.py --output_dir ./result/ --tag tl300
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from utils import (load_csv, resolve_csv_path,
                   to_float, mean_or_dash, count_status, SCALES)


DEFAULT_SCALES = SCALES


SOLVERS = [
    {
        "name": "CG-Benchmark",
        "ub": "pb_ub",
        "lb": "pb_lb",
        "gap": "pb_gap",
        "time": "pb_time",
        "status": "pb_status",
    },
    {
        "name": "CG-Benchmark+Mix",
        "ub": "pb_mix_ub",
        "lb": "pb_mix_lb",
        "gap": "pb_mix_gap",
        "time": "pb_mix_time",
        "status": "pb_mix_status",
    },
    {
        "name": "MIP",
        "ub": "mip_ub",
        "lb": "mip_lb",
        "gap": "mip_gap",
        "time": "mip_time",
        "status": "mip_status",
    },
    {
        "name": "MIP+Mix",
        "ub": "mip_mix_ub",
        "lb": "mip_mix_lb",
        "gap": "mip_mix_gap",
        "time": "mip_mix_time",
        "status": "mip_mix_status",
    },
]


def summarize_heuristic_gap(rows):
    """
    Compute optimality gap of VMPack+MixVM201Pro against certified MIP optima.

    Only instances where MIP proves optimality (mip_status == 'Optimal')
    are included. Returns (avg_gap, max_gap, hit_rate, avg_time_ms, n_certified).

    - n_certified: number of instances (out of 100) for which VanillaMIP
      certifies optimality. This is the true denominator and must match the
      Optimal count in Table exact_comparison's status vector.
    - hit_rate: "{hits}/{n_certified}" where hits is the number of certified
      instances on which the heuristic itself attains the optimum (NOT the
      number of certified instances).
    """
    gaps_pct = []
    hits = 0
    n_certified = 0
    times_ms = []

    for r in rows:
        if r.get("mip_status") != "Optimal":
            continue
        n_certified += 1
        h = to_float(r.get("heuristic_npms"))
        opt = to_float(r.get("mip_ub"))
        t = to_float(r.get("heuristic_time"))

        if h is None or opt is None or opt <= 0:
            continue

        gaps_pct.append((h - opt) / opt * 100.0)
        if h == opt:
            hits += 1
        if t is not None:
            times_ms.append(t * 1000.0)

    if not gaps_pct:
        return "-", "-", "-", "-", n_certified

    avg_gap = f"{np.mean(gaps_pct):.2f}"
    max_gap = f"{max(gaps_pct):.2f}"
    hit_rate = f"{hits}/{n_certified}"
    avg_time = f"{np.mean(times_ms):.2f}" if times_ms else "-"
    return avg_gap, max_gap, hit_rate, avg_time, n_certified


def summarize_solver(rows, solver):
    """
    Summarize one solver over one scale.
    """
    ubs = [to_float(r.get(solver["ub"])) for r in rows]
    lbs = [to_float(r.get(solver["lb"])) for r in rows]
    gaps = [to_float(r.get(solver["gap"])) for r in rows]
    times = [
        to_float(r.get(solver["time"])) * 1000
        for r in rows
        if to_float(r.get(solver["time"])) is not None
    ]

    # Absolute gap may be missing in older CSV files. Recompute when UB and LB exist.
    recomputed_gaps = []
    for r in rows:
        gap = to_float(r.get(solver["gap"]))
        ub = to_float(r.get(solver["ub"]))
        lb = to_float(r.get(solver["lb"]))
        if gap is not None:
            recomputed_gaps.append(gap)
        elif ub is not None and lb is not None:
            recomputed_gaps.append(max(0.0, ub - lb))

    relative_gaps = []
    for r in rows:
        ub = to_float(r.get(solver["ub"]))
        lb = to_float(r.get(solver["lb"]))
        if ub is not None and lb is not None and ub > 0:
            relative_gaps.append(max(0.0, (ub - lb) / ub))

    opt, feas, nosol, oom, other = count_status(rows, solver["status"])

    return {
        "avg_ub": mean_or_dash(ubs),
        "avg_lb": mean_or_dash(lbs),
        "avg_abs_gap": mean_or_dash(recomputed_gaps),
        "avg_rel_gap": mean_or_dash([g * 100 for g in relative_gaps], "{:.2f}"),
        "avg_time_ms": mean_or_dash(times),
        "status": f"{opt}/{feas}/{nosol}/{oom}",
        "other": other,
    }


def generate_latex_table(output_dir, tag, scales):
    """
    Print LaTeX table to stdout.
    """
    print("\n% ===== Exact-verifiable Gap Summary =====")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Gap summary on the exact-verifiable subset. "
          r"The table reports average upper bounds, lower bounds, absolute gaps, "
          r"relative gaps, runtimes, and solver status counts under the prescribed "
          r"time limit. Status is reported as Opt/Feas/NoSol/OOM.}")
    print(r"\label{tab:exact_gap_summary}")
    print(r"\fontsize{8}{10}\selectfont")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"\textbf{Scale} & \textbf{Solver} & \textbf{Avg. UB} & \textbf{Avg. LB} & "
          r"\textbf{Abs. Gap} & \textbf{Rel. Gap (\%)} & \textbf{Time (ms)} & \textbf{Status} \\")
    print(r"\midrule")

    for scale in scales:
        csv_path = resolve_csv_path(output_dir, scale, tag)
        if csv_path is None:
            print(f"% [SKIP] Missing CSV for {scale}, tag={tag}")
            continue

        rows = load_csv(csv_path)
        first = True

        for solver in SOLVERS:
            summary = summarize_solver(rows, solver)
            scale_cell = scale if first else ""
            first = False

            print(
                f"{scale_cell} & {solver['name']} & "
                f"{summary['avg_ub']} & {summary['avg_lb']} & "
                f"{summary['avg_abs_gap']} & {summary['avg_rel_gap']} & "
                f"{summary['avg_time_ms']} & {summary['status']} \\\\"
            )

        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def generate_heuristic_gap_table(output_dir, tag, scales):
    """
    Print Table: heuristic optimality gap on the exact-verifiable subset.

    Columns: Scale, Average optimality gap, Maximum optimality gap,
             Optimal-hit rate, Certified instances.
    Only instances with mip_status == 'Optimal' are included in the gap.
    """
    print("\n% ===== Heuristic Optimality Gap (Table 16) =====")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Optimality gaps of VMPack+MixVM201Pro on the exact-verifiable "
          r"subset, averaged only over instances where VanillaMIP certifies "
          r"optimality within the time limit. The \emph{Certified instances} "
          r"column gives the number of instances (out of 100) for which "
          r"VanillaMIP certifies optimality (matching the Optimal count in the "
          r"status vector of Table~\ref{tab:exact_comparison}). The "
          r"\emph{Optimal-hit rate} column reports, among those certified "
          r"instances, how many are also solved to optimality by "
          r"VMPack+MixVM201Pro.}")
    print(r"\label{tab:exact_gap_subset}")
    print(r"\fontsize{9}{12}\selectfont")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Scale} & \textbf{Average optimality gap} & "
          r"\textbf{Maximum optimality gap} & \textbf{Optimal-hit rate} & "
          r"\textbf{Certified instances} \\")
    print(r"\midrule")

    for scale in scales:
        csv_path = resolve_csv_path(output_dir, scale, tag)
        if csv_path is None:
            continue

        rows = load_csv(csv_path)
        avg_gap, max_gap, hit_rate, _, n_certified = summarize_heuristic_gap(rows)

        print(
            f"\t\t{scale} & {avg_gap}\\% & {max_gap}\\% & {hit_rate} & {n_certified}/100 \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def save_summary_csv(output_dir, tag, scales):
    """
    Save gap summary as CSV.
    """
    rows_out = []

    for scale in scales:
        csv_path = resolve_csv_path(output_dir, scale, tag)
        if csv_path is None:
            continue

        rows = load_csv(csv_path)

        for solver in SOLVERS:
            summary = summarize_solver(rows, solver)
            rows_out.append({
                "scale": scale,
                "solver": solver["name"].replace("\\&", "&"),
                "avg_ub": summary["avg_ub"],
                "avg_lb": summary["avg_lb"],
                "avg_abs_gap": summary["avg_abs_gap"],
                "avg_rel_gap_percent": summary["avg_rel_gap"],
                "avg_time_ms": summary["avg_time_ms"],
                "status_opt_feas_nosol_oom": summary["status"],
            })

    suffix = f"_{tag}" if tag else ""
    gap_dir = Path(output_dir) / 'gap'
    gap_dir.mkdir(parents=True, exist_ok=True)
    out_path = gap_dir / f"exact_gap_summary{suffix}.csv"

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "scale",
            "solver",
            "avg_ub",
            "avg_lb",
            "avg_abs_gap",
            "avg_rel_gap_percent",
            "avg_time_ms",
            "status_opt_feas_nosol_oom",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n[INFO] Gap summary CSV saved to: {out_path}")


def main():
    import io
    import contextlib

    parser = argparse.ArgumentParser(
        description="Generate exact-verifiable subset gap summary table."
    )
    parser.add_argument("--output_dir", type=str, default="./result/")
    parser.add_argument("--tag", type=str, default="",
                        help="CSV tag, e.g., tl5, tl10.")
    parser.add_argument("--scales", type=str, default="S1,S2,M1,M2,L1,L2",
                        help="Comma-separated scales for exact-verifiable subset.")
    parser.add_argument("--save_csv", action="store_true",
                        help="Save summary CSV in addition to printing LaTeX.")
    parser.add_argument("--tex_dir", type=str, default=None,
                        help="If set, write each LaTeX table to a .tex file in this directory "
                             "and only print a one-line confirmation per table. If unset, "
                             "tables are printed to stdout as before.")

    args = parser.parse_args()

    scales = [x.strip() for x in args.scales.split(",") if x.strip()]
    tex_dir = args.tex_dir
    if tex_dir:
        Path(tex_dir).mkdir(parents=True, exist_ok=True)

    def _run_table(fn, tex_name, *a, **kw):
        """Run one table function, capturing stdout to a .tex file if tex_dir set."""
        if tex_dir is None:
            fn(*a, **kw)
            return
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*a, **kw)
        suffix = f"_{args.tag}" if args.tag else ""
        out_path = Path(tex_dir) / f"{tex_name}{suffix}.tex"
        content = buf.getvalue()
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [TEX] {out_path.name}  ({len(content)} chars)")

    _run_table(generate_latex_table, 'gap_summary', args.output_dir, args.tag, scales)
    _run_table(generate_heuristic_gap_table, 'heuristic_gap', args.output_dir, args.tag, scales)

    if args.save_csv:
        save_summary_csv(args.output_dir, args.tag, scales)


if __name__ == "__main__":
    main()
