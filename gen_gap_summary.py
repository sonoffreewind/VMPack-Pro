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


def summarize_time_budgets(output_dir, time_limits, scales):
    """Read independent runs strictly: no fallback to another budget's CSV."""
    from collections import defaultdict
    grouped = defaultdict(list)
    expected = {}
    for scale in scales:
        for budget in time_limits:
            path = Path(output_dir) / 'scale' / f'{scale}_tl{budget:g}.csv'
            rows = load_csv(path)
            ids = [r['instance'] for r in rows]
            if not rows or len(set(ids)) != len(ids):
                raise ValueError(f'Empty or duplicate instance rows: {path}')
            identity = {(r['instance'], r.get('instance_hash', r.get('data_file', '')),
                         r.get('funcase'), r.get('seed')) for r in rows}
            if scale in expected and expected[scale] != identity:
                raise ValueError(f'Instance sets differ across budgets: {scale}')
            expected[scale] = identity
            if any(to_float(r.get('timelimit')) != budget for r in rows):
                raise ValueError(f'Budget does not match filename: {path}')
            grouped[scale, budget] = rows

    summaries = []
    for (scale, budget), rows in grouped.items():
        for solver in SOLVERS:
            prefix = solver['ub'][:-3]
            valid = [r for r in rows if _budget_bound(r.get(solver['ub']), positive=True)
                     is not None]
            ubs = [to_float(r[solver['ub']]) for r in valid]
            lbs, gaps = [], []
            certified = 0
            for r in rows:
                ub = _budget_bound(r.get(solver['ub']), positive=True)
                lb = _budget_bound(r.get(solver['lb']))
                if lb is not None:
                    lbs.append(lb)
                if ub is not None and lb is not None:
                    if lb > ub + 1e-6:
                        raise ValueError(f'LB exceeds UB: {scale}, {budget}, {prefix}')
                    gaps.append(100 * (ub - lb) / ub)
                    # Do not credit an optimum returned after the requested budget.
                    if (abs(ub - lb) <= 1e-6 and r.get(solver['status']) != 'Error'
                            and to_float(r.get(solver['time']), float('inf')) <= budget):
                        certified += 1
            # Compare initializations on the same instances, never different feasible subsets.
            base = prefix.removesuffix('_mix')
            paired = [(to_float(r.get(f'{base}_ub')), to_float(r.get(f'{base}_mix_ub')))
                      for r in rows if _budget_bound(r.get(f'{base}_ub'), True) is not None
                      and _budget_bound(r.get(f'{base}_mix_ub'), True) is not None]
            times = [to_float(r.get(solver['time'])) for r in rows]
            summaries.append({
                'scale': scale, 'budget_s': budget, 'solver': solver['name'],
                'n': len(rows), 'n_feasible': len(valid), 'n_certified': certified,
                'certified_percent': 100 * certified / len(rows),
                'n_gap': len(gaps), 'avg_ub': mean_or_dash(ubs), 'avg_lb': mean_or_dash(lbs),
                'avg_rel_gap_percent': mean_or_dash(gaps), 'avg_time_ms': mean_or_dash(
                    [t * 1000 for t in times if t is not None]),
                'n_over_budget': sum(t is not None and t > budget for t in times),
                'n_error': sum(r.get(solver['status']) == 'Error' for r in rows),
                'heuristic_npms': mean_or_dash([to_float(r.get('heuristic_npms')) for r in rows]),
                'heuristic_time_ms': mean_or_dash([to_float(r.get('heuristic_time')) * 1000
                                                 for r in rows if to_float(r.get('heuristic_time')) is not None]),
                'paired_n': len(paired),
                'paired_pm_gain': mean_or_dash([a - b for a, b in paired]),
            })
    return summaries


def _budget_bound(value, positive=False):
    value = to_float(value)
    if value is None or not np.isfinite(value) or abs(value) >= 1e90:
        return None
    valid = value > 0 if positive else value >= 0
    return value if valid else None


def generate_time_budget_report(output_dir, time_limits, scales, save_csv=False, tex_dir=None):
    summaries = summarize_time_budgets(output_dir, time_limits, scales)
    if save_csv:
        path = Path(output_dir) / 'gap' / 'time_budget_summary.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f'[CSV] {path}')
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Independent time-budget runs. Values describe the final returned state; overruns are counted separately. Gap is $(UB-LB)/UB$, excluding missing bounds. Certification is credited only within budget. NoMix and Mix use NoMixPack and MixVM201Pro initialization, respectively.}',
             r'\begin{tabular}{llrrrrrrrrr}', r'\toprule',
             r'Scale / Budget & Method & UB & LB & Gap (\%) & $n_{gap}$ & Feasible / $n$ & Certified / $n$ & Overruns & Errors & Time (ms) \\',
             r'\midrule']
    for r in summaries:
        lines.append(f"{r['scale']} / {r['budget_s']:g}s & {r['solver']} & {r['avg_ub']} & "
                     f"{r['avg_lb']} & {r['avg_rel_gap_percent']} & {r['n_gap']} & "
                     f"{r['n_feasible']}/{r['n']} & {r['n_certified']}/{r['n']} & "
                     f"{r['n_over_budget']} & {r['n_error']} & {r['avg_time_ms']} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    content = '\n'.join(lines) + '\n'
    if tex_dir:
        Path(tex_dir).mkdir(parents=True, exist_ok=True)
        path = Path(tex_dir) / 'time_budget_comparison.tex'
        path.write_text(content, encoding='utf-8')
        print(f'[TEX] {path}')
    else:
        print(content)
    return summaries


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
    parser.add_argument('--time_limits', default='',
                        help='Independent budgets, e.g. 10,60,300; read exact matching CSVs.')
    parser.add_argument("--tex_dir", type=str, default=None,
                        help="If set, write each LaTeX table to a .tex file in this directory "
                             "and only print a one-line confirmation per table. If unset, "
                             "tables are printed to stdout as before.")

    args = parser.parse_args()

    scales = [x.strip() for x in args.scales.split(",") if x.strip()]
    if args.time_limits:
        budgets = sorted(set(float(x) for x in args.time_limits.split(',')))
        if not budgets or any(not np.isfinite(x) or x <= 0 for x in budgets):
            parser.error('time_limits must be positive finite numbers')
        generate_time_budget_report(args.output_dir, budgets, scales, args.save_csv, args.tex_dir)
        return
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
