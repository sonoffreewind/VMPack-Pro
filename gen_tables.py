"""
gen_tables.py
Generate LaTeX tables from experiment CSV results.

  gen_method_comparison: Ours vs CG-Benchmark vs MIP across all scales
  gen_heuristic_empowerment: NoMix vs Mix impact on CG‑Benchmark and Gurobi
"""
import os
import csv
import numpy as np
from pathlib import Path

from utils import (load_csv, resolve_csv_path,
                   is_valid, to_float, to_int, SCALES)
from column_registry import COLUMN_REGISTRY, SCALE_UP


def _read_wide_csv(output_dir, fun_case, scale_name):
    """
    Read the new organized wide-format CSV for a synthetic scale.

    Path: {output_dir}/{fun_case}/random_{scale_name}.csv
    Returns list of dicts (one per instance), or [] if not found.
    """
    # Scale names: S1 → random_s1, S2 → random_s2, etc.
    scale_lower = scale_name.lower()
    path = os.path.join(output_dir, fun_case, f"random_{scale_lower}.csv")
    if os.path.exists(path):
        return load_csv(path)
    return []


def _read_wide_csv_l2(output_dir, fun_case):
    """
    Read the new organized wide-format CSV for L2 (UP=1000) scale.

    Path: {output_dir}/{fun_case}/random_l2.csv
    Returns list of dicts, or [] if not found.
    """
    path = os.path.join(output_dir, fun_case, "random_l2.csv")
    if os.path.exists(path):
        return load_csv(path)
    return []


def _wide_npms_list(rows, algo_prefix):
    """Extract npms values for a given algorithm from wide-format rows."""
    col = f"{algo_prefix}_npms"
    return [to_float(r.get(col)) for r in rows if to_float(r.get(col)) is not None]


def _wide_time_ms_list(rows, algo_prefix):
    """Extract time in ms for a given algorithm from wide-format rows."""
    col = f"{algo_prefix}_time"
    return [to_float(r.get(col)) * 1000 for r in rows if to_float(r.get(col)) is not None]


def _load_optimal_map(output_dir, fun_case='improvevmpack'):
    """
    Load the optimal values from unified CSV or organized wide CSVs.

    Priority:
      1. {fun_case}_unified.csv (old format, has optimal_npms column)
      2. {fun_case}/random_*.csv (new format, has MIP*_status + MIP*_npms columns)

    Returns dict: {(instance, UP): optimal_npms}
    """
    opt_map = {}

    # Try old unified CSV first
    path = os.path.join(output_dir, f"{fun_case}_unified.csv")
    if os.path.exists(path):
        rows = load_csv(path)
        for r in rows:
            opt = to_float(r.get('optimal_npms'))
            if opt is not None and opt > 0:
                inst = to_int(r.get('instance'))
                up = to_int(r.get('UP'))
                if inst is not None and up is not None:
                    opt_map[(inst, up)] = opt
        if opt_map:
            print(f"  [INFO] Loaded {len(opt_map)} optimal values from {fun_case}_unified.csv")
            return opt_map

    # Fallback: read from new organized CSVs (scan solver columns)
    solver_prefixes = []
    for algo in COLUMN_REGISTRY[fun_case]['algorithms']:
        if algo['type'] == 'solver':
            solver_prefixes.append(algo['prefix'])

    for scale_name, up in SCALE_UP.items():
        scale_lower = scale_name.lower()
        csv_path = os.path.join(output_dir, fun_case, f"random_{scale_lower}.csv")
        if not os.path.exists(csv_path):
            continue
        rows = load_csv(csv_path)
        for r in rows:
            inst = to_int(r.get('seq'), -1) - 1  # seq is 1-based → 0-based instance ID
            if inst < 0:
                continue
            # Check each solver for optimal status
            for prefix in solver_prefixes:
                status_col = f"{prefix}_status"
                npms_col = f"{prefix}_npms"
                if r.get(status_col) == 'Optimal':
                    opt_val = to_float(r.get(npms_col))
                    if opt_val is not None and opt_val > 0:
                        opt_map[(inst, up)] = opt_val
                        break

    if opt_map:
        print(f"  [INFO] Loaded {len(opt_map)} optimal values from {fun_case}/random_*.csv")
    return opt_map


def _gap_str(avg_npms, opt_val):
    """Format gap vs optimal as percentage string, or '--' if no optimal."""
    if opt_val is not None and opt_val > 0 and avg_npms > 0:
        gap = (avg_npms - opt_val) / opt_val * 100
        return f"{gap:.2f}\\%"
    return "--"


def _bool_from_csv(v):
    """Parse common CSV boolean representations."""
    return str(v).strip().lower() in ('true', '1', 'yes', 'y')


def _infer_timelimit(rows):
    vals = [to_float(r.get('timelimit')) for r in rows if to_float(r.get('timelimit')) is not None]
    if not vals:
        return None
    return vals[0] if all(abs(v - vals[0]) < 1e-9 for v in vals) else None

def _gurobi_status_label(rows, status_col='mip_status'):
    """Return Gurobi status counts as Opt/Feas/NoSol/OOM format."""
    opt = sum(1 for r in rows if r.get(status_col) == 'Optimal')
    feas = sum(1 for r in rows if r.get(status_col) == 'Feasible')
    nosol = sum(1 for r in rows if r.get(status_col) == 'NoSolution')
    oom = sum(1 for r in rows if 'OOM' in str(r.get(status_col, '')))
    return f"{opt}/{feas}/{nosol}/{oom}"


def _vanilla_mip_stats(rows):
    """Compute VanillaMIP statistics from a batch of rows.
    Returns (mip_mean_str, mip_time_str, gurobi_status).
    """
    mip_feas_rows = [r for r in rows if r.get('mip_status') in ('Optimal', 'Feasible')]
    mip_feas_npms = [float(r['mip_npms']) for r in mip_feas_rows
                     if r.get('mip_npms') not in ('', 'None', None)]
    mip_feas_times = [float(r['mip_time']) * 1000 for r in mip_feas_rows
                      if r.get('mip_time') not in ('', 'None', None)]
    mip_mean = np.mean(mip_feas_npms) if mip_feas_npms else '-'
    mip_time_mean = np.mean(mip_feas_times) if mip_feas_times else '-'

    gurobi_status = _gurobi_status_label(rows, 'mip_status')

    mip_mean_str = f"{mip_mean:.2f}" if isinstance(mip_mean, float) else mip_mean
    mip_time_str = f"{mip_time_mean:.2f}" if isinstance(mip_time_mean, float) else mip_time_mean
    return mip_mean_str, mip_time_str, gurobi_status


def gen_method_comparison(output_dir, tag=''):
    """
    Method comparison across all scales (Ours vs CG‑Benchmark vs MIP).
    One unified table with S1, S2, M1, M2, L1, L2 rows.
    """
    print("\n% ===== Method Comparison =====")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Time-limited comparison between VMPack+MixVM201Pro and exact solvers across synthetic bottleneck instance scales. Exact-solver PM counts denote certified optima only when optimality is proven within the 1-second limit; otherwise, they denote the best incumbent solutions found within the limit. CG-Benchmark denotes the column-generation-based pattern benchmark defined in Section~\ref{sec:pattern_benchmark}. The table is used both to calibrate heuristic quality on fully certified scales and to show the certification burden of exact optimization on larger scales. The VanillaMIP status vector is reported as [Optimal / Feasible / NoSol / OOM].}")
    print(r"\label{tab:exact_comparison}")
    print(r"\fontsize{9}{12}\selectfont")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{cccccccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Scale}} & \multicolumn{3}{c}{\textbf{\#PMs}} & \multicolumn{3}{c}{\textbf{Time(ms)}} & \multirow{2}{*}{\textbf{\shortstack{VanillaMIP Status \\ (O/F/N/OOM)}}} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"& \textbf{\shortstack{VMPack+\\MixVM201Pro}} & \textbf{CG-Benchmark} & \textbf{VanillaMIP} & \textbf{\shortstack{VMPack+\\MixVM201Pro}} & \textbf{CG-Benchmark} & \textbf{VanillaMIP} & \\")
    print(r"\midrule")

    scales = SCALES
    for scale in scales:
        csv_path = resolve_csv_path(output_dir, scale, tag)
        if csv_path is None or not os.path.exists(csv_path):
            continue

        rows = load_csv(csv_path)

        # #PMs
        h_npms = [float(r['heuristic_npms']) for r in rows]
        pb_ubs = [to_float(r.get('pb_ub')) for r in rows if to_float(r.get('pb_ub')) is not None]

        # Time (ms)
        h_times_ms = [float(r['heuristic_time']) * 1000 for r in rows]
        pb_times_ms = [float(r['pb_time']) * 1000 for r in rows]

        # MIP stats
        mip_mean_str, mip_time_str, gurobi_status = _vanilla_mip_stats(rows)

        h_mean = np.mean(h_npms)
        pb_mean = np.mean(pb_ubs)
        h_time = np.mean(h_times_ms)
        pb_time = np.mean(pb_times_ms)

        print(f"\t\t{scale} & {h_mean:.2f} & {pb_mean:.2f} & {mip_mean_str} & "
              f"{h_time:.2f} & {pb_time:.2f} & {mip_time_str} & "
              f"{gurobi_status} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table}")


def gen_heuristic_empowerment(output_dir, tag=''):
    """
    Heuristic Empowerment on Exact Solvers.

    This table quantifies how much benefit a heuristic warm start provides to the
    exact solvers.  Specifically, it contrasts the number of bins, runtime and
    generated columns for the column‑generation‑based benchmark (abbreviated as
    CG‑Benchmark) under two initialization schemes: the baseline NoMix initialization and
    the MixVM201Pro initialization.  For the assignment‑based VanillaMIP solver it
    reports the change in runtime and status counts.  Both solvers are run
    under identical time limits, and the differences are computed as
    “Mix – NoMix”.
    """
    print("\n% ===== Heuristic Empowerment on Exact Solvers =====")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Impact of MixVM201Pro warm starts on time-limited CG-Benchmark and "
          r"VanillaMIP solver performance under a 1-second time limit, matching the "
          r"exact-solver comparison in Table~\ref{tab:exact_comparison}. Here CG-Benchmark "
          r"denotes the column-generation-based pattern benchmark, and CG-Benchmark+Mix "
          r"initializes its pattern set with MixVM201Pro configurations. The status vector "
          r"is reported as [Optimal / Feasible / NoSol / OOM].}")

    print(r"\label{tab:heuristic_lift}")
    print(r"\fontsize{8}{11}\selectfont")
    print(r"\begin{tabular}{cccc|cc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Scale}} & \multicolumn{3}{c|}{\textbf{CG-Benchmark (NoMix $\rightarrow$ Mix)}} & \multicolumn{2}{c}{\textbf{VanillaMIP (NoMix $\rightarrow$ Mix)}} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}")
    print(r"& \textbf{Bins} & \textbf{Time (ms)} & \textbf{Cols} & \textbf{Time (ms)} & \textbf{Status} \\")
    print(r"\midrule")

    scales = SCALES
    for scale in scales:
        csv_path = resolve_csv_path(output_dir, scale, tag)
        if csv_path is None or not os.path.exists(csv_path):
            continue

        rows = load_csv(csv_path)

        # CG‑Benchmark: Bins (upper bound on PM count)
        pb_ubs = [to_float(r.get('pb_ub')) for r in rows if to_float(r.get('pb_ub')) is not None]
        pb_mix_ubs = [to_float(r.get('pb_mix_ub')) for r in rows if to_float(r.get('pb_mix_ub')) is not None]

        # CG‑Benchmark: Time (ms)
        pb_times_ms = [to_float(r.get('pb_time')) * 1000 for r in rows if to_float(r.get('pb_time')) is not None]
        pb_mix_times_ms = [to_float(r.get('pb_mix_time')) * 1000 for r in rows if to_float(r.get('pb_mix_time')) is not None]

        # CG‑Benchmark: Columns (n_cols)
        pb_n_cols = [int(to_float(r.get('pb_n_cols'))) for r in rows if to_float(r.get('pb_n_cols')) is not None]
        pb_mix_n_cols = [int(to_float(r.get('pb_mix_n_cols'))) for r in rows if to_float(r.get('pb_mix_n_cols')) is not None]

        # Gurobi: Time (ms) — only Optimal + Feasible
        mip_feas_rows = [r for r in rows if r.get('mip_status') in ('Optimal', 'Feasible')]
        mip_times_ms = [to_float(r.get('mip_time')) * 1000 for r in mip_feas_rows if to_float(r.get('mip_time')) is not None]
        mip_mix_feas_rows = [r for r in rows if r.get('mip_mix_status') in ('Optimal', 'Feasible')]
        mip_mix_times_ms = [to_float(r.get('mip_mix_time')) * 1000 for r in mip_mix_feas_rows if to_float(r.get('mip_mix_time')) is not None]

        # Format PB columns
        pb_bins_str = f"{np.mean(pb_ubs):.2f} $\\rightarrow$ {np.mean(pb_mix_ubs):.2f}" if pb_ubs and pb_mix_ubs else "---"
        pb_time_str = f"{np.mean(pb_times_ms):.2f} $\\rightarrow$ {np.mean(pb_mix_times_ms):.2f}" if pb_times_ms and pb_mix_times_ms else "---"
        pb_cols_str = f"{np.mean(pb_n_cols):.0f} $\\rightarrow$ {np.mean(pb_mix_n_cols):.0f}" if pb_n_cols and pb_mix_n_cols else "---"

        # Format GU columns
        mip_time_str = f"{np.mean(mip_times_ms):.2f}" if mip_times_ms else "---"
        mip_mix_time_str = f"{np.mean(mip_mix_times_ms):.2f}" if mip_mix_times_ms else "---"
        gu_time_str = f"{mip_time_str} $\\rightarrow$ {mip_mix_time_str}"

        # GU Status
        gu_nomix_status = _gurobi_status_label(rows, 'mip_status')
        gu_mix_status = _gurobi_status_label(rows, 'mip_mix_status')
        gu_status_str = f"{gu_nomix_status} $\\rightarrow$ {gu_mix_status}"

        print(f"\t\t{scale} & {pb_bins_str} & {pb_time_str} & {pb_cols_str} & {gu_time_str} & {gu_status_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def gen_scale_sweep_table(output_dir="./result/", tag=""):
    """
    Generate Table X: scale sweep (UP sweep) across S1→L2 for mixalgos.

    Shows how average PM count, Gap vs Optimal (%), and running time scale
    with instance size. Falls back to ρ when optimal is not proven.
    Includes the MixVM201Pro vs MixVM301 reduction percentage.
    """
    up_scales = [
        ("S1", 10), ("S2", 20), ("M1", 50),
        ("M2", 100), ("L1", 500), ("L2", 1000),
    ]

    # Expected VMs: E[n] = 7*(UP+1) for two-class setting
    e_n = {10: 77, 20: 147, 50: 357, 100: 707, 500: 3507, 1000: 7007}

    # Algorithms to report (with their column registry prefixes)
    report_algos = [
        ('NoMixPack', 'NoMixPack'),
        ('MixVM301', 'MixVM301'),
        ('MixVM201', 'MixVM201'),
        # ('MixVM201Priority', 'MixVM201Priority'),
        ('MixVM201Pro', 'MixVM201Pro'),
        ('MixPack', 'MixPack'),
        # ('SafeMix', 'SafeMix'),
        ('BFD', 'BFD'),
    ]

    # Load optimal values from unified CSV
    opt_map = _load_optimal_map(output_dir, fun_case='mixalgos')

    print("\n% ===== Scale Sweep Performance (Table X) =====")
    print(r"\begin{table*}[htbp]")
    print(r"\centering")
    print(r"\caption{Performance of standalone mixed-packing heuristics across "
          r"instance scales on two-class synthetic bottleneck instances. "
          r"$\overline{A(L)}$ denotes the average PM count, "
          r"$\overline{Gap}_{\mathrm{Opt}}$ the average optimality gap (\%) "
          r"against certified optima on the exact-verifiable scales (averaged only over "
          r"certified instances within the 1-second time limit; denominators in "
          r"Table~\ref{tab:exact_gap_subset}), and $\overline{T}$ the average running "
          r"time in milliseconds. The last column reports the reduction in average PM "
          r"count achieved by MixVM201Pro relative to MixVM301. As in "
          r"Table~\ref{tab:heuristic_mixalgos}, the certified optima are proven by "
          r"branch-and-bound and are independent of the warm-start source.}")
    print(r"\label{tab:scale_sweep}")
    print(r"\fontsize{8}{11}\selectfont")
    print(r"\setlength{\tabcolsep}{3pt}")

    # Build column spec dynamically: Scale | E[n] | for each algo: A(L) | Gap/Opt or ρ | T(ms) | Δ%
    n_algos = len(report_algos)
    col_spec = "lr" + "rrr" * n_algos + "r"  # +1 for Δ% column
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{" + col_spec + r"}")
    print(r"\toprule")

    # Header row 1
    header1 = r"\multirow{2}{*}{\textbf{Scale}} & \multirow{2}{*}{\textbf{E[n]}}"
    for algo_name, _ in report_algos:
        display = algo_name.replace('VMPack_', 'VMPack+')
        header1 += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{display}}}}}"
    header1 += r" & \multirow{2}{*}{\textbf{$\Delta_{\texttt{Pro}}$}} \\"
    print(header1)

    # Header row 2
    header2 = r" & & "
    for i in range(n_algos):
        header2 += r"$\overline{A(L)}$ & $\overline{\textbf{Gap}}_{\textbf{Opt}}$ & $\overline{T}$ (ms)"
        if i < n_algos - 1:
            header2 += " & "
    header2 += r" & \\"
    print(header2)

    print(r"\midrule")

    for scale_name, up in up_scales:
        # Read new wide-format CSV
        rows = _read_wide_csv(output_dir, 'mixalgos', scale_name)
        if not rows:
            print(f"  % [SKIP] {scale_name} not found")
            continue

        line = f"{scale_name} & {e_n[up]}"

        # Store values for Δ% calculation
        v301_npms = None
        v201pro_npms = None

        for algo_name, prefix in report_algos:
            npms_list = _wide_npms_list(rows, prefix)
            times_ms = _wide_time_ms_list(rows, prefix)

            if not npms_list:
                line += " & --- & --- & ---"
                continue

            avg_npms = np.mean(npms_list)
            avg_time = np.mean(times_ms)

            # Compute Gap vs Optimal (fallback to ρ when optimal not proven)
            gaps = []
            for r in rows:
                inst = to_int(r.get('seq'), 1) - 1  # seq is 1-based
                opt_val = opt_map.get((inst, up))
                h_npms = to_float(r.get(f'{prefix}_npms'))
                if opt_val is not None and opt_val > 0 and h_npms is not None and h_npms > 0:
                    gap_pct = (h_npms - opt_val) / opt_val * 100
                    gaps.append(gap_pct)
                else:
                    # Fallback: use ρ (npms/lb)
                    lb = to_float(r.get('lb'))
                    if lb is not None and lb > 0 and h_npms is not None and h_npms > 0:
                        gaps.append((h_npms / lb - 1) * 100)
                    else:
                        gaps.append(0)

            avg_gap = np.mean(gaps) if gaps else 0

            line += f" & {avg_npms:.2f} & {avg_gap:.2f} & {avg_time:.2f}"

            if algo_name == 'MixVM301':
                v301_npms = avg_npms
            elif algo_name == 'MixVM201Pro':
                v201pro_npms = avg_npms

        # Add Δ% column (MixVM201Pro vs MixVM301)
        if v301_npms is not None and v201pro_npms is not None and v301_npms > 0:
            reduction = (v301_npms - v201pro_npms) / v301_npms * 100
            line += f" & ${reduction:.2f}\\%$"
        else:
            line += " & ---"

        line += r" \\"
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")


def gen_runtime_scaling_table(output_dir="./result/", tag=""):
    """
    Generate Table Y: runtime scaling across S1→L2 for mixalgos.

    Shows how average running time scales with instance size for each heuristic.
    """
    up_scales = [
        ("S1", 10), ("S2", 20), ("M1", 50),
        ("M2", 100), ("L1", 500), ("L2", 1000),
    ]

    e_n = {10: 77, 20: 147, 50: 357, 100: 707, 500: 3507, 1000: 7007}

    report_algos = [
        ('NoMixPack', 'NoMixPack'),
        ('MixVM301', 'MixVM301'),
        ('MixVM201', 'MixVM201'),
        ('MixVM201Priority', 'MixVM201Priority'),
        ('MixVM201Pro', 'MixVM201Pro'),
        ('MixPack', 'MixPack'),
        ('SafeMix', 'SafeMix'),
        ('BFD', 'BFD'),
        ('FFD', 'FFD'),
    ]

    print("\n% ===== Runtime Scaling (Table Y) =====")
    print(r"\begin{table*}[htbp]")
    print(r"\centering")
    print(r"\caption{Average running time (ms) across instance scales for "
          r"standalone mixed-packing heuristics on two-class synthetic bottleneck instances. "
          r"The expected number of VMs $E[n]$ is shown for reference. "
          r"All times are below 100 ms even on the largest scale, "
          r"confirming the low computational overhead of the proposed methods.}")
    print(r"\label{tab:runtime_scaling}")
    print(r"\fontsize{8}{11}\selectfont")
    print(r"\setlength{\tabcolsep}{3pt}")

    n_algos = len(report_algos)
    col_spec = "lrr" + "r" * n_algos
    print(r"\begin{tabular}{" + col_spec + r"}")
    print(r"\toprule")
    print(r"\textbf{Scale} & \textbf{E[n]}", end="")
    for algo_name, _ in report_algos:
        display = algo_name.replace('VMPack_', 'VMPack+')
        print(f" & \\textbf{{{display}}}", end="")
    print(r" \\")
    print(r"\midrule")

    for scale_name, up in up_scales:
        rows = _read_wide_csv(output_dir, 'mixalgos', scale_name)
        if not rows:
            print(f"  % [SKIP] {scale_name} not found")
            continue

        line = f"{scale_name} & {e_n[up]}"

        for algo_name, prefix in report_algos:
            times_ms = _wide_time_ms_list(rows, prefix)
            if not times_ms:
                line += " & ---"
                continue
            avg_time = np.mean(times_ms)
            line += f" & {avg_time:.2f}"

        line += r" \\"
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def gen_heuristic_comparison(output_dir, tag=''):
    """
    Generate LaTeX table for heuristic comparison experiments.
    Reads random_l2.csv from organized directories, produces a combined table
    with columns: Algorithm, Avg #PMs, Avg Time, Avg ρ, Max ρ, η_CPU, η_MEM.

    The optimality gap (Gap_Opt) is NOT included here: it belongs to the
    exact-verifiable subset (different denominator) and is reported separately
    in Table exact_gap_subset (see gen_gap_summary.py). Mixing it into this
    L2 main table conflates instance sets (opinion 5, round 6).
    """
    configs = {
        'mixalgos': {
            'label': 'Base Heuristics (L0/L2)',
            'algos': [
                ('NoMixPack', 'NoMixPack'),
                ('MixVM301', 'MixVM301'),
                ('MixVM201', 'MixVM201'),
                ('MixVM201Priority', 'MixVM201Priority'),
                ('MixVM201Pro', 'MixVM201Pro'),
                ('MixPack', 'MixPack'),
                ('SafeMix', 'SafeMix'),
                ('BFD', 'BFD'),
                ('FFD', 'FFD'),
            ],
        },
        'improvevmpack': {
            'label': 'VMPack Variants (L0/L1/L2)',
            'algos': [
                ('VMPack_NoMixPack', 'VMPack_NoMixPack'),
                ('VMPack_MixVM301', 'VMPack_MixVM301'),
                ('VMPack_MixVM201', 'VMPack_MixVM201'),
                ('VMPack_MixVM201Priority', 'VMPack_MixVM201Priority'),
                ('VMPack_MixVM201Pro', 'VMPack_MixVM201Pro'),
                ('VMPack_MixPack', 'VMPack_MixPack'),
                ('VMPack_SafeMix', 'VMPack_SafeMix'),
                ('BFD', 'BFD'),
                ('FFD', 'FFD'),
            ],
        },
    }

    for cfg_name, cfg in configs.items():
        # Read from new organized directory: {fun_case}/random_l2.csv
        rows = _read_wide_csv_l2(output_dir, cfg_name)
        if not rows:
            print(f"  [SKIP] {cfg_name}/random_l2.csv not found")
            continue

        print(f"\n% ===== Heuristic Comparison: {cfg_name} =====")
        print(r"\begin{table}[htbp]")
        print(r"\centering")
        if cfg_name == 'mixalgos':
            print(r"\caption{Standalone heuristic performance on L2-scale synthetic "
                  r"bottleneck instances with $C_1=0$ and $C_0<3C_2$. Optimality gaps "
                  r"against certified optima are reported separately in "
                  r"Table~\ref{tab:exact_gap_subset}.}")
        else:
            print(r"\caption{Performance of VMPack variants after replacing the "
                  r"bottleneck-stage mixed-packing strategy on L2-scale synthetic "
                  r"instances. Optimality gaps against certified optima are reported "
                  r"separately in Table~\ref{tab:exact_gap_subset}.}")
        print(rf"\label{{tab:heuristic_{cfg_name}}}")
        print(r"\fontsize{9}{12}\selectfont")
        print(r"\begin{tabular}{lcccccc}")
        print(r"\toprule")
        print(r"Heuristic & $\overline{A(L)}$ & $\overline{T}$ (ms) & $\overline{\rho}$ & $\max \rho$ & $\eta_{\mathit{CPU}}$ & $\eta_{\mathit{MEM}}$ \\")
        print(r"\midrule")

        for algo_name, prefix in cfg['algos']:
            npms_list = _wide_npms_list(rows, prefix)
            times_ms = _wide_time_ms_list(rows, prefix)

            if not npms_list:
                continue

            lbs = [to_float(r.get('lb')) for r in rows if to_float(r.get('lb')) is not None]
            rhos = [npms_list[j] / lbs[j] if lbs[j] > 0 else 0 for j in range(min(len(npms_list), len(lbs)))]

            avg_npms = np.mean(npms_list)
            std_npms = np.std(npms_list, ddof=1)  # sample std
            avg_time = np.mean(times_ms)
            avg_rho = np.mean(rhos) if rhos else 0
            max_rho = max(rhos) if rhos else 0

            # CPU/MEM utilization
            total_cpu = sum(to_float(r.get('total_cpu'), 0) for r in rows)
            total_mem = sum(to_float(r.get('total_mem'), 0) for r in rows)
            total_bins = sum(npms_list)

            # C = 2^(T+1) = 256
            C = 256

            cpu_util = total_cpu / (total_bins * C) if total_bins > 0 else 0
            mem_util = total_mem / (total_bins * 2 * C) if total_bins > 0 else 0

            # Display name: VMPack_NoMixPack -> VMPack+NoMixPack (matches paper
            # body text and avoids a bare underscore in LaTeX text mode).
            display_name = algo_name.replace('VMPack_', 'VMPack+')
            print(f"\t\t{display_name} & {avg_npms:.2f} ($\\pm${std_npms:.2f}) & {avg_time:.2f} & {avg_rho:.3f} & "
                  f"{max_rho:.3f} & {cpu_util*100:.2f}\\% & {mem_util*100:.2f}\\% \\\\")

        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

def _summarize_huawei_trace(trace_dir):
    """
    Summarize Huawei active-set VMPack export results for BOTH scenarios.

    Returns a list of two dicts (mixalgos + improvevmpack). Each dict carries:
        trace, variant, raw_requests, retained_requests, coverage,
        generated_batches, bottleneck_batches

    Bottleneck semantics:
      - mixalgos:      L1 VMs discarded -> C1=0 by construction -> all bottleneck.
                       Count via the summary's `valid` flag.
      - improvevmpack: L1 VMs retained  -> C1>0 in general -> non-bottleneck.
                       The summary has no C0/C1/C2 columns, so we report
                       bottleneck_batches = 0 (every three-class snapshot has
                       active L1 VMs in the Huawei trace).
    """
    raw_requests = 125430
    retained_requests = 124771
    coverage = retained_requests / raw_requests * 100 if raw_requests > 0 else 0.0

    summaries = []
    for scenario in ["mixalgos", "improvevmpack"]:
        path = os.path.join(trace_dir, f"huawei_vmpack_instances_{scenario}_summary.csv")
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        generated_batches = len(rows) if rows else 0
        if scenario == "mixalgos":
            # L1 filtered out -> C1=0 by construction -> all batches bottleneck.
            bottleneck_batches = sum(
                1 for r in rows if _bool_from_csv(r.get("valid", "True"))
            ) if rows else 0
        else:
            # Three-class snapshots retain L1 VMs -> C1>0 -> non-bottleneck.
            # Huawei summary has no C0/C1/C2, so we cannot recompute per-row;
            # report 0 (all three-class batches are non-bottleneck by construction).
            bottleneck_batches = 0

        summaries.append({
            "trace": "Huawei",
            "variant": scenario,
            "raw_requests": raw_requests,
            "retained_requests": retained_requests,
            "coverage": coverage,
            "generated_batches": generated_batches,
            "bottleneck_batches": bottleneck_batches,
        })
    return summaries


def _summarize_microsoft_trace(trace_dir):
    """
    Summarize Microsoft vmtable VMPack instance generation for BOTH scenarios.

    Returns a list of two dicts (mixalgos + improvevmpack). Each dict carries:
        trace, variant, raw_requests, retained_requests, coverage,
        generated_batches, bottleneck_batches

    Bottleneck semantics:
      - mixalgos:      L1 VMs discarded -> C1=0 by construction -> bottleneck
                       counted via the summary's `mixalgos_bottleneck` flag.
      - improvevmpack: L1 VMs retained  -> non-bottleneck by construction
                       (reported as 0; the `mixalgos_bottleneck` flag is only
                       meaningful for the two-class variant).
    """
    # Read raw/retained from analysis report (improvevmpack scenario)
    report_path = os.path.join(trace_dir, "microsoft_vmtable_analysis_report_improvevmpack.txt")
    if not os.path.exists(report_path):
        report_path = os.path.join(trace_dir, "microsoft_vmtable_analysis_report.txt")
    raw_requests = None
    retained_requests = None
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Total VM rows:"):
                    raw_requests = int(line.split(":")[1].strip())
                elif line.startswith("Retained VM rows:"):
                    retained_requests = int(line.split(":")[1].strip())

    summaries = []
    for scenario in ["mixalgos", "improvevmpack"]:
        path = os.path.join(trace_dir, f"microsoft_vmpack_instances_{scenario}_summary.csv")
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        generated_batches = len(rows) if rows else 0
        if scenario == "mixalgos":
            bottleneck_batches = sum(
                1 for r in rows if _bool_from_csv(r.get("mixalgos_bottleneck"))
            ) if rows else 0
        else:
            # Three-class batches retain L1 VMs -> C1>0 -> non-bottleneck.
            bottleneck_batches = 0

        # raw/retained are the same across scenarios (same trace); fill if missing.
        rr = raw_requests
        rtr = retained_requests
        if rr is None or rtr is None:
            if rows:
                rtr = sum(to_int(r.get("num_vms")) for r in rows) if rtr is None else rtr
            else:
                rtr = 0 if rtr is None else rtr
            rr = rtr if rr is None else rr

        coverage = rtr / rr * 100 if rr > 0 else 0.0
        summaries.append({
            "trace": "Microsoft",
            "variant": scenario,
            "raw_requests": rr,
            "retained_requests": rtr,
            "coverage": coverage,
            "generated_batches": generated_batches,
            "bottleneck_batches": bottleneck_batches,
        })
    return summaries


def gen_public_trace_summary_table(huawei_dir="./huawei_trace_output/",
                                   microsoft_dir="./microsoft_vmtable_output/"):
    """
    Generate Table: public trace filtering and instance generation summary.

    Reports BOTH the two-class bottleneck variant (mixalgos, L1 discarded,
    C1=0 by construction) and the three-class general variant (improvevmpack,
    L1 retained). Each trace contributes two rows.
    """
    summaries = []

    for s in _summarize_huawei_trace(huawei_dir):
        summaries.append(s)
    for s in _summarize_microsoft_trace(microsoft_dir):
        summaries.append(s)

    if not summaries:
        print("\n% [SKIP] Table 8: no public trace summary files found.")
        return

    print("\n% ===== Public Trace Filtering Summary =====")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Filtering statistics for constructed trace-derived variants. "
          r"The \emph{mixalgos} variant discards $L_1$ VMs, so $C_1=0$ by "
          r"construction and all resulting batches are bottleneck batches by design; "
          r"the \emph{improvevmpack} variant keeps $L_0,L_1,L_2$ VMs and generally "
          r"does not satisfy the bottleneck condition. Coverage is computed as "
          r"retained requests divided by raw requests and measures only the "
          r"finite-type dyadic filtering rate, not the natural frequency of "
          r"bottleneck windows.}")
    print(r"\label{tab:trace_filtering}")
    print(r"\fontsize{8}{10}\selectfont")
    print(r"\begin{tabular}{llrrrrr}")
    print(r"\toprule")
    print(r"\textbf{Trace} & \textbf{Variant} & \textbf{Raw requests} & "
          r"\textbf{Retained} & \textbf{Coverage} & \textbf{Batches} & "
          r"\textbf{Bottleneck by design} \\")
    print(r"\midrule")

    for s in summaries:
        variant_label = "Two-class (bottleneck)" if s["variant"] == "mixalgos" \
                        else "Three-class (general)"
        bottleneck = s["bottleneck_batches"]
        print(
            f"{s['trace']} & {variant_label} & "
            f"{s['raw_requests']} & "
            f"{s['retained_requests']} & "
            f"{s['coverage']:.2f}\\% & "
            f"{s['generated_batches']} & "
            f"{bottleneck} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def _load_trace_wtl(output_dir, trace_name, tag):
    """
    Load win/tie/loss CSV generated by run_trace_experiments.py.

    Expected names:
        huawei_trace_wtl_huawei.csv
        microsoft_trace_wtl_microsoft.csv

    If tag is provided:
        huawei_trace_wtl_<tag>.csv
    """
    trace_lower = trace_name.lower()

    candidates = []
    # Search trace/ subdirectory first, then root
    for base_dir in [os.path.join(output_dir, 'trace'), output_dir]:
        if tag:
            candidates.append(os.path.join(base_dir, f"{trace_lower}_trace_wtl_{tag}.csv"))
        candidates.append(os.path.join(base_dir, f"{trace_lower}_trace_wtl_{trace_lower}.csv"))
        candidates.append(os.path.join(base_dir, f"{trace_lower}_trace_wtl.csv"))

    for path in candidates:
        if os.path.exists(path):
            return load_csv(path)

    return []


def _print_trace_detail_panel(rows, trace_label, trace_algos, panel_label=""):
    """Print one panel (5 rows) of the trace detail table."""
    print(f"\t\t\\multirow{{5}}{{*}}{{{trace_label} trace}}")

    for algo in trace_algos:
        algo_rows = [r for r in rows if r['algorithm'] == algo]
        if not algo_rows:
            continue

        n = len(algo_rows)
        npms_list = [float(r['npms']) for r in algo_rows]
        times_ms = [float(r['time']) * 1000 for r in algo_rows]
        lbs = [float(r['lb']) for r in algo_rows]
        rhos = [npms_list[j] / lbs[j] if lbs[j] > 0 else 0 for j in range(n)]

        avg_npms = np.mean(npms_list)
        std_npms = np.std(npms_list, ddof=1)
        avg_time = np.mean(times_ms)
        avg_rho = np.mean(rhos)
        max_rho = max(rhos)

        # CPU/MEM utilization
        total_cpu = sum(float(r['total_cpu']) for r in algo_rows)
        total_mem = sum(float(r['total_mem']) for r in algo_rows)
        total_bins = sum(npms_list)

        first_row = algo_rows[0]
        if is_valid(first_row.get('C')):
            C = int(float(first_row['C']))
        else:
            C = 256
        M = int(float(first_row.get('M', 512)))

        eta_cpu = total_cpu / (total_bins * C) * 100 if total_bins * C > 0 else 0
        eta_mem = total_mem / (total_bins * M) * 100 if total_bins * M > 0 else 0

        # Display name
        display_name = algo.replace('VMPack_', 'VMPack+')

        print(f"\t\t& {display_name}"
              f" & {avg_npms:.2f} ($\\pm${std_npms:.2f}) & {avg_time:.2f}"
              f" & {avg_rho:.3f} & {max_rho:.3f}"
              f" & {eta_cpu:.2f}\\% & {eta_mem:.2f}\\% \\\\")


def gen_trace_detail_table(output_dir="./result/", tag=""):
    """
    Generate Table 13: per-trace performance of VMPack variants and baselines.

    Produces a two-panel table:
      Panel (a): bottleneck-filtered instances (C1=0, C0<3C2)
      Panel (b): full three-class instances (no bottleneck filter)

    Reads *_trace_detail_*.csv files for both scenarios.
    """
    trace_names = [("Huawei", "huawei"), ("Microsoft", "microsoft")]

    # Algorithms in the order they appear in the article table
    trace_algos = ['FFD', 'BFD', 'VMPack_MixVM301', 'VMPack_MixVM201Pro', 'VMPack_MixPack']

    scenarios = [
        ("Two-class bottleneck variant ($L_0$+$L_2$ only, $C_1=0$ by construction)", ""),
        ("Three-class general variant ($L_0$+$L_1$+$L_2$, non-bottleneck)", "_improvevmpack"),
    ]

    print("\n% ===== Trace Detail Performance (Table 13) =====")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Performance of VMPack variants and engineering baselines "
          r"on constructed trace-derived variants. "
          r"Panel (a) reports results on the two-class bottleneck variant "
          r"($L_0$+$L_2$ only, $C_1=0$ by construction); panel (b) reports results "
          r"on the three-class general variant ($L_0$+$L_1$+$L_2$, generally "
          r"non-bottleneck). The two panels are used for different purposes: panel (a) "
          r"tests the bottleneck mechanism after projecting trace data into the studied "
          r"two-class regime, whereas panel (b) checks the full VMPack pipeline on "
          r"general trace-derived batches.}")
    print(r"\label{tab:trace_results}")
    print(r"\fontsize{8}{11}\selectfont")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"\textbf{Trace source} & \textbf{Algorithm}"
          r" & $\overline{A(L)}$ & $\overline{T}$ (ms)"
          r" & $\overline{\rho}_{\mathrm{LB}}$ & $\max \rho_{\mathrm{LB}}$"
          r" & $\eta_{\mathit{CPU}}$ & $\eta_{\mathit{MEM}}$ \\")
    print(r"\midrule")

    for panel_idx, (scenario_label, scenario_suffix) in enumerate(scenarios):
        panel_letter = chr(ord('a') + panel_idx)
        print(f"\t\\multicolumn{{8}}{{l}}{{\\textbf{{({panel_letter}) {scenario_label}}}}} \\\\")
        print(r"\midrule")

        for trace_idx, (trace_label, trace_key) in enumerate(trace_names):
            if scenario_suffix:
                fname = f"{trace_key}_trace_detail_{trace_key}{scenario_suffix}.csv"
            else:
                fname = f"{trace_key}_trace_detail_{trace_key}.csv"

            # Search trace/ subdirectory first, then root
            detail_path = os.path.join(output_dir, 'trace', fname)
            if not os.path.exists(detail_path):
                detail_path = os.path.join(output_dir, fname)

            if not os.path.exists(detail_path):
                print(f"  % [SKIP] {fname} not found")
                continue

            rows = load_csv(detail_path)
            _print_trace_detail_panel(rows, trace_label, trace_algos)

            # Print midrule between traces and between panels (but not after the last row)
            is_last = (trace_idx == len(trace_names) - 1) and (panel_idx == len(scenarios) - 1)
            if not is_last:
                print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def gen_trace_win_tie_loss_table(output_dir="./result/", tag=""):
    """
    Generate Table 14: win/tie/loss comparison on trace-derived instances.
    """
    all_rows = []

    for trace_name in ["Huawei", "Microsoft"]:
        rows = _load_trace_wtl(output_dir, trace_name, tag)
        all_rows.extend(rows)

    if not all_rows:
        print("\n% [SKIP] Table 14: no trace win/tie/loss files found.")
        return

    print("\n% ===== Trace Win/Tie/Loss Summary =====")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Win/tie/loss comparison on the constructed two-class "
          r"trace-derived bottleneck variant. A win means that the target "
          r"algorithm uses fewer PMs on an instance than the baseline algorithm.}")
    print(r"\label{tab:trace_win_tie_loss}")
    print(r"\fontsize{8}{10}\selectfont")
    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"\textbf{Trace} & \textbf{Comparison} & \textbf{Instances} & "
          r"\textbf{Baseline wins} & \textbf{Ties} & \textbf{Target wins} \\")
    print(r"\midrule")

    for r in all_rows:
        trace = r.get("trace", "")
        baseline = r.get("baseline_algo", "")
        target = r.get("target_algo", "")
        compared = to_int(r.get("compared_instances"))
        baseline_win = to_int(r.get("baseline_win"))
        tie = to_int(r.get("tie"))
        target_win = to_int(r.get("target_win"))

        # Display names: VMPack_MixVM301 -> VMPack+MixVM301 (matches paper body
        # and avoids bare underscore in LaTeX text mode).
        baseline_disp = baseline.replace('VMPack_', 'VMPack+')
        target_disp = target.replace('VMPack_', 'VMPack+')
        comparison = f"{baseline_disp} vs. {target_disp}"

        print(
            f"{trace} & {comparison} & "
            f"{compared} & {baseline_win} & {tie} & {target_win} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def _run_table(fn, tex_dir, tex_name, *args, **kwargs):
    """Run one gen_* table function, capturing its stdout.

    If tex_dir is set, the table is written to {tex_dir}/{tex_name}.tex and a
    one-line confirmation is printed. Otherwise the table is printed to stdout
    as before.
    """
    import io
    import contextlib

    if tex_dir is None:
        fn(*args, **kwargs)
        return

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)

    out_path = Path(tex_dir) / f"{tex_name}.tex"
    content = buf.getvalue()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [TEX] {out_path.name}  ({len(content)} chars)")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./result/')
    parser.add_argument('--tag', type=str, default='',
                        help='Optional suffix tag used in CSV filenames, e.g. tl5/tl10')
    parser.add_argument('--huawei_dir', type=str, default='./huawei_trace_output/',
                        help='Directory containing Huawei trace processing outputs.')
    parser.add_argument('--microsoft_dir', type=str, default='./microsoft_vmtable_output/',
                        help='Directory containing Microsoft vmtable processing outputs.')
    parser.add_argument('--only_trace_tables', action='store_true',
                        help='Generate only Table 8 and Table 14.')
    parser.add_argument('--tex_dir', type=str, default=None,
                        help='If set, write each LaTeX table to a .tex file in this directory '
                             'and only print a one-line confirmation per table. If unset, '
                             'tables are printed to stdout as before.')

    args = parser.parse_args()

    tex_dir = args.tex_dir
    if tex_dir:
        Path(tex_dir).mkdir(parents=True, exist_ok=True)

    if not args.only_trace_tables:
        # heuristic_comparison prints two tables (mixalgos + improvevmpack) in one call
        _run_table(gen_heuristic_comparison, tex_dir, 'heuristic_comparison',
                   args.output_dir, tag=args.tag)
        _run_table(gen_method_comparison, tex_dir, 'method_comparison',
                   args.output_dir, tag=args.tag)
        _run_table(gen_heuristic_empowerment, tex_dir, 'heuristic_empowerment',
                   args.output_dir, tag=args.tag)
        _run_table(gen_scale_sweep_table, tex_dir, 'scale_sweep',
                   args.output_dir, tag=args.tag)
        _run_table(gen_runtime_scaling_table, tex_dir, 'runtime_scaling',
                   args.output_dir, tag=args.tag)

    _run_table(gen_public_trace_summary_table, tex_dir, 'public_trace_summary',
               huawei_dir=args.huawei_dir, microsoft_dir=args.microsoft_dir)

    _run_table(gen_trace_detail_table, tex_dir, 'trace_detail',
               output_dir=args.output_dir, tag=args.tag)

    _run_table(gen_trace_win_tie_loss_table, tex_dir, 'trace_win_tie_loss',
               output_dir=args.output_dir, tag=args.tag)


if __name__ == '__main__':
    main()
