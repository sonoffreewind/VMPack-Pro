"""
export_unified_results.py

Export unified wide-format CSVs into organized directories using the
column registry (column_registry.py) for programmatic column naming.

Directory structure (per plan.md):
    result/
    ├── mixalgos/          # L0/L2 scenario
    │   ├── random_s1.csv  (UP=10)
    │   ├── random_s2.csv  (UP=20)
    │   ├── random_m1.csv  (UP=50)
    │   ├── random_m2.csv  (UP=100)
    │   ├── random_l1.csv  (UP=500)
    │   ├── random_l2.csv  (UP=1000)
    │   ├── huawei.csv
    │   └── microsoft.csv
    └── improvevmpack/     # L0/L1/L2 scenario
        ├── random_s1.csv
        ├── random_s2.csv
        ├── random_m1.csv
        ├── random_m2.csv
        ├── random_l1.csv
        ├── random_l2.csv
        ├── huawei.csv
        └── microsoft.csv

Usage:
    python export_unified_results.py --fun_case improvevmpack --with_solvers
    python export_unified_results.py --fun_case mixalgos
    python export_unified_results.py --all
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import globalvars as gv
from basic import CpuSize
from column_registry import (
    COLUMN_REGISTRY, SCALE_UP, SCALES, SCALE_FILE_NAMES, get_column_names,
)
from data import LoadExamples, DataTypes, GetFilePath
from utils import to_float, load_csv


# ── Solver column name mapping (maxtime CSV → column registry) ────────
# The maxtime CSV uses lowercase prefixes (pb_*, mip_*) while the
# column registry uses capitalized prefixes (CG_* and MIP_*).  This mapping
# converts the raw CSV field names into their canonical form used in
# COLUMN_REGISTRY.  The “pb” prefix corresponds to the column‑generation‑based
# CG‑Benchmark (previously referred to as P&B).
SOLVER_COL_MAP = {
    # CG‑Benchmark (NoMix)
    'pb_npms': 'CG_npms', 'pb_time': 'CG_time',
    'pb_gap': 'CG_gap', 'pb_status': 'CG_status',
    'pb_n_cols': 'CG_n_cols',
    # CG‑Benchmark+Mix
    'pb_mix_npms': 'CG_Mix_npms', 'pb_mix_time': 'CG_Mix_time',
    'pb_mix_gap': 'CG_Mix_gap', 'pb_mix_status': 'CG_Mix_status',
    'pb_mix_n_cols': 'CG_Mix_n_cols',
    # VanillaMIP (NoMix)
    'mip_npms': 'MIP_npms', 'mip_time': 'MIP_time',
    'mip_gap': 'MIP_gap', 'mip_status': 'MIP_status',
    'mip_bestbound': 'MIP_bestbound',
    # VanillaMIP+Mix
    'mip_mix_npms': 'MIP_Mix_npms', 'mip_mix_time': 'MIP_Mix_time',
    'mip_mix_gap': 'MIP_Mix_gap', 'mip_mix_status': 'MIP_Mix_status',
    'mip_mix_bestbound': 'MIP_Mix_bestbound',
}

# Checkpoint column mapping (maxtime CSV → column registry)
CHECKPOINT_TIMES = ['1s', '5s', '10s']
CHECKPOINT_BASES = ['npms', 'gap', 'status']


def _remap_solver_col(old_col):
    """Remap a maxtime CSV column name to column registry naming.

    Handles both base columns and checkpoint columns.
    """
    if old_col in SOLVER_COL_MAP:
        return SOLVER_COL_MAP[old_col]

    # Check checkpoint columns: mip_1s_npms → MIP_1s_npms, mip_mix_1s_gap → MIP_Mix_1s_gap
    for prefix_old, prefix_new in [('mip_mix_', 'MIP_Mix_'), ('mip_', 'MIP_')]:
        if old_col.startswith(prefix_old):
            for cp in CHECKPOINT_TIMES:
                for base in CHECKPOINT_BASES:
                    suffix = f'{cp}_{base}'
                    if old_col == f'{prefix_old}{suffix}':
                        return f'{prefix_new}{suffix}'
    return None


def find_optimal_npms(row):
    """
    Scan solver columns in a wide row to find the best PROVEN optimal value.

    Priority order (by likelihood of proving optimality):
      MIP+Mix > MIP > PB+Mix > PB

    Returns (optimal_npms, source_solver) or (None, None).
    """
    solvers = [
        ('MIP_Mix', 'MIP_Mix_status', 'MIP_Mix_npms'),
        ('MIP',     'MIP_status',     'MIP_npms'),
        ('PB_Mix',  'PB_Mix_status',  'PB_Mix_npms'),
        ('PB',      'PB_status',      'PB_npms'),
    ]
    for name, status_col, ub_col in solvers:
        if row.get(status_col) == 'Optimal':
            ub = to_float(row.get(ub_col))
            if ub is not None and ub > 0:
                return ub, name
    return None, None


def compute_lb_from_demands(vm_demands):
    """Compute LB from VM demands array."""
    C = gv.C
    C0 = CpuSize(vm_demands[0])
    C1 = CpuSize(vm_demands[1])
    C2 = CpuSize(vm_demands[2])
    total_cpu = C0 + C1 + C2
    total_mem = C0 + 2 * C1 + 4 * C2
    lb = int(np.ceil(max(total_cpu / C, total_mem / (2 * C))))
    return lb, total_cpu, total_mem


def pivot_heuristic_rows(rows):
    """
    Convert long-format heuristic rows into wide-format dicts.

    Input:  list of dicts, each with {instance, algorithm, npms, time, lb, ...}
    Output: dict {instance_id: wide_row}
    """
    by_instance = {}
    for r in rows:
        inst = int(r['instance'])
        if inst not in by_instance:
            by_instance[inst] = {
                'instance': inst,
                'lb': float(r.get('lb', 0)),
                'total_cpu': float(r.get('total_cpu', 0)),
                'total_mem': float(r.get('total_mem', 0)),
            }
        algo = r['algorithm']
        by_instance[inst][f'{algo}_npms'] = float(r['npms'])
        by_instance[inst][f'{algo}_time'] = float(r['time'])

    return by_instance


def pivot_trace_rows(rows):
    """
    Convert long-format trace detail rows into wide-format dicts.

    Trace rows have the same structure as heuristic rows but with
    additional trace metadata (trace name, C0, C1, C2, etc.).
    """
    by_instance = {}
    for r in rows:
        inst = int(r['instance'])
        if inst not in by_instance:
            by_instance[inst] = {
                'instance': inst,
                'lb': float(r.get('lb', 0)),
                'total_cpu': float(r.get('total_cpu', 0)),
                'total_mem': float(r.get('total_mem', 0)),
            }
        algo = r['algorithm']
        by_instance[inst][f'{algo}_npms'] = float(r['npms'])
        by_instance[inst][f'{algo}_time'] = float(r['time'])

    return by_instance


def load_vm_demands(data_dir, n_inst, T, UP, fun_case):
    """
    Load the original JSON instances and return a dict:
        {instance_id: [[L0...], [L1...], [L2...]]}
    where each Ls is a list of T integer counts.
    """
    gv.InitialGlobalVars(T, UP)
    filepath = GetFilePath(data_dir, n_inst, DataTypes.RANDOM, fun_case)
    if not os.path.exists(filepath):
        print(f"  [WARN] Instance JSON not found: {filepath}")
        return {}

    Ls = LoadExamples(filepath)
    demands = {}
    for i, L in enumerate(Ls):
        demands[i] = [[int(L[s][t]) for t in range(T)] for s in range(3)]
    return demands


def load_trace_instances_json(trace_key, fun_case, data_dir):
    """
    Load trace instance data from the trace's VMPack instance JSON.

    Searches data/ subdirectories first, then result/ subdirectories.
    Tries fun_case-specific files first (e.g. *_mixalgos.json), then
    falls back to improvevmpack JSON.

    Returns dict: {instance_id: [[L0...], [L1...], [L2...]]}
    or empty dict if file not found.
    """
    # Search in data/ first, then result/
    search_roots = [
        Path(data_dir),
        Path(data_dir).parent / 'result',  # fallback: result/
    ]

    trace_subdirs = [
        f"{trace_key}_trace",
        f"{trace_key}_vmtable",
    ]

    candidates = []
    for root in search_roots:
        for subdir in trace_subdirs:
            candidates.append(root / subdir / f"{trace_key}_vmpack_instances_{fun_case}.json")
            candidates.append(root / subdir / f"{trace_key}_vmpack_instances_improvevmpack.json")

    filepath = None
    for c in candidates:
        if c.exists():
            filepath = c
            break

    if filepath is None:
        print(f"  [INFO] Trace instance JSON not found for {trace_key}/{fun_case}")
        return {}

    with open(filepath, 'r') as f:
        raw = json.load(f)

    if isinstance(raw, dict) and 'instances' in raw:
        raw = raw['instances']

    T = 7
    gv.InitialGlobalVars(T, 1000)

    demands = {}
    for i, item in enumerate(raw):
        vm_demands = np.array(item, dtype=int)
        # Only include bottleneck instances for mixalgos
        if fun_case == 'mixalgos':
            C0 = CpuSize(vm_demands[0])
            C1 = CpuSize(vm_demands[1])
            C2 = CpuSize(vm_demands[2])
            if not (C1 == 0 and C0 < 3 * C2):
                continue
        idx = len(demands)
        demands[idx] = [[int(vm_demands[s][t]) for t in range(vm_demands.shape[1])] for s in range(3)]

    return demands


def merge_solver_columns(wide, mt_csv_path, fun_case):
    """
    Merge solver columns from maxtime CSV into wide rows.

    Handles column name remapping (lowercase → uppercase registry naming)
    and computes optimal_npms + gap information.
    """
    if not mt_csv_path.exists():
        return 0

    mt_rows = load_csv(mt_csv_path)
    mt_index = {}
    for r in mt_rows:
        inst = int(r['instance'])
        mt_index[inst] = r

    print(f"  Loaded solvers: {len(mt_rows)} rows from {mt_csv_path.name}")

    n_merged = 0
    for inst_id, wrow in wide.items():
        if inst_id in mt_index:
            mr = mt_index[inst_id]
            n_merged += 1

            # Remap and merge solver columns
            for old_col, val in mr.items():
                new_col = _remap_solver_col(old_col)
                if new_col is not None:
                    wrow[new_col] = val

    # Compute optimal_npms
    n_optimal = 0
    for inst_id, wrow in wide.items():
        opt_val, opt_src = find_optimal_npms(wrow)
        if opt_val is not None:
            wrow['optimal_npms'] = opt_val
            wrow['optimal_source'] = opt_src
            n_optimal += 1

    if n_optimal > 0:
        print(f"  Optimal proven: {n_optimal}/{len(wide)} instances")

    return n_merged


def write_csv(output_path, wide_rows, fun_case, include_solvers=True):
    """
    Write wide-format rows to CSV using column registry ordering.
    """
    if not wide_rows:
        print(f"  [SKIP] No data to write")
        return

    # Get column names from registry
    col_names = get_column_names(fun_case, include_solvers=include_solvers)

    # Convert wide_rows dict to sorted list by instance
    sorted_instances = sorted(wide_rows.keys())
    output_rows = []
    for seq, inst_id in enumerate(sorted_instances, start=1):
        wrow = wide_rows[inst_id]
        out = {'seq': seq}

        # instance column: check if we have the VM demand data
        instance_data = wrow.get('_instance_data')
        if instance_data is not None:
            out['instance'] = json.dumps(instance_data)
        else:
            out['instance'] = ''

        # Copy other columns
        for col in col_names[2:]:  # skip seq and instance (handled above)
            if col in wrow:
                val = wrow[col]
                if isinstance(val, float):
                    out[col] = f"{val:.6g}"
                else:
                    out[col] = val
            else:
                out[col] = ''

        output_rows.append(out)

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"  [OK] {output_path}")
    print(f"       {len(output_rows)} rows × {len(col_names)} columns")


def _find_csv(output_dir, subdir, filename):
    """Find a CSV file in subdirectory first, then root."""
    p = Path(output_dir) / subdir / filename
    if p.exists():
        return p
    p = Path(output_dir) / filename
    return p if p.exists() else None


def export_synthetic_scale(scale_name, fun_case, args, output_subdir):
    """Export one synthetic scale CSV (random_s1..random_l2)."""
    up = SCALE_UP[scale_name]
    filename = SCALE_FILE_NAMES[scale_name]
    T_val = 7

    # ── 1. Load heuristic CSV ──
    heur_csv = _find_csv(args.output_dir, 'heuristic', f"{fun_case}_up{up}.csv")
    if heur_csv is None:
        print(f"  [SKIP] {fun_case} UP={up}: heuristic CSV not found")
        return

    heur_rows = load_csv(heur_csv)
    wide = pivot_heuristic_rows(heur_rows)
    print(f"  {filename}: loaded heuristic {len(heur_rows)} rows, {len(wide)} instances")

    # ── 2. Add VM demands from JSON ──
    demands = load_vm_demands(args.data_dir, args.n_inst, T_val, up, fun_case)
    for inst_id, wrow in wide.items():
        if inst_id in demands:
            wrow['_instance_data'] = demands[inst_id]

    # ── 3. Merge solver columns (if requested) ──
    if args.with_solvers:
        if fun_case == 'improvevmpack':
            mt_csv = _find_csv(args.output_dir, 'scale', f"{scale_name}_maxtime.csv")
        else:
            mt_csv = _find_csv(args.output_dir, 'scale', f"{scale_name}_{fun_case}_maxtime.csv")

        if mt_csv is not None:
            merge_solver_columns(wide, mt_csv, fun_case)
        else:
            print(f"  [SKIP] maxtime CSV not found for {scale_name} {fun_case}")

    # ── 4. Write CSV ──
    out_path = output_subdir / f"{filename}.csv"
    write_csv(out_path, wide, fun_case, include_solvers=True)


def export_trace(trace_key, fun_case, args, output_subdir):
    """Export one trace CSV (huawei.csv, microsoft.csv)."""
    T_val = 7

    # Determine trace detail CSV path based on fun_case
    if fun_case == 'mixalgos':
        trace_detail = _find_csv(args.output_dir, 'trace', f"{trace_key}_trace_detail_{trace_key}.csv")
    else:
        trace_detail = _find_csv(args.output_dir, 'trace', f"{trace_key}_trace_detail_{trace_key}_improvevmpack.csv")

    if trace_detail is None:
        print(f"  [SKIP] {trace_key} ({fun_case}): trace detail not found")
        return

    trace_rows = load_csv(trace_detail)
    wide = pivot_trace_rows(trace_rows)
    print(f"  {trace_key}: loaded {len(trace_rows)} rows, {len(wide)} instances")

    # Load instance data from trace JSON (best-effort)
    demands = load_trace_instances_json(trace_key, fun_case, Path(args.data_dir))
    for inst_id, wrow in wide.items():
        if inst_id in demands:
            wrow['_instance_data'] = demands[inst_id]
    if len(demands) != len(wide):
        print(f"  [INFO] {trace_key}: instance data available for {len(demands)}/{len(wide)} instances")

    # Write CSV (trace files have only heuristic columns)
    out_path = output_subdir / f"{trace_key}.csv"
    write_csv(out_path, wide, fun_case, include_solvers=False)


def export_fun_case(fun_case, args):
    """Export all CSVs for one fun_case."""
    output_subdir = Path(args.output_dir) / fun_case
    output_subdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 68}")
    print(f"  Export: {fun_case}")
    print(f"{'=' * 68}")

    # ── Synthetic scales ──
    for scale_name in SCALES:
        export_synthetic_scale(scale_name, fun_case, args, output_subdir)

    # ── Traces ──
    for trace_key in ['huawei', 'microsoft']:
        export_trace(trace_key, fun_case, args, output_subdir)


def main():
    parser = argparse.ArgumentParser(
        description="Export unified wide-format CSVs into organized directories.")
    parser.add_argument('--fun_case', type=str, default='improvevmpack',
                        choices=['mixalgos', 'improvevmpack'],
                        help="Which instance type to export")
    parser.add_argument('--all', action='store_true',
                        help="Export both fun_cases")
    parser.add_argument('--with_solvers', action='store_true',
                        help="Merge solver columns from maxtime CSVs")
    parser.add_argument('--output_dir', type=str, default='./result/',
                        help="Result directory (contains heuristic/solver CSVs)")
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help="Directory containing JSON instance files")
    parser.add_argument('--n_inst', type=int, default=100,
                        help="Number of instances per scale")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed used for generation")

    args = parser.parse_args()

    if args.all:
        for fc in ['mixalgos', 'improvevmpack']:
            args.with_solvers = (fc == 'improvevmpack')
            export_fun_case(fc, args)
    else:
        export_fun_case(args.fun_case, args)


if __name__ == '__main__':
    main()
