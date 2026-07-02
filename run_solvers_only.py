"""
run_solvers_only.py

Run exact solvers (column‑generation benchmark and VanillaMIP) with or
without MixVM201Pro warm‑start, using a single maxtime run and checkpoint
callbacks.

This script orchestrates the following solvers:

  • **CG‑Benchmark (column‑generation pattern benchmark)**: A pattern‑based
    column‑generation formulation used as a time‑limited exact‑solver
    benchmark.  It solves the master problem by column generation and then
    solves an integer master problem over the generated patterns.  Because
    it performs only a single branch‑and‑bound on the restricted master
    problem, it is **not** a full branch‑and‑price algorithm.  In practice
    the CG‑Benchmark converges within 1 second on the scales considered in
    this paper, so its incumbent solution and pattern set are identical
    across the 1 s, 5 s and 10 s checkpoints.

  • **VanillaMIP (assignment‑based formulation)**: The standard integer
    assignment model for the two‑resource VM allocation problem.  We use
    a Gurobi callback to record solver state at specified checkpoint times
    (e.g., 1 s, 5 s, 10 s) without restarting the solver.

  • **MixVM201Pro warm‑start**: Both solvers can be optionally initialized
    with the MixVM201Pro heuristic.  This provides an incumbent solution
    or initial pattern set that may accelerate convergence.

Key concepts:

  • All solvers run **once** per instance.  There is no restart between
    checkpoints.
  • VanillaMIP uses Gurobi callbacks to record solver states at checkpoint
    times without re‑running.
  • The CG‑Benchmark typically converges in less than 1 second on the tested
    scales, so its results are reused across checkpoints.
  • Heuristics run once and are used only for warm‑start and performance
    reporting.

Output files:

  * ``{scale}_maxtime.csv`` – full results with checkpoint columns.
  * ``{scale}.csv`` – 1‑second checkpoint data (backward‑compatible).
  * ``{scale}_tl5.csv`` – 5‑second checkpoint data.
  * ``{scale}_tl10.csv`` – 10‑second checkpoint data.

Usage examples:

    python run_solvers_only.py                                     # Default: improvevmpack
    python run_solvers_only.py --fun_case mixalgos                 # mixalgos instances
    python run_solvers_only.py --maxtime 10                        # Default checkpoints: 1,5,10
    python run_solvers_only.py --maxtime 10 --checkpoint_times 1,5,10
"""
import argparse, csv, json, os, platform, time
from datetime import datetime
from pathlib import Path

import numpy as np

import globalvars as gv
from data import LoadExamples, DataTypes, GetFilePath
from heuristics import VMPack_MixVM201Pro, VMPack_NoMixPack
from pricebranch import PriceBranch
from vanilla_mip import VanillaMIP
from utils import to_float


EXPERIMENT_CONFIGS = {
    'S1': {'T': 7, 'UP': 10,   'group': 'small'},
    'S2': {'T': 7, 'UP': 20,   'group': 'small'},
    'M1': {'T': 7, 'UP': 50,   'group': 'medium'},
    'M2': {'T': 7, 'UP': 100,  'group': 'medium'},
    'L1': {'T': 7, 'UP': 500,  'group': 'large'},
    'L2': {'T': 7, 'UP': 1000, 'group': 'large'},
}

# Column groups for checkpoint extraction
CHECKPOINT_COLS = {
    'mip': ['mip_npms', 'mip_lb', 'mip_ub', 'mip_gap', 'mip_time',
            'mip_status', 'mip_nodecount', 'mip_bestbound'],
    'mip_mix': ['mip_mix_npms', 'mip_mix_lb', 'mip_mix_ub', 'mip_mix_gap',
                'mip_mix_time', 'mip_mix_status', 'mip_mix_nodecount',
                'mip_mix_bestbound'],
}

PB_COLS = ['pb_npms', 'pb_lb', 'pb_ub', 'pb_gap', 'pb_time',
           'pb_status', 'pb_n_cols']
PB_MIX_COLS = ['pb_mix_npms', 'pb_mix_lb', 'pb_mix_ub', 'pb_mix_gap',
               'pb_mix_time', 'pb_mix_status', 'pb_mix_n_cols']


def load_1s_heuristic(scale_name, output_dir, fun_case='improvevmpack'):
    """Read heuristic_npms and heuristic_time from existing 1s CSV."""
    # For mixalgos, check fun_case-qualified name first
    if fun_case != 'improvevmpack':
        path = Path(output_dir) / f"{scale_name}_{fun_case}.csv"
        if path.exists():
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            npms = [float(r['heuristic_npms']) for r in rows]
            times = [float(r['heuristic_time']) for r in rows]
            return npms, times
    path = Path(output_dir) / f"{scale_name}.csv"
    if not path.exists():
        return None, None
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    npms = [float(r['heuristic_npms']) for r in rows]
    times = [float(r['heuristic_time']) for r in rows]
    return npms, times


def run_pricebranch(vm_demands, timelimit, ub_fn):
    """Run PriceBranch and return result dict."""
    t0 = time.time()
    result, _, gap_info = PriceBranch(
        vm_demands, ub_heuristic_fn=ub_fn,
        timelimit=timelimit, verbose=False)
    elapsed = time.time() - t0
    npms = result if isinstance(result, (int, float, np.integer)) else -1
    lb = gap_info.get('lb')
    ub = gap_info.get('ub')
    gap = gap_info.get('gap')
    n_cols = gap_info.get('n_cols')
    status = ('Optimal' if gap_info.get('cg_certified') and gap == 0
              else 'Feasible' if ub and ub > 0 else 'NoSolution')
    return {
        'npms': npms, 'lb': lb, 'ub': ub, 'gap': gap,
        'time': elapsed, 'status': status, 'n_cols': n_cols,
    }


def run_vanilla_mip(vm_demands, timelimit, ub_fn, checkpoint_times=None):
    """Run VanillaMIP with optional checkpoint callback. Returns (result_dict, checkpoints_dict)."""
    t0 = time.time()
    result, _, gap_info = VanillaMIP(
        vm_demands, timelimit=timelimit, verbose=False,
        ub_heuristic_fn=ub_fn, checkpoint_times=checkpoint_times)
    elapsed = time.time() - t0
    npms = result if isinstance(result, (int, float, np.integer)) else -1
    lb = gap_info.get('lb')
    ub = gap_info.get('ub')
    gap = gap_info.get('gap')
    status = gap_info.get('status', 'Unknown')
    nodes = gap_info.get('nodecount')
    bb = gap_info.get('bestbound')
    if status == 'TimeLimit':
        status = 'Feasible' if (ub and ub > 0) else 'NoSolution'
    result_dict = {
        'npms': npms, 'lb': lb, 'ub': ub, 'gap': gap,
        'time': elapsed, 'status': status, 'nodecount': nodes,
        'bestbound': bb,
    }
    checkpoints = gap_info.get('checkpoints', {})
    return result_dict, checkpoints


def write_checkpoint_csvs(results, scale_name, output_dir, checkpoint_times, tag, args, fun_case='improvevmpack'):
    """
    From the full results (with checkpoint columns), write 4 CSVs:
      - {scale}_maxtime.csv or {scale}_{fun_case}_maxtime.csv
      - {scale}.csv or {scale}_{fun_case}.csv  (1s checkpoint)
      - {scale}_tl5.csv  or {scale}_{fun_case}_tl5.csv
      - {scale}_tl10.csv or {scale}_{fun_case}_tl10.csv

    For fun_case='mixalgos', a fun_case infix is added to avoid collisions.
    """
    if not results:
        return

    # Build base name with optional fun_case infix
    base = f"{scale_name}_{fun_case}" if fun_case != 'improvevmpack' else scale_name

    # Ensure output goes to scale/ subdirectory
    out_dir = output_dir / 'scale'
    out_dir.mkdir(parents=True, exist_ok=True)

    # CG‑Benchmark runtime sanity check: all CG‑Benchmark runs should complete
    # within 1 second for checkpoint reuse to be valid.  Because the
    # column‑generation benchmark is reused across 1 s/5 s/10 s checkpoints,
    # unexpectedly long runtimes would invalidate the reuse assumption.
    pb_times = [to_float(r.get('pb_time')) for r in results if to_float(r.get('pb_time')) is not None]
    if pb_times and max(pb_times) > 1.0:
        print(
            f"  [WARN] CG‑Benchmark exceeded 1 s on some instances "
            f"(max={max(pb_times):.2f} s). Checkpoint reuse for 1 s/5 s/10 s may be inaccurate."
        )

    # Collect all fieldnames across all rows (checkpoint cols are dynamic)
    all_fieldnames = list(results[0].keys())
    for r in results:
        for k in r.keys():
            if k not in all_fieldnames:
                all_fieldnames.append(k)

    # 1) Write maxtime CSV (complete)
    mt_path = out_dir / f"{base}_{tag}.csv"
    with open(mt_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved (maxtime): {mt_path}")

    # 2) Build checkpoint-specific rows
    #    For each checkpoint time, remap mip/mip_mix columns to the "final" names.
    #    Suffix convention: 1s → '' (backward-compatible), Ns → '_tlN'.
    for cp_time in checkpoint_times:
        cp_suffix = '' if cp_time == 1 else f'_tl{cp_time}'

        cp_rows = []
        for row in results:
            cp_row = dict(row)  # copy

            # CG‑Benchmark columns stay as‑is (same at all checkpoints)
            # MIP columns: replace with checkpoint data
            for prefix, cols in CHECKPOINT_COLS.items():
                cp_key = f"{prefix}_{cp_time}s_"  # e.g. mip_1s_npms
                for col in cols:
                    base_name = col[len(prefix) + 1:] if col.startswith(prefix + '_') else col
                    cp_col = f"{prefix}_{cp_time}s_{base_name}"
                    if cp_col in row:
                        cp_row[col] = row[cp_col]

            cp_rows.append(cp_row)

        cp_path = out_dir / f"{base}{cp_suffix}.csv"
        with open(cp_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_fieldnames)
            w.writeheader()
            w.writerows(cp_rows)
        print(f"  Saved ({cp_time}s cp): {cp_path}")


def build_result_row(i, L, scale_name, cfg, args,
                     h_npms, h_time, checkpoint_times, fun_case='improvevmpack'):
    """Build one CSV row running all solvers once with checkpoint callback."""
    T_val = cfg['T']
    vm_demands = np.array([[int(L[s][t]) for t in range(T_val)] for s in range(3)])

    row = {
        'instance': i, 'scale': scale_name, 'group': cfg['group'],
        'funcase': fun_case, 'T': T_val, 'UP': cfg['UP'],
        'C': gv.C, 'M': gv.M,
        'n_inst': args.n_inst, 'seed': args.seed,
        'timelimit': args.maxtime,
        'data_file': '', 'data_generated': False,
        'heuristic_npms': h_npms, 'heuristic_time': h_time,
    }

    # ----- CG‑Benchmark (NoMix) — runs once, converges <1 s -----
    pb = run_pricebranch(vm_demands, args.maxtime, None)
    row.update({'pb_npms': pb['npms'], 'pb_lb': pb['lb'], 'pb_ub': pb['ub'],
                'pb_gap': pb['gap'], 'pb_time': pb['time'],
                'pb_status': pb['status'], 'pb_n_cols': pb['n_cols']})

    # ----- CG‑Benchmark+Mix — runs once, converges <1 s -----
    pb_mix = run_pricebranch(vm_demands, args.maxtime, VMPack_MixVM201Pro)
    row.update({'pb_mix_npms': pb_mix['npms'], 'pb_mix_lb': pb_mix['lb'],
                'pb_mix_ub': pb_mix['ub'], 'pb_mix_gap': pb_mix['gap'],
                'pb_mix_time': pb_mix['time'], 'pb_mix_status': pb_mix['status'],
                'pb_mix_n_cols': pb_mix['n_cols']})

    # ----- VanillaMIP (NoMix) — runs once with checkpoint callback -----
    mip, mip_cp = run_vanilla_mip(
        vm_demands, args.maxtime, None, checkpoint_times=checkpoint_times)
    row.update({'mip_npms': mip['npms'], 'mip_lb': mip['lb'], 'mip_ub': mip['ub'],
                'mip_gap': mip['gap'], 'mip_time': mip['time'],
                'mip_status': mip['status'], 'mip_nodecount': mip['nodecount'],
                'mip_bestbound': mip['bestbound']})
    # Write checkpoint columns
    for cp_time, cp_data in mip_cp.items():
        suffix = f"{int(cp_time)}s" if isinstance(cp_time, (int, float)) and cp_time != 'final' else 'maxtime'
        row[f'mip_{suffix}_npms'] = cp_data.get('ub')
        row[f'mip_{suffix}_lb'] = cp_data.get('lb')
        row[f'mip_{suffix}_ub'] = cp_data.get('ub')
        row[f'mip_{suffix}_gap'] = cp_data.get('gap')
        row[f'mip_{suffix}_time'] = cp_data.get('real_time', cp_data.get('time'))
        status = 'Optimal' if cp_data.get('gap') == 0 and cp_data.get('ub') is not None else \
                 'Feasible' if cp_data.get('ub') is not None else 'NoSolution'
        row[f'mip_{suffix}_status'] = status
        row[f'mip_{suffix}_nodecount'] = cp_data.get('nodecount')
        row[f'mip_{suffix}_bestbound'] = cp_data.get('bestbound')

    # ----- VanillaMIP+Mix — runs once with checkpoint callback -----
    mip_mix, mip_mix_cp = run_vanilla_mip(
        vm_demands, args.maxtime, VMPack_MixVM201Pro, checkpoint_times=checkpoint_times)
    row.update({'mip_mix_npms': mip_mix['npms'], 'mip_mix_lb': mip_mix['lb'],
                'mip_mix_ub': mip_mix['ub'], 'mip_mix_gap': mip_mix['gap'],
                'mip_mix_time': mip_mix['time'], 'mip_mix_status': mip_mix['status'],
                'mip_mix_nodecount': mip_mix['nodecount'],
                'mip_mix_bestbound': mip_mix['bestbound']})
    for cp_time, cp_data in mip_mix_cp.items():
        suffix = f"{int(cp_time)}s" if isinstance(cp_time, (int, float)) and cp_time != 'final' else 'maxtime'
        row[f'mip_mix_{suffix}_npms'] = cp_data.get('ub')
        row[f'mip_mix_{suffix}_lb'] = cp_data.get('lb')
        row[f'mip_mix_{suffix}_ub'] = cp_data.get('ub')
        row[f'mip_mix_{suffix}_gap'] = cp_data.get('gap')
        row[f'mip_mix_{suffix}_time'] = cp_data.get('real_time', cp_data.get('time'))
        status = 'Optimal' if cp_data.get('gap') == 0 and cp_data.get('ub') is not None else \
                 'Feasible' if cp_data.get('ub') is not None else 'NoSolution'
        row[f'mip_mix_{suffix}_status'] = status
        row[f'mip_mix_{suffix}_nodecount'] = cp_data.get('nodecount')
        row[f'mip_mix_{suffix}_bestbound'] = cp_data.get('bestbound')

    # Compute differences
    h_val = row.get('heuristic_npms', '*')
    pb_val = row.get('pb_npms', '*')
    mip_val = row.get('mip_npms', '*')
    row['h_minus_pb'] = (h_val - pb_val if isinstance(h_val, (int, float))
                         and isinstance(pb_val, (int, float)) and pb_val > 0 else '*')
    row['h_minus_mip'] = (h_val - mip_val if isinstance(h_val, (int, float))
                          and isinstance(mip_val, (int, float)) and mip_val > 0 else '*')

    return row


def main():
    parser = argparse.ArgumentParser(
        description='Run exact solvers with checkpoint callback (single maxtime run).')
    parser.add_argument('--fun_case', type=str, default='improvevmpack',
                        choices=['improvevmpack', 'mixalgos'],
                        help="Which instance type to run (default: improvevmpack)")
    parser.add_argument('--maxtime', type=float, default=10,
                        help='Hard time limit for MIP solvers (default: 10)')
    parser.add_argument('--tag', type=str, default='maxtime',
                        help='Output tag for the maxtime CSV (default: maxtime)')
    parser.add_argument('--checkpoint_times', type=str, default='1,5,10',
                        help='Comma-separated checkpoint seconds (default: 1,5,10)')
    parser.add_argument('--n_inst', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./result/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    checkpoint_times = sorted(set(
        int(x) for x in args.checkpoint_times.split(',') if x.strip()
    ))

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scale_name, cfg in EXPERIMENT_CONFIGS.items():
        T_val, UP = cfg['T'], cfg['UP']
        funcase = args.fun_case
        gv.InitialGlobalVars(T_val, UP)

        if not args.quiet:
            print(f"\n{'─' * 60}")
            print(f"  Scale: {scale_name} (fun_case={funcase}, maxtime={args.maxtime}s, "
                  f"checkpoints={checkpoint_times})")
            print(f"{'─' * 60}")

        # Load or generate instances
        filepath = GetFilePath(args.data_dir, args.n_inst, DataTypes.RANDOM, funcase)
        if os.path.exists(filepath):
            Ls = LoadExamples(filepath)
        else:
            print(f"  Generating {args.n_inst} instances...")
            from data import GenExamples
            Ls = GenExamples(args.n_inst, DataTypes.RANDOM, funcase)

        # Load heuristic results from 1s CSV (or run inline)
        h_npms_list, h_times_list = load_1s_heuristic(scale_name, args.output_dir, fun_case=funcase)
        if h_npms_list is None:
            if not args.quiet:
                print(f"  [WARN] No 1s CSV found, running heuristic inline")
            h_npms_list, h_times_list = [], []
            for L in Ls:
                vm_demands = np.array([[int(L[s][t]) for t in range(T_val)] for s in range(3)])
                t0 = time.time()
                r = VMPack_MixVM201Pro(vm_demands)
                el = time.time() - t0
                npms = r[0] if isinstance(r, tuple) else r
                h_npms_list.append(float(npms))
                h_times_list.append(el)

        results = []
        for i, L in enumerate(Ls):
            row = build_result_row(
                i, L, scale_name, cfg, args,
                h_npms_list[i] if i < len(h_npms_list) else -1,
                h_times_list[i] if i < len(h_times_list) else 0,
                checkpoint_times,
                fun_case=funcase,
            )
            results.append(row)

            if not args.quiet and (i + 1) % 10 == 0:
                h = row.get('heuristic_npms', '?')
                pb_s = row.get('pb_status', '?')
                mip_s = row.get('mip_status', '?')
                mip_mix_s = row.get('mip_mix_status', '?')
                print(f"  [{i+1:>3}/{args.n_inst}] H={h:<4} | "
                      f"PB={pb_s:<10} | MIP={mip_s:<10} | MIP+Mix={mip_mix_s:<10}")

        # Save all CSVs
        write_checkpoint_csvs(results, scale_name, output_dir,
                              checkpoint_times, args.tag, args,
                              fun_case=funcase)

        # Save metadata
        meta_base = f"{scale_name}_{funcase}" if funcase != 'improvevmpack' else scale_name
        meta_dir = output_dir / 'scale'
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{meta_base}_{args.tag}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'mode': 'solvers_only_maxtime',
                'scale': scale_name,
                'tag': args.tag,
                'maxtime': args.maxtime,
                'checkpoint_times': list(checkpoint_times),
                'n_inst': args.n_inst,
                'T': T_val, 'UP': UP,
                'seed': args.seed,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  ALL SOLVER EXPERIMENTS COMPLETE (maxtime={args.maxtime}s)")
    print(f"  Checkpoints: {checkpoint_times}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
