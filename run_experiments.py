"""
run_experiments.py
Numerical experiment runner.

This script executes two kinds of experiments:

  • Scale comparison (5‑group exact comparison) on each scale:
      1. MixVM201Pro full‑pipeline heuristic.
      2. CG‑Benchmark (column‑generation pattern benchmark) with NoMix initialization.
      3. VanillaMIP (assignment‑based solver) with NoMix upper bound.
      4. CG‑Benchmark initialized with MixVM201Pro patterns.
      5. VanillaMIP initialized with MixVM201Pro upper bound.

  • Heuristic comparison across scales or instance groups.

Use --mode to choose between scale (exact comparison) and heuristic (heuristic
comparison) runs.  See the command‑line examples below.

Usage:
  python run_experiments.py --scale-group all --n_inst 100
  python run_experiments.py --scale-group small --n_inst 10
  python run_experiments.py --scale S1 --n_inst 10
  python run_experiments.py --scale S1 --n_inst 10 --timelimit 60 --quiet
"""
import argparse
import os
import time
import json
import platform
from datetime import datetime
import numpy as np
import csv


import globalvars as gv
from data import GenExamples, LoadExamples, DataTypes, GetFilePath, SaveExamples
from heuristics import (
    VMPack_MixVM201Pro, VMPack_NoMixPack,
    NoMixPack, MixVM301, MixVM201, MixVM201Pro, MixPack,
    SafeMix, VMPack_SafeMix,
    BFD, FFD,
    VMPack_MixVM301, VMPack_MixVM201, VMPack_MixPack,
    # Priority-only variant for ablation
    MixVM201Priority, VMPack_MixVM201Priority,
)
from utils import bootstrap_mean_ci, cliffs_delta
# CG‑Benchmark / VanillaMIP are imported lazily in run_scale_experiment()
# so that --skip_gurobi and heuristic-only runs do not require gurobipy.


# ═══════════════════════════════════════════════════════════════════
# Experiment Configuration
# ═══════════════════════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    'S1': {'T': 7, 'UP': 10,   'group': 'small',  'desc': 'Small (T=7, UP=10, ~100 VMs)'},
    'S2': {'T': 7, 'UP': 20,   'group': 'small',  'desc': 'Small (T=7, UP=20, ~190 VMs)'},

    'M1': {'T': 7, 'UP': 50,   'group': 'medium', 'desc': 'Medium (T=7, UP=50, ~460 VMs)'},
    'M2': {'T': 7, 'UP': 100,  'group': 'medium', 'desc': 'Medium (T=7, UP=100, ~910 VMs)'},

    'L1': {'T': 7, 'UP': 500,  'group': 'large',  'desc': 'Large (T=7, UP=500, ~4500 VMs)'},
    'L2': {'T': 7, 'UP': 1000, 'group': 'large',  'desc': 'Large (T=7, UP=1000, ~9000 VMs)'},
}

DEFAULT_TIME_LIMIT = 1  # seconds

# Heuristic comparison experiment configurations
# Each config specifies: data file, algorithm list, and fun_case type.
HEURISTIC_CONFIGS = {
    'mixalgos': {
        'desc': 'Base heuristic comparison (L0/L2 only)',
        'fun_case': 'mixalgos',
        'algos': {
            'NoMixPack':      NoMixPack,
            'MixVM301':       MixVM301,
            'MixVM201':       MixVM201,
            'MixVM201Pro':    MixVM201Pro,
            'MixPack':        MixPack,
            'SafeMix':        SafeMix,
            'MixVM201Priority': MixVM201Priority,
            'BFD':            BFD,
            'FFD':            FFD,
        },
    },
    'improvevmpack': {
        'desc': 'VMPack variant comparison (L0/L1/L2)',
        'fun_case': 'improvevmpack',
        'algos': {
            'VMPack_NoMixPack':   VMPack_NoMixPack,
            'VMPack_MixVM301':    VMPack_MixVM301,
            'VMPack_MixVM201':    VMPack_MixVM201,
            'VMPack_MixVM201Pro': VMPack_MixVM201Pro,
            'VMPack_MixPack':     VMPack_MixPack,
            'VMPack_SafeMix':     VMPack_SafeMix,
            'VMPack_MixVM201Priority': VMPack_MixVM201Priority,
            'BFD':      BFD,
            'FFD':      FFD,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# Runner Functions
# ═══════════════════════════════════════════════════════════════════

def run_heuristic(vm_demands, heuristic_fn):
    """Run a heuristic. Returns (npms, elapsed)."""
    t0 = time.time()
    result = heuristic_fn(vm_demands)
    elapsed = time.time() - t0
    npms = result[0] if isinstance(result, tuple) else result
    return npms, elapsed


def run_pricebranch(vm_demands, timelimit, ub_heuristic_fn=None):
    """Run the column‑generation benchmark (CG‑Benchmark).

    This helper wraps the `pricebranch.PriceBranch` solver, which implements
    a column‑generation pattern benchmark for the mixed‑packing problem.  It
    returns a tuple `(npms, lb, ub, gap, elapsed, status, n_cols)` where `npms`
    is the best incumbent bin count, `lb` is the certified lower bound, `ub`
    is the incumbent upper bound (the best bin count), `gap` is the solver
    incumbent gap, `elapsed` is the runtime in seconds, `status` indicates
    whether an optimal solution was proven, and `n_cols` reports the number
    of patterns generated.
    """
    from pricebranch import PriceBranch
    t0 = time.time()
    result, _, gap_info = PriceBranch(
        vm_demands,
        ub_heuristic_fn=ub_heuristic_fn,
        timelimit=timelimit,
        verbose=False,
    )
    elapsed = time.time() - t0

    npms = result if isinstance(result, (int, float, np.integer)) else -1
    lb = gap_info.get('lb', None)
    ub = gap_info.get('ub', None)
    gap = gap_info.get('gap', None)
    n_cols = gap_info.get('n_cols', None)

    if gap is not None and gap == 0:
        status = 'Optimal'
    elif ub is not None and ub > 0:
        status = 'Feasible'
    else:
        status = 'NoSolution'

    return npms, lb, ub, gap, elapsed, status, n_cols


def run_vanilla_mip(vm_demands, timelimit, ub_heuristic_fn=None):
    """Run VanillaMIP. Returns (npms, lb, ub, gap, elapsed, status, nodecount, bestbound)."""
    from vanilla_mip import VanillaMIP
    t0 = time.time()
    result, _, gap_info = VanillaMIP(
        vm_demands,
        timelimit=timelimit,
        verbose=False,
        ub_heuristic_fn=ub_heuristic_fn,
    )
    elapsed = time.time() - t0

    npms = result if isinstance(result, (int, float, np.integer)) else -1
    lb = gap_info.get('lb', None)
    ub = gap_info.get('ub', None)
    gap = gap_info.get('gap', None)
    status = gap_info.get('status', 'Unknown')
    nodecount = gap_info.get('nodecount', None)
    bestbound = gap_info.get('bestbound', None)

    # Refine status: if TimeLimit but a feasible solution was found, mark as Feasible
    if status == 'TimeLimit':
        if ub is not None and ub > 0:
            status = 'Feasible'
        else:
            status = 'NoSolution'

    return npms, lb, ub, gap, elapsed, status, nodecount, bestbound


# ═══════════════════════════════════════════════════════════════════
# Data Loader
# ═══════════════════════════════════════════════════════════════════

def _load_or_generate_instances(n_inst, T, UP, funcase, data_dir, save_generated=True):
    """
    Load pre-generated instances from data_dir if available.
    If the file is missing, generate instances on the fly and optionally save
    them immediately for reproducibility.
    """
    gv.InitialGlobalVars(T, UP)
    filepath = GetFilePath(data_dir, n_inst, DataTypes.RANDOM, funcase)

    if os.path.exists(filepath):
        Ls = LoadExamples(filepath)
        print(f"  Loaded {len(Ls)} instances from {filepath}")
        return Ls, filepath, False

    print(f"  Data file not found: {filepath}")
    print(f"  Generating {n_inst} instances (T={T}, UP={UP}, funcase={funcase})...")
    Ls = GenExamples(n_inst, DataTypes.RANDOM, funcase)

    if save_generated:
        SaveExamples(data_dir, Ls, DataTypes.RANDOM, funcase)
        print(f"  Generated instances saved to {filepath}")

    return Ls, filepath, True

# ═══════════════════════════════════════════════════════════════════
# Heuristic Comparison Runner
# ═══════════════════════════════════════════════════════════════════

def run_heuristic_comparison(n_inst, T, UP, output_dir, data_dir='./data/',
                             configs=None, quiet=False, seed=42, tag=None):
    """
    Run heuristic comparison experiments (mixalgos + improvevmpack).
    For each config, runs all algorithms on the same instances and saves
    per-instance results to CSV for later table generation.

    CSV format (per-instance):
        instance, algorithm, npms, time, lb, total_cpu, total_mem
    """
    if configs is None:
        configs = HEURISTIC_CONFIGS

    from basic import CpuSize

    for cfg_name, cfg in configs.items():
        fun_case = cfg['fun_case']
        algos = cfg['algos']

        if not quiet:
            print(f"\n{'─' * 60}")
            print(f"  Heuristic Comparison: {cfg_name} — {cfg['desc']}")
            print(f"  Algorithms: {', '.join(algos.keys())}")
            print(f"{'─' * 60}")

        Ls, data_file, generated = _load_or_generate_instances(n_inst, T, UP, fun_case, data_dir)


        results = []
        for i, L in enumerate(Ls):
            # Compute lower bound and total demands for this instance
            C = gv.C
            vm0, vm1, vm2 = L[0], L[1], L[2]
            C0, C1, C2 = CpuSize(vm0), CpuSize(vm1), CpuSize(vm2)
            lb = float(np.ceil(max((C0 + C1 + C2) / C, (C0 + 2 * C1 + 4 * C2) / (2 * C))))
            total_cpu = float(C0 + C1 + C2)
            total_mem = float(C0 + 2 * C1 + 4 * C2)

            for algo_name, algo_fn in algos.items():
                npms_val, elapsed = run_heuristic(
                    np.array([[int(L[s][t]) for t in range(T)] for s in range(3)]),
                    algo_fn,
                )
                results.append({
                    'instance': i,
                    'algorithm': algo_name,
                    'funcase': fun_case,
                    'T': T,
                    'UP': UP,
                    'C': gv.C,
                    'M': gv.M,
                    'n_inst': n_inst,
                    'seed': seed,
                    'data_file': data_file,
                    'data_generated': generated,
                    'npms': npms_val,
                    'time': elapsed,
                    'lb': lb,
                    'total_cpu': total_cpu,
                    'total_mem': total_mem,
                })

            if not quiet and (i + 1) % 10 == 0:
                print(f"  [{i+1:>3}/{n_inst}] done")

        _save_results(results, cfg_name, output_dir, tag=tag, subdir='heuristic')
        _save_metadata(output_dir, _tagged_name(f"{cfg_name}_metadata", tag, "json"), {
            'mode': 'heuristic',
            'config': cfg_name,
            'funcase': fun_case,
            'n_inst': n_inst,
            'T': T,
            'UP': UP,
            'C': gv.C,
            'M': gv.M,
            'seed': seed,
            'data_dir': data_dir,
            'data_file': data_file,
            'data_generated': generated,
            'algorithms': list(algos.keys()),
        })
        _print_heuristic_summary(results, cfg_name, quiet=quiet)

        # Per-instance win/tie/loss + Wilcoxon signed-rank test for the key
        # bottleneck comparison in BOTH scenarios:
        #   - mixalgos (standalone two-class): MixVM301 vs MixVM201Pro
        #   - improvevmpack (full pipeline):   VMPack_MixVM301 vs VMPack_MixVM201Pro
        # Provides instance-level robustness evidence matching the trace WTL.
        _save_wtl_and_test(results, cfg_name, output_dir, tag=tag)


def _compute_wtl(by_inst, baseline_algo, target_algo):
    """Per-instance win/tie/loss for two algorithms over the same instances.

    baseline_win: baseline uses fewer PMs; tie: equal; target_win: target fewer.
    Returns the WTL counts plus the paired PM-count lists (for the Wilcoxon test).
    """
    baseline_vals, target_vals = [], []
    b_win = t_win = tie = 0
    for _, vals in sorted(by_inst.items()):
        if baseline_algo not in vals or target_algo not in vals:
            continue
        b = vals[baseline_algo]
        t = vals[target_algo]
        baseline_vals.append(b)
        target_vals.append(t)
        if b < t:
            b_win += 1
        elif t < b:
            t_win += 1
        else:
            tie += 1
    return {
        "baseline_algo": baseline_algo,
        "target_algo": target_algo,
        "compared_instances": len(baseline_vals),
        "baseline_win": b_win,
        "tie": tie,
        "target_win": t_win,
        "_baseline_vals": baseline_vals,
        "_target_vals": target_vals,
    }


def _wilcoxon_signed_rank_pvalue(diffs):
    """Two-sided Wilcoxon signed-rank test p-value (normal approximation),
    implemented without scipy so no new dependency is introduced.

    Handles the zero-difference case (ties at 0 excluded from ranking).
    For n < 1 returns None. Uses the standard normal approximation with
    tie correction; exact tables are not used (n is large here, >=100).
    """
    import math
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 1:
        return None
    abs_d = sorted(abs(d) for d in nz)
    # Assign ranks with average ranks for ties in |d|
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1] == abs_d[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    # Sum of ranks of positive differences
    w_plus = 0.0
    for d, r in zip(sorted(nz, key=abs), ranks):
        if d > 0:
            w_plus += r
    # Tie correction: count ties among |d| groups
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1] == abs_d[i]:
            j += 1
        t = j - i + 1
        tie_term += t * (t * t - 1)
        i = j + 1
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return None
    z = (w_plus - mean_w) / math.sqrt(var_w)
    # Two-sided p-value via standard normal CDF (error function)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p


def _save_wtl_and_test(results, cfg_name, output_dir, tag=None):
    """Compute WTL + Wilcoxon p-value for the key bottleneck comparison and
    save to a CSV in result/heuristic/.

    Comparison (scenario-dependent):
      - mixalgos (standalone two-class):      MixVM301  vs MixVM201Pro
      - improvevmpack (full VMPack pipeline): VMPack_MixVM301 vs VMPack_MixVM201Pro
    """
    # Reorganize per-instance {algorithm: npms}
    by_inst = {}
    for r in results:
        by_inst.setdefault(r['instance'], {})[r['algorithm']] = r['npms']

    # Pick the baseline/target pair appropriate to this scenario.
    if cfg_name == 'mixalgos':
        baseline = "MixVM301"
        target = "MixVM201Pro"
    else:  # improvevmpack
        baseline = "VMPack_MixVM301"
        target = "VMPack_MixVM201Pro"
    wtl = _compute_wtl(by_inst, baseline, target)

    # Wilcoxon signed-rank test on paired PM counts
    diffs = [b - t for b, t in zip(wtl["_baseline_vals"], wtl["_target_vals"])]
    p_value = _wilcoxon_signed_rank_pvalue(diffs)

    # 95% bootstrap CI for mean paired PM difference (baseline - target)
    # and Cliff's delta effect size (non-parametric, paired).
    mean_diff, ci_lo, ci_hi = bootstrap_mean_ci(diffs, n_boot=1000, alpha=0.05, seed=42)
    delta = cliffs_delta(diffs)

    out_dir = os.path.join(output_dir, 'heuristic')
    os.makedirs(out_dir, exist_ok=True)
    base_name = _tagged_name(f"{cfg_name}_wtl", tag, "csv")
    csv_path = os.path.join(out_dir, base_name)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["baseline_algo", "target_algo", "compared_instances",
                    "baseline_win", "tie", "target_win",
                    "wilcoxon_pvalue", "mean_baseline_npms", "mean_target_npms",
                    "mean_diff_pm", "ci95_low", "ci95_high", "cliffs_delta"])
        mb = sum(wtl["_baseline_vals"]) / len(wtl["_baseline_vals"]) if wtl["_baseline_vals"] else 0
        mt = sum(wtl["_target_vals"]) / len(wtl["_target_vals"]) if wtl["_target_vals"] else 0
        p_str = f"{p_value:.6g}" if p_value is not None else "N/A"
        w.writerow([baseline, target, wtl["compared_instances"],
                    wtl["baseline_win"], wtl["tie"], wtl["target_win"],
                    p_str, f"{mb:.4f}", f"{mt:.4f}",
                    f"{mean_diff:.4f}", f"{ci_lo:.4f}", f"{ci_hi:.4f}", f"{delta:.4f}"])
    print(f"  WTL saved to {csv_path}")
    print(f"    {baseline} vs {target}: "
          f"baseline_win={wtl['baseline_win']}, tie={wtl['tie']}, "
          f"target_win={wtl['target_win']}, "
          f"Wilcoxon p={p_str}, "
          f"95% CI ΔPM=[{ci_lo:.3f}, {ci_hi:.3f}], Cliff's δ={delta:.3f}")


def _print_heuristic_summary(results, cfg_name, quiet=False):
    """Print summary for heuristic comparison results."""
    if not results:
        return

    # Group by algorithm
    algos = sorted(set(r['algorithm'] for r in results))
    n_inst = max(r['instance'] for r in results) + 1

    print(f"\n  ├─── {cfg_name} Summary ───┤")
    print(f"  │ Instances: {n_inst}")
    if not quiet:
        print(f"  │ {'Algorithm':<22} {'Avg #PMs':>9} {'Avg Time(ms)':>12} {'Avg ρ':>8} {'CPU%':>7} {'MEM%':>7}")
        print(f"  │ {'-'*65}")
        for algo in algos:
            algo_rows = [r for r in results if r['algorithm'] == algo]
            npms_list = [r['npms'] for r in algo_rows]
            times_ms = [r['time'] * 1000 for r in algo_rows]
            rhos = [r['npms'] / r['lb'] if r['lb'] > 0 else 0 for r in algo_rows]
            # Utilization: total demand / (npms * capacity)
            total_cpu = sum(r['total_cpu'] for r in algo_rows)
            total_mem = sum(r['total_mem'] for r in algo_rows)
            total_bins = sum(r['npms'] for r in algo_rows)
            C = gv.C
            cpu_util = total_cpu / (total_bins * C) if total_bins > 0 else 0
            mem_util = total_mem / (total_bins * 2 * C) if total_bins > 0 else 0

            print(f"  │ {algo:<22} {np.mean(npms_list):>9.2f} {np.mean(times_ms):>12.2f} "
                  f"{np.mean(rhos):>8.4f} {cpu_util*100:>6.1f}% {mem_util*100:>6.1f}%")
    print(f"  └────────────────────────┘")

def run_scale_experiment(configs, n_inst, output_dir, data_dir='./data/',
                         timelimit=DEFAULT_TIME_LIMIT, quiet=False, seed=42, tag=None):
    """
    Run the five‑group exact comparison on the given scale configurations.

    For each scale the following five methods are compared:
      1. MixVM201Pro full‑pipeline heuristic (denoted “Mix” in the tables).
      2. CG‑Benchmark with the baseline NoMix initialization.
      3. VanillaMIP with the baseline NoMix upper bound.
      4. CG‑Benchmark initialized with MixVM201Pro patterns (denoted “CG‑Benchmark+Mix”).
      5. VanillaMIP initialized with MixVM201Pro upper bound (denoted “VanillaMIP+Mix”).

    The CG‑Benchmark is a column‑generation‑based pattern benchmark used for
    calibration; it is not intended as a practical online algorithm.
    """
    for scale_name, cfg in configs.items():
        group = cfg['group']
        if not quiet:
            print(f"\n{'─' * 60}")
            print(f"  Scale: {scale_name} — {cfg['desc']}")
            print(f"{'─' * 60}")

        T, UP = cfg['T'], cfg['UP']
        funcase = 'improvevmpack'

        Ls, data_file, generated = _load_or_generate_instances(n_inst, T, UP, funcase, data_dir)

        results = []
        for i, L in enumerate(Ls):
            vm_demands = np.array([[int(L[s][t]) for t in range(T)] for s in range(3)])

            # 1. MixVM201Pro heuristic
            h_npms, h_time = run_heuristic(vm_demands, VMPack_MixVM201Pro)

            # 2. CG‑Benchmark (default: NoMix initial columns)
            pb_npms, pb_lb, pb_ub, pb_gap, pb_time, pb_status, pb_n_cols = \
                run_pricebranch(vm_demands, timelimit, ub_heuristic_fn=None)

            # 3. VanillaMIP (default: NoMix upper bound)
            mip_npms, mip_lb, mip_ub, mip_gap, mip_time, mip_status, mip_nodes, mip_bb = \
                run_vanilla_mip(vm_demands, timelimit, ub_heuristic_fn=None)

            # 4. CG‑Benchmark with MixVM201Pro initial columns
            pb_mix_npms, pb_mix_lb, pb_mix_ub, pb_mix_gap, pb_mix_time, pb_mix_status, pb_mix_n_cols = \
                run_pricebranch(vm_demands, timelimit, ub_heuristic_fn=VMPack_MixVM201Pro)

            # 5. VanillaMIP with MixVM201Pro upper bound
            mip_mix_npms, mip_mix_lb, mip_mix_ub, mip_mix_gap, mip_mix_time, mip_mix_status, mip_mix_nodes, mip_mix_bb = \
                run_vanilla_mip(vm_demands, timelimit, ub_heuristic_fn=VMPack_MixVM201Pro)

            results.append({
                'instance': i,
                'scale': scale_name,
                'group': group,
                'funcase': funcase,
                'T': T,
                'UP': UP,
                'C': gv.C,
                'M': gv.M,
                'n_inst': n_inst,
                'seed': seed,
                'timelimit': timelimit,
                'data_file': data_file,
                'data_generated': generated,

                # 1. Proposed heuristic
                'heuristic_npms': h_npms,
                'heuristic_time': h_time,

                # 2. CG‑Benchmark with default initialization
                'pb_npms': pb_npms,
                'pb_lb': pb_lb,
                'pb_ub': pb_ub,
                'pb_gap': pb_gap,
                'pb_time': pb_time,
                'pb_status': pb_status,
                'pb_n_cols': pb_n_cols,

                # 3. VanillaMIP with default upper bound
                'mip_npms': mip_npms,
                'mip_lb': mip_lb,
                'mip_ub': mip_ub,
                'mip_gap': mip_gap,
                'mip_time': mip_time,
                'mip_status': mip_status,
                'mip_nodecount': mip_nodes,
                'mip_bestbound': mip_bb,

                # 4. CG‑Benchmark initialized with MixVM201Pro
                'pb_mix_npms': pb_mix_npms,
                'pb_mix_lb': pb_mix_lb,
                'pb_mix_ub': pb_mix_ub,
                'pb_mix_gap': pb_mix_gap,
                'pb_mix_time': pb_mix_time,
                'pb_mix_status': pb_mix_status,
                'pb_mix_n_cols': pb_mix_n_cols,

                # 5. VanillaMIP initialized with MixVM201Pro
                'mip_mix_npms': mip_mix_npms,
                'mip_mix_lb': mip_mix_lb,
                'mip_mix_ub': mip_mix_ub,
                'mip_mix_gap': mip_mix_gap,
                'mip_mix_time': mip_mix_time,
                'mip_mix_status': mip_mix_status,
                'mip_mix_nodecount': mip_mix_nodes,
                'mip_mix_bestbound': mip_mix_bb,

                # Convenience differences.  These are only meaningful when
                # the compared solver produced a feasible incumbent.
                'h_minus_pb': h_npms - pb_npms if isinstance(pb_npms, (int, float, np.integer)) and pb_npms > 0 else '*',
                'h_minus_mip': h_npms - mip_npms if isinstance(mip_npms, (int, float, np.integer)) and mip_npms > 0 else '*',
            })


            if not quiet and (i + 1) % 10 == 0:
                pb_ub_str = str(pb_ub) if pb_ub is not None else "None"
                pb_mix_ub_str = str(pb_mix_ub) if pb_mix_ub is not None else "None"
                mip_status_short = str(mip_status)[:10] if mip_status else "None"
                # Replace labels: PB → CG and Gurobi → MIP for clarity
                print(f"  [{i+1:>3}/{n_inst}] Mix={h_npms:<4} | CG={pb_ub_str:<4} | "
                      f"CG+Mix={pb_mix_ub_str:<4} | MIP={mip_status_short} | MIP+Mix={str(mip_mix_status)[:10]}")

        _save_results(results, scale_name, output_dir, tag=tag, subdir='scale')
        _save_metadata(output_dir, _tagged_name(f"{scale_name}_metadata", tag, "json"), {
            'mode': 'scale',
            'config': scale_name,
            'tag': tag,
            'scale': scale_name,
            'group': group,
            'desc': cfg.get('desc', ''),
            'funcase': funcase,
            'n_inst': n_inst,
            'T': T,
            'UP': UP,
            'C': gv.C,
            'M': gv.M,
            'seed': seed,
            'timelimit': timelimit,
            'data_dir': data_dir,
            'data_file': data_file,
            'data_generated': generated,
            # record solver names using CG‑Benchmark for clarity
            'solvers': [
                'VMPack_MixVM201Pro',
                'CG‑Benchmark',
                'VanillaMIP',
                'CG‑Benchmark+MixVM201Pro',
                'VanillaMIP+MixVM201Pro',
            ],
        })
        _print_summary(results, scale_name, quiet=quiet)



# ═══════════════════════════════════════════════════════════════════
# Output Helpers
# ═══════════════════════════════════════════════════════════════════

def _save_metadata(output_dir, filename, metadata, subdir='heuristic'):
    """Save experiment metadata for reproducibility."""
    output_dir = os.path.join(output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    meta = dict(metadata)
    meta['timestamp'] = datetime.now().isoformat(timespec='seconds')
    meta['python_version'] = platform.python_version()
    meta['platform'] = platform.platform()

    # Gurobi may not be installed in all environments. Record version if possible.
    try:
        import gurobipy as gp
        meta['gurobi_version'] = ".".join(map(str, gp.gurobi.version()))
    except Exception:
        meta['gurobi_version'] = None

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"  Metadata saved to {path}")

def _tagged_name(base_name, tag, ext):
    """Build output filename with optional tag."""
    if tag:
        return f"{base_name}_{tag}.{ext}"
    return f"{base_name}.{ext}"

def _save_results(results, scale_name, output_dir, tag=None, subdir='heuristic'):
    """Save per-instance results to CSV in subdirectory."""
    output_dir = os.path.join(output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, _tagged_name(scale_name, tag, "csv"))

    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Results saved to {csv_path}")


def _print_summary(results, scale_name, quiet=False):
    """Print unified summary for scale experiment results."""
    n = len(results)
    pb_opt = sum(1 for r in results if r['pb_status'] == 'Optimal')
    pb_tl = sum(1 for r in results if 'TimeLimit' in str(r['pb_status']))
    mip_opt = sum(1 for r in results if r['mip_status'] == 'Optimal')
    mip_feas = sum(1 for r in results if r['mip_status'] == 'Feasible')
    mip_nosol = sum(1 for r in results if r['mip_status'] == 'NoSolution')
    mip_oom = sum(1 for r in results if 'OOM' in str(r['mip_status']))

    # Mix-enhanced results
    pb_mix_opt = sum(1 for r in results if r['pb_mix_status'] == 'Optimal')
    mip_mix_opt = sum(1 for r in results if r['mip_mix_status'] == 'Optimal')
    mip_mix_feas = sum(1 for r in results if r['mip_mix_status'] == 'Feasible')
    mip_mix_nosol = sum(1 for r in results if r['mip_mix_status'] == 'NoSolution')
    mip_mix_oom = sum(1 for r in results if 'OOM' in str(r['mip_mix_status']))

    h_times = [r['heuristic_time'] for r in results]
    pb_times = [r['pb_time'] for r in results]
    pb_mix_times = [r['pb_mix_time'] for r in results]
    mip_times = [r['mip_time'] for r in results if r['mip_time'] is not None]
    mip_mix_times = [r['mip_mix_time'] for r in results if r['mip_mix_time'] is not None]

    print(f"\n  ├─── {scale_name} Summary ───┤")
    print(f"  │ Instances                : {n}")
    if not quiet:
        print(f"  │ CG‑Benchmark Optimal      : {pb_opt}/{n} ({pb_opt/n*100:.1f}%)")
        if pb_tl:
            print(f"  │ CG‑Benchmark TimeLimit    : {pb_tl}/{n}")
        print(f"  │ CG‑Benchmark+Mix Optimal  : {pb_mix_opt}/{n}")
        print(f"  │ VanillaMIP Opt/Feas/No/OOM : {mip_opt}/{mip_feas}/{mip_nosol}/{mip_oom}")
        print(f"  │ VanillaMIP+Mix Opt/Feas/No/OOM: {mip_mix_opt}/{mip_mix_feas}/{mip_mix_nosol}/{mip_mix_oom}")
        print(f"  │ MixVM201Pro Avg Time     : {np.mean(h_times)*1000:.2f} ms")
        print(f"  │ CG‑Benchmark Avg Time     : {np.mean(pb_times)*1000:.2f} ms")
        print(f"  │ CG‑Benchmark+Mix Avg Time : {np.mean(pb_mix_times)*1000:.2f} ms")
        if mip_times:
            print(f"  │ VanillaMIP Avg Time      : {np.mean(mip_times)*1000:.2f} ms")
        if mip_mix_times:
            print(f"  │ VanillaMIP+Mix Avg Time  : {np.mean(mip_mix_times)*1000:.2f} ms")
    print(f"  └────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run Numerical Experiments")
    parser.add_argument('--mode', type=str, default='scale',
                        choices=['scale', 'heuristic'],
                        help='Experiment mode: scale (5-group exact comparison) or heuristic (heuristic comparison)')
    parser.add_argument('--scale-group', type=str, default='all',
                        choices=['small', 'medium', 'large', 'all'],
                        help='Which scale group to run (for --mode scale)')
    parser.add_argument('--n_inst', type=int, default=100,
                        help='Number of instances per scale (default: 100)')
    parser.add_argument('--output_dir', type=str, default='./result/',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--scale', type=str, default=None,
                        help='Run only specific scale (e.g., S1, M1, L1)')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='Directory for pre-generated instance files')
    parser.add_argument('--timelimit', type=float, default=DEFAULT_TIME_LIMIT,
                        help='Time limit in seconds for exact solvers (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-instance progress output, only show scale summary')
    # Heuristic comparison specific args
    parser.add_argument('--T', type=int, default=7,
                        help='Number of CPU sizes for heuristic comparison (default: 7)')
    parser.add_argument('--UP', type=int, default=1000,
                        help='Upper bound for heuristic comparison (default: 1000)')
    parser.add_argument('--heuristic-config', type=str, default=None,
                        choices=['mixalgos', 'improvevmpack'],
                        help='Run only specific heuristic config (default: both)')
    parser.add_argument('--tag', type=str, default='',
                        help='Optional suffix tag for output files, e.g. tl5/tl10')
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = args.output_dir

    if args.mode == 'scale':
        # Build config dict based on filters
        if args.scale:
            configs = {k: v for k, v in EXPERIMENT_CONFIGS.items() if k == args.scale}
        elif args.scale_group == 'all':
            configs = EXPERIMENT_CONFIGS
        else:
            configs = {k: v for k, v in EXPERIMENT_CONFIGS.items()
                       if v['group'] == args.scale_group}

        # Run unified scale experiment (5 groups)
        if configs:
            run_scale_experiment(configs, args.n_inst, output_dir, args.data_dir,
                                 timelimit=args.timelimit, quiet=args.quiet, seed=args.seed, tag=args.tag)

    elif args.mode == 'heuristic':
        hconfigs = HEURISTIC_CONFIGS
        if args.heuristic_config:
            hconfigs = {k: v for k, v in HEURISTIC_CONFIGS.items()
                        if k == args.heuristic_config}
        run_heuristic_comparison(args.n_inst, args.T, args.UP, output_dir,
                             args.data_dir, configs=hconfigs, quiet=args.quiet, seed=args.seed, tag=args.tag)

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Results saved in: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
