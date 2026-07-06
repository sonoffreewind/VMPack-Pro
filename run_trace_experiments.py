"""
run_trace_experiments.py

Run heuristic algorithms on public-trace-derived VMPack instances.

Input JSON format:
    [
      [vm0, vm1, vm2],
      [vm0, vm1, vm2],
      ...
    ]

Each vm0/vm1/vm2 is a length-T list.

Example:
    python run_trace_experiments.py \
      --input ./huawei_trace_output/huawei_vmpack_instances.json \
      --trace_name Huawei \
      --T 7 \
      --UP 1000 \
      --output_dir ./result/ \
      --tag huawei

    python run_trace_experiments.py \
      --input ./microsoft_vmtable_output/microsoft_vmpack_instances.json \
      --trace_name Microsoft \
      --T 7 \
      --UP 1000 \
      --output_dir ./result/ \
      --tag microsoft
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

import globalvars as gv
from basic import CpuSize
from heuristics import (
    FFD,
    BFD,
    VMPack_MixVM301,
    VMPack_MixVM201Pro,
    VMPack_MixPack,
    SafeMix,
    VMPack_SafeMix,
)
from utils import bootstrap_mean_ci, cliffs_delta


ALGORITHMS = {
    "FFD": FFD,
    "BFD": BFD,
    "VMPack_MixVM301": VMPack_MixVM301,
    "VMPack_MixVM201Pro": VMPack_MixVM201Pro,
    "VMPack_MixPack": VMPack_MixPack,
    # SafeMix two-class heuristic and full-pipeline version
    "SafeMix": SafeMix,
    "VMPack_SafeMix": VMPack_SafeMix,
}


def load_trace_instances(input_path, T):
    """
    Load VMPack-style instances from JSON.

    Expected format:
        [
          [vm0, vm1, vm2],
          ...
        ]
    """
    input_path = Path(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "instances" in payload:
        raise ValueError(
            "The input appears to be active-set snapshot JSON, not VMPack JSON. "
            "Please use --export_vmpack_json in process_huawei_trace.py first."
        )

    instances = []
    for idx, item in enumerate(payload):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"Invalid instance at index {idx}: expected [vm0, vm1, vm2].")

        arr = np.array(item, dtype=int)
        if arr.shape != (3, T):
            raise ValueError(
                f"Invalid instance shape at index {idx}: got {arr.shape}, expected (3, {T})."
            )

        instances.append(arr)

    return instances


def lower_bound(vm_demands):
    """
    Compute standard 2-D resource lower bound.
    """
    C = gv.C
    vm0, vm1, vm2 = vm_demands[0], vm_demands[1], vm_demands[2]

    C0 = CpuSize(vm0)
    C1 = CpuSize(vm1)
    C2 = CpuSize(vm2)

    total_cpu = C0 + C1 + C2
    total_mem = C0 + 2 * C1 + 4 * C2

    lb = int(np.ceil(max(total_cpu / C, total_mem / (2 * C))))

    return lb, total_cpu, total_mem, C0, C1, C2


def filter_bottleneck_instances(instances):
    """
    Keep only instances where C1==0 and C0 < 3*C2 (the VMPack bottleneck condition).

    This matches the paper's claim that trace experiments are conducted on
    bottleneck batches.
    """
    filtered = []
    dropped = 0
    for vm_demands in instances:
        C0 = CpuSize(vm_demands[0])
        C1 = CpuSize(vm_demands[1])
        C2 = CpuSize(vm_demands[2])
        if C1 == 0 and C0 < 3 * C2:
            filtered.append(vm_demands)
        else:
            dropped += 1
    if dropped:
        print(f"[INFO] Bottleneck filter: dropped {dropped}/{len(instances)} instances, "
              f"kept {len(filtered)}.")
    return filtered


def run_one_algorithm(vm_demands, algo_fn):
    """
    Run one heuristic and normalize its return value to PM count.
    """
    t0 = time.time()
    result = algo_fn(vm_demands)
    elapsed = time.time() - t0

    if isinstance(result, tuple):
        npms = result[0]
    else:
        npms = result

    if isinstance(npms, list):
        npms = len(npms)

    return int(npms), elapsed


def summarize_rows(rows):
    """
    Aggregate per-instance rows by algorithm.
    """
    summaries = []

    algorithms = sorted(set(r["algorithm"] for r in rows))
    for algo in algorithms:
        algo_rows = [r for r in rows if r["algorithm"] == algo]

        npms = [float(r["npms"]) for r in algo_rows]
        times = [float(r["time"]) for r in algo_rows]
        lbs = [float(r["lb"]) for r in algo_rows if float(r["lb"]) > 0]

        ratios = [
            float(r["npms"]) / float(r["lb"])
            for r in algo_rows
            if float(r["lb"]) > 0
        ]

        total_cpu = sum(float(r["total_cpu"]) for r in algo_rows)
        total_mem = sum(float(r["total_mem"]) for r in algo_rows)
        total_bins = sum(float(r["npms"]) for r in algo_rows)

        cpu_util = total_cpu / (total_bins * gv.C) if total_bins > 0 else 0
        mem_util = total_mem / (total_bins * gv.M) if total_bins > 0 else 0

        summaries.append({
            "algorithm": algo,
            "n_instances": len(algo_rows),
            "avg_npms": np.mean(npms) if npms else None,
            "avg_time_ms": np.mean(times) * 1000 if times else None,
            "avg_ratio": np.mean(ratios) if ratios else None,
            "max_ratio": max(ratios) if ratios else None,
            "cpu_util_percent": cpu_util * 100,
            "mem_util_percent": mem_util * 100,
        })

    return summaries


def _wilcoxon_signed_rank_pvalue(diffs):
    """Two-sided Wilcoxon signed-rank test p-value (normal approximation),
    implemented without scipy so no new dependency is introduced.

    Same implementation as run_experiments._wilcoxon_signed_rank_pvalue;
    duplicated here to avoid a circular import.
    """
    import math
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 1:
        return None
    abs_d = sorted(abs(d) for d in nz)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1] == abs_d[i]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    w_plus = 0.0
    for d, r in zip(sorted(nz, key=abs), ranks):
        if d > 0:
            w_plus += r
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
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p


def compute_win_tie_loss(rows, baseline_algo, target_algo):
    """
    Compare two algorithms instance by instance.

    Output convention:
        baseline_win: baseline uses fewer PMs
        tie: same PM count
        target_win: target uses fewer PMs

    Also computes a two-sided Wilcoxon signed-rank p-value on the paired
    PM-count differences (baseline - target), reported as `wilcoxon_pvalue`.
    """
    by_instance = {}

    for r in rows:
        inst = int(r["instance"])
        algo = r["algorithm"]
        by_instance.setdefault(inst, {})[algo] = int(r["npms"])

    baseline_win = 0
    tie = 0
    target_win = 0
    compared = 0
    baseline_vals = []
    target_vals = []

    for _, vals in by_instance.items():
        if baseline_algo not in vals or target_algo not in vals:
            continue

        compared += 1
        baseline_npms = vals[baseline_algo]
        target_npms = vals[target_algo]
        baseline_vals.append(baseline_npms)
        target_vals.append(target_npms)

        if baseline_npms < target_npms:
            baseline_win += 1
        elif baseline_npms > target_npms:
            target_win += 1
        else:
            tie += 1

    # Wilcoxon signed-rank test on paired differences (baseline - target).
    # A positive difference means baseline uses more PMs, i.e. target is better.
    diffs = [b - t for b, t in zip(baseline_vals, target_vals)]
    p_value = _wilcoxon_signed_rank_pvalue(diffs)

    # 95% bootstrap CI for mean paired PM difference and Cliff's delta.
    mean_diff, ci_lo, ci_hi = bootstrap_mean_ci(diffs, n_boot=1000, alpha=0.05, seed=42)
    delta = cliffs_delta(diffs)

    return {
        "baseline_algo": baseline_algo,
        "target_algo": target_algo,
        "compared_instances": compared,
        "baseline_win": baseline_win,
        "tie": tie,
        "target_win": target_win,
        "wilcoxon_pvalue": p_value,
        "mean_diff_pm": round(mean_diff, 4),
        "ci95_low": round(ci_lo, 4),
        "ci95_high": round(ci_hi, 4),
        "cliffs_delta": round(delta, 4),
    }


def tagged_name(base, tag, ext):
    """
    Build output filename with optional tag.
    """
    if tag:
        return f"{base}_{tag}.{ext}"
    return f"{base}.{ext}"


def save_csv(rows, output_path):
    """
    Save list of dictionaries to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[WARN] No rows to save: {output_path}")
        return

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved: {output_path}")


def print_latex_table13(trace_name, summaries):
    """
    Print LaTeX-style rows for trace performance table.
    """
    print("\n% ===== Trace Performance Summary =====")
    print("% Trace & Algorithm & Avg. #PMs & Time(ms) & Avg. rho & Max rho & CPU util & MEM util \\\\")
    for s in summaries:
        print(
            f"{trace_name} & {s['algorithm']} & "
            f"{s['avg_npms']:.2f} & {s['avg_time_ms']:.2f} & "
            f"{s['avg_ratio']:.3f} & {s['max_ratio']:.3f} & "
            f"{s['cpu_util_percent']:.2f}\\% & {s['mem_util_percent']:.2f}\\% \\\\"
        )


def print_latex_table14(trace_name, win_rows):
    """
    Print LaTeX-style rows for win/tie/loss table.
    """
    print("\n% ===== Trace Win/Tie/Loss Summary =====")
    print("% Trace & Baseline & Target & Baseline wins & Ties & Target wins \\\\")
    for r in win_rows:
        print(
            f"{trace_name} & {r['baseline_algo']} & {r['target_algo']} & "
            f"{r['baseline_win']} & {r['tie']} & {r['target_win']} \\\\"
        )


def run_trace_experiment(args):
    """
    Main experiment runner.
    """
    gv.InitialGlobalVars(args.T, args.UP)

    instances = load_trace_instances(args.input, args.T)

    if args.bottleneck_only:
        instances = filter_bottleneck_instances(instances)

    if args.max_instances is not None:
        instances = instances[:args.max_instances]

    output_dir = Path(args.output_dir) / 'trace'
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, vm_demands in enumerate(instances):
        lb, total_cpu, total_mem, C0, C1, C2 = lower_bound(vm_demands)
        num_vms = int(np.sum(vm_demands))
        bottleneck_mixalgos = bool(C1 == 0 and C0 < 3 * C2)

        for algo_name, algo_fn in ALGORITHMS.items():
            npms, elapsed = run_one_algorithm(vm_demands, algo_fn)

            rows.append({
                "trace": args.trace_name,
                "instance": idx,
                "algorithm": algo_name,
                "T": args.T,
                "UP": args.UP,
                "C": gv.C,
                "M": gv.M,
                "num_vms": num_vms,
                "C0": C0,
                "C1": C1,
                "C2": C2,
                "total_cpu": total_cpu,
                "total_mem": total_mem,
                "lb": lb,
                "npms": npms,
                "time": elapsed,
                "rho": npms / lb if lb > 0 else "",
                "mixalgos_bottleneck": bottleneck_mixalgos,
            })

        if not args.quiet and (idx + 1) % 10 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(instances)} instances")

    detail_path = output_dir / tagged_name(f"{args.trace_name.lower()}_trace_detail", args.tag, "csv")
    save_csv(rows, detail_path)

    summaries = summarize_rows(rows)
    summary_path = output_dir / tagged_name(f"{args.trace_name.lower()}_trace_summary", args.tag, "csv")
    save_csv(summaries, summary_path)

    win_rows = []

    comparisons = [
        ("VMPack_MixVM301", "VMPack_MixVM201Pro"),
        ("VMPack_MixVM201Pro", "VMPack_MixPack"),
        ("BFD", "VMPack_MixVM201Pro"),
        ("FFD", "VMPack_MixVM201Pro"),
        ("VMPack_MixVM201Pro", "VMPack_SafeMix"),
    ]

    for baseline, target in comparisons:
        win_row = compute_win_tie_loss(rows, baseline, target)
        win_row["trace"] = args.trace_name
        win_rows.append(win_row)

    win_path = output_dir / tagged_name(f"{args.trace_name.lower()}_trace_wtl", args.tag, "csv")
    save_csv(win_rows, win_path)

    metadata = {
        "trace_name": args.trace_name,
        "input": str(args.input),
        "n_instances": len(instances),
        "T": args.T,
        "UP": args.UP,
        "C": gv.C,
        "M": gv.M,
        "tag": args.tag,
        "algorithms": list(ALGORITHMS.keys()),
    }
    metadata_path = output_dir / tagged_name(f"{args.trace_name.lower()}_trace_metadata", args.tag, "json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved: {metadata_path}")

    if args.print_latex:
        print_latex_table13(args.trace_name, summaries)
        print_latex_table14(args.trace_name, win_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run heuristic experiments on public-trace-derived VMPack instances."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input VMPack JSON generated from public trace.")
    parser.add_argument("--trace_name", type=str, required=True,
                        help="Trace name, e.g., Huawei or Microsoft.")
    parser.add_argument("--T", type=int, default=7,
                        help="Number of CPU sizes.")
    parser.add_argument("--UP", type=int, default=1000,
                        help="UP parameter used for metadata consistency.")
    parser.add_argument("--output_dir", type=str, default="./result/",
                        help="Output directory.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional output tag.")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Use only the first N instances for debugging.")
    parser.add_argument("--bottleneck_only", action="store_true",
                        help="Keep only bottleneck instances (C1==0 and C0<3*C2).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress logs.")
    parser.add_argument("--print_latex", action="store_true",
                        help="Print LaTeX-style table rows.")

    args = parser.parse_args()
    run_trace_experiment(args)


if __name__ == "__main__":
    main()
