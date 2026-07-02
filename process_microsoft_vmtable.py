import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_resource_columns(df, cpu_col=None, memory_col=None):
    """
    Find CPU and memory columns automatically if they are not provided.

    Priority:
        CPU: prefer 'virtual_core_count' or 'core_count_bucket' over generic 'cpu'
        MEM: prefer 'memory_gb_bucket' or 'mem_gb' over generic 'memory'
    """
    if cpu_col is not None and memory_col is not None:
        return cpu_col, memory_col

    cpu_candidates = [
        col for col in df.columns
        if "cpu" in col.lower() or "core" in col.lower()
    ]
    memory_candidates = [
        col for col in df.columns
        if "memory" in col.lower() or "mem" in col.lower()
    ]

    if not cpu_candidates or not memory_candidates:
        raise ValueError(
            f"Cannot find CPU/memory columns automatically. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Prefer bucket columns (e.g. vm_virtual_core_count_bucket, vm_memory_gb_bucket)
    # over generic average/max reading columns.
    cpu_bucket = [c for c in cpu_candidates if "bucket" in c.lower()]
    mem_bucket = [c for c in memory_candidates if "bucket" in c.lower()]

    cpu_col = cpu_bucket[0] if cpu_bucket else cpu_candidates[0]
    memory_col = mem_bucket[0] if mem_bucket else memory_candidates[0]

    return cpu_col, memory_col




def load_vmtable(vmtable_file, cpu_col=None, memory_col=None):
    """
    Load Microsoft trace_data_vmtable_vmtable.csv and identify CPU/memory columns.

    trace_data_vmtable_vmtable.csv has NO header row (first row is data) with 11 columns:
        vm_id, subscription_id, deployment_id,
        timestamp_vm_created, timestamp_vm_deleted,
        max_cpu, avg_cpu, p95_max_cpu,
        vm_category, vm_virtual_core_count_bucket, vm_memory_gb_bucket
    """
    print(f"[INFO] Loading vmtable: {vmtable_file}")

    column_names = [
        'vm_id',
        'subscription_id',
        'deployment_id',
        'timestamp_vm_created',
        'timestamp_vm_deleted',
        'max_cpu',
        'avg_cpu',
        'p95_max_cpu',
        'vm_category',
        'vm_virtual_core_count_bucket',
        'vm_memory_gb_bucket',
    ]

    df = pd.read_csv(vmtable_file, names=column_names, header=None)
    raw_total = len(df)
    print(f"[INFO] Raw rows: {raw_total}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    cpu_col, memory_col = find_resource_columns(df, cpu_col, memory_col)
    print(f"[INFO] Using CPU column: {cpu_col}")
    print(f"[INFO] Using memory column: {memory_col}")

    df[cpu_col] = pd.to_numeric(df[cpu_col], errors="coerce")
    df[memory_col] = pd.to_numeric(df[memory_col], errors="coerce")
    df = df.dropna(subset=[cpu_col, memory_col])
    df = df[(df[cpu_col] > 0) & (df[memory_col] > 0)]

    return df, cpu_col, memory_col, raw_total


def count_cpu_memory_pairs(df, cpu_col, memory_col):
    """
    Count frequencies of (cpu, memory) pairs.
    """
    pair_counts = (
        df.groupby([cpu_col, memory_col])
        .size()
        .reset_index(name="count")
        .sort_values([cpu_col, memory_col])
    )
    return pair_counts


def save_pair_counts(pair_counts, output_csv):
    """
    Save (cpu, memory) frequency counts to CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pair_counts.to_csv(output_csv, index=False)
    print(f"[INFO] Pair counts saved to: {output_csv}")


def plot_pair_counts(pair_counts, cpu_col, memory_col, output_fig, max_labels=None):
    """
    Plot frequency bar chart for (cpu, memory) pairs.

    The x-axis is sorted by CPU ascending and then memory ascending.
    """
    output_fig = Path(output_fig)
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    plot_df = pair_counts.copy()
    if max_labels is not None and len(plot_df) > max_labels:
        plot_df = plot_df.head(max_labels)

    labels = [
        f"({cpu:g},{mem:g})"
        for cpu, mem in zip(plot_df[cpu_col], plot_df[memory_col])
    ]
    counts = plot_df["count"].tolist()

    if not labels:
        print("[WARN] No pairs to plot.")
        return

    fig_width = max(10, len(labels) * 0.35)
    plt.figure(figsize=(fig_width, 6))

    bars = plt.bar(
        range(len(labels)),
        counts,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.5,
    )

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("(cpu, memory) pairs")
    plt.ylabel("Frequency")
    plt.title("Frequency of Microsoft VM Types by (cpu, memory)")

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Frequency figure saved to: {output_fig}")


def add_vmpack_type_columns(df, cpu_col, memory_col, T):
    """
    Add model type columns for VMPack-style instances (vectorized).

    A VM is retained if:
        cpu is a power of two,
        memory / cpu is one of {1, 2, 4},
        t = log2(cpu) is in [0, T-1],
        s = log2(memory / cpu) is in {0, 1, 2}.
    """
    cpu = df[cpu_col]
    mem = df[memory_col]

    # Round to nearest integer (bucket columns are integer-like)
    cpu_int = cpu.round().astype(int)
    mem_int = mem.round().astype(int)

    # Vectorized power-of-two check: n > 0 and (n & (n-1)) == 0
    cpu_pow2 = (cpu_int > 0) & ((cpu_int & (cpu_int - 1)) == 0)

    # ratio = memory / cpu, must be in {1, 2, 4}
    ratio = mem / cpu
    ratio_ok = ratio.isin([1.0, 2.0, 4.0])

    # t = log2(cpu), s = log2(ratio), must be in valid ranges
    t = np.log2(cpu).round().astype(int)
    s = np.log2(ratio).round().astype(int)
    t_ok = (t >= 0) & (t < T)
    s_ok = (s >= 0) & (s <= 2)

    mask = cpu_pow2 & ratio_ok & t_ok & s_ok

    retained = df[mask].copy()
    retained["_cpu_int"] = cpu_int[mask]
    retained["_memory_int"] = mem_int[mask]
    retained["_s"] = s[mask]
    retained["_t"] = t[mask]

    if retained.empty:
        return pd.DataFrame(columns=list(df.columns) + ["_cpu_int", "_memory_int", "_s", "_t"])
    return retained


def chunk_to_instance(chunk_df, T):
    """
    Convert one VM chunk to a VMPack instance [vm0, vm1, vm2].

    vm_s[t] stores the number of VMs with:
        memory / cpu = 2^s
        cpu = 2^t
    """
    L = np.zeros((3, T), dtype=int)

    for _, row in chunk_df.iterrows():
        s = int(row["_s"])
        t = int(row["_t"])
        L[s, t] += 1

    return [L[0].tolist(), L[1].tolist(), L[2].tolist()]


def cpu_size(vm):
    """
    Compute total CPU demand of one VM type vector.
    """
    return sum((2 ** t) * int(vm[t]) for t in range(len(vm)))


def instance_summary(instance):
    """
    Compute simple summary statistics for one instance.
    """
    vm0, vm1, vm2 = instance
    C0 = cpu_size(vm0)
    C1 = cpu_size(vm1)
    C2 = cpu_size(vm2)

    total_cpu = C0 + C1 + C2
    total_mem = C0 + 2 * C1 + 4 * C2
    num_vms = sum(vm0) + sum(vm1) + sum(vm2)

    return {
        "num_vms": int(num_vms),
        "C0": int(C0),
        "C1": int(C1),
        "C2": int(C2),
        "total_cpu": int(total_cpu),
        "total_mem": int(total_mem),
        "mixalgos_bottleneck": bool(C1 == 0 and C0 < 3 * C2),
    }


def generate_instances(
    retained_df,
    T,
    batch_size,
    n_instances=None,
    shuffle=False,
    seed=42,
    allowed_classes=None,
):
    """
    Generate VMPack-style instances from retained Microsoft VM rows.

    By default, rows are split sequentially into chunks of batch_size.
    If shuffle=True, retained rows are shuffled before chunking.

    Parameters
    ----------
    allowed_classes : list of int, optional
        If given (e.g. [0, 2]), only VM rows with _s in this set are used.
        This is used for the mixalgos scenario where we skip L1 VMs.
    """
    if retained_df.empty:
        return [], []

    working_df = retained_df.copy()

    if allowed_classes is not None:
        working_df = working_df[working_df["_s"].isin(allowed_classes)]

    if shuffle:
        working_df = working_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        working_df = working_df.reset_index(drop=True)

    instances = []
    summaries = []

    max_instances = len(working_df) // batch_size
    if n_instances is not None:
        max_instances = min(max_instances, n_instances)

    for i in range(max_instances):
        start = i * batch_size
        end = start + batch_size
        chunk = working_df.iloc[start:end]

        instance = chunk_to_instance(chunk, T)
        summary = instance_summary(instance)
        summary["instance_id"] = i
        summary["batch_size"] = batch_size

        instances.append(instance)
        summaries.append(summary)

    return instances, summaries


def save_instances_json(instances, output_json):
    """
    Save generated instances in the same JSON structure used by data.py.
    """
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(instances, f, separators=(",", ":"))

    print(f"[INFO] VMPack instances saved to: {output_json}")


def save_instance_summary(summaries, output_csv):
    """
    Save per-instance summary statistics.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summaries).to_csv(output_csv, index=False)
    print(f"[INFO] Instance summary saved to: {output_csv}")


def save_analysis_report(
    output_txt,
    total_rows,
    retained_rows,
    cpu_col,
    memory_col,
    pair_counts,
    retained_pair_counts,
    summaries,
):
    """
    Save a text report for Microsoft vmtable analysis.
    """
    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    coverage = retained_rows / total_rows * 100 if total_rows > 0 else 0
    bottleneck_batches = sum(1 for s in summaries if s["mixalgos_bottleneck"])

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Microsoft vmtable analysis report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total VM rows: {total_rows}\n")
        f.write(f"Retained VM rows: {retained_rows}\n")
        f.write(f"Coverage: {coverage:.2f}%\n")
        f.write(f"CPU column: {cpu_col}\n")
        f.write(f"Memory column: {memory_col}\n\n")

        f.write("Top raw (cpu, memory) pairs:\n")
        top_raw = pair_counts.sort_values("count", ascending=False).head(20)
        for _, row in top_raw.iterrows():
            f.write(f"  ({row[cpu_col]}, {row[memory_col]}): {row['count']}\n")

        f.write("\nTop retained (cpu, memory) pairs:\n")
        if retained_pair_counts.empty:
            f.write("  None\n")
        else:
            top_retained = retained_pair_counts.sort_values("count", ascending=False).head(20)
            for _, row in top_retained.iterrows():
                f.write(f"  ({row['_cpu_int']}, {row['_memory_int']}): {row['count']}\n")

        f.write("\nGenerated instances:\n")
        f.write(f"  Total batches: {len(summaries)}\n")
        f.write(f"  Bottleneck batches: {bottleneck_batches}\n")

        if summaries:
            avg_vms = sum(s["num_vms"] for s in summaries) / len(summaries)
            f.write(f"  Average VMs per batch: {avg_vms:.2f}\n")

    print(f"[INFO] Analysis report saved to: {output_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Microsoft trace_data_vmtable_vmtable.csv and generate VMPack-style instances."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to trace_data_vmtable_vmtable.csv")
    parser.add_argument("--output_dir", type=str, default="./microsoft_vmtable_output/")
    parser.add_argument("--cpu_col", type=str, default=None, help="CPU column name")
    parser.add_argument("--memory_col", type=str, default=None, help="Memory column name")
    parser.add_argument("--T", type=int, default=7, help="Number of CPU sizes")
    parser.add_argument("--batch_size", type=int, default=1000, help="VMs per generated instance")
    parser.add_argument("--n_instances", type=int, default=None, help="Maximum number of instances")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle retained VMs before batching")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--max_plot_labels",
        type=int,
        default=None,
        help="Maximum number of pair labels to plot. Default plots all pairs.",
    )
    parser.add_argument(
        "--scenario", type=str, default="improvevmpack",
        choices=["mixalgos", "improvevmpack"],
        help="Instance generation scenario: "
             "'mixalgos' keeps only L0 and L2 VMs (two-class bottleneck); "
             "'improvevmpack' keeps all three classes.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, cpu_col, memory_col, raw_total = load_vmtable(args.input, args.cpu_col, args.memory_col)

    pair_counts = count_cpu_memory_pairs(df, cpu_col, memory_col)
    save_pair_counts(pair_counts, output_dir / "microsoft_cpu_memory_counts.csv")
    plot_pair_counts(
        pair_counts,
        cpu_col,
        memory_col,
        output_dir / "microsoft_cpu_memory_counts.png",
        max_labels=args.max_plot_labels,
    )

    retained_df = add_vmpack_type_columns(df, cpu_col, memory_col, args.T)

    retained_pair_counts = (
        retained_df.groupby(["_cpu_int", "_memory_int"])
        .size()
        .reset_index(name="count")
        .sort_values(["_cpu_int", "_memory_int"])
        if not retained_df.empty
        else pd.DataFrame(columns=["_cpu_int", "_memory_int", "count"])
    )

    retained_pair_counts.to_csv(
        output_dir / "microsoft_retained_cpu_memory_counts.csv",
        index=False,
    )

    print(f"[INFO] Retained rows: {len(retained_df)} / {len(df)} "
          f"({len(retained_df) / len(df) * 100:.2f}%)")

    # Determine allowed VM classes based on scenario
    if args.scenario == "mixalgos":
        allowed_classes = [0, 2]  # L0 and L2 only, skip L1
        scenario_suffix = "mixalgos"
        print(f"[INFO] Scenario '{args.scenario}': keeping only L0 (s=0) and L2 (s=2) VMs.")
    else:
        allowed_classes = None  # all three classes
        scenario_suffix = "improvevmpack"

    instances, summaries = generate_instances(
        retained_df,
        T=args.T,
        batch_size=args.batch_size,
        n_instances=args.n_instances,
        shuffle=args.shuffle,
        seed=args.seed,
        allowed_classes=allowed_classes,
    )

    save_instances_json(instances, output_dir / f"microsoft_vmpack_instances_{scenario_suffix}.json")
    save_instance_summary(summaries, output_dir / f"microsoft_vmpack_instances_{scenario_suffix}_summary.csv")

    save_analysis_report(
        output_dir / f"microsoft_vmtable_analysis_report_{scenario_suffix}.txt",
        total_rows=raw_total,
        retained_rows=len(retained_df),
        cpu_col=cpu_col,
        memory_col=memory_col,
        pair_counts=pair_counts,
        retained_pair_counts=retained_pair_counts,
        summaries=summaries,
    )

    print("\n[SUMMARY]")
    print(f"  Raw VM rows (file line count): {raw_total}")
    print(f"  Loaded numeric rows: {len(df)}")
    print(f"  Retained VM rows: {len(retained_df)}")
    print(f"  Generated instances: {len(instances)}")
    if summaries:
        bottleneck_batches = sum(1 for s in summaries if s["mixalgos_bottleneck"])
        print(f"  Bottleneck batches: {bottleneck_batches}")


if __name__ == "__main__":
    main()

'''
python process_microsoft_vmtable.py ^
  --input "raw_data/trace_data_vmtable_vmtable.csv" ^
  --output_dir "./microsoft_vmtable_output/" ^
  --T 7 ^
  --batch_size 1000 ^
  --n_instances 100 ^
  --shuffle ^
  --seed 42
'''