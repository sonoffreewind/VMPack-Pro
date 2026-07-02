import csv,math
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import Counter, OrderedDict

import matplotlib.pyplot as plt


def detect_delimiter(input_path):
    """
    Detect whether the input file is comma-separated or tab-separated.
    """
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)

    return "\t" if sample.count("\t") > sample.count(",") else ","


def parse_number(x):
    """
    Parse a numeric field. If the value is integer-like, return int;
    otherwise return float.
    """
    v = float(x)
    if v.is_integer():
        return int(v)
    return v


def load_trace(input_path):
    """
    Load Huawei VM trace records.

    Expected columns:
        vmid, cpu, memory, time, type

    type:
        0 = create
        1 = delete

    Returns
    -------
    records : list of dict
        Each record contains:
            vmid, cpu, memory, time, type
    """
    input_path = Path(input_path)
    delimiter = detect_delimiter(input_path)

    records = []
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        required_cols = {"vmid", "cpu", "memory", "time", "type"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            try:
                records.append({
                    "vmid": str(row["vmid"]),
                    "cpu": parse_number(row["cpu"]),
                    "memory": parse_number(row["memory"]),
                    "time": float(row["time"]),
                    "type": int(row["type"]),
                })
            except (ValueError, TypeError, KeyError):
                # Skip malformed rows.
                continue

    return records


def count_create_vm_by_cpu_mem(records):
    """
    Count type=0 VM creation events by (cpu, memory).
    """
    counter = Counter()

    for r in records:
        if r["type"] == 0:
            counter[(r["cpu"], r["memory"])] += 1

    return counter


def save_counter_csv(counter, output_csv):
    """
    Save a Counter[(cpu, memory)] to CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(counter.items(), key=lambda x: (x[0][0], x[0][1]))

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpu", "memory", "count"])
        for (cpu, memory), count in sorted_items:
            writer.writerow([cpu, memory, count])


def plot_counter(counter, output_fig, title):
    """
    Plot a frequency bar chart for Counter[(cpu, memory)].
    The x-axis is sorted by cpu ascending and then memory ascending.
    """
    output_fig = Path(output_fig)
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(counter.items(), key=lambda x: (x[0][0], x[0][1]))

    labels = [f"({cpu:g},{mem:g})" for (cpu, mem), _ in sorted_items]
    counts = [count for _, count in sorted_items]

    if not labels:
        print(f"[WARN] No data to plot for {output_fig}")
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
    plt.ylabel("Frequency")
    plt.xlabel("(cpu, memory) pairs")
    plt.title(title)

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


def active_set_to_instance(active_set):
    """
    Convert the current active VM set into an instance represented by
    frequency counts of (cpu, memory).

    Parameters
    ----------
    active_set : dict
        vmid -> (cpu, memory)

    Returns
    -------
    instance_counter : Counter
        Counter[(cpu, memory)] = frequency
    """
    counter = Counter()
    for cpu, memory in active_set.values():
        counter[(cpu, memory)] += 1
    return counter

def is_power_of_two_int(n):
    """
    Check whether n is a positive power of two.
    """
    try:
        n = int(round(float(n)))
    except Exception:
        return False
    return n > 0 and (n & (n - 1)) == 0


def map_cpu_memory_to_vmpack_index(cpu, memory, T):
    """
    Map one VM type (cpu, memory) to (s, t) in the VMPack model.

    Conditions:
        cpu = 2^t
        memory / cpu in {1, 2, 4}
        t in [0, T-1]
        s in {0, 1, 2}

    Returns
    -------
    (s, t) if valid, otherwise None
    """
    try:
        cpu_int = int(round(float(cpu)))
        mem_int = int(round(float(memory)))
    except Exception:
        return None

    if cpu_int <= 0 or mem_int <= 0:
        return None

    if not is_power_of_two_int(cpu_int):
        return None

    if mem_int % cpu_int != 0:
        return None

    ratio = mem_int // cpu_int
    if ratio not in (1, 2, 4):
        return None

    if not is_power_of_two_int(ratio):
        return None

    t = int(round(math.log2(cpu_int)))
    s = int(round(math.log2(ratio)))

    if not (0 <= t < T):
        return None
    if not (0 <= s <= 2):
        return None

    return s, t


def snapshot_to_vmpack_instance(snapshot, T, strict=False, allowed_classes=None):
    """
    Convert one active-set snapshot to a VMPack instance [vm0, vm1, vm2].

    Parameters
    ----------
    snapshot : dict
        One snapshot produced by generate_active_set_instances().
        Must contain the key "counts".
    T : int
        Number of CPU sizes.
    strict : bool
        If True, discard the whole snapshot when any VM type is not mappable.
    allowed_classes : list of int, optional
        If given (e.g. [0, 2]), only VMs with s in this set are kept.
        Used for mixalgos scenario where L1 is skipped.

    Returns
    -------
    instance : list or None
        [vm0, vm1, vm2] if successful, otherwise None.
    info : dict
        Summary info for this snapshot.
    """
    L = np.zeros((3, T), dtype=int)

    total_vms = 0
    retained_vms = 0
    dropped_vms = 0
    dropped_types = 0

    for item in snapshot["counts"]:
        cpu = item["cpu"]
        memory = item["memory"]
        count = int(item["count"])
        total_vms += count

        mapped = map_cpu_memory_to_vmpack_index(cpu, memory, T)
        if mapped is None:
            if strict:
                return None, {
                    "valid": False,
                    "reason": "unmappable_vm_type",
                    "total_vms": total_vms,
                    "retained_vms": retained_vms,
                    "dropped_vms": dropped_vms + count,
                    "dropped_types": dropped_types + 1,
                }
            dropped_vms += count
            dropped_types += 1
            continue

        s, t = mapped

        # Scenario filtering: skip L1 VMs in mixalgos mode
        if allowed_classes is not None and s not in allowed_classes:
            dropped_vms += count
            dropped_types += 1
            continue

        L[s][t] += count
        retained_vms += count

    info = {
        "valid": True,
        "reason": "",
        "total_vms": total_vms,
        "retained_vms": retained_vms,
        "dropped_vms": dropped_vms,
        "dropped_types": dropped_types,
    }

    return [L[0].tolist(), L[1].tolist(), L[2].tolist()], info


def export_vmpack_instances(instances, T, output_json, strict=False, summary_csv=None, allowed_classes=None):
    """
    Export active-set snapshots to VMPack JSON format.

    The JSON format is a plain list:
        [
          [vm0, vm1, vm2],
          ...
        ]

    If strict=False, unmappable VM types are skipped.
    If strict=True, any snapshot containing an unmappable VM type is discarded.

    Parameters
    ----------
    allowed_classes : list of int, optional
        If given (e.g. [0, 2]), only VMs with s in this set are kept.
    """
    exported = []
    summary_rows = []

    for inst in instances:
        converted, info = snapshot_to_vmpack_instance(inst, T, strict=strict, allowed_classes=allowed_classes)
        if converted is None:
            continue

        exported.append(converted)
        summary_rows.append({
            "instance_id": inst["instance_id"],
            "snapshot_time": inst["snapshot_time"],
            "num_active_vms": inst["num_active_vms"],
            "num_vm_types": inst["num_vm_types"],
            "total_vms": info["total_vms"],
            "retained_vms": info["retained_vms"],
            "dropped_vms": info["dropped_vms"],
            "dropped_types": info["dropped_types"],
            "valid": info["valid"],
        })

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(exported, f, separators=(",", ":"))

    print(f"[INFO] VMPack JSON saved to: {output_json}")
    print(f"[INFO] Exported VMPack instances: {len(exported)}")

    if summary_csv is not None:
        summary_csv = Path(summary_csv)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"[INFO] VMPack summary saved to: {summary_csv}")

    return exported, summary_rows

def generate_active_set_instances(
    records,
    snapshot_interval=None,
    snapshot_every_events=None,
    min_active_vms=1,
    include_final_snapshot=True,
):
    """
    Generate trace-derived packing instances using active-set snapshots.

    The event stream is scanned in chronological order:
        - type=0: add VM to active set
        - type=1: remove VM from active set

    A snapshot instance is generated in either of the following ways:
        1. Every fixed time interval, controlled by snapshot_interval.
        2. Every fixed number of processed events, controlled by snapshot_every_events.

    At least one of snapshot_interval or snapshot_every_events should be set.
    If both are set, both conditions can trigger snapshots.

    Parameters
    ----------
    records : list of dict
        Trace records sorted by time.
    snapshot_interval : float or None
        Generate one snapshot whenever current time reaches the next sampling time.
        The unit is the same as the trace time column.
    snapshot_every_events : int or None
        Generate one snapshot every N processed events.
    min_active_vms : int
        Discard snapshots with fewer than this number of active VMs.
    include_final_snapshot : bool
        Whether to save one final snapshot after all events are processed.

    Returns
    -------
    instances : list of dict
        Each instance contains:
            instance_id
            snapshot_time
            num_active_vms
            counts
    stats : dict
        Summary statistics.
    """
    if snapshot_interval is None and snapshot_every_events is None and not include_final_snapshot:
        raise ValueError(
            "At least one snapshot rule should be enabled: "
            "snapshot_interval, snapshot_every_events, or include_final_snapshot."
        )

    active_set = OrderedDict()
    instances = []

    create_events = 0
    delete_events = 0
    unmatched_delete_events = 0
    duplicate_create_events = 0

    if records:
        next_snapshot_time = records[0]["time"] + snapshot_interval if snapshot_interval else None
    else:
        next_snapshot_time = None

    def save_snapshot(snapshot_time):
        """
        Save the current active set as one instance if it is large enough.
        """
        if len(active_set) < min_active_vms:
            return

        counter = active_set_to_instance(active_set)

        counts = [
            {
                "cpu": cpu,
                "memory": memory,
                "count": count,
            }
            for (cpu, memory), count in sorted(counter.items(), key=lambda x: (x[0][0], x[0][1]))
        ]

        instances.append({
            "instance_id": len(instances),
            "snapshot_time": snapshot_time,
            "num_active_vms": len(active_set),
            "num_vm_types": len(counter),
            "counts": counts,
        })

    for idx, r in enumerate(records, start=1):
        vmid = r["vmid"]
        cpu = r["cpu"]
        memory = r["memory"]
        t = r["time"]
        event_type = r["type"]

        if event_type == 0:
            create_events += 1
            if vmid in active_set:
                duplicate_create_events += 1
            active_set[vmid] = (cpu, memory)

        elif event_type == 1:
            delete_events += 1
            if vmid in active_set:
                del active_set[vmid]
            else:
                unmatched_delete_events += 1

        # Time-based snapshots.
        if snapshot_interval is not None:
            while next_snapshot_time is not None and t >= next_snapshot_time:
                save_snapshot(next_snapshot_time)
                next_snapshot_time += snapshot_interval

        # Event-count-based snapshots.
        if snapshot_every_events is not None and idx % snapshot_every_events == 0:
            save_snapshot(t)

    if include_final_snapshot and records:
        save_snapshot(records[-1]["time"])

    stats = {
        "raw_records": len(records),
        "create_events": create_events,
        "delete_events": delete_events,
        "unmatched_delete_events": unmatched_delete_events,
        "duplicate_create_events": duplicate_create_events,
        "generated_instances": len(instances),
        "min_active_vms": min_active_vms,
        "snapshot_interval": snapshot_interval,
        "snapshot_every_events": snapshot_every_events,
        "include_final_snapshot": include_final_snapshot,
    }

    if instances:
        stats["avg_active_vms"] = sum(x["num_active_vms"] for x in instances) / len(instances)
        stats["max_active_vms"] = max(x["num_active_vms"] for x in instances)
        stats["min_active_vms_in_instances"] = min(x["num_active_vms"] for x in instances)
    else:
        stats["avg_active_vms"] = 0
        stats["max_active_vms"] = 0
        stats["min_active_vms_in_instances"] = 0

    return instances, stats


def save_instances_json(instances, stats, output_json):
    """
    Save active-set snapshot instances to JSON.
    """
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": stats,
        "instances": instances,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_instances_summary_csv(instances, output_csv):
    """
    Save a compact summary of generated active-set instances.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_id",
            "snapshot_time",
            "num_active_vms",
            "num_vm_types",
        ])

        for inst in instances:
            writer.writerow([
                inst["instance_id"],
                inst["snapshot_time"],
                inst["num_active_vms"],
                inst["num_vm_types"],
            ])


def save_each_instance_counter_csv(instances, output_dir):
    """
    Save each generated instance as an individual CSV file.
    Each CSV contains columns:
        cpu, memory, count
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for inst in instances:
        output_csv = output_dir / f"instance_{inst['instance_id']:04d}.csv"

        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cpu", "memory", "count"])
            for item in inst["counts"]:
                writer.writerow([item["cpu"], item["memory"], item["count"]])


def main():
    parser = argparse.ArgumentParser(
        description="Process Huawei VM trace: count creation events and generate active-set snapshot instances."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input Huawei VM trace file path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./huawei_trace_output/",
        help="Output directory.",
    )

    parser.add_argument(
        "--snapshot_interval",
        type=float,
        default=None,
        help="Time interval for active-set snapshots. Unit follows the trace time column.",
    )

    parser.add_argument(
        "--snapshot_every_events",
        type=int,
        default=None,
        help="Generate one active-set snapshot every N processed events.",
    )

    parser.add_argument(
        "--min_active_vms",
        type=int,
        default=1,
        help="Discard snapshots with fewer than this number of active VMs.",
    )

    parser.add_argument(
        "--no_final_snapshot",
        action="store_true",
        help="Disable saving the final active-set snapshot.",
    )

    parser.add_argument(
        "--save_each_instance_csv",
        action="store_true",
        help="Save each generated instance as a separate CSV file.",
    )

    parser.add_argument(
        "--export_vmpack_json",
        action="store_true",
        help="Export generated active-set snapshots to VMPack JSON format.",
    )

    parser.add_argument(
        "--vmpack_strict",
        action="store_true",
        help="Discard snapshots containing any unmappable VM types.",
    )

    parser.add_argument(
        "--vmpack_json_name",
        type=str,
        default="huawei_vmpack_instances.json",
        help="Output filename for exported VMPack JSON.",
    )

    parser.add_argument(
        "--vmpack_summary_name",
        type=str,
        default="huawei_vmpack_instances_summary.csv",
        help="Output filename for exported VMPack summary CSV.",
    )

    parser.add_argument(
        "--T",
        type=int,
        default=7,
        help="Number of CPU sizes used by the VMPack model.",
    )

    parser.add_argument(
        "--scenario", type=str, default="improvevmpack",
        choices=["mixalgos", "improvevmpack"],
        help="Instance generation scenario: "
             "'mixalgos' keeps only L0 and L2 VMs (two-class bottleneck); "
             "'improvevmpack' keeps all three classes.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading trace from: {input_path}")
    records = load_trace(input_path)
    print(f"[INFO] Loaded records: {len(records)}")

    # Part 1: Count creation events by (cpu, memory).
    create_counter = count_create_vm_by_cpu_mem(records)

    create_stats_csv = output_dir / "huawei_create_vm_stats.csv"
    create_stats_fig = output_dir / "huawei_create_vm_stats.png"

    save_counter_csv(create_counter, create_stats_csv)
    plot_counter(
        create_counter,
        create_stats_fig,
        title="Frequency of Huawei VM Creation Events by (cpu, memory)",
    )

    print(f"[INFO] Creation-event statistics saved to: {create_stats_csv}")
    print(f"[INFO] Creation-event frequency figure saved to: {create_stats_fig}")

    # Part 2: Generate active-set snapshot instances.
    instances, stats = generate_active_set_instances(
        records,
        snapshot_interval=args.snapshot_interval,
        snapshot_every_events=args.snapshot_every_events,
        min_active_vms=args.min_active_vms,
        include_final_snapshot=not args.no_final_snapshot,
    )

    # Determine allowed VM classes based on scenario
    if args.scenario == "mixalgos":
        allowed_classes = [0, 2]  # L0 and L2 only, skip L1
        scenario_suffix = "mixalgos"
        print(f"[INFO] Scenario '{args.scenario}': keeping only L0 (s=0) and L2 (s=2) VMs.")
    else:
        allowed_classes = None  # all three classes
        scenario_suffix = "improvevmpack"

    instances_json = output_dir / "huawei_active_set_instances.json"
    instances_summary_csv = output_dir / "huawei_active_set_instances_summary.csv"

    save_instances_json(instances, stats, instances_json)
    save_instances_summary_csv(instances, instances_summary_csv)

    if args.export_vmpack_json:
        export_vmpack_instances(
            instances,
            T=args.T,
            output_json=output_dir / f"huawei_vmpack_instances_{scenario_suffix}.json",
            strict=args.vmpack_strict,
            summary_csv=output_dir / f"huawei_vmpack_instances_{scenario_suffix}_summary.csv",
            allowed_classes=allowed_classes,
        )

    if args.save_each_instance_csv:
        save_each_instance_counter_csv(instances, output_dir / "instances_csv")

    print(f"[INFO] Active-set instances saved to: {instances_json}")
    print(f"[INFO] Active-set instance summary saved to: {instances_summary_csv}")

    print("\n[SUMMARY]")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
'''
python process_huawei_trace.py \
  --input huawei_vm_records.csv \
  --output_dir ./huawei_trace_output/ \
  --snapshot_every_events 10000 \
  --min_active_vms 100 \
  --export_vmpack_json \
  --T 7
'''