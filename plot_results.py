"""
plot_results.py

Generate paper figures from experiment CSV files.

Supported figures:
  fig5 : Runtime-quality scatter for heuristic algorithms (both configs).
  fig6 : Win/tie/loss heatmap from trace experiments.
  fig7 : Warm-start impact on time-limited exact solvers.
  fig4 : Quality trend across UP/scale for heuristic algorithms.

Examples:
  python plot_results.py --figure fig5 --output_dir ./result/
  python plot_results.py --figure fig6 --output_dir ./result/
  python plot_results.py --figure fig7 --output_dir ./result/ --tag tl10
  python plot_results.py --figure all  --output_dir ./result/
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from utils import load_csv, is_valid, to_float, SCALES, UP_SWEEP


def setup_matplotlib():
    """
    Configure matplotlib style for paper figures.
    """
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def ensure_fig_dir(output_dir):
    """
    Create figure output directory.
    """
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_figure(path):
    """
    Save figure as both PDF and PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved: {path.with_suffix('.pdf')}")
    print(f"[INFO] Saved: {path.with_suffix('.png')}")


def group_heuristic_rows(rows):
    """
    Group heuristic CSV rows by algorithm.
    """
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get("algorithm", "")].append(r)
    return grouped


def plot_fig4_quality_trend(output_dir, tag, config):
    """
    Figure 4 (pm_usage_scales):
    Plot average PM usage over UP values for heuristic algorithms.

    Shows 4 curves: BFD, VMPack+MixVM301, VMPack+MixVM201Pro, VMPack+MixPack
    (using improvevmpack config data).

    Reads from wide-format CSVs in {output_dir}/improvevmpack/random_*.csv
    """
    setup_matplotlib()
    fig_dir = ensure_fig_dir(output_dir)

    # Map UP values → wide-format scale filenames
    up_to_scale = {
        10: 'random_s1', 20: 'random_s2', 50: 'random_m1',
        100: 'random_m2', 500: 'random_l1', 1000: 'random_l2',
    }

    # Only show these 4 algorithms (paper Figure 4)
    target_algos = ['BFD', 'VMPack_MixVM301', 'VMPack_MixVM201Pro', 'VMPack_MixPack']
    algo_labels = {
        'BFD': 'BFD',
        'VMPack_MixVM301': 'VMPack+MixVM301',
        'VMPack_MixVM201Pro': 'VMPack+MixVM201Pro',
        'VMPack_MixPack': 'VMPack+MixPack',
    }
    algo_colors = {
        'BFD': '#d62728',
        'VMPack_MixVM301': '#ff7f0e',
        'VMPack_MixVM201Pro': '#2ca02c',
        'VMPack_MixPack': '#1f77b4',
    }
    algo_markers = {
        'BFD': 'v',
        'VMPack_MixVM301': 's',
        'VMPack_MixVM201Pro': 'o',
        'VMPack_MixPack': 'D',
    }

    per_algo = defaultdict(list)
    per_algo_rho = defaultdict(list)
    available_ups = []

    for up in UP_SWEEP:
        scale_file = up_to_scale.get(up)
        if scale_file is None:
            continue

        csv_path = Path(output_dir) / 'improvevmpack' / f'{scale_file}.csv'
        if not csv_path.exists():
            continue

        rows = load_csv(csv_path)
        available_ups.append(up)

        for algo in target_algos:
            npms_vals = [to_float(r.get(f"{algo}_npms")) for r in rows
                         if to_float(r.get(f"{algo}_npms")) is not None]
            lb_vals = [to_float(r.get("lb")) for r in rows
                       if to_float(r.get("lb")) is not None and to_float(r.get(f"{algo}_npms")) is not None]
            if npms_vals:
                per_algo[algo].append(np.mean(npms_vals))
                # rho = npms / lb (lower-bound ratio); use paired values
                rhos = [n / l for n, l in zip(npms_vals, lb_vals) if l > 0]
                per_algo_rho[algo].append(np.mean(rhos) if rhos else np.nan)
            else:
                per_algo[algo].append(np.nan)
                per_algo_rho[algo].append(np.nan)

    if not available_ups:
        raise FileNotFoundError(
            f"No CSV files found for Figure 4. Expected improvevmpack/random_*.csv files."
        )

    plt.figure(figsize=(7.2, 4.5))

    # Plot lower-bound ratio ρ instead of raw PM count: this放大 the differences
    # between the close-performing VMPack variants (MixVM301/MixVM201Pro/MixPack),
    # which would otherwise overlap on a raw-PM-count log axis.
    for algo in target_algos:
        values = per_algo_rho.get(algo, [])
        if len(values) != len(available_ups):
            continue
        plt.plot(
            available_ups,
            values,
            marker=algo_markers.get(algo, 'o'),
            linewidth=1.8,
            markersize=5,
            color=algo_colors.get(algo, None),
            label=algo_labels.get(algo, algo),
        )

    plt.xscale("log")
    plt.xticks(available_ups, [str(x) for x in available_ups])
    plt.xlabel("UP")
    plt.ylabel(r"Average $\overline{\rho}_{LB}$ (PM count / lower bound)")
    plt.title("Normalized PM usage across algorithms and scales")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)

    save_figure(fig_dir / "pm_usage_scales")


def plot_fig5_runtime_quality_tradeoff(output_dir, tag, config):
    """
    Figure 5:
    Plot runtime-quality trade-off for heuristic algorithms.

    x-axis: average runtime in milliseconds
    y-axis: average approximation ratio (lower-bound ratio)
    point: one algorithm on one config (mixalgos or improvevmpack)

    Reads from wide-format CSVs in {output_dir}/{cfg}/random_*.csv
    """
    setup_matplotlib()
    fig_dir = ensure_fig_dir(output_dir)

    configs = ["mixalgos", "improvevmpack"]
    markers = {"mixalgos": "o", "improvevmpack": "s"}
    colors = {"mixalgos": "#3182bd", "improvevmpack": "#e6550d"}

    plt.figure(figsize=(7.2, 5.0))

    for cfg in configs:
        # Collect all scale CSVs for this config
        all_rows = []
        for scale_lower in ['s1', 's2', 'm1', 'm2', 'l1', 'l2']:
            csv_path = Path(output_dir) / cfg / f'random_{scale_lower}.csv'
            if csv_path.exists():
                all_rows.extend(load_csv(csv_path))

        if not all_rows:
            continue

        # Group by algorithm (wide-format: each heuristic algo has {prefix}_npms and {prefix}_time)
        # Only include heuristic algorithms (exclude solver columns like PB*, MIP*).
        # Use column_registry to know which prefixes are heuristics vs solvers.
        from column_registry import COLUMN_REGISTRY
        algo_prefixes = set()
        if cfg in COLUMN_REGISTRY:
            for algo in COLUMN_REGISTRY[cfg]['algorithms']:
                if algo['type'] == 'heuristic':
                    algo_prefixes.add(algo['prefix'])
        # Per-algorithm label offset (dx, dy) to avoid overlapping.
        # Adjust these values until labels no longer collide.
        label_offset = {
            'NoMixPack':     (0.2, 0),
            'MixVM301':      (0.2, 0),
            'MixVM201':      (0.2, 0.002),
            'MixVM201Pro':   (0.05, -0.007),
            'MixPack':       (0.2, 0),
            'BFD':           (0.2, 0),
            'FFD':           (0.2, 0),
            'VMPack_NoMixPack':  (0.2, 0),
            'VMPack_MixVM301':   (-1.5, 0.008),
            'VMPack_MixVM201':   (0.2, 0.002),
            'VMPack_MixVM201Pro':(0.2, -0.001),
            'VMPack_MixPack':    (0.2, 0.003),
        }
        for prefix in sorted(algo_prefixes):
            ratios = []
            times = []

            for r in all_rows:
                npms = to_float(r.get(f"{prefix}_npms"))
                lb = to_float(r.get("lb"))
                runtime = to_float(r.get(f"{prefix}_time"))

                if npms is None or lb is None or lb <= 0 or runtime is None:
                    continue

                ratios.append(npms / lb)
                times.append(runtime * 1000)

            if not ratios or not times:
                continue

            x = np.mean(times)
            y = np.mean(ratios)

            # Shorten algorithm names for display
            label = prefix.replace("VMPack_", "VMPack+")
            
            dx, dy = label_offset.get(prefix, (0.2, 0))
            plt.scatter(x, y, s=90, marker=markers[cfg],
                        color=colors[cfg], edgecolor="black",
                        linewidth=0.5, zorder=5)
            plt.text(x + dx, y + dy, f" {label}", va="center", fontsize=7)
    plt.xlabel("Average runtime (ms)")
    plt.ylabel(r"Average $\overline{\rho}_{LB}$")
    plt.grid(True, linestyle="--", alpha=0.35)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3182bd",
               markersize=8, label="Standalone (L0/L2)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e6550d",
               markersize=8, label="VMPack pipeline (L0/L1/L2)"),
    ]
    plt.legend(handles=legend_elements, frameon=False)

    save_figure(fig_dir / "runtime_quality_scatter")


def plot_fig6_wtl_heatmap(output_dir, tag):
    """
    Figure 6:
    Win/tie/loss heatmap against VMPack+MixVM301 baseline.

    Rows: trace source (Huawei, Microsoft) × algorithm (MixVM201Pro, MixPack)
    Columns: Win / Tie / Loss
    Cell value: number of instances
    """
    setup_matplotlib()
    fig_dir = ensure_fig_dir(output_dir)

    # Load WTL data from both traces (search trace/ subdirectory first, then root)
    all_rows = []
    for trace_name in ["Huawei", "Microsoft"]:
        trace_lower = trace_name.lower()
        candidates = [
            Path(output_dir) / "trace" / f"{trace_lower}_trace_wtl_{tag}.csv",
            Path(output_dir) / "trace" / f"{trace_lower}_trace_wtl_{trace_lower}.csv",
            Path(output_dir) / "trace" / f"{trace_lower}_trace_wtl.csv",
            Path(output_dir) / f"{trace_lower}_trace_wtl_{tag}.csv",
            Path(output_dir) / f"{trace_lower}_trace_wtl_{trace_lower}.csv",
            Path(output_dir) / f"{trace_lower}_trace_wtl.csv",
        ]
        for path in candidates:
            if path.exists():
                all_rows.extend(load_csv(path))
                break

    if not all_rows:
        print("[WARN] No WTL data found for Figure 6")
        return

    # Build heatmap data
    # We want comparisons: VMPack_MixVM301 vs VMPack_MixVM201Pro
    #                     VMPack_MixVM301 vs VMPack_MixPack
    row_labels = []
    win_vals = []
    tie_vals = []
    loss_vals = []

    for trace_name in ["Huawei", "Microsoft"]:
        for target_name in ["VMPack_MixVM201Pro", "VMPack_MixPack"]:
            for r in all_rows:
                if (r.get("trace") == trace_name
                        and r.get("baseline_algo") == "VMPack_MixVM301"
                        and r.get("target_algo") == target_name):
                    row_labels.append(f"{trace_name}\n{target_name.replace('VMPack_', '')}")
                    win_vals.append(int(r.get("target_win", 0)))
                    tie_vals.append(int(r.get("tie", 0)))
                    loss_vals.append(int(r.get("baseline_win", 0)))
                    break

    if not row_labels:
        print("[WARN] No baseline WTL comparisons found for Figure 6")
        return

    # Plot heatmap
    data = np.array([win_vals, tie_vals, loss_vals])  # shape (3, n_rows)
    n_rows = len(row_labels)

    fig, ax = plt.subplots(figsize=(5.0, 2.0 + 0.5 * n_rows))

    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=max(max(win_vals), max(tie_vals), max(loss_vals), 1))

    for i in range(n_rows):
        for j, val in enumerate([win_vals[i], tie_vals[i], loss_vals[i]]):
            color = cmap(norm(val))
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color,
                                  edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if norm(val) > 0.5 else "black")

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Win", "Tie", "Loss"], fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.invert_yaxis()

    # Remove top/right spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.set_title("Win/Tie/Loss vs VMPack+MixVM301", fontsize=11, pad=10)

    plt.tight_layout()
    save_figure(fig_dir / "wtl_heatmap")


def status_rate(rows, status_col, mode="feasible_or_optimal"):
    """
    Compute solver status rate for a status column.

    mode:
      'feasible_or_optimal' — count Optimal or Feasible (has a solution)
      'optimal'             — count only Optimal (proven optimum)
    """
    if not rows:
        return 0.0

    good = 0
    for r in rows:
        status = r.get(status_col, "")
        if mode == "optimal":
            if status == "Optimal":
                good += 1
        else:
            if status in ("Optimal", "Feasible"):
                good += 1

    return good / len(rows)


def plot_fig7_warm_start_effect(output_dir, tags):
    """
    Figure 7:
    Plot warm-start impact using scale CSV files across multiple time limits.

    It generates a 3‑panel figure summarizing the warm‑start effect on the
    exact solvers:

      1. **CG‑Benchmark average generated columns**
         (time‑independent).  Columns correspond to the number of patterns
         generated by the column‑generation benchmark under NoMix versus
         MixVM201Pro initialization.

      2. **CG‑Benchmark average runtime** (time‑independent).

      3. **VanillaMIP feasible/optimal rate** at 1 s, 5 s and 10 s.  This
         panel reports the fraction of instances for which the assignment‑based
         solver finds a feasible or proven optimal solution within the given
         time limit, contrasting the baseline (NoMix) and warm‑started
         (MixVM201Pro) runs.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing experiment CSV files.
    tags : list of str
        Time-limit tags, e.g. ['tl5', 'tl10'].
        The untagged (1s) baseline is always included automatically.
    """
    setup_matplotlib()
    fig_dir = ensure_fig_dir(output_dir)

    # Always include the 1s baseline, plus any additional tags
    all_tags = [''] + list(tags)  # '' = untagged 1s baseline

    scale_labels = []
    pb_cols = []
    pb_mix_cols = []
    pb_times = []
    pb_mix_times = []

    # Per-time-limit MIP rates: list of (mip_rates, mip_mix_rates) per tag
    mip_rates_by_tag = {tag: [] for tag in all_tags}
    mip_mix_rates_by_tag = {tag: [] for tag in all_tags}

    for scale in SCALES:
        # Read from wide-format improvevmpack CSV (has all checkpoint data)
        scale_lower = scale.lower()
        wide_path = Path(output_dir) / 'improvevmpack' / f'random_{scale_lower}.csv'
        if not wide_path.exists():
            continue

        rows_1s = load_csv(wide_path)
        scale_labels.append(scale)

        cols = [
            to_float(r.get("CG_n_cols"))
            for r in rows_1s
            if to_float(r.get("CG_n_cols")) is not None
        ]
        mix_cols = [
            to_float(r.get("CG_Mix_n_cols"))
            for r in rows_1s
            if to_float(r.get("CG_Mix_n_cols")) is not None
        ]

        times = [
            to_float(r.get("CG_time")) * 1000
            for r in rows_1s
            if to_float(r.get("CG_time")) is not None
        ]
        mix_times = [
            to_float(r.get("CG_Mix_time")) * 1000
            for r in rows_1s
            if to_float(r.get("CG_Mix_time")) is not None
        ]

        pb_cols.append(np.mean(cols) if cols else np.nan)
        pb_mix_cols.append(np.mean(mix_cols) if mix_cols else np.nan)
        pb_times.append(np.mean(times) if times else np.nan)
        pb_mix_times.append(np.mean(mix_times) if mix_times else np.nan)

        # Load MIP rates for each time limit from checkpoint columns
        for tag in all_tags:
            if tag == '':
                status_col = 'MIP_1s_status'
                mix_status_col = 'MIP_Mix_1s_status'
            else:
                # tag is like 'tl10' -> extract '10'
                time_val = tag.replace('tl', '')
                status_col = f'MIP_{time_val}s_status'
                mix_status_col = f'MIP_Mix_{time_val}s_status'

            mip_rates_by_tag[tag].append(
                status_rate(rows_1s, status_col, mode="optimal") * 100 if status_col in rows_1s[0] else np.nan)
            mip_mix_rates_by_tag[tag].append(
                status_rate(rows_1s, mix_status_col, mode="optimal") * 100 if mix_status_col in rows_1s[0] else np.nan)

    if not scale_labels:
        raise FileNotFoundError(f"No scale CSV files found for tags={all_tags}")

    x = np.arange(len(scale_labels))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))

    # --- Panel 1: CG‑Benchmark generated columns (time‑independent) ---
    axes[0].bar(x - width / 2, pb_cols, width, label="NoMix", color="#9ecae1", hatch='...',edgecolor="black",)
    axes[0].bar(x + width / 2, pb_mix_cols, width, label="Mix", color="#3182bd", hatch=r'\\\\',edgecolor="black",)
    axes[0].set_title("CG‑Benchmark generated columns")
    axes[0].set_ylabel("Average columns")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scale_labels)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[0].set_ylim(0, 60)

    # --- Panel 2: CG‑Benchmark runtime (time‑independent) ---
    axes[1].bar(x - width / 2, pb_times, width, label="NoMix", color="#a1d99b", hatch='...',edgecolor="black",)
    axes[1].bar(x + width / 2, pb_mix_times, width, label="Mix", color="#31a354", hatch=r'\\\\',edgecolor="black",)
    axes[1].set_title("CG‑Benchmark runtime")
    axes[1].set_ylabel("Average time (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scale_labels)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)

    # --- Panel 3: VanillaMIP feasible rate (all time limits) ---
    # Build label/color maps dynamically from the actual tags passed in.
    # '' (1s baseline) always present; others are 'tlN' → 'Ns'.
    tag_labels = {'': '1s'}
    for t in all_tags:
        if t and t not in tag_labels:
            tag_labels[t] = f"{t.replace('tl', '')}s"

    # Cycle through a fixed color palette so panels stay consistent.
    # _nomix_palette = ['#9ecae1', '#a1d99b', '#fdae6b', '#bcbddc', '#fcae91']
    # _mix_palette   = ['#3182bd', '#41ab5d', '#e6550d', '#756bb1', '#fb6a4a']
    _nomix_palette = ['#8cbaeb', '#8fd175', '#fdc086', '#beaed4', '#fd8d8d']
    _mix_palette   = ['#1f78b4', '#238b45', '#e6550d', '#6a3d9a', '#cb181d']

    tag_colors_nomix = {t: _nomix_palette[i % len(_nomix_palette)] for i, t in enumerate(all_tags)}
    tag_colors_mix   = {t: _mix_palette[i % len(_mix_palette)] for i, t in enumerate(all_tags)}

    n_tags = len(all_tags)
    group_width = 0.80
    bar_width = group_width / n_tags

    for ti, tag in enumerate(all_tags):
        offset = (ti - (n_tags - 1) / 2) * bar_width
        axes[2].bar(
            x + offset,
            mip_rates_by_tag[tag],
            bar_width,
            label=f"NoMix ({tag_labels[tag]})" if tag else "NoMix (1s)",
            color=tag_colors_nomix[tag],
            linewidth=0.5, 
            hatch='...',
            edgecolor="black",
        )
        axes[2].bar(
            x + offset,
            mip_mix_rates_by_tag[tag],
            bar_width,
            label=f"Mix ({tag_labels[tag]})" if tag else "Mix (1s)",
            color=tag_colors_mix[tag],
            linewidth=0.5,
            alpha=0.6,  # slightly transparent to differentiate from NoMix
            hatch=r'\\\\',
            edgecolor="black",
        )

    axes[2].set_title("VanillaMIP optimal rate")
    axes[2].set_ylabel(r"Certified optimal rate (\%)")
    axes[2].set_ylim(0, 119)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(scale_labels)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)

    # Custom legend for panel 3 — built dynamically from the actual tags.
    # Place above the panel.
    axes[2].legend(loc="upper center",
                   ncol=3, frameon=False, fontsize=7.5)

    # Place the CG‑Benchmark legend (panels 1‑2) inside panel 1 (upper‑right corner).
    # Manually create the legend with explicit patch handles.
    axes[0].legend(loc="upper left", frameon=True,
                   fancybox=False, edgecolor="black", facecolor="white",
                   framealpha=1.0, fontsize=9)

    # Panel 2 also needs its own legend.
    axes[1].legend(loc="upper left", frameon=True,
                   fancybox=False, edgecolor="black", facecolor="white",
                   framealpha=1.0, fontsize=9)

    save_figure(fig_dir / "time_limit_warm_start")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 4/5/6/7 from experiment CSV files."
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        choices=["fig4", "fig5", "fig6", "fig7", "all"],
        help="Which figure to generate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result/",
        help="Directory containing experiment CSV files.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional CSV tag, e.g., tl5, tl10.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="improvevmpack",
        choices=["mixalgos", "improvevmpack"],
        help="Heuristic experiment config for Figure 5 (Figure 4 always uses improvevmpack).",
    )

    args = parser.parse_args()

    if args.figure in ("fig4", "all"):
        plot_fig4_quality_trend(args.output_dir, args.tag, args.config)

    if args.figure in ("fig5", "all"):
        plot_fig5_runtime_quality_tradeoff(args.output_dir, args.tag, args.config)

    if args.figure in ("fig6", "all"):
        plot_fig6_wtl_heatmap(args.output_dir, args.tag)

    if args.figure in ("fig7", "all"):
        # Figure 7 uses multi-tag comparison; extract non-empty tags
        fig7_tags = [t for t in args.tag.split(",") if t.strip()] if args.tag else []
        plot_fig7_warm_start_effect(args.output_dir, fig7_tags)


if __name__ == "__main__":
    main()
