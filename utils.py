"""
utils.py
Common utility functions used by gen_tables.py, gen_gap_summary.py, and plot_results.py.
"""
import csv
from pathlib import Path

import numpy as np

# ── Shared constants ────────────────────────────────────────────────
SCALES = ["S1", "S2", "M1", "M2", "L1", "L2"]
UP_SWEEP = [10, 20, 50, 100, 500, 1000]
DEFAULT_N_INST = 100
DEFAULT_SEED = 42
DEFAULT_T = 7


def load_csv(path):
    """Load a CSV file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tagged_csv_name(base, tag):
    """Build tagged CSV filename, e.g. 'S1_tl10.csv' or 'S1.csv'."""
    if tag:
        return f"{base}_{tag}.csv"
    return f"{base}.csv"


def bootstrap_mean_ci(diffs, n_boot=1000, alpha=0.05, seed=42):
    """95% bootstrap confidence interval for the mean of paired differences.

    ``diffs`` is a list of paired differences (baseline - target).  Returns
    (mean, ci_low, ci_high); if ``diffs`` is empty, returns (0.0, 0.0, 0.0).
    Uses a fixed RNG seed for reproducibility (seed is NOT taken from system
    entropy, so results are deterministic across runs).
    """
    diffs = list(diffs)
    if not diffs:
        return 0.0, 0.0, 0.0
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(arr)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return float(arr.mean()), lo, hi


def cliffs_delta(diffs):
    """Cliff's delta effect size for paired differences (baseline - target).

    delta = #{b>t}/n - #{b<t}/n, ranging in [-1, 1].  Positive delta means the
    baseline tends to use more PMs than the target (target is better).  Returns
    0.0 for empty input.
    """
    diffs = list(diffs)
    if not diffs:
        return 0.0
    n = len(diffs)
    more = sum(1 for d in diffs if d > 0)  # baseline > target -> target better
    less = sum(1 for d in diffs if d < 0)  # baseline < target -> baseline better
    return (more - less) / n



def resolve_csv_path(output_dir, base, tag):
    """Prefer tagged CSV. Fall back to untagged CSV. Search subdirectories."""
    output_dir = Path(output_dir)
    tagged = output_dir / tagged_csv_name(base, tag)
    plain = output_dir / f"{base}.csv"
    if tagged.exists():
        return tagged
    if plain.exists():
        return plain
    # Search in subdirectories (scale/, trace/)
    for subdir in ['scale', 'trace']:
        tagged_sub = output_dir / subdir / tagged_csv_name(base, tag)
        if tagged_sub.exists():
            return tagged_sub
        plain_sub = output_dir / subdir / f"{base}.csv"
        if plain_sub.exists():
            return plain_sub
    return None


def is_valid(v):
    """Return True if a CSV value is neither empty nor textual None."""
    return v not in ("", "None", None, "*")


def to_float(v, default=None):
    """Safely convert CSV value to float."""
    if not is_valid(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def to_int(v, default=0):
    """Safely convert CSV value to int."""
    value = to_float(v, None)
    if value is None:
        return default
    return int(round(value))


def mean_or_dash(values, fmt="{:.2f}"):
    """Format mean value or dash."""
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return "-"
    return fmt.format(float(np.mean(values)))


def count_status(rows, status_col):
    """Count Optimal / Feasible / NoSolution / OOM."""
    opt = sum(1 for r in rows if r.get(status_col) == "Optimal")
    feas = sum(1 for r in rows if r.get(status_col) == "Feasible")
    nosol = sum(1 for r in rows if r.get(status_col) == "NoSolution")
    oom = sum(1 for r in rows if "OOM" in str(r.get(status_col, "")))
    other = len(rows) - opt - feas - nosol - oom
    return opt, feas, nosol, oom, other
