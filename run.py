#!/usr/bin/env python
"""
run.py — Master experiment orchestrator for the CAOR paper
"Improving VMPack: Heuristic Mixed Packing Algorithms for
 Two Specific Virtual Machine Classes."

Reproduces all experimental results from Section 5 (Computational Experiments).
Runs every step in paper order. Use --steps to select specific steps.

Usage:
    python run.py                              # Run everything (needs Gurobi, ~hours)
    python run.py --steps generate_data,heuristic  # Specific steps
    python run.py --steps all --n_inst 10       # Quick test
    python run.py --skip_gurobi                 # Skip Gurobi steps
    python run.py --steps gen_figures           # Only figures

Paper experiment flow:
    Step 1  generate_data     Section 5.2.1  Synthetic bottleneck instances
    Step 2  process_traces    Section 5.2.2  Public trace processing
    Step 3  heuristic         Section 5.5    Heuristic comparison (UP sweep)
    Step 4  scale_maxtime     Section 5.7    1-run maxtime exact solver (Gurobi)
    Step 5  trace_experiments Section 5.6    Trace-derived experiments
    Step 6  export_unified     —             Export organized wide-format CSVs
    Step 7  gen_tables        All sections   Generate LaTeX tables
    Step 8  gen_figures       All sections   Generate figures
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from utils import SCALES, UP_SWEEP, DEFAULT_N_INST, DEFAULT_SEED, DEFAULT_T

# ── Project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Use conda py312 environment (has all dependencies)
PYTHON = r"E:\ProgramData\Miniconda3\envs\py312\python.exe"
RESULT_DIR = PROJECT_ROOT / "result"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = RESULT_DIR / "log"   # per-step subprocess logs (when --quiet)

# ── Paper constants ────────────────────────────────────────────────
HEURISTIC_CONFIGS = ["mixalgos", "improvevmpack"]


# ═══════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════

class StepResult:
    def __init__(self, name, success, elapsed, note=""):
        self.name = name; self.success = success
        self.elapsed = elapsed; self.note = note


def run_cmd(cmd, desc="", cwd=None, timeout=None, quiet=False, log_path=None):
    """Run a subprocess command; print a one-line header.

    When quiet=True, the subprocess stdout/stderr is captured to log_path
    (a file) instead of streaming to the terminal. On failure, the captured
    log is printed so the error is visible. When quiet=False, output streams
    to the terminal as before.
    """
    if cwd is None:
        cwd = PROJECT_ROOT
    print(f"  ▸ {desc or ' '.join(cmd)}", flush=True)
    t0 = time.time()

    if quiet and log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as logf:
            rc = subprocess.run(cmd, cwd=str(cwd), timeout=timeout,
                                stdout=logf, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        if rc.returncode != 0:
            # Dump the captured log on failure so the error is visible.
            print(f"    [FAIL] exit {rc.returncode}. Log ({log_path.name}):")
            with open(log_path, 'r', encoding='utf-8') as logf:
                for line in logf:
                    print(f"    {line.rstrip()}")
            raise RuntimeError(f"exit {rc.returncode}: {' '.join(cmd)}")
        print(f"    [OK] {elapsed:.1f}s")
    else:
        rc = subprocess.run(cmd, cwd=str(cwd), timeout=timeout)
        elapsed = time.time() - t0
        if rc.returncode != 0:
            raise RuntimeError(f"exit {rc.returncode}: {' '.join(cmd)}")
        print(f"    [OK] {elapsed:.1f}s")
    return elapsed


def run_step(cmd, desc, args, log_name):
    """Run a subprocess step with quiet/streaming chosen by args.quiet.

    When --quiet, output goes to result/log/{log_name}.log and the terminal
    shows only the one-line [OK]/[FAIL] summary.
    """
    log_path = LOG_DIR / f"{log_name}.log"
    return run_cmd(cmd, desc=desc, quiet=args.quiet, log_path=log_path)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Step 1 — Generate synthetic instances (Section 5.2.1)
# ═══════════════════════════════════════════════════════════════════

def step_generate_data(args):
    """Generate synthetic two-class and three-class bottleneck instances."""
    print(f"\n{'=' * 68}\n  Step 1: Generate Synthetic Instance Data\n{'=' * 68}")
    ensure_dir(DATA_DIR)

    results = []
    for up in UP_SWEEP:
        for cfg in HEURISTIC_CONFIGS:
            fname = f"{args.n_inst}_r_{DEFAULT_T}_{up}_{cfg}.json"
            # Map UP to scale subdirectory
            if up == 10: sub = 'random_s1'
            elif up == 20: sub = 'random_s2'
            elif up == 50: sub = 'random_m1'
            elif up == 100: sub = 'random_m2'
            elif up == 500: sub = 'random_l1'
            elif up == 1000: sub = 'random_l2'
            else: sub = ''
            fpath = DATA_DIR / sub / fname
            if fpath.exists() and not args.force:
                print(f"  [SKIP] {fname} -> {sub}/")
                continue

            cmd = [
                PYTHON, "-c", f"""
import numpy as np; np.random.seed({args.seed})
import globalvars as gv; gv.InitialGlobalVars({DEFAULT_T}, {up})
from data import GenExamples, DataTypes, SaveExamples
Ls = GenExamples({args.n_inst}, DataTypes.RANDOM, '{cfg}')
data_path = r'{DATA_DIR}'
SaveExamples(data_path, Ls, DataTypes.RANDOM, '{cfg}')
print(f'  Generated {{len(Ls)}} instances')
"""
            ]
            elapsed = run_step(cmd, f"Generate {cfg} T={DEFAULT_T} UP={up}", args, f"generate_{cfg}_up{up}")
            results.append(StepResult(f"data_{cfg}_up{up}", True, elapsed))
    return results


# ═══════════════════════════════════════════════════════════════════
# Step 2 — Process public traces (Section 5.2.2)
# ═══════════════════════════════════════════════════════════════════

def step_process_traces(args):
    """Process Huawei and Microsoft traces into VMPack instances."""
    print(f"\n{'=' * 68}\n  Step 2: Process Public Trace Data\n{'=' * 68}")
    results = []

    # Huawei
    hw_in = args.huawei_input
    if hw_in and os.path.exists(hw_in):
        hw_out = DATA_DIR / "huawei_trace"
        ensure_dir(hw_out)
        for scenario in HEURISTIC_CONFIGS:
            cmd = [
                PYTHON, "process_huawei_trace.py",
                "--input", hw_in, "--output_dir", str(hw_out),
                "--snapshot_every_events", "2000",
                "--min_active_vms", "50",
                "--export_vmpack_json", "--T", str(DEFAULT_T),
                "--scenario", scenario,
            ]
            elapsed = run_step(cmd, f"Huawei trace ({scenario})", args, f"huawei_{scenario}")
            results.append(StepResult(f"trace_huawei_{scenario}", True, elapsed))
    else:
        print(f"  [SKIP] Huawei trace input not found (use --huawei_input)")

    # Microsoft
    ms_in = args.microsoft_input
    if ms_in and os.path.exists(ms_in):
        ms_out = DATA_DIR / "microsoft_vmtable"
        ensure_dir(ms_out)
        for scenario in HEURISTIC_CONFIGS:
            cmd = [
                PYTHON, "process_microsoft_vmtable.py",
                "--input", ms_in, "--output_dir", str(ms_out),
                "--T", str(DEFAULT_T), "--batch_size", "1000",
                "--n_instances", str(args.n_inst),
                "--shuffle", "--seed", str(args.seed),
                "--scenario", scenario,
            ]
            elapsed = run_step(cmd, f"Microsoft vmtable ({scenario})", args, f"microsoft_{scenario}")
            results.append(StepResult(f"trace_microsoft_{scenario}", True, elapsed))
    else:
        print(f"  [SKIP] Microsoft vmtable input not found (use --microsoft_input)")

    return results


# ═══════════════════════════════════════════════════════════════════
# Step 3 — Heuristic comparison (Section 5.5)
# ═══════════════════════════════════════════════════════════════════

def step_heuristic(args):
    """Run all heuristic comparison experiments (UP sweep)."""
    print(f"\n{'=' * 68}\n  Step 3: Heuristic Comparison\n{'=' * 68}")
    ensure_dir(RESULT_DIR)
    results = []

    for up in UP_SWEEP:
        tag = f"up{up}"
        all_exist = all(
            (RESULT_DIR / "heuristic" / f"{cfg}_{tag}.csv").exists() for cfg in HEURISTIC_CONFIGS
        )
        if all_exist and not args.force:
            print(f"  [SKIP] UP={up}")
            continue

        for cfg in HEURISTIC_CONFIGS:
            cmd = [
                PYTHON, "run_experiments.py",
                "--mode", "heuristic", "--heuristic-config", cfg,
                "--T", str(DEFAULT_T), "--UP", str(up),
                "--n_inst", str(args.n_inst), "--seed", str(args.seed),
                "--output_dir", str(RESULT_DIR),
                "--data_dir", str(DATA_DIR), "--tag", tag,
            ]
            if args.quiet:
                cmd.append("--quiet")
            elapsed = run_step(cmd, f"Heuristic {cfg} UP={up}", args, f"heuristic_{cfg}_up{up}")
            results.append(StepResult(f"heuristic_{cfg}_up{up}", True, elapsed))

    return results


# ═══════════════════════════════════════════════════════════════════
# Steps 4 — Scale experiment (Sections 5.7, 5.8)
# Single maxtime run with checkpoint callbacks replaces old 1s/5s/10s
# ═══════════════════════════════════════════════════════════════════

def step_scale_maxtime(args):
    """Run scale experiments once with maxtime and checkpoint callbacks.

    Runs for both improvevmpack (default) and mixalgos fun_cases.
    The mixalgos run produces optimal solutions for Table 5.

    Checkpoint times come from --checkpoint_times (default: 1,5,10).
    The 1s checkpoint is written as {scale}.csv (backward-compatible);
    other checkpoints use {scale}_tlN.csv where N is the time in seconds.

    Outputs (improvevmpack):
      {scale}_maxtime.csv  — Full results with all checkpoint columns
      {scale}.csv          — 1s checkpoint
      {scale}_tl5.csv      — 5s checkpoint (etc.)

    Outputs (mixalgos, fun_case-qualified naming):
      {scale}_mixalgos_maxtime.csv
      {scale}_mixalgos.csv (1s)
      {scale}_mixalgos_tl5.csv (etc.)
    """
    print(f"\n{'=' * 68}\n  Step: Scale Experiment (maxtime={args.maxtime}s, checkpoints={args.checkpoint_times})\n{'=' * 68}")
    ensure_dir(RESULT_DIR)
    results = []

    fun_cases = ['improvevmpack']
    if not args.skip_gurobi:
        fun_cases.append('mixalgos')

    for fun_case in fun_cases:
        # Check if already done (in scale/ subdirectory)
        scale_dir = RESULT_DIR / 'scale'
        if fun_case == 'mixalgos':
            all_exist = all(
                (scale_dir / f"{s}_mixalgos_maxtime.csv").exists()
                for s in SCALES
            )
            label = "mixalgos"
        else:
            all_exist = all(
                (scale_dir / f"{s}_maxtime.csv").exists()
                for s in SCALES
            )
            label = "improvevmpack"

        if all_exist and not args.force:
            print(f"  [SKIP] All scales {label} maxtime already done")
            continue

        cmd = [
            PYTHON, "run_solvers_only.py",
            "--fun_case", fun_case,
            "--maxtime", str(args.maxtime),
            "--tag", "maxtime",
            "--checkpoint_times", args.checkpoint_times,
            "--n_inst", str(args.n_inst),
            "--seed", str(args.seed),
            "--output_dir", str(RESULT_DIR),
            "--data_dir", str(DATA_DIR),
        ]
        if args.quiet: cmd.append("--quiet")
        elapsed = run_step(cmd, f"Scale experiments {label} (maxtime={args.maxtime}s, all scales)", args, f"scale_{label}_maxtime")
        results.append(StepResult(f"scale_{label}_maxtime", True, elapsed))

    return results


# ═══════════════════════════════════════════════════════════════════
# Step 7 — Trace experiments (Section 5.6)
# ═══════════════════════════════════════════════════════════════════

def step_trace_experiments(args):
    """Run heuristic experiments on trace-derived instances.

    Both scenarios are supported and emit separate output files:
      * mixalgos      - only L0/L2 VMs (C1=0 by construction). Tagged with the
        plain trace key (e.g. huawei), so its CSVs keep the default names
        consumed by gen_tables.py / plot_results.py
        (e.g. huawei_trace_detail_huawei.csv).
      * improvevmpack - all three VM classes. Tagged with
        <trace_key>_improvevmpack so it never collides with the mixalgos
        outputs and can be supplemented/inspected later.

    Use --trace_scenarios to run a subset, e.g. --trace_scenarios improvevmpack.
    """
    print(f"\n{'=' * 68}\n  Step 7: Trace-Derived Experiments\n{'=' * 68}")
    ensure_dir(RESULT_DIR)
    results = []

    scenarios = [s.strip() for s in args.trace_scenarios.split(",") if s.strip()]
    # Per-scenario output tag: mixalgos keeps the plain trace key (default CSV
    # names consumed by gen_tables.py), improvevmpack appends a scenario suffix.
    scenario_tag_suffix = {"mixalgos": "", "improvevmpack": "_improvevmpack"}

    for trace_key, trace_label in [("huawei", "Huawei"), ("microsoft", "Microsoft")]:
        # Directories that may hold the per-scenario VMPack instance JSON.
        trace_dirs = [
            DATA_DIR / f"{trace_key}_trace",
            DATA_DIR / f"{trace_key}_vmtable",
            RESULT_DIR / "microsoft_vmtable" if trace_key == "microsoft" else RESULT_DIR / f"{trace_key}_trace",
            PROJECT_ROOT / f"{trace_key}_trace_output",
            PROJECT_ROOT / f"{trace_key}_vmtable_output",
        ]

        for scenario in scenarios:
            if scenario not in scenario_tag_suffix:
                print(f"  [SKIP] Unknown scenario '{scenario}' for {trace_label}")
                continue

            json_name = f"{trace_key}_vmpack_instances_{scenario}.json"
            instance_file = None
            for d in trace_dirs:
                cand = d / json_name
                if cand.exists():
                    instance_file = str(cand)
                    break

            if not instance_file:
                print(f"  [SKIP] No {scenario} VMPack instances for {trace_label} "
                      f"(run step process_traces with --scenario {scenario} first)")
                continue

            tag = f"{trace_key}{scenario_tag_suffix[scenario]}"
            trace_dir = RESULT_DIR / 'trace'
            detail_csv = trace_dir / f"{trace_key}_trace_detail_{tag}.csv"
            if detail_csv.exists() and not args.force:
                print(f"  [SKIP] {trace_label} ({scenario}) trace already done")
                continue

            cmd = [
                PYTHON, "run_trace_experiments.py",
                "--input", instance_file, "--trace_name", trace_label,
                "--T", str(DEFAULT_T), "--UP", "1000",
                "--output_dir", str(RESULT_DIR), "--tag", tag,
            ]
            # Only the L0/L2 (mixalgos) scenario applies the bottleneck filter.
            # The full three-class (improvevmpack) scenario runs unfiltered to
            # evaluate the algorithm on general-purpose trace instances.
            if scenario == "mixalgos":
                cmd.append("--bottleneck_only")
            elapsed = run_step(cmd, f"Trace: {trace_label} ({scenario})", args, f"trace_{trace_key}_{scenario}")
            results.append(StepResult(f"trace_{trace_key}_{scenario}", True, elapsed))

    return results


# ═══════════════════════════════════════════════════════════════════
# Step 8 — Generate LaTeX tables
# ═══════════════════════════════════════════════════════════════════

def step_gen_tables(args):
    """Generate all LaTeX tables."""
    print(f"\n{'=' * 68}\n  Step 8: Generate LaTeX Tables\n{'=' * 68}")
    results = []

    # Always write .tex files to result/tables/ (quiet-friendly); stdout is
    # captured to result/log/ when --quiet so the terminal stays clean.
    tex_dir = RESULT_DIR / "tables"
    ensure_dir(tex_dir)

    cmd = [
        PYTHON, "gen_tables.py", "--output_dir", str(RESULT_DIR),
        "--huawei_dir", str(DATA_DIR / "huawei_trace"),
        "--microsoft_dir", str(DATA_DIR / "microsoft_vmtable"),
        "--tex_dir", str(tex_dir),
    ]
    elapsed = run_step(cmd, "gen_tables.py", args, "gen_tables")
    results.append(StepResult("gen_tables", True, elapsed))

    for tl_tag, label in [("", "1s"), ("tl5", "5s"), ("tl10", "10s"), ("maxtime", "maxtime")]:
        cmd = [
            PYTHON, "gen_gap_summary.py",
            "--output_dir", str(RESULT_DIR), "--tag", tl_tag, "--save_csv",
            "--tex_dir", str(tex_dir),
        ]
        elapsed = run_step(cmd, f"gen_gap_summary ({label})", args, f"gen_gap_{label}")
        results.append(StepResult(f"gap_{label}", True, elapsed))

    return results


# ═══════════════════════════════════════════════════════════════════
# Step 9 — Generate figures
# ═══════════════════════════════════════════════════════════════════

def step_gen_figures(args):
    """Generate all paper figures."""
    print(f"\n{'=' * 68}\n  Step 9: Generate Figures\n{'=' * 68}")
    ensure_dir(RESULT_DIR)
    results = []

    # Figure 4: PM usage across scales
    cmd = [PYTHON, "plot_results.py", "--figure", "fig4",
           "--output_dir", str(RESULT_DIR)]
    elapsed = run_step(cmd, "Figure 4 (pm_usage_scales)", args, "fig4")
    results.append(StepResult("fig4", True, elapsed))

    # Figure 5: Runtime-quality trade-off
    cmd = [PYTHON, "plot_results.py", "--figure", "fig5",
           "--output_dir", str(RESULT_DIR)]
    elapsed = run_step(cmd, "Figure 5 (runtime_quality_scatter)", args, "fig5")
    results.append(StepResult("fig5", True, elapsed))

    # Figure 6: WTL heatmap
    # cmd = [PYTHON, "plot_results.py", "--figure", "fig6",
    #        "--output_dir", str(RESULT_DIR)]
    # elapsed = run_step(cmd, "Figure 6 (wtl_heatmap)", args, "fig6")
    # results.append(StepResult("fig6", True, elapsed))

    # Figure 7: Warm-start sensitivity (multi-tag)
    cmd = [PYTHON, "plot_results.py", "--figure", "fig7",
           "--output_dir", str(RESULT_DIR), "--tag", "tl5,tl10"]
    elapsed = run_step(cmd, "Figure 7 (time_limit_warm_start)", args, "fig7")
    results.append(StepResult("fig7", True, elapsed))

    return results


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def step_export_unified(args):
    """Generate unified wide-format CSVs for both mixalgos and improvevmpack.

    Merges heuristic (long-format) CSVs, exact-solver maxtime CSVs, and raw
    instance data into one wide-format CSV per fun_case.
    """
    print(f"\n{'=' * 68}\n  Step: Export Unified CSVs\n{'=' * 68}")
    results = []

    for fun_case in ['mixalgos', 'improvevmpack']:
        cmd = [
            PYTHON, "export_unified_results.py",
            "--fun_case", fun_case,
            "--output_dir", str(RESULT_DIR),
            "--data_dir", str(DATA_DIR),
            "--n_inst", str(args.n_inst),
            "--seed", str(args.seed),
        ]
        # Both fun_cases now have solver data (mixalgos has MIP+Mix)
        cmd.append("--with_solvers")
        elapsed = run_step(cmd, f"Export unified CSV ({fun_case})", args, f"export_unified_{fun_case}")
        results.append(StepResult(f"unified_{fun_case}", True, elapsed))

    return results


STEP_REGISTRY = {
    "generate_data":     step_generate_data,
    "process_traces":    step_process_traces,
    "heuristic":         step_heuristic,
    "scale_maxtime":     step_scale_maxtime,
    "trace_experiments": step_trace_experiments,
    "export_unified":    step_export_unified,
    "gen_tables":        step_gen_tables,
    "gen_figures":       step_gen_figures,
}

GUROBI_STEPS = {"scale_maxtime"}


def main():
    parser = argparse.ArgumentParser(
        description="CAOR Paper — Master Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available steps: {', '.join(STEP_REGISTRY.keys())}

Examples:
  python run.py                                    # Everything
  python run.py --steps generate_data,heuristic    # Select steps
  python run.py --skip_gurobi                      # No Gurobi
  python run.py --steps all --n_inst 10 --quiet    # Quick test
        """,
    )
    parser.add_argument("--steps", type=str, default="all",
                        help="Comma-separated steps to run")
    parser.add_argument("--skip_gurobi", action="store_true",
                        help="Skip Gurobi-requiring steps")
    parser.add_argument("--n_inst", type=int, default=DEFAULT_N_INST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--maxtime", type=int, default=10,
                        help="Hard time limit for MIP solvers (default: 10s)")
    parser.add_argument("--checkpoint_times", type=str, default="1,5,10",
                        help="Comma-separated checkpoint seconds for Gurobi callback "
                             "(default: 1,5,10). Used by scale_maxtime step.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs exist")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--huawei_input", type=str, default="",
                        help="Path to Huawei-East-1.csv")
    parser.add_argument("--microsoft_input", type=str, default="",
                        help="Path to trace_data_vmtable_vmtable.csv")
    parser.add_argument("--trace_scenarios", type=str, default="mixalgos,improvevmpack",
                        help="Comma-separated trace scenarios to run in step "
                             "trace_experiments (mixalgos and/or improvevmpack). "
                             "mixalgos writes the default CSV names; improvevmpack "
                             "writes scenario-suffixed CSVs.")
    args = parser.parse_args()

    # Resolve steps
    if args.steps == "all":
        steps = list(STEP_REGISTRY.keys())
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    if args.skip_gurobi:
        steps = [s for s in steps if s not in GUROBI_STEPS]
        print("[INFO] Gurobi steps excluded")

    for s in steps:
        if s not in STEP_REGISTRY:
            print(f"[ERROR] Unknown step: {s}")
            print(f"  Available: {', '.join(STEP_REGISTRY.keys())}")
            sys.exit(1)

    ensure_dir(RESULT_DIR)
    ensure_dir(DATA_DIR)

    # Run
    all_results = []
    t_start = time.time()
    print(f"\n{'=' * 68}")
    print(f"  CAOR Paper — Experiment Runner")
    print(f"  Steps: {', '.join(steps)}")
    print(f"  n_inst={args.n_inst}, seed={args.seed}")
    print(f"  Start: {datetime.now().isoformat(timespec='seconds')}")
    print(f"{'=' * 68}")

    for name in steps:
        fn = STEP_REGISTRY[name]
        t0 = time.time()
        try:
            res = fn(args)
            all_results.extend(res if isinstance(res, list) else [res])
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            all_results.append(StepResult(name, False, time.time() - t0, str(e)))

    # Summary
    total_t = time.time() - t_start
    n_ok = sum(1 for r in all_results if r.success)
    n_fail = len(all_results) - n_ok
    print(f"\n{'=' * 68}")
    print(f"  SUMMARY: {n_ok} OK, {n_fail} FAIL  |  {total_t/60:.1f} min")
    print(f"  Output: {RESULT_DIR}")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
