# Reproduction Guide

This document maps every table and figure in the paper to the exact command
that generates it, so that the reported results can be reproduced end to end.

## Environment

- **Python**: 3.12 (conda env `py312` at `E:\ProgramData\Miniconda3\envs\py312`)
- **Gurobi**: 13.0.1 (license required; `gurobipy`)
- **Other deps**: `numpy`, `sortedcontainers`, `matplotlib`, `pandas`
- Install: `pip install -r requirements.txt` (and `gurobipy` separately)

All commands are run from the project root directory. The master orchestrator
is `run.py`; it uses the `py312` interpreter internally (hardcoded in `run.py`).

## One-command full reproduction

```bash
python run.py --steps heuristic,scale_maxtime,trace_experiments,export_unified,gen_tables,gen_figures --maxtime 10 --checkpoint_times 1,5,10 --force --quiet
```

This regenerates every CSV, LaTeX table, and figure from the pre-generated
instances in `data/`. Estimated time: ~1 hour (Gurobi runs dominate).

To also regenerate instances and re-process raw traces (slow, only if `data/`
is missing):
```bash
python run.py --maxtime 10 --checkpoint_times 1,5,10
```

## Step-by-step: table/figure → command mapping

Each step writes outputs to `result/`. The LaTeX tables land in
`result/tables/*.tex`; figures in `result/figures/*.{pdf,png}`.

### Instance data (Section 5.2.1)

| Artifact | Command | Output |
|---|---|---|
| Synthetic instances (S1–L2, both scenarios) | `python run.py --steps generate_data` | `data/random_{s1..l2}/{n}_r_7_{UP}_{fun_case}.json` |

### Trace preprocessing (Section 5.2.2)

The raw trace files are not bundled in the repository. Download them into `raw_data/` first (see README "Step 6: Process Public Traces" for URLs and sources), then run:

| Artifact | Command | Output |
|---|---|---|
| Huawei trace → VMPack instances | `python process_huawei_trace.py --input raw_data/Huawei-East-1.csv --output_dir ./data/huawei_trace/ --snapshot_every_events 2000 --min_active_vms 50 --export_vmpack_json --T 7 --scenario {mixalgos\|improvevmpack}` | `data/huawei_trace/huawei_vmpack_instances_{scenario}.json` |
| Microsoft vmtable → VMPack instances | `python process_microsoft_vmtable.py --input raw_data/trace_data_vmtable_vmtable.csv --output_dir ./data/microsoft_vmtable/ --T 7 --batch_size 1000 --n_instances 100 --shuffle --seed 42 --scenario {mixalgos\|improvevmpack}` | `data/microsoft_vmtable/microsoft_vmpack_instances_{scenario}.json` |

Run both scenarios (`mixalgos` and `improvevmpack`) for each trace. Or via
`python run.py --steps process_traces --huawei_input raw_data/Huawei-East-1.csv --microsoft_input raw_data/trace_data_vmtable_vmtable.csv`.

### Heuristic comparison (Section 5.5)

| Table/Figure | Command | Output file |
|---|---|---|
| Table 11 (`tab:heuristic_mixalgos`) | `python run.py --steps heuristic,export_unified,gen_tables` | `result/tables/heuristic_comparison.tex` (1st table) |
| Table 12 (`tab:heuristic_improvevmpack`) | (same as above) | `result/tables/heuristic_comparison.tex` (2nd table) |
| Table 4 (`tab:scale_sweep`) | (same as above) | `result/tables/scale_sweep.tex` |
| Table runtime_scaling (`tab:runtime_scaling`) | (same as above) | `result/tables/runtime_scaling.tex` |
| Figure 4 (`fig:pm_usage_scales`) | `python run.py --steps gen_figures` | `result/figures/pm_usage_scales.{pdf,png}` |
| Figure 5 (`fig:runtime_quality_scatter`) | (same as above) | `result/figures/runtime_quality_scatter.{pdf,png}` |
| **WTL + Wilcoxon** (§5.5.2 body) | (produced by `heuristic` step) | `result/heuristic/improvevmpack_wtl_up{UP}.csv` |

### Trace results (Section 5.6)

| Table/Figure | Command | Output file |
|---|---|---|
| Table 13 (`tab:trace_results`) | `python run.py --steps trace_experiments,gen_tables` | `result/tables/trace_detail.tex` |
| Table 14 (`tab:trace_win_tie_loss`) | (same as above) | `result/tables/trace_win_tie_loss.tex` |
| Table trace_filtering (`tab:trace_filtering`) | (same as above) | `result/tables/public_trace_summary.tex` |
| Figure 6 (`fig:wtl_heatmap`) | `python run.py --steps gen_figures` | `result/figures/wtl_heatmap.{pdf,png}` |

### Exact-solver comparison (Sections 5.7, 5.8)

| Table/Figure | Command | Output file |
|---|---|---|
| Table 15 (`tab:exact_comparison`) | `python run.py --steps scale_maxtime,gen_tables` | `result/tables/method_comparison.tex` |
| Table 16 (`tab:exact_gap_summary`) | (same + `gen_gap_summary`) | `result/tables/gap_summary_maxtime.tex` |
| Table exact_gap_subset (`tab:exact_gap_subset`) | (same) | `result/tables/heuristic_gap_maxtime.tex` |
| Table 18 (`tab:heuristic_lift`) | (same as Table 15) | `result/tables/heuristic_empowerment.tex` |
| Figure 7 (`fig:time_limit_warm_start`) | `python run.py --steps gen_figures` | `result/figures/time_limit_warm_start.{pdf,png}` |

**Exact-solver parameters** (Gurobi, both VanillaMIP and P&B subproblems):
`Threads=1`, `TimeLimit=10`, `MemLimit=24`, `Seed=42`. The `scale_maxtime` step
runs each solver once per instance with Gurobi checkpoint callbacks recording
state at 1s/5s/10s.

### Figures

Generated figures are written to `result/figures/*.{pdf,png}` by `python run.py --steps gen_figures`.

## Output directory structure

```
result/
├── heuristic/          # per-instance heuristic CSVs (long format) + WTL CSVs
├── scale/              # exact-solver CSVs ({scale}_maxtime.csv + checkpoint variants)
├── trace/              # trace experiment CSVs (detail/summary/wtl)
├── mixalgos/           # unified wide-format CSVs (two-class scenario)
├── improvevmpack/      # unified wide-format CSVs (three-class scenario, with solvers)
├── gap/                # gap summary CSVs
├── tables/             # generated LaTeX table fragments (*.tex)
├── figures/            # generated figures (PDF + PNG)
└── log/                # subprocess logs (when --quiet)
```

## Reproducibility notes

- **Random seed**: `np.random.seed(42)` for instance generation; Gurobi `Seed=42`.
- **Determinism**: heuristics are fully deterministic. Exact solvers have
  minimal nondeterminism (Gurobi internal heuristics) but `Seed=42` + `Threads=1`
  keeps runs reproducible; wall-clock times vary slightly with machine load.
- **Instance counts**: 100 instances per scale, after filtering.
- **Git**: see `Data availability` in the paper for the commit hash / release tag.
