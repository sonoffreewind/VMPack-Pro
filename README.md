# Improving the Bottleneck Stage of VMPack for Structured Virtual Machine Allocation

[![License](https://img.shields.io/badge/license-MIT-blue.svg )](https://opensource.org/licenses/MIT )

> This repository contains the source code and experimental data for the manuscript titled *"Improving the Bottleneck Stage of VMPack for Structured Virtual Machine Allocation."* 

## Description


This work investigates the Virtual Machine Allocation (VMA) problem, modeled as a 2D Vector Bin Packing problem, considering practical constraints such as a finite set of VM types and resource sizes that are powers of two.

The primary contribution of this work is the development and analysis of several novel heuristic algorithms (MixVM201, MixVM201Pro, MixPack) designed to improve upon the performance of the existing VMPack algorithm, particularly in its identified bottleneck scenarios. The proposed heuristics aim to enhance physical machine (PM) utilization by effectively mixing different VM classes, reusing residual capacity, and applying ratio balancing strategies. The three core heuristic algorithms are:
- **MixVM201**: Reuses residual capacity via a ratio balance strategy.
- **MixVM201Pro**: Maintains stricter ratio balance between VM classes.
- **MixPack**: Refines packing logic to achieve an improved asymptotic upper bound.

This repository provides the implementation of all proposed heuristics and the necessary scripts to reproduce the empirical results presented in the paper.

## Code Structure

The repository is organized into several Python files:

### Core algorithms
- `globalvars.py`: Defines and initializes global variables such as `T`, `S`, `C`, `M`, and `UP`.
- `basic.py`: Contains helper functions, including `CpuSize`, `FindVMsWithBound`, `FillOnePm`, and solution validation utilities.
- `data.py`: Includes methods for generating, loading, and saving synthetic VM request instances.
- `heuristics.py`: Contains all heuristic algorithms, including `NoMixPack`, `MixVM301`, `MixVM201`, `MixVM201Pro`, `MixPack`, `VMPack`, `VMPackPro`, `FFD`, and `BFD`.
- `pricebranch.py`: Implements a column-generation-based exact/near-exact method that solves the LP relaxation and then an integer restricted master problem.
- `vanilla_mip.py`: Implements a standard assignment-based MIP formulation solved by Gurobi, with checkpoint callback support for multi-time-limit experiments.
- `utils.py`: Shared utility functions (CSV loading, number conversion, constants) used by table/figure generation scripts.
- `column_registry.py`: Unified column registry defining column names for all algorithms in wide-format CSVs.
- `export_unified_results.py`: Exports organized wide-format CSVs into `result/{fun_case}/` directories using the column registry.

### Experiment orchestration
- **`run.py`**: Master experiment orchestrator. Runs all experiments in paper order.
- `run_experiments.py`: Runs synthetic heuristic comparisons and scale comparisons, saving per-instance CSV results.
- `run_solvers_only.py`: Runs exact solvers (P&B, VanillaMIP) once with maxtime and Gurobi checkpoint callbacks, producing 1s/5s/10s checkpoint data from a single run.
- `run_trace_experiments.py`: Runs heuristic algorithms on public-trace-derived instances.

### Result generation
- `gen_tables.py`: Generates LaTeX tables from the organized wide-format CSVs.
- `gen_gap_summary.py`: Generates exact-verifiable subset gap summary tables.
- `plot_results.py`: Generates Figures 4–7 from experiment CSV files.
- `export_unified_results.py`: Pivots long-format heuristic CSVs into wide-format, merges solver columns, and outputs to `result/{fun_case}/` directories.

### Data processing
- `process_huawei_trace.py`: Processes Huawei VM create/delete traces, produces creation-event statistics, active-set snapshots, and VMPack-compatible JSON instances.
- `process_microsoft_vmtable.py`: Processes the Microsoft Azure `trace_data_vmtable_vmtable.csv` trace, analyzes VM type frequencies, filters VMPack-compatible VM types, and generates VMPack-compatible JSON instances.

> A developer-oriented mapping between each Section 5 subsection and the corresponding code functions, input/output files, CSV formats, and generated LaTeX tables is provided in [`experiment_framework.md`](experiment_framework.md).

## Project Structure

The repository is organized as follows:
```
vm-mixpack-algorithms/
├── run.py                         ← Master experiment runner
├── run_experiments.py
├── run_solvers_only.py
├── run_trace_experiments.py
├── export_unified_results.py      ← Export organized wide-format CSVs
├── column_registry.py             ← Column naming registry
├── gen_tables.py
├── gen_gap_summary.py
├── plot_results.py
├── process_huawei_trace.py
├── process_microsoft_vmtable.py
├── heuristics.py
├── pricebranch.py
├── vanilla_mip.py
├── utils.py
├── data.py
├── globalvars.py
├── basic.py
├── data/
│   ├── random_s1/          ← Synthetic instances (UP=10)
│   ├── random_s2/          ← Synthetic instances (UP=20)
│   ├── random_m1/          ← Synthetic instances (UP=50)
│   ├── random_m2/          ← Synthetic instances (UP=100)
│   ├── random_l1/          ← Synthetic instances (UP=500)
│   ├── random_l2/          ← Synthetic instances (UP=1000)
│   ├── huawei_trace/       ← Processed Huawei trace data
│   └── microsoft_vmtable/  ← Processed Microsoft trace data
├── result/
│   ├── mixalgos/           ← Wide-format CSVs (L0/L2 scenario)
│   ├── improvevmpack/      ← Wide-format CSVs (L0/L1/L2 scenario)
│   ├── heuristic/          ← Heuristic comparison CSVs (intermediate)
│   ├── scale/              ← Exact solver scale CSVs
│   ├── trace/              ← Trace experiment outputs
│   ├── gap/                ← Optimality gap summaries
│   ├── tables/             ← LaTeX table files (*.tex)
│   ├── log/                ← Subprocess output logs (when --quiet)
│   └── figures/            ← Generated figures (PDF + PNG)
├── article/
│   ├── article.tex         ← Paper LaTeX source
│   └── figures/            ← Paper figures
└── requirements.txt
```

## Algorithm Implementations

This repository contains our implementations of several heuristic algorithms. The algorithms `MixVM201`, `MixVM201Pro`, and `MixPack` are novel contributions of our work.

The `VMPack` algorithm implemented herein is based on the description provided in the following published paper:

> Guo, L., Lu, C., & Wu, G. (2023). Approximation algorithms for a virtual machine allocation problem with finite types. *Information Processing Letters*, 180, 106339. [https://doi.org/10.1016/j.ipl.2022.106339](https://doi.org/10.1016/j.ipl.2022.106339)

Furthermore, the heuristic we term `MixVM301` is a formalization and implementation of the 301 mixed packing strategy that was employed within the VMPack framework in that same publication.

Our implementations of both `VMPack` and `MixVM301` were developed independently based on the methodology described in the original publication and were not copied from any other source.


## Requirements

The code is written in Python and requires the following libraries:
- Python 3.x
- `numpy`
- `sortedcontainers` (used in `BFD` for efficient data handling)
- `gurobipy` (required for `pricebranch.py`, `vanilla_mip.py`)

You can install the required libraries using pip:
```bash
pip install numpy sortedcontainers gurobipy
```
Alternatively, you can install all dependencies from the provided file:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start — Reproduce All Results

The recommended way to reproduce the paper's experimental results is to use `run.py`:

```bash
# Run all experiments (needs Gurobi, takes several hours)
python run.py --maxtime 10 --checkpoint_times 1,5,10

# Skip Gurobi-requiring steps (heuristics + trace + tables + figures only)
python run.py --skip_gurobi

# Run specific steps
python run.py --steps generate_data,heuristic

# Quick test with 10 instances
python run.py --maxtime 10 --checkpoint_times 1,5,10 --n_inst 10 --quiet

# Generate figures from existing data (no experiments needed)
python run.py --steps gen_figures

# Process traces (needs trace CSV files)
python run.py --steps process_traces --huawei_input raw_data/Huawei-East-1.csv --microsoft_input raw_data/trace_data_vmtable_vmtable.csv
```

Available steps (in paper order):
| Step | Description | Section | Needs Gurobi? |
|---|---|---|---|
| `generate_data` | Generate synthetic bottleneck instances | 5.2.1 | No |
| `process_traces` | Process Huawei/Microsoft traces | 5.2.2 | No |
| `heuristic` | Heuristic comparison (UP sweep) | 5.5 | No |
| `scale_maxtime` | Scale experiments (single maxtime run with checkpoint callbacks) | 5.7, 5.8 | **Yes** |
| `trace_experiments` | Trace-derived heuristic experiments | 5.6 | No |
| `export_unified` | Export organized wide-format CSVs (`result/{fun_case}/`) | — | No |
| `gen_tables` | Generate LaTeX tables | all | No |
| `gen_figures` | Generate Figures 4–7 | all | No |

### Step-by-Step Usage (Legacy)

#### Step 1: Generate Test Data

Use `run.py` to generate test instances:

```bash
python run.py --steps generate_data
```

Or use `data.py` directly via `run_experiments.py` which auto-generates missing data.

#### Step 2: Run Heuristic Experiments

```bash
# Full heuristic UP sweep (UP=10,20,50,100,500,1000)
python run.py --steps heuristic

# Or directly using run_experiments.py
python run_experiments.py --mode heuristic --T 7 --UP 1000 --n_inst 100 --tag up1000
```

#### Step 3: Run Exact Solver Experiments

```bash
# Single maxtime run with checkpoint callbacks (produces 1s/5s/10s data)
python run_solvers_only.py --maxtime 10 --checkpoint_times 1,5,10 --n_inst 100
```

**Gurobi parameters** (set identically in `vanilla_mip.py` and the P&B pricing
subproblems in `pricebranch.py`, for fair single-threaded comparison):
`Threads=1`, `TimeLimit=10`, `MemLimit=24`, `Seed=42`.
- `maxtime=10` is the hard wall-clock budget for each Gurobi run.
- `checkpoint_times=1,5,10` records solver state (incumbent, bound, gap, status)
  at 1s/5s/10s via a Gurobi callback, without re-running the solver.
- P&B converges within 1s, so its result is shared across all checkpoints.
- Run both scenarios: `--fun_case improvevmpack` (default) and `--fun_case mixalgos`.

#### Step 4: Export Organized Wide-Format CSVs

After running heuristics and (optionally) exact solvers, export organized wide-format CSVs:

```bash
# Export both fun_cases (mixalgos + improvevmpack)
python run.py --steps export_unified

# Or directly:
python export_unified_results.py --fun_case mixalgos
python export_unified_results.py --fun_case improvevmpack --with_solvers
```

Output: `result/mixalgos/random_*.csv` and `result/improvevmpack/random_*.csv`
(8 files per fun_case: random_s1~l2 + huawei + microsoft, one row per instance)

#### Step 5: Generate LaTeX Tables

```bash
python gen_tables.py --output_dir ./result/ --tex_dir ./result/tables/
python gen_gap_summary.py --output_dir ./result/ --tag maxtime --save_csv --tex_dir ./result/tables/
```

#### Step 6: Generate Figures

```bash
python plot_results.py --figure all --output_dir ./result/
```

#### Step 6: Process Public Traces

The two raw trace files are **not included in this repository** (they are too
large to host). Download them manually into a `raw_data/` directory at the
repository root before running the commands below.

| File | Source | Download |
|---|---|---|
| `raw_data/trace_data_vmtable_vmtable.csv` | Microsoft Azure 2017 public VM trace ([Cortez et al., 2017](https://dl.acm.org/doi/10.1145/3127479.3127600)) | <https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/trace_data_vmtable_vmtable.csv.gz> (decompress the `.gz`; the extracted file is already named `trace_data_vmtable_vmtable.csv`) |
| `raw_data/Huawei-East-1.csv` | Huawei-East-1 VM trace ([Liu et al., IJCAI 2022](https://doi.org/10.24963/ijcai.2022/860)) | <https://github.com/mail-ecnu/VMAgent/blob/master/vmagent/data/Huawei-East-1.csv> |

After downloading, the directory should contain:
```
raw_data/
├── trace_data_vmtable_vmtable.csv   # ~784 MB, 2,695,548 rows, 11 columns, no header
└── Huawei-East-1.csv                # ~5.3 MB, 125,430 VM create/delete events
```

Two scenarios must be run for each trace: `mixalgos` (two-class bottleneck,
$L_1$ discarded, $C_1=0$ by construction) and `improvevmpack` (three-class
general, $L_1$ retained).

```bash
# Huawei trace (run for both scenarios)
python process_huawei_trace.py --input raw_data/Huawei-East-1.csv \
    --output_dir ./data/huawei_trace/ --snapshot_every_events 2000 \
    --min_active_vms 50 --export_vmpack_json --T 7 --scenario mixalgos
python process_huawei_trace.py --input raw_data/Huawei-East-1.csv \
    --output_dir ./data/huawei_trace/ --snapshot_every_events 2000 \
    --min_active_vms 50 --export_vmpack_json --T 7 --scenario improvevmpack

# Microsoft vmtable (run for both scenarios)
python process_microsoft_vmtable.py --input raw_data/trace_data_vmtable_vmtable.csv \
    --output_dir ./data/microsoft_vmtable/ --T 7 --batch_size 1000 \
    --n_instances 100 --shuffle --seed 42 --scenario mixalgos
python process_microsoft_vmtable.py --input raw_data/trace_data_vmtable_vmtable.csv \
    --output_dir ./data/microsoft_vmtable/ --T 7 --batch_size 1000 \
    --n_instances 100 --shuffle --seed 42 --scenario improvevmpack
```

**Trace preprocessing protocol** (matches Section 5.2.2 of the paper):
- **Huawei-East-1** (125,430 VM create/delete events): events processed in
  chronological order; an active-set snapshot is taken every 2,000 events;
  snapshots with fewer than 50 active VMs are discarded; yields 121 snapshots.
  Each VM is mapped to type $(s,t)$ if its CPU is a power of two $2^t$
  ($t \in [0,6]$) and memory-to-CPU ratio is in $\{1,2,4\}$ ($s \in \{0,1,2\}$).
- **Microsoft 2017 vmtable** (2,695,548 VM records, 11 columns, no header):
  VMs filtered by the dyadic power-of-two rule on `vm_virtual_core_count_bucket`
  and `vm_memory_gb_bucket` (72.13% retained); retained VMs shuffled (seed=42)
  and grouped into batches of 1,000; yields 100 batches.
- In the `mixalgos` scenario, $s=1$ VMs are discarded to construct two-class
  bottleneck instances; in `improvevmpack`, all three classes are retained.

#### Step 7: Trace Experiments

```bash
python run_trace_experiments.py \
    --input ./data/huawei_trace/huawei_vmpack_instances_improvevmpack.json \
    --trace_name Huawei --T 7 --UP 1000 --output_dir ./result/ --bottleneck_only
```

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

> **Note:** This implementation is provided for academic and research purposes only. 

## Disclosure

A patent application (Application No. 202510593589.9) related to the underlying methods is pending with China Communications Information & Technology Group Co., Ltd.

## How to Cite

If you use this work or the provided code in your research, please cite our manuscript. To maintain anonymity during peer review, author information will be provided upon formal publication. The entry can be updated with the complete publication details at that time.

```bibtex
@misc{Anonymous_2025_preprint,
  title     = {Improving VMPack: Heuristic Mixed Packing Algorithms for Two Specific Virtual Machine Classes},
  author    = {Anonymous Author(s)},
  note      = {Manuscript under review},
  year      = {2025}
}
```
