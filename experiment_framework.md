# Experiment Framework — Section 5 of article_v10.tex

本文档是面向开发者的论文-代码映射详表，按照论文第5章《Computational Experiments》的小节顺序，系统地梳理了代码工程中的实验功能、函数设计、计算流程、输入输出文件。公开读者复现实验只需看 [README.md](README.md) 的 Quick Start；本文档供需要深入理解各实验 step 内部实现细节的开发者使用。

---

## 总览：实验的 4 个层次

| 层次 | 实例来源 | 目的 | 对应论文章节 |
|---|---|---|---|
| 1. Synthetic bottleneck | 随机生成（T=7, UP 控制规模） | 验证算法在理论 bottleneck 结构下的表现 | 5.2.1, 5.5 |
| 2. Public-trace-filtered | 华为/Microsoft 真实 trace | 外部验证——真实数据下结论是否成立 | 5.2.2, 5.6 |
| 3. Exact-verifiable subset | S1/S2/M1（1s 内可证明最优） | 校准启发式算法与真正最优解的差距 | 5.2.3, 5.7 |
| 4. Time-limited exact solver | 全部 6 个 scale | 展示 exact solver 的 scalability 与 warm-start 效果 | 5.7, 5.8 |

---

## 5.1 Experimental design and research questions（实验设计与研究问题）

### 功能
定义 5 个研究问题（RQ1–RQ5），指导后续所有实验。

### 代码工程
**无对应代码。** 这一节是纯文本描述，不涉及计算。

### 输入文件
无。

### 输出文件
无。

---

## 5.2 Instance families（实例族）

### 5.2.1 Synthetic bottleneck instances（合成 bottleneck 实例）

#### 功能
生成符合 bottleneck 条件的合成 VM 请求实例，用于测试算法在受控条件下的表现。

#### 参数设置
- T = 7（CPU 规格数：2^0 ~ 2^6）
- C = 256, M = 512（PM 容量）
- VM class 根据 memory-to-CPU ratio 划分：L0(1), L1(2), L2(4)
- 每个 VM type 的请求数 n_{t,s} ~ Uniform(1, UP)
- UP ∈ {10, 20, 50, 100, 500, 1000} 对应 S1–L2 六个 scale
- 每个 scale 100 个实例（过滤后）
- 两种 subfamily：
  - **mixalgos**（two-class）：仅 L0 + L2，过滤条件 C1=0, C0<3C2
  - **improvevmpack**（three-class）：L0 + L1 + L2，过滤触发 VMPack bottleneck

#### 对应代码

| 文件 | 函数/类 | 功能 |
|---|---|---|
| `data.py` | `DataTypes.RANDOM` | 随机分布类型标识 |
| `data.py` | `GenExamples(n, data_type, fun_case)` | 生成 n 个实例（auto-generates missing data via run_experiments.py） |
| `data.py` | `LoadExamples(path)` | 从 JSON 文件加载实例 |
| `data.py` | `SaveExamples(path, Ls, data_type, fun_case)` | 保存实例到 JSON |
| `data.py` | `GetFileName(n, data_type, fun_case)` | 获取标准文件名 |
| `globalvars.py` | `InitialGlobalVars(T, UP)` | 初始化全局变量（C, M 等） |
| `basic.py` | `CpuSize(vms)` | 计算 VM 类别的总 CPU 需求 |
| `run.py` | `step_generate_data(args)` | 统一入口：遍历所有 UP 值 + 两种 subfamily 批量生成 |

#### 输入文件
无（实例由代码随机生成）。

#### 输出文件
```
data/random_s1/          ← UP=10 的实例
data/random_s2/          ← UP=20 的实例
data/random_m1/          ← UP=50 的实例
data/random_m2/          ← UP=100 的实例
data/random_l1/          ← UP=500 的实例
data/random_l2/          ← UP=1000 的实例
│   └── {n_inst}_r_{T}_{UP}_{fun_case}.json
```
例：
- `data/random_s1/100_r_7_10_mixalgos.json` — 100 个 S1 two-class 实例
- `data/random_l2/100_r_7_1000_improvevmpack.json` — 100 个 L2 three-class 实例

#### JSON 格式
```json
[
  [  // instance 0
    [  // L0 VMs (class s=0)
      [cpu_size, memory_size, count],  // VM type (t=0)
      ...
    ],
    [  // L1 VMs (class s=1)
      ...
    ],
    [  // L2 VMs (class s=2)
      ...
    ]
  ],
  ...
]
```

---

### 5.2.2 Public-trace-filtered instances（公共 trace 过滤实例）

#### 功能
从真实云 trace 数据中提取 VM 请求，过滤出符合论文假设的实例，用于外部验证。

#### 数据来源
- **Huawei trace**（raw_data/Huawei-East-1.csv）：241,743 条 VM 创建/删除事件
- **Microsoft vmtable**（raw_data/trace_data_vmtable_vmtable.csv）：2,695,548 行 VM 记录，11 列

#### 过滤条件
1. CPU 和 memory 需求均为 2 的幂
2. memory-to-CPU ratio ∈ {1, 2, 4} → 映射到 L0(1), L1(2), L2(4)
3. 按时间窗口或连续请求分批，计算 C0, C1, C2
4. 保留 bottleneck batch：C1=0 且 C0<3C2

---

#### Huawei trace 处理

##### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `process_huawei_trace.py` | `load_create_events(input_file)` | 读取 CSV，提取 create VM 事件 |
| `process_huawei_trace.py` | `snapshot_active_set(events, time)` | 在指定时刻生成 active set 快照 |
| `process_huawei_trace.py` | `snapshot_to_vmpack_instance(active_set, T, allowed_classes)` | 将 active set 转为 VMPack 实例 |
| `process_huawei_trace.py` | `export_vmpack_instances(...)` | 批量转换并导出 VMPack 实例 |

##### 运行方式
```bash
# 直接运行
python process_huawei_trace.py --input raw_data/Huawei-East-1.csv \
    --output_dir ./data/huawei_trace/ \
    --snapshot_every_events 2000 --min_active_vms 50 \
    --export_vmpack_json --T 7 --scenario improvevmpack

# 通过 run.py
python run.py --steps process_traces --huawei_input raw_data/Huawei-East-1.csv
```

##### 输入文件
- `raw_data/Huawei-East-1.csv`（241,743 行，VM 创建/删除事件）

##### 输出文件
```
data/huawei_trace/
├── huawei_active_set_instances.json            # 121 个 active set 快照
├── huawei_active_set_instances_summary.csv     # 快照摘要
├── huawei_create_vm_stats.csv                  # VM 创建事件统计
├── huawei_create_vm_stats.png                  # 创建事件可视化
├── huawei_vmpack_instances_mixalgos.json       # 121 个 VMPack 实例（mixalgos 场景）
├── huawei_vmpack_instances_mixalgos_summary.csv
├── huawei_vmpack_instances_improvevmpack.json  # 121 个 VMPack 实例（improvevmpack 场景）
└── huawei_vmpack_instances_improvevmpack_summary.csv
```

---

#### Microsoft vmtable 处理

##### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `process_microsoft_vmtable.py` | `load_vmtable(input_file)` | 读取无表头 CSV，映射 11 列 |
| `process_microsoft_vmtable.py` | `find_resource_columns(df)` | 自动识别 CPU bucket 和 MEM bucket 列 |
| `process_microsoft_vmtable.py` | `filter_dyadic_vmtypes(df, T)` | 过滤 dyadic VM type |
| `process_microsoft_vmtable.py` | `generate_instances(...)` | 生成 VMPack 实例 |
| `process_microsoft_vmtable.py` | `export_vmpack_instances(...)` | 导出 JSON |

##### 运行方式
```bash
# 直接运行
python process_microsoft_vmtable.py --input raw_data/trace_data_vmtable_vmtable.csv \
    --output_dir ./data/microsoft_vmtable/ \
    --T 7 --batch_size 1000 --shuffle --seed 42 \
    --scenario improvevmpack

# 通过 run.py
python run.py --steps process_traces --microsoft_input raw_data/trace_data_vmtable_vmtable.csv
```

##### 输入文件
- `raw_data/trace_data_vmtable_vmtable.csv`（2,695,548 行，11 列，**无表头**）

##### 输出文件
```
data/microsoft_vmtable/
├── microsoft_cpu_memory_counts.csv                    # CPU-MEM 分布
├── microsoft_cpu_memory_counts.png                    # 分布可视化
├── microsoft_retained_cpu_memory_counts.csv            # 过滤后分布
├── microsoft_vmpack_instances_mixalgos.json            # 1944 个实例（mixalgos）
├── microsoft_vmpack_instances_mixalgos_summary.csv
├── microsoft_vmpack_instances_improvevmpack.json       # 1944 个实例（improvevmpack）
├── microsoft_vmpack_instances_improvevmpack_summary.csv
├── microsoft_vmtable_analysis_report_mixalgos.txt      # 分析报告
└── microsoft_vmtable_analysis_report_improvevmpack.txt
```

---

### 5.2.3 Exact-verifiable subset（精确可验证子集）

#### 功能
不单独生成数据，而是从 synthetic scale 中取 S1, S2, M1 作为 exact-verifiable subset（在 1s 时间限制下 VanillaMIP 可证明最优的 scale）。

#### 对应代码
直接复用 5.2.1 的合成数据。`gen_gap_summary.py` 默认对 S1, S2, M1 计算 gap。

---

## 5.3 Compared algorithms（对比算法）

### 功能
定义四组算法及其角色。

### 算法清单

| 组 | 算法 | 代码实现位置 | 论文角色 |
|---|---|---|---|
| Engineering heuristics | FFD, BFD | `heuristics.py` | 快速工程基线 |
| Direct prior work | VMPack+MixVM301 | `heuristics.py` (`VMPack` + `MixVM301`) | 原始 VMPack bottleneck 策略 |
| Proposed family | NoMixPack, MixVM301, MixVM201, MixVM201Pro, MixPack | `heuristics.py` | 渐进式机制变体 |
| Exact solvers | VanillaMIP, P&B | `vanilla_mip.py`, `pricebranch.py` | 精确求解器基准 |

### 启发式算法函数签名

```python
# heuristics.py
def NoMixPack(vm_demands)                          -> (npms, detailed)
def MixVM301(vm_demands)                           -> (npms, detailed)
def MixVM201(vm_demands)                           -> (npms, detailed)
def MixVM201Pro(vm_demands)                        -> (npms, detailed)
def MixPack(vm_demands)                            -> (npms, detailed)
def VMPack(vm_demands)                             -> (npms, detailed)
def BFD(vm_demands)                                -> (npms, detailed)
def FFD(vm_demands)                                -> (npms, detailed)

# VMPack 嵌入版本（在 VMPack 框架内替换 bottleneck 策略）
def VMPack_NoMixPack(vm_demands)
def VMPack_MixVM301(vm_demands)
def VMPack_MixVM201(vm_demands)
def VMPack_MixVM201Pro(vm_demands)
def VMPack_MixPack(vm_demands)
```

### 精确求解器函数签名

```python
# vanilla_mip.py
def VanillaMIP(vm_demands, timelimit, ub_heuristic_fn)  -> (npms, detailed, gap_info)

# pricebranch.py
def PriceBranch(vm_demands, timelimit, ub_heuristic_fn) -> (npms, detailed, gap_info)
```

---

## 5.4 Evaluation metrics and implementation details（评估指标与实现细节）

### 功能
定义所有评估指标和实验环境参数。

### 指标计算公式

| 指标 | 公式 | 代码实现位置 |
|---|---|---|
| PM count A(L) | 算法返回的 PM 数量 | 各算法的 `npms` 返回值 |
| Lower bound lb(L) | ceil(max((C0+C1+C2)/C, (C0+2C1+4C2)/(2C))) | `run_experiments.py:compute_lb()` |
| Lower-bound ratio ρ_LB | A(L) / lb(L) | `gen_tables.py` 中计算 |
| Exact optimality gap | (A(L) - z*) / z* × 100% | `gen_gap_summary.py:summarize_heuristic_gap()` |
| CPU utilization η_CPU | total_cpu / (A(L) × C) | `gen_tables.py` 中计算 |
| MEM utilization η_MEM | total_mem / (A(L) × 2C) | `gen_tables.py` 中计算 |
| Win/tie/loss | 算法 A vs B 的 PM 数比较 | `run_trace_experiments.py` 中计算 |

### 环境参数

| 参数 | 值 | 设置位置 |
|---|---|---|
| Python | 3.12 | conda env py312 |
| CPU | AMD Ryzen 9 7945HX | — |
| RAM | 32 GB | — |
| Gurobi | 13.0.1 | `vanilla_mip.py`, `pricebranch.py` |
| Gurobi Threads | 1 | `gurobipy` 参数设置 |
| Gurobi TimeLimit | maxtime=10s (hard limit), checkpoint=1s/5s/10s | `--maxtime` + callback |
| Gurobi MemLimit | 24 GB | `gurobipy` 参数设置 |
| Gurobi Seed | 42 | `gurobipy` 参数设置 |
| NumPy seed | 42 | `np.random.seed(42)` |

---

## 5.5 Results on synthetic bottleneck instances（合成实例结果）

### 5.5.1 Standalone mixed-packing heuristics（独立混合打包启发式）

#### 功能
在 two-class bottleneck 实例（L0+L2, UP=1000, 100 个实例）上对比所有启发式算法。

#### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `run_experiments.py` | `run_heuristic_comparison(n_inst, T, UP, output_dir, configs)` | 运行一组启发式算法并保存结果 |
| `run_experiments.py` | `run_heuristic(vm_demands, heuristic_fn)` | 单个启发式算法运行 |
| `run_experiments.py` | `HEURISTIC_CONFIGS['mixalgos']` | 配置：7 个算法 + fun_case='mixalgos' |
| `gen_tables.py` | `gen_heuristic_comparison(output_dir, tag)` | 生成 LaTeX Table 11 |

#### 运行方式
```bash
# 单个 UP 值
python run_experiments.py --mode heuristic --heuristic-config mixalgos \
    --T 7 --UP 1000 --n_inst 100 --tag up1000

# 全部 UP 值（通过 run.py）
python run.py --steps heuristic
```

#### 输入文件
- `data/100_r_7_1000_mixalgos.json`（合成 two-class 实例）

#### 输出文件
```
result/heuristic/
├── mixalgos.csv                     # UP=1000 的完整结果（7 算法 × 100 实例）
├── mixalgos_up10.csv                # UP=10
├── mixalgos_up20.csv                # UP=20
├── mixalgos_up50.csv                # UP=50
├── mixalgos_up100.csv               # UP=100
├── mixalgos_up500.csv               # UP=500
└── mixalgos_up1000.csv              # UP=1000
```

#### CSV 格式
```csv
instance,algorithm,funcase,T,UP,C,M,n_inst,seed,npms,time,lb,total_cpu,total_mem
0,NoMixPack,mixalgos,7,1000,256,512,100,42,734,0.005,611,502104,632656
0,MixVM301,mixalgos,7,1000,256,512,100,42,654,0.014,611,502104,632656
...
```

#### 输出 LaTeX 表格
- **Table 11** (`tab:heuristic_mixalgos`)：7 行算法 × 6 列指标

---

### Figure 4: PM usage across scales（PM 用量趋势图）

#### 功能
展示 BFD、VMPack+MixVM301、VMPack+MixVM201Pro、VMPack+MixPack 四条 PM 数量曲线，横轴为 6 个 scale（S1–L2）。

#### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `plot_results.py` | `plot_fig4_quality_trend(output_dir, tag, config)` | 读取 improvevmpack_up*.csv，绘制 4 条曲线 |
| `run.py` | `step_gen_figures(args)` | 统一调用 |

#### 输入文件
- `result/heuristic/improvevmpack_up{10,20,50,100,500,1000}.csv`

#### 输出文件
```
result/figures/
├── pm_usage_scales.pdf      # 矢量图
└── pm_usage_scales.png      # 位图
```
```

---

### 5.5.2 Embedding the proposed heuristics into VMPack（嵌入 VMPack）

#### 功能
在 three-class bottleneck 实例（L0+L1+L2, UP=1000）上测试 VMPack 嵌入变体。

#### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `run_experiments.py` | `HEURISTIC_CONFIGS['improvevmpack']` | 配置：7 个 VMPack 算法 |
| `gen_tables.py` | `gen_heuristic_comparison(output_dir, tag)` | 生成 LaTeX Table 12 |

#### 运行方式
```bash
python run_experiments.py --mode heuristic --heuristic-config improvevmpack \
    --T 7 --UP 1000 --n_inst 100
```

#### 输入文件
- `data/100_r_7_1000_improvevmpack.json`（合成 three-class 实例）

#### 输出文件
```
result/heuristic/
├── improvevmpack.csv                # UP=1000 的完整结果
├── improvevmpack_up10.csv
├── improvevmpack_up20.csv
├── improvevmpack_up50.csv
├── improvevmpack_up100.csv
├── improvevmpack_up500.csv
└── improvevmpack_up1000.csv
```

#### 输出 LaTeX 表格
- **Table 12** (`tab:heuristic_improvevmpack`)：7 行 VMPack 算法 × 6 列指标

---

### Figure 5: Runtime-quality trade-off（运行时-质量散点图）

#### 功能
在二维空间中展示所有算法-scale 对的平均运行时（x轴）和平均 ρ_LB（y轴）。mixalgos 用圆圈标记，improvevmpack 用方块标记。

#### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `plot_results.py` | `plot_fig5_runtime_quality_tradeoff(output_dir, tag, config)` | 读取 mixalgos.csv + improvevmpack.csv，绘制散点图 |

#### 输入文件
- `result/heuristic/mixalgos.csv`
- `result/heuristic/improvevmpack.csv`

#### 输出文件
```
result/figures/
├── runtime_quality_scatter.pdf
└── runtime_quality_scatter.png
```

---

## 5.6 Results on public-trace-filtered instances（trace 过滤实例结果）

### 功能
在 trace 过滤实例上运行 5 个算法（FFD, BFD, VMPack+MixVM301, VMPack+MixVM201Pro, VMPack+MixPack），输出性能对比和 win/tie/loss 统计。

#### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `run_trace_experiments.py` | `main()` | 读取 VMPack JSON 实例，运行所有算法 |
| `run_trace_experiments.py` | `compute_wtl(results, baseline_algo)` | 计算 win/tie/loss |
| `gen_tables.py` | `gen_trace_detail_table(output_dir, tag)` | 生成 Table 13 |
| `gen_tables.py` | `gen_trace_win_tie_loss_table(output_dir, tag)` | 生成 Table 14 |

#### 运行方式
```bash
python run_trace_experiments.py \
    --input ./data/huawei_trace/huawei_vmpack_instances_improvevmpack.json \
    --trace_name Huawei --T 7 --UP 1000 --output_dir ./result/trace/

# 通过 run.py
python run.py --steps trace_experiments
```

#### 输入文件
- `data/huawei_trace/huawei_vmpack_instances_improvevmpack.json`
- `data/microsoft_vmtable/microsoft_vmpack_instances_improvevmpack.json`

#### 输出文件
```
result/trace/
├── huawei_trace_detail_huawei.csv           # 华为 121 实例 × 5 算法 详细结果
├── huawei_trace_summary_huawei.csv          # 华为汇总统计
├── huawei_trace_wtl_huawei.csv              # 华为 win/tie/loss
├── microsoft_trace_detail_microsoft.csv     # 微软 100 实例 × 5 算法 详细结果
├── microsoft_trace_summary_microsoft.csv    # 微软汇总统计
└── microsoft_trace_wtl_microsoft.csv        # 微软 win/tie/loss
```

#### WTL CSV 格式
```csv
baseline_algo,target_algo,compared_instances,baseline_win,tie,target_win,trace
VMPack_MixVM301,VMPack_MixVM201Pro,121,0,30,91,Huawei
...
```

#### 输出 LaTeX 表格
- **Table 13** (`tab:trace_results`)：华为 + 微软的 5 算法性能表
- **Table 14** (`tab:trace_win_tie_loss`)：华为 + 微软的 win/tie/loss 对比

---

### Figure 6: WTL heatmap（Win/Tie/Loss 热力图）

#### 功能
以 VMPack+MixVM301 为 baseline，展示 MixVM201Pro 和 MixPack 在两个 trace 上的 Win/Tie/Loss 分布。

#### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `plot_results.py` | `plot_fig6_wtl_heatmap(output_dir, tag)` | 读取 WTL CSV，绘制热力图 |

#### 输入文件
- `result/trace/huawei_trace_wtl_huawei.csv`
- `result/trace/microsoft_trace_wtl_microsoft.csv`

#### 输出文件
```
result/figures/
├── wtl_heatmap.pdf
└── wtl_heatmap.png
```

---

## 5.7 Exact-solver comparison and exact-verifiable subset（精确求解器对比）

### 功能
在全部 6 个 scale 上运行 VanillaMIP 和 P&B（含 NoMix/Mix 两种初始化），输出 PM 数、运行时间、求解状态，并计算全部 6 个 scale 上的最优性 gap。

#### 重要：时间限制实验的设计变更

当前代码使用 **单次 maxtime 运行 + Gurobi checkpoint callback** 替代了原来的三次独立运行（1s/5s/10s）：

| 算法 | 运行方式 | 说明 |
|---|---|---|
| 启发式（FFD/BFD/MixVM201Pro...） | 自然结束（无时间限制） | 几十毫秒完成，结果唯一 |
| P&B / P&B+Mix | 自然结束（无时间限制） | <1s 收敛，所有时间点结果相同 |
| VanillaMIP / MIP+Mix | **maxtime=10s** | Gurobi TimeLimit=10，callback 在 1s/5s/10s 记录中间状态 |

**`_CheckpointCallback`**（`vanilla_mip.py`）通过 Gurobi callback API 在指定时间点记录：
- incumbent 上界（ub）
- best bound 下界（lb）
- MIP gap
- node count
- wall-clock time

如果 solver 在 checkpoint 时间之前就收敛了（如 S1 在 0.01s 完成），callback 自动用最终状态填充缺失的 checkpoint。

**计时说明：** VanillaMIP 的 `mip_time` 是 wall-clock 总时间，包含启发式 warm-start + 模型构建 + Gurobi 优化。Gurobi 的 `TimeLimit` 只限制优化阶段本身，因此当模型构建时间较长时（如 L2 scale ~200ms），`mip_time` 可能略微超过 `TimeLimit` 设置值（如 1215ms > 1000ms）。论文中的 "1s"、"5s"、"10s" 指 Gurobi 的 TimeLimit 参数值，而非算法总时间。启发式和 P&B 的时间始终是自然完成时间。

#### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `vanilla_mip.py` | `_CheckpointCallback(checkpoint_times)` | Gurobi callback：在指定时间点记录 solver 状态 |
| `vanilla_mip.py` | `VanillaMIP(vm_demands, timelimit, ub_heuristic_fn, checkpoint_times)` | 运行 VanillaMIP（支持 checkpoint） |
| `run_solvers_only.py` | `main()` | 单次 maxtime 运行，自动生成多时间点 CSV |
| `run_solvers_only.py` | `run_vanilla_mip(...)` | 封装 VanillaMIP + 提取 checkpoint 数据 |
| `run_solvers_only.py` | `write_checkpoint_csvs(...)` | 从 maxtime CSV 提取 1s/5s/10s 子文件 |
| `pricebranch.py` | `PriceBranch(vm_demands, timelimit, ub_heuristic_fn)` | 运行 P&B（自然结束） |
| `gen_gap_summary.py` | `summarize_heuristic_gap(rows)` | 计算启发式 vs MIP 最优的 gap |
| `gen_gap_summary.py` | `generate_heuristic_gap_table(output_dir, tag, scales)` | 生成 Table 16 |

#### 5 组对比实验

| 编号 | 算法 | 初始化 |
|---|---|---|
| 1 | VMPack+MixVM201Pro（启发式） | — |
| 2 | PriceBranch | NoMix（空初始列） |
| 3 | VanillaMIP | NoMix（无上界） |
| 4 | PriceBranch + Mix | MixVM201Pro 初始列 |
| 5 | VanillaMIP + Mix | MixVM201Pro 初始上界 |

#### 运行方式
```bash
# 单次 maxtime 运行（自动生成 1s/5s/10s checkpoint CSV）
python run_solvers_only.py --maxtime 10 --tag maxtime --checkpoint_times 1,5,10 --n_inst 100

# 通过 run.py
python run.py --steps scale_maxtime
```

#### 输入文件
- `data/100_r_7_{UP}_improvevmpack.json`（6 个 UP 值）

#### 输出文件
一次运行生成 4 个 CSV 文件 × 6 个 scale：
```
result/scale/
├── S1_maxtime.csv               # 完整数据（含所有 checkpoint 列）
├── S1.csv                       # 1s checkpoint（向后兼容）
├── S1_tl10.csv                  # 10s checkpoint（向后兼容）
├── S1_tl5.csv                   # 5s checkpoint
├── S2_maxtime.csv, S2.csv, ...
├── M1_maxtime.csv, M1.csv, ...
├── M2_maxtime.csv, M2.csv, ...
├── L1_maxtime.csv, L1.csv, ...
├── L2_maxtime.csv, L2.csv, ...
├── S1_mixalgos_maxtime.csv      # mixalgos 场景加 _mixalgos 后缀
├── L1_mixalgos_maxtime.csv
└── L2_mixalgos_maxtime.csv
```

#### CSV 列说明（maxtime CSV 约 100 列）

| 类别 | 列前缀 | 内容 |
|---|---|---|
| 元信息 | instance, scale, group, funcase, T, UP, C, M, n_inst, seed, timelimit | 实验参数 |
| 启发式 | heuristic_npms, heuristic_time | VMPack+MixVM201Pro 结果 |
| P&B (NoMix) | pb_npms, pb_lb, pb_ub, pb_gap, pb_time, pb_status, pb_n_cols | P&B 自然结束 |
| P&B (Mix) | pb_mix_npms, pb_mix_lb, pb_mix_ub, pb_mix_gap, pb_mix_time, pb_mix_status, pb_mix_n_cols | P&B+Mix 自然结束 |
| VanillaMIP (NoMix) maxtime | mip_npms, mip_time, mip_status | maxtime 最终结果 |
| VanillaMIP checkpoint | mip_1s_npms, mip_1s_time, mip_1s_status, mip_5s_*, mip_10s_* | 1s/5s/10s 中间状态 |
| VanillaMIP (Mix) maxtime | mip_mix_npms, mip_mix_time, mip_mix_status | maxtime 最终结果 |
| VanillaMIP (Mix) checkpoint | mip_mix_1s_npms, mip_mix_1s_time, mip_mix_1s_status, mip_mix_5s_*, mip_mix_10s_* | 1s/5s/10s 中间状态 |
| 差值 | h_minus_pb, h_minus_mip | 启发式减求解器 |

#### 输出 LaTeX 表格
- **Table 15** (`tab:exact_comparison`)：6 scale × 3 算法（Ours/P&B/MIP）的 PM 数和时间
- **Table 16** (`tab:exact_gap_subset`)：S1/S2/M1/M2/L1/L2 的最优性 gap
- **Table 18** (`tab:heuristic_lift`)：6 scale × 2 求解器的 NoMix→Mix 提升

#### Gap summary 输出
```
result/gap/
├── exact_gap_summary.csv          # 1s gap（从 checkpoint 数据提取）
├── exact_gap_summary_tl10.csv     # 10s gap
├── exact_gap_summary_tl5.csv      # 5s gap
└── exact_gap_summary_maxtime.csv  # maxtime gap
```

---

## 5.8 Mechanism-level attribution and warm-start analysis（机制归因与 Warm-start 分析）

### 5.8.1 Mechanism-level attribution（机制归因）

#### 功能
通过算法序列的渐进式对比，归因每个设计机制的贡献。**不涉及新实验**，直接引用已有表格的数据。

#### 5 个机制对比

| 对比 | 证据来源 | 归因的机制 |
|---|---|---|
| NoMixPack vs MixVM301 | Table 11 | mixing 的价值 |
| MixVM301 vs MixVM201 | Table 11 | flexible residual-capacity reuse |
| MixVM201 vs MixVM201Pro | Table 11 | priority + Δ constraint |
| MixVM201Pro vs MixPack | Table 11 + Theorem | 简化 vs 理论保证 |
| VMPack+MixVM301 vs VMPack+MixVM201Pro | Table 12 | 替换 bottleneck 策略 |
| NoMix init vs Mix init | Table 18 | warm-start |

#### 对应代码
无新增实验代码。直接使用 `gen_tables.py` 输出的 Table 11/12/18 数据。

---

### 5.8.2 Warm-start analysis（Warm-start 分析）

#### 功能
测试 MixVM201Pro 的 packing pattern 是否能加速 P&B 和 VanillaMIP。

#### 对应代码
复用 5.7 节的 scale 实验数据（pb_npms vs pb_mix_npms, mip_npms vs mip_mix_npms）。

#### 运行方式
已在 5.7 节的 1s/5s/10s 实验中完成。

---

### Figure 7: Time-limit warm-start sensitivity（时间限制敏感性）

#### 功能
三面板图：Panel 1 = P&B 生成列数（时间无关），Panel 2 = P&B 运行时间（时间无关），Panel 3 = VanillaMIP optimal rate（1s/5s/10s 三组对比）。

#### 对应代码

| 文件 | 函数 | 功能 |
|---|---|---|
| `plot_results.py` | `plot_fig7_warm_start_effect(output_dir, tags)` | 读取 1s + t5 + tl0 数据，绘制三面板图 |

#### 输入文件
- `result/scale/S1.csv`, ..., `L2.csv`（1s）
- `result/scale/S1_tl10.csv`, ..., `L2_tl10.csv`（10s）
- `result/scale/S1_tl5.csv`, ..., `L2_tl5.csv`（5s）

#### 输出文件
```
result/figures/
├── time_limit_warm_start.pdf
└── time_limit_warm_start.png
```

---

## 5.9 Summary of experimental findings（实验结论总结）

### 功能
纯文本总结，不涉及新实验。

---

## result/ 目录结构与文件读写工作流

### 目录总览

```
result/
├── mixalgos/              ← 最终宽表（二类场景，每行一个算例）
├── improvevmpack/         ← 最终宽表（三类场景，每行一个算例）
├── figures/               ← 生成的图片（PDF + PNG）
├── gap/                   ← 最优性 gap 汇总 CSV
├── heuristic/             ← 启发式实验原始结果（长格式 CSV）
├── trace/                 ← Trace 实验原始结果（长格式 CSV + 元数据）
├── huawei_trace/          ← 华为 trace 处理后的 JSON 实例
└── microsoft_vmtable/     ← 微软 trace 处理后的 JSON 实例
```

### mixalgos/ 与 improvevmpack/ —— 最终宽表

这是**最核心的两个目录**，每个场景各 8 个文件：

```
mixalgos/ 或 improvevmpack/
├── random_s1.csv ~ random_l2.csv     # 6 档合成实例
├── huawei.csv                        # 华为 trace 实例
└── microsoft.csv                     # 微软 trace 实例
```

**每行一个算例**，列顺序由 `column_registry.py` 定义：

| 列类别 | 列名示例 | 说明 |
|---|---|---|
| 标识 | `seq`, `instance` | 序号和实例 JSON 字符串 |
| 资源下界 | `lb`, `total_cpu`, `total_mem` | 资源需求 |
| 启发式结果 | `MixVM201Pro_npms`, `MixVM201Pro_time` | 前缀+后缀命名 |
| 求解器结果 | `MIP_Mix_npms`, `MIP_Mix_status`, `MIP_Mix_bestbound` | 含 checkpoint |
| Checkpoint | `MIP_Mix_1s_npms`, `MIP_Mix_5s_gap`, `MIP_Mix_10s_status` | 1s/5s/10s |

列数对比：
- **mixalgos**：5(公共) + 7×2(启发式) + 14(MIP+Mix) = **33 列**
- **improvevmpack**：5(公共) + 7×2(启发式) + 38(4 个求解器) = **57 列**

---

### run.py 各步骤的文件读写工作流

#### Step 1: `generate_data`

```
写：data/{n_inst}_r_{T}_{UP}_{fun_case}.json
```

随机生成合成实例 JSON，供后续所有步骤使用。

---

#### Step 2: `process_traces`

```
读：raw_data/Huawei-East-1.csv, raw_data/trace_data_vmtable_vmtable.csv
写：data/huawei_trace/     ← 华为处理后的 JSON 实例
写：data/microsoft_vmtable/ ← 微软处理后的 JSON 实例
```

处理原始 trace CSV，提取 VMPack 兼容的实例 JSON（两种 fun_case 各一份）。

---

#### Step 3: `heuristic`

```
读：data/random_{s1..l2}/{n_inst}_r_{T}_{UP}_{fun_case}.json    ← 合成实例
写：result/heuristic/{fun_case}_up{UP}.csv                       ← 长格式原始结果
写：result/heuristic/{fun_case}_metadata_up{UP}.json
```

运行 7 个启发式算法 × 100 实例 × 6 档 UP × 2 种 fun_case。

输出文件是**长格式**（每个算法一行，instance 重复），供后续 `export_unified` 步骤 pivot 为宽表。

---

#### Step 4: `scale_maxtime`

```
读：data/random_{s1..l2}/{n_inst}_r_{T}_{UP}_{fun_case}.json     ← 合成实例
写：result/scale/{Scale}_maxtime.csv                 ← 含全部 checkpoint 列
写：result/scale/{Scale}.csv, {Scale}_tl5.csv, {Scale}_tl10.csv
写：result/scale/{Scale}{_fun_case}_maxtime_metadata.json
```

运行 4 个求解器（P&B / P&B+Mix / MIP / MIP+Mix）× 100 实例 × 6 档 UP × 2 种 fun_case。

Mixalgos 场景输出带 `_mixalgos` 后缀的文件名（如 `L2_mixalgos_maxtime.csv`）。

---

#### Step 5: `trace_experiments`

```
读：data/huawei_trace/huawei_vmpack_instances_{fun_case}.json
读：data/microsoft_vmtable/microsoft_vmpack_instances_{fun_case}.json
写：result/trace/{trace}_trace_detail_{tag}.csv       ← 详细结果
写：result/trace/{trace}_trace_summary_{tag}.csv       ← 汇总统计
写：result/trace/{trace}_trace_wtl_{tag}.csv           ← win/tie/loss
写：result/trace/{trace}_trace_metadata_{tag}.json
```

运行 5 个启发式（FFD/BFD/VMPack+MixVM301/VMPack+MixVM201Pro/VMPack+MixPack）× trace 实例。

tag 区分场景：mixalgos 用 `{trace_key}`，improvevmpack 用 `{trace_key}_improvevmpack`。

---

#### Step 6: `export_unified`

```
读：result/heuristic/{fun_case}_up{UP}.csv              ← 长格式启发式结果
读：result/scale/{Scale}{_fun_case}_maxtime.csv          ← 求解器结果（--with_solvers 时）
读：result/trace/{trace}_trace_detail_{tag}.csv          ← trace 结果
读：data/random_{s1..l2}/{n_inst}_r_{T}_{UP}_{fun_case}.json  ← 实例 JSON（获取 VM 需求）
写：result/{fun_case}/random_{s1..l2}.csv               ← 宽表（每行一个算例）
写：result/{fun_case}/huawei.csv, microsoft.csv          ← trace 宽表
```

**核心聚合步骤**：
1. 将长格式启发式 CSV **pivot** 为宽格式（每个算法一列）
2. 从 `data/` JSON 获取每实例的 VM 需求（`L0_t0` ~ `L2_t6` 列）
3. 从 `scale/` maxtime CSV **merge** 求解器列（列名映射：`pb_*` → `PB_*`）
4. 从 `trace/` 读取 trace 结果并 pivot 为宽格式
5. 输出到 `result/{fun_case}/` 目录，列顺序由 `column_registry.py` 控制

---

#### Step 7: `gen_tables`

```
读：result/{fun_case}/random_l2.csv                     ← 宽表（启发式对比表）
读：result/{fun_case}/random_{s1..l2}.csv               ← 宽表（最优解映射）
读：result/scale/{Scale}{_fun_case}_maxtime.csv         ← 求解器数据（resolve_csv_path 搜索子目录）
读：result/trace/*.csv                                   ← trace 数据
读：data/huawei_trace/*, data/microsoft_vmtable/*        ← trace 摘要
写：输出 LaTeX 表格到 stdout
写：result/gap/exact_gap_summary*.csv                    ← gap 汇总（由 gen_gap_summary 输出）
```

生成所有 LaTeX 表格。关键路径：
- `gen_heuristic_comparison()` → 读取 `{fun_case}/random_l2.csv` 宽表
- `gen_scale_sweep_table()` → 读取 `mixalgos/random_*.csv` 宽表
- `gen_method_comparison()` → 通过 `resolve_csv_path` 搜索 `scale/` 子目录
- `gen_gap_summary.py` → 通过 `resolve_csv_path` 搜索 `scale/` 子目录

---

#### Step 8: `gen_figures`

```
读：result/{fun_case}/random_*.csv                       ← 宽表（Figure 4/5/7）
读：result/trace/*_wtl_*.csv                              ← WTL 数据
写：result/figures/*.pdf, *.png
```

Figure 4：从 `improvevmpack/random_*.csv` 宽表读取
Figure 5：从 `mixalgos/random_*.csv`, `improvevmpack/random_*.csv` 宽表读取
Figure 6：从 `trace/*_wtl_*.csv` 读取
Figure 7：从 `improvevmpack/random_*.csv` 宽表读取（含 checkpoint 列）

---

### 文件读写流程图

```
data/random_{s1..l2}/  ──→ Step1: generate_data ──→ data/random_{s1..l2}/*.json
                                                          │
raw_data/Huawei-East-1.csv ──→ Step2: process_traces ──→ data/huawei_trace/*
raw_data/trace_data_vmtable_vmtable.csv ──→               data/microsoft_vmtable/*
                                                          │
data/random_{s1..l2}/*.json ──→ Step3: heuristic ──→ result/heuristic/*.csv
                                                          │
data/random_{s1..l2}/*.json ──→ Step4: scale_maxtime ──→ result/scale/*_maxtime.csv
                                                          │
data/huawei_trace/* ─────→ Step5: trace_experiments ──→ result/trace/*.csv
data/microsoft_vmtable/*                                   │
                                                          │
result/heuristic/*.csv ──→ Step6: export_unified ──→ result/{fun_case}/random_*.csv
result/scale/*_maxtime.csv ──┘                          result/{fun_case}/huawei.csv
result/trace/*.csv ──────────┘                           result/{fun_case}/microsoft.csv
data/random_{s1..l2}/*.json ──┘
                                                          │
result/{fun_case}/random_*.csv ──→ Step7: gen_tables ──→ LaTeX stdout
result/scale/*.csv ────────────────┘                     result/gap/*.csv
result/trace/*.csv ────────────────┘
                                                          │
result/{fun_case}/random_*.csv ──→ Step8: gen_figures ──→ result/figures/*.pdf/png
result/trace/*_wtl_*.csv ────────┘
```

## 文件索引速查表

### 核心算法文件

| 文件 | 核心函数/类 | 行数 | 用途 |
|---|---|---|---|
| `basic.py` | `CpuSize()`, `FindVMsWithBound()`, `FillOnePm()`, `ValidatePMs()` | ~100 | 基础工具函数 |
| `data.py` | `GenExamples()`, `LoadExamples()`, `SaveExamples()`, `GetFileName()` | ~150 | 合成数据生成与管理 |
| `globalvars.py` | `InitialGlobalVars(T, UP)` | ~15 | 全局变量初始化 |
| `utils.py` | 共享常量（SCALES, UP_SWEEP）+ 工具函数 | ~80 | 通用工具函数 |
| `heuristics.py` | 10+ 启发式算法函数 | ~600 | 所有启发式算法实现 |
| `vanilla_mip.py` | `VanillaMIP()`, `_CheckpointCallback` | ~350 | 基于 Gurobi 的 MIP 求解器（含 checkpoint callback） |
| `pricebranch.py` | `PriceBranch()` | ~250 | 列生成 P&B 求解器 |

### 实验编排文件

| 文件 | 核心函数 | 行数 | 用途 |
|---|---|---|---|
| `run.py` | 9 个 step_* 函数 | ~350 | **统一实验入口** |
| `run_experiments.py` | `run_heuristic_comparison()`, `run_scale_experiment()` | ~620 | 合成实例实验运行器 |
| `run_solvers_only.py` | `write_checkpoint_csvs()`, `build_result_row()` | ~400 | 单次 maxtime 求解器实验（含 checkpoint） |
| `run_trace_experiments.py` | `compute_wtl()`, `main()` | ~250 | Trace 实例实验运行器 |

### 结果生成文件

| 文件 | 核心函数 | 行数 | 用途 |
|---|---|---|---|
| `gen_tables.py` | `gen_heuristic_comparison()`, `gen_method_comparison()`, `gen_heuristic_empowerment()`, `gen_trace_detail_table()`, `gen_trace_win_tie_loss_table()`, `gen_public_trace_summary_table()` | ~640 | 生成所有 LaTeX 表格 |
| `gen_gap_summary.py` | `summarize_heuristic_gap()`, `generate_heuristic_gap_table()`, `generate_latex_table()` | ~380 | 生成最优性 gap 表格 |
| `plot_results.py` | `plot_fig4_quality_trend()`, `plot_fig5_runtime_quality_tradeoff()`, `plot_fig6_wtl_heatmap()`, `plot_fig7_warm_start_effect()` | ~620 | 生成所有图片 |
| `utils.py` | 共享常量 + 工具函数 | ~80 | 通用工具函数 |

### 数据处理文件

| 文件 | 核心函数 | 行数 | 用途 |
|---|---|---|---|
| `process_huawei_trace.py` | `load_create_events()`, `snapshot_active_set()`, `snapshot_to_vmpack_instance()`, `export_vmpack_instances()` | ~550 | 华为 trace 处理 |
| `process_microsoft_vmtable.py` | `load_vmtable()`, `find_resource_columns()`, `filter_dyadic_vmtypes()`, `generate_instances()` | ~450 | 微软 vmtable 处理 |

---

## 输出目录结构（整理后）

```
result/
├── heuristic/                   启发式算法对比 CSV
│   ├── mixalgos.csv
│   ├── mixalgos_up10.csv ... mixalgos_up1000.csv
│   ├── improvevmpack.csv
│   └── improvevmpack_up10.csv ... improvevmpack_up1000.csv
├── scale/                       Scale 实验 CSV（含元数据 JSON）
│   ├── S1.csv, S2.csv, ..., L2.csv          # 1s checkpoint
│   ├── S1_maxtime.csv, ..., L2_maxtime.csv  # maxtime（含所有 checkpoint 列）
│   ├── S1_tl10.csv, ..., L2_tl10.csv        # 10s checkpoint
│   ├── S1_tl5.csv, ..., L2_tl5.csv          # 5s checkpoint
├── trace/                       Trace 实验结果
│   ├── huawei_trace_detail_huawei.csv
│   ├── huawei_trace_summary_huawei.csv
│   ├── huawei_trace_wtl_huawei.csv
│   ├── microsoft_trace_detail_microsoft.csv
│   ├── microsoft_trace_summary_microsoft.csv
│   └── microsoft_trace_wtl_microsoft.csv
├── gap/                         Gap 汇总
│   ├── exact_gap_summary.csv
│   ├── exact_gap_summary_tl10.csv
│   └── exact_gap_summary_tl5.csv
├── figures/                     生成的图片
│   ├── pm_usage_scales.pdf/.png
│   ├── runtime_quality_scatter.pdf/.png
│   ├── wtl_heatmap.pdf/.png
│   └── time_limit_warm_start.pdf/.png
├── huawei_trace/                华为 trace 处理输出
│   ├── huawei_vmpack_instances_*.json
│   └── huawei_active_set_instances.json
└── microsoft_vmtable/           微软 trace 处理输出
    ├── microsoft_vmpack_instances_*.json
    └── microsoft_vmtable_analysis_report_*.txt
```

## 数据流图（Data Flow）

```
data.py GenExamples()
    ↓
data/*.json ──────────────────────────────────────────┐
    ↓                                                   │
run_experiments.py (heuristic mode)                    │
    ↓                                                   │
result/heuristic/*.csv ──→ gen_tables.py ──→ LaTeX Table 11/12
    ↓
plot_results.py ──→ Figure 4 (pm_usage_scales)
    ↓
plot_results.py ──→ Figure 5 (runtime_quality_scatter)

data/*.json ──────────────────────────────────────────┐
    ↓                                                   │
run_solvers_only.py (--maxtime 10 --checkpoint_times 1,5,10)
    ↓
result/*_maxtime.csv  ← 完整数据（含 1s/5s/10s checkpoint 列）
    ↓输出子文件
result/{S1,S2,...,L2}.csv        (1s checkpoint)
result/{S1,S2,...,L2}_tl10.csv   (10s checkpoint)
result/{S1,S2,...,L2}_tl5.csv    (5s checkpoint)
    ↓
gen_tables.py ──→ LaTeX Table 15/18
    ↓                        ↓
plot_results.py          gen_gap_summary.py
    ↓                        ↓
Figure 7 (warm_start)    LaTeX Table 16 (6 scales)

process_huawei_trace.py ──→ data/huawei_trace/*.json ──┐
process_microsoft_vmtable.py ──→ data/microsoft_vmtable/*.json ──┤
    ↓                                                      │
run_trace_experiments.py                                   │
    ↓                                                      │
result/trace/*.csv ──→ gen_tables.py ──→ LaTeX Table 13/14
    ↓
plot_results.py ──→ Figure 6 (wtl_heatmap)
```
