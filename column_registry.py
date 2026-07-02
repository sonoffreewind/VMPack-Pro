"""
column_registry.py

Unified column registry for experiment result CSVs.
Maps algorithm names to their column prefixes and suffixes,
enabling programmatic read/write of the wide-format CSV files.

Directory structure:
    result/
    ├── mixalgos/          # L0/L2 scenario
    │   ├── random_s1.csv  (UP=10)
    │   ├── random_s2.csv  (UP=20)
    │   ├── random_m1.csv  (UP=50)
    │   ├── random_m2.csv  (UP=100)
    │   ├── random_l1.csv  (UP=500)
    │   ├── random_l2.csv  (UP=1000)
    │   ├── huawei.csv
    │   └── microsoft.csv
    └── improvevmpack/     # L0/L1/L2 scenario
        ├── random_s1.csv
        ├── random_s2.csv
        ├── random_m1.csv
        ├── random_m2.csv
        ├── random_l1.csv
        ├── random_l2.csv
        ├── huawei.csv
        └── microsoft.csv

Each CSV row = one instance, with columns:
    seq, instance (JSON string), lb, total_cpu, total_mem,
    then algorithm-specific columns (prefix_suffix naming).
"""

# ── Scale ↔ UP mapping ────────────────────────────────────────────────
SCALE_UP = {
    'S1': 10, 'S2': 20, 'M1': 50, 'M2': 100, 'L1': 500, 'L2': 1000,
}
SCALES = ['S1', 'S2', 'M1', 'M2', 'L1', 'L2']

# CSV filename (without .csv) for each scale/trace
SCALE_FILE_NAMES = {
    'S1': 'random_s1', 'S2': 'random_s2',
    'M1': 'random_m1', 'M2': 'random_m2',
    'L1': 'random_l1', 'L2': 'random_l2',
}

# ── Column registry ───────────────────────────────────────────────────

COLUMN_REGISTRY = {
    'mixalgos': {
        'fun_case': 'mixalgos',
        'algorithms': [
            {'name': 'NoMixPack',     'prefix': 'NoMixPack',     'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'MixVM301',      'prefix': 'MixVM301',      'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'MixVM201',      'prefix': 'MixVM201',      'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'MixVM201Pro',   'prefix': 'MixVM201Pro',   'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'MixPack',       'prefix': 'MixPack',       'cols': ['npms', 'time'], 'type': 'heuristic'},
            # Priority-only variant of MixVM201Pro for ablation study
            {'name': 'MixVM201Priority', 'prefix': 'MixVM201Priority', 'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'BFD',           'prefix': 'BFD',           'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'FFD',           'prefix': 'FFD',           'cols': ['npms', 'time'], 'type': 'heuristic'},
    {'name': 'VanillaMIP(mix)', 'prefix': 'MIP_Mix',
             'cols': ['npms', 'time', 'gap', 'status', 'bestbound',
                      '1s_npms', '1s_gap', '1s_status',
                      '5s_npms', '5s_gap', '5s_status',
                      '10s_npms', '10s_gap', '10s_status'],
             'type': 'solver'},
    # SafeMix heuristic wrapper that selects the best of MixVM301, MixVM201Pro and MixPack
    {'name': 'SafeMix',        'prefix': 'SafeMix',      'cols': ['npms', 'time'], 'type': 'heuristic'},
        ],
        'trace_algorithms': [
            {'name': 'FFD',              'prefix': 'FFD',              'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'BFD',              'prefix': 'BFD',              'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'VMPack_MixVM301',  'prefix': 'VMPack_MixVM301',  'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'VMPack_MixVM201Pro', 'prefix': 'VMPack_MixVM201Pro', 'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'VMPack_MixPack',   'prefix': 'VMPack_MixPack',   'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'VMPack_SafeMix',   'prefix': 'VMPack_SafeMix',   'cols': ['npms', 'time'], 'type': 'heuristic'},
        ],
    },
    'improvevmpack': {
        'fun_case': 'improvevmpack',
        'algorithms': [
    {'name': 'VMPack_NoMixPack',   'prefix': 'VMPack_NoMixPack',   'cols': ['npms', 'time'], 'type': 'heuristic'},
    {'name': 'VMPack_MixVM301',    'prefix': 'VMPack_MixVM301',    'cols': ['npms', 'time'], 'type': 'heuristic'},
    {'name': 'VMPack_MixVM201',    'prefix': 'VMPack_MixVM201',    'cols': ['npms', 'time'], 'type': 'heuristic'},
    {'name': 'VMPack_MixVM201Pro', 'prefix': 'VMPack_MixVM201Pro', 'cols': ['npms', 'time'], 'type': 'heuristic'},
    {'name': 'VMPack_MixPack',     'prefix': 'VMPack_MixPack',     'cols': ['npms', 'time'], 'type': 'heuristic'},
    # SafeMix full-pipeline heuristic wrapper
    {'name': 'VMPack_SafeMix',     'prefix': 'VMPack_SafeMix',     'cols': ['npms', 'time'], 'type': 'heuristic'},
    # Priority-only variant of MixVM201Pro in the full pipeline
    {'name': 'VMPack_MixVM201Priority', 'prefix': 'VMPack_MixVM201Priority', 'cols': ['npms', 'time'], 'type': 'heuristic'},
    {'name': 'BFD',               'prefix': 'BFD',               'cols': ['npms', 'time'], 'type': 'heuristic'},
    {'name': 'FFD',               'prefix': 'FFD',               'cols': ['npms', 'time'], 'type': 'heuristic'},
    # Column‑generation pattern benchmark (CG‑Benchmark) and its Mix initialization
    {'name': 'CG-Benchmark',               'prefix': 'CG',                'cols': ['npms', 'time', 'gap', 'status', 'n_cols'], 'type': 'solver'},
    {'name': 'CG-Benchmark(mix)',          'prefix': 'CG_Mix',            'cols': ['npms', 'time', 'gap', 'status', 'n_cols'], 'type': 'solver'},
    {'name': 'VanillaMIP',        'prefix': 'MIP',
     'cols': ['npms', 'time', 'gap', 'status', 'bestbound',
              '1s_npms', '1s_gap', '1s_status',
              '5s_npms', '5s_gap', '5s_status',
              '10s_npms', '10s_gap', '10s_status'],
     'type': 'solver'},
    {'name': 'VanillaMIP(mix)',   'prefix': 'MIP_Mix',
     'cols': ['npms', 'time', 'gap', 'status', 'bestbound',
              '1s_npms', '1s_gap', '1s_status',
              '5s_npms', '5s_gap', '5s_status',
              '10s_npms', '10s_gap', '10s_status'],
     'type': 'solver'},
        ],
        'trace_algorithms': [
            {'name': 'FFD',              'prefix': 'FFD',              'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'BFD',              'prefix': 'BFD',              'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'VMPack_MixVM301',  'prefix': 'VMPack_MixVM301',  'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'VMPack_MixVM201Pro', 'prefix': 'VMPack_MixVM201Pro', 'cols': ['npms', 'time'], 'type': 'heuristic'},
            {'name': 'VMPack_MixPack',   'prefix': 'VMPack_MixPack',   'cols': ['npms', 'time'], 'type': 'heuristic'},
        ],
    },
}


def get_column_names(fun_case, include_solvers=True):
    """Get the full list of CSV column names for a given fun_case.

    Args:
        fun_case: 'mixalgos' or 'improvevmpack'
        include_solvers: If True, include solver columns (for synthetic CSVs).
                         If False, only heuristic columns (for trace CSVs).

    Returns:
        List of column name strings.
    """
    registry = COLUMN_REGISTRY[fun_case]
    algos = registry['algorithms'] if include_solvers else registry['trace_algorithms']

    cols = ['seq', 'instance', 'lb', 'total_cpu', 'total_mem']
    for algo in algos:
        for suffix in algo['cols']:
            cols.append(f"{algo['prefix']}_{suffix}")
    return cols


def get_algo_cols(fun_case, algo_name, include_solvers=True):
    """Get the column names for a specific algorithm.

    Args:
        fun_case: 'mixalgos' or 'improvevmpack'
        algo_name: Algorithm name as in the registry (e.g. 'MixVM201Pro')
        include_solvers: If True, search in algorithms list; else search trace_algorithms.

    Returns:
        List of column name strings, or empty list if not found.
    """
    registry = COLUMN_REGISTRY[fun_case]
    algos = registry['algorithms'] if include_solvers else registry['trace_algorithms']
    for algo in algos:
        if algo['name'] == algo_name:
            return [f"{algo['prefix']}_{suffix}" for suffix in algo['cols']]
    return []


def get_algo_prefix(fun_case, algo_name, include_solvers=True):
    """Get the column prefix for a specific algorithm.

    Args:
        fun_case: 'mixalgos' or 'improvevmpack'
        algo_name: Algorithm name as in the registry.

    Returns:
        Prefix string, or None if not found.
    """
    registry = COLUMN_REGISTRY[fun_case]
    algos = registry['algorithms'] if include_solvers else registry['trace_algorithms']
    for algo in algos:
        if algo['name'] == algo_name:
            return algo['prefix']
    return None
