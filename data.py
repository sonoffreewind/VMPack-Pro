import numpy as np
import os
from enum import Flag, auto
from basic import *
import json

# ── UP → scale subdirectory mapping ──────────────────────────────
UP_TO_SUBDIR = {
    10: 'random_s1', 20: 'random_s2', 50: 'random_m1',
    100: 'random_m2', 500: 'random_l1', 1000: 'random_l2',
}

def _scale_subdir(up):
    """Map UP value to scale subdirectory name (e.g. 10 → 'random_s1')."""
    subdir = UP_TO_SUBDIR.get(up)
    if subdir is None:
        raise ValueError(f"Unknown UP value: {up}. Valid: {sorted(UP_TO_SUBDIR.keys())}")
    return subdir

class DataTypes(Flag):
    NONE = 0
    RANDOM = auto()
    WEIBULL = auto()
    HUAWEI = auto()
    # Default mode is to generate data using random mode
    DEFAULT = RANDOM

def TestExample(L):
    L = [np.copy(L[0]), np.copy(L[1]), np.copy(L[2])]
    T,C,ZERO = gv.T, gv.C,gv.ZERO
    vm0,vm1,vm2 = L[0],L[1],L[2]
    # calcute the parameters
    C0,C1,C2 = CpuSize(vm0),CpuSize(vm1),CpuSize(vm2)
    n0,n1,n2 = np.floor(C0/2**T), np.floor(C1/2**(T-1)), np.floor(C2/2**(T-1))

    # enlarge the vms
    for t in range(T-3,-1,-1):
        # enlarge (t,1) type vm
        while vm1[t] > 0:
            if n1 >= min(n0,n2): return False
            if np.floor((C0-2**T*n1)/(3*2**(T-1))) >= np.floor((C2-2**(T-1)*n1)/2**(T-1)): return False
            # enlarge a vm of type (t,1)
            isEnlarge, _ = EnlargeVM(t,L)
            if not isEnlarge: break
            # update vm0,vm1,vm2
            C0,C1,C2 = CpuSize(vm0),CpuSize(vm1),CpuSize(vm2)
            n0,n1,n2 = np.floor(C0/2**T), np.floor(C1/2**(T-1)), np.floor(C2/2**(T-1))
    return True

def GenRandomExamples(n:int,funcase:str):
    Up = gv.UP
    Ls = []
    it = 0
    if funcase == 'mixalgos':
        while it < n:
            # Generate random VMs with improved distribution
            vm0 = np.random.randint(low = 1, high = Up+1, size = gv.T)  # Ensure non-zero values
            vm1 = np.zeros(gv.T)
            vm2 = np.random.randint(low = 1, high = Up+1, size = gv.T)
            # Check capacity constraint
            if CpuSize(vm0) >= 3*CpuSize(vm2): continue
            Ls.append([vm0, vm1, vm2])
            it += 1    
    elif funcase == 'improvevmpack':
        while it < n:
            # Generate random VMs with improved distribution
            vm0 = np.random.randint(low = 1, high = Up+1, size = gv.T)  # Ensure non-zero values
            vm1 = np.random.randint(low = 1, high = Up+1, size = gv.T)
            vm2 = np.random.randint(low = 1, high = Up+1, size = gv.T)
            # Check capacity constraint
            if TestExample([vm0, vm1, vm2]):
                Ls.append([vm0, vm1, vm2])
                it += 1    
    else:
        raise ValueError("Invalid function case. Use 'mixalgos' or 'improvevmpack'.")
        
    return Ls

def GenWeibullExamples(n:int, fun_case:str):
    pass

def GenHuaweiExamples(n:int, fun_case:str):
    pass

def GenExamples(n:int, datatype: DataTypes, fun_case:str):
    """Generate test data
    Args:
        n: Number of data items to generate
        datatype: Type of the data
        fun_case: Purpose of the data. 
            - 'mixalgos' means comparing mixed heuristics for types 1 and 3 VMs
            - 'improvevmpack' means testing two vmpack examples
    Returns:
        The generated data
    """

    if datatype == DataTypes.RANDOM:
        return GenRandomExamples(n,fun_case)
    elif datatype == DataTypes.WEIBULL:
        return GenWeibullExamples(n,fun_case)
    elif datatype == DataTypes.HUAWEI:
        return GenHuaweiExamples(n,fun_case)
    else:
        return None

# read csv data: R = [L0--bins of heuristic, L2--bins of opt], VM0--vm lists of type 1:1, VM2--vm lists of type 1:2
def LoadExamples(file_path):
    try:
        # Directly use the provided file_path as the file path.
        filename = os.path.normpath(file_path)
        # check if the filename is a directory
        if not filename.endswith('.json'):
            raise ValueError("The file is not a json file.")
        with open(filename, 'r', encoding='utf-8') as f:
            Ls = json.load(f)
            return [[np.array(L[0]),np.array(L[1]),np.array(L[2])] for L in Ls]
    except ValueError as e:
        print(e)
        return []
    except Exception as e:
        print(f"Error loading examples: {e}")
        return []

def SaveExamples(path, Ls, datatype: DataTypes, fun_case:str):
    subdir = _scale_subdir(gv.UP)
    save_path = os.path.join(path, subdir)
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, GetFileName(len(Ls), datatype, fun_case))
    SaveJson(filename, Ls)

def SaveJson(filename, results):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NumpyArrayEncoder, separators=(',', ':'))

def GetFileName(n, datatype: DataTypes, fun_case:str):
    # get the file name by the parameters
    TYPE_STR = {DataTypes.RANDOM:'r', DataTypes.WEIBULL:'w', DataTypes.HUAWEI:'h'}
    return f'{n}_{TYPE_STR[datatype]}_{gv.T}_{gv.UP}_{fun_case}.json'

def GetFilePath(data_dir, n, datatype: DataTypes, fun_case:str):
    """
    Get the full path to a JSON instance file, including scale subdirectory.

    For RANDOM data, the file is stored under data_dir/{scale_subdir}/{filename}.
    For other data types, it's stored directly under data_dir.
    """
    filename = GetFileName(n, datatype, fun_case)
    if datatype == DataTypes.RANDOM:
        subdir = _scale_subdir(gv.UP)
        return os.path.join(data_dir, subdir, filename)
    return os.path.join(data_dir, filename)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle NumPy scalar types, such as np.int64, np.float32, etc.
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        # Recursively process nested structures
        elif isinstance(obj, dict):
            return {key: self.default(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.default(item) for item in obj]
        return super().default(obj)
