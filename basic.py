import numpy as np
import globalvars as gv

def CpuSize(Q:np.array):
    Q = np.array(Q)
    # if there are negative values in Q or Q is not of length T, raise error
    if np.any(Q < 0) or Q.size != gv.T: raise ValueError("Q must be a vector of length T")
    return sum(Q*gv.CPU)

def FindVMsWithBound(c:int, *args):
    # raise error if c<0
    T,C,CPU = gv.T, gv.C, gv.CPU
    if c < 0:
        raise ValueError("c must be non-negative")
    if len(args) == 2 and isinstance(args[0], int) and (isinstance(args[1], np.ndarray) or isinstance(args[1], list)):
        s,L = args
        # raise error if s not in [0,1,2]
        if s not in [0,1,2]: raise ValueError("s must be in [0,1,2]")
        vm = L[int(s)]
    else:
        vm = args[0]
    c0 = c
    Q = []
    for i in range(T-1,-1,-1):
        y = min(int(np.floor(c/CPU[i])), vm[i])   # y = min( X>>(n-1-i), vm[i])
        Q.append(y)
        c -= y*CPU[i]
        if c < 0 or y < 0: raise ValueError("c and y must be non-negative")
    Q.reverse()
    return np.array(Q), int(c0-c)

def FillOnePm(cpu0, cpu2, L):
    C = gv.C
    if cpu0 < 0:
        Q2, cpu_q_2 = FindVMsWithBound(cpu2, 2, L)
        mem_q_2 = 4*cpu_q_2
        cpu0 = min(C-cpu_q_2, 2*C - mem_q_2)
        Q0,_ = FindVMsWithBound(cpu0, 0, L)
    elif cpu2 < 0:
        Q0, cpu_q_0 = FindVMsWithBound(cpu0, 0, L)
        mem_q_0 = cpu_q_0
        cpu2 = min(C-cpu_q_0, (2*C - mem_q_0)//4)
        Q2,_ = FindVMsWithBound(cpu2, 2, L)
    else:
        Q0,_ = FindVMsWithBound(cpu0, 0, L)
        Q2,_ = FindVMsWithBound(cpu2, 2, L)
    if any(Q0 < 0) or any(Q2 < 0):
        raise ValueError("Q must be an positive integer array")
    
    cpu = CpuSize(Q0) + CpuSize(Q2)
    mem = CpuSize(Q0) + 4 * CpuSize(Q2)
    if cpu > gv.C or mem > gv.M:
        raise AssertionError(f"Infeasible PM: cpu={cpu}, mem={mem}")
    
    return Q0,Q2

# find vm set V in the L[s] such that CpuSize(V) == c
def CollectVMs(c,s,L:np.array):
    Q, cpu_r = FindVMsWithBound(c, s, L)
    return Q if cpu_r == c else None

def EnlargeVM(t, L):
    T = gv.T
    isEnlarge = False
    records = []
    while True:
        c_i = 2**t
        if c_i >= 2**(T-2): return isEnlarge,records
        Q0 = CollectVMs(2*c_i, 0, L)
        Q2 = CollectVMs(c_i, 2, L)
        if Q0 is None or Q2 is None: return isEnlarge,records
        if gv.RECORD_ENLARGE: 
            records.append([-Q0, np.zeros(T) ,-Q2])
            records[-1][1][t] = -1; records[-1][1][t+2] = 1
        L[0] -= Q0
        L[1][t] -= 1; t = t + 2; L[1][t] += 1
        L[2] -= Q2
        isEnlarge = True
    return isEnlarge, records

def ValidatePMs(PMs, vm_demands):
    """
    Validate that PMs are feasible and cover the original VM demands.

    Checks:
        1. No negative VM counts in any PM.
        2. Each PM satisfies CPU and memory capacity constraints.
        3. The union of all PMs covers the original vm_demands.

    Args:
        PMs: List of PMs, each PM is [Q0, Q1, Q2].
        vm_demands: Original VM demands as (S, T) array.

    Raises:
        RuntimeError: If any validation check fails.
    """
    S, T, C, M, CPU = gv.S, gv.T, gv.C, gv.M, gv.CPU
    cover = np.zeros((S, T), dtype=int)

    for k, pm in enumerate(PMs):
        cpu = 0
        mem = 0
        for s in range(S):
            for t in range(T):
                cnt = int(round(pm[s][t]))
                if cnt < 0:
                    raise RuntimeError(f"Negative count in PM {k}, type ({s},{t}).")
                cover[s][t] += cnt
                cpu += CPU[t] * cnt
                mem += (2**s) * CPU[t] * cnt
        if cpu > C or mem > M:
            raise RuntimeError(f"Infeasible PM {k}: cpu={cpu}, mem={mem}.")

    if not np.array_equal(cover, vm_demands):
        raise RuntimeError(
            f"Expanded PMs do not exactly match original demands.\n"
            f"Demand:\n{vm_demands}\nCover:\n{cover}\n"
            f"Diff:\n{cover - vm_demands}"
        )