from sortedcontainers import SortedList
import numpy as np
from basic import *

def NoMixPack(L):
    L = np.copy(L)
    C,ZERO = gv.C, gv.ZERO
    vm0,vm2 = L[0],L[2]
    C0,C2 = CpuSize(vm0),CpuSize(vm2)
    npms,PMs = 0,[]
    while C0 > 0:
        Q0,cpu_q_0 = FindVMsWithBound(C, 0, L)
        vm0 -= Q0
        if gv.RETURN_PMS:
            PMs.append([Q0,ZERO,ZERO])
        else:
            npms += 1
        C0 -= cpu_q_0
    while C2 > 0:
        Q2,cpu_q_2 = FindVMsWithBound(C//2, 2, L)
        vm2 -= Q2
        if gv.RETURN_PMS:
            PMs.append([ZERO,ZERO,Q2]) 
        else:
            npms += 1
        C2 -= cpu_q_2
    return PMs if gv.RETURN_PMS else npms
 
def MixVM301(L):
    L = np.copy(L)
    T,C,ZERO = gv.T, gv.C, gv.ZERO
    vm0,vm2 = L[0],L[2]
    C0,C2 = CpuSize(vm0),CpuSize(vm2)
    npms,PMs = 0,[]
    while C0 >= 3*C//4 and C2 >= C//4:
        Q0,Q2 = FillOnePm(3*C//4,C//4,L)
        vm0 -= Q0
        vm2 -= Q2
        if gv.RETURN_PMS:
            PMs.append([Q0,ZERO,Q2]) 
        else:
            npms += 1
        C0 -= CpuSize(Q0)
        C2 -= CpuSize(Q2)
    # pack vms by NF strategy
    PM = [np.zeros(T),np.zeros(T),np.zeros(T)]; cpu_k = 0; mem_k = 0
    for s in [2,0]:
        for t in range(T-1,-1,-1):
            c_i, m_i = 2**t, 2**(t+s)
            while L[s][t] > 0:
                # if the remainning cpu or mem size of the last PM is not enough, then crate a new PM
                if c_i + cpu_k > C or m_i + mem_k > 2*C: 
                    if gv.RETURN_PMS:
                        PMs.append(PM) 
                    else:
                        npms += 1
                    PM = [np.zeros(T),np.zeros(T),np.zeros(T)]; cpu_k = 0; mem_k = 0
                PM[s][t] += 1; cpu_k += c_i; mem_k += m_i
                L[s][t] -= 1
    if cpu_k > 0 or mem_k > 0:
        if gv.RETURN_PMS:
            PMs.append(PM) 
        else:
            npms += 1
    return PMs if gv.RETURN_PMS else npms

def MixVM201(L):
    L = np.copy(L)
    C,ZERO = gv.C,gv.ZERO
    vm0,vm2 = L[0],L[2]
    C0,C2 = CpuSize(vm0),CpuSize(vm2)
    npms,PMs = 0,[]
    while C0 + C2 > 0:
        if C0 <= 2*C2:
            R,cpu_r = FindVMsWithBound(2*C/3, 0, L)
            cpu2 = min(C-cpu_r, (2*C - cpu_r)//4)
            Q0,Q2 = FillOnePm(-1, cpu2, L)            
        else:
            R,cpu_r = FindVMsWithBound(C/3, 2, L)
            cpu0 = min(C-cpu_r, 2*C - 4*cpu_r)
            Q0,Q2 = FillOnePm(cpu0, -1, L)
        vm0 -= Q0
        vm2 -= Q2
        if gv.RETURN_PMS:
            PMs.append([Q0,ZERO,Q2]) 
        else:
            npms += 1
        C0 -= CpuSize(Q0)
        C2 -= CpuSize(Q2)
    return PMs if gv.RETURN_PMS else npms

def MixVM201Pro(L):
    L = np.copy(L)
    C,ZERO = gv.C,gv.ZERO
    vm0,vm2 = L[0],L[2]
    C0,C2 = CpuSize(vm0),CpuSize(vm2)
    npms,PMs = 0,[]
    while C0 + C2 > 0:
        if C0 <= 2*C/3 and C2 <= C/3:
            Q0,Q2 = np.copy(vm0),np.copy(vm2)
        elif C0 <= 2*C2:
            R,cpu_r = FindVMsWithBound(C2-C/3, 2, L)
            cpu0 = 2*C - 4*C2 + 4*cpu_r
            Q0,Q2 = FillOnePm(cpu0, -1, L)
        else:
            R,cpu_r = FindVMsWithBound((3*C0-2*C)/3, 0, L)
            cpu2 = C - C0 + cpu_r
            Q0,Q2 = FillOnePm(-1, cpu2, L)
        vm0 -= Q0
        vm2 -= Q2
        if gv.RETURN_PMS:
            PMs.append([Q0,ZERO,Q2]) 
        else:
            npms += 1
        C0 -= CpuSize(Q0)
        C2 -= CpuSize(Q2)
    return PMs if gv.RETURN_PMS else npms

def MixPack(L):
    L = np.copy(L)
    C,ZERO = gv.C,gv.ZERO
    vm0,vm2 = L[0],L[2]
    C0,C2 = CpuSize(vm0),CpuSize(vm2)
    npms,PMs = 0,[]
    while C0 + C2 > 0:
        if C0 <= 2*C/3 and C2 <= C/3:
            Q0,Q2 = np.copy(vm0),np.copy(vm2)
        elif C0 <= 2*C2:
            Q0,Q2 = FillOnePm(-1, C//2, L)
        else:
            R,cpu_r = FindVMsWithBound((3*C0-2*C)/3, 0, L)
            cpu2 = C - C0 + cpu_r
            Q0,Q2 = FillOnePm(-1, cpu2, L)
        vm0 -= Q0
        vm2 -= Q2
        if gv.RETURN_PMS:
            PMs.append([Q0,ZERO,Q2]) 
        else:
            npms += 1
        C0 -= CpuSize(Q0)
        C2 -= CpuSize(Q2)
    return PMs if gv.RETURN_PMS else npms


def SafeMix(L):
    """
    SafeMix wrapper: select the best packing result among MixVM301, MixVM201Pro and MixPack.

    This heuristic runs the three core mixed-packing strategies (301, 201Pro and MixPack) on a
    given two-class VM demand instance and returns the solution with the lowest PM count.
    If multiple strategies yield the same PM count, the first in the order (MixVM301, MixVM201Pro,
    MixPack) is chosen. The function preserves the return type of the chosen heuristic (either
    a PM list or a scalar PM count) to remain drop-in compatible with other heuristics.
    """
    # Copy the input to avoid side effects
    L_copy = np.copy(L)
    strategies = [MixVM301, MixVM201Pro, MixPack]
    best_result = None
    best_npms = None
    for fn in strategies:
        # Work on a fresh copy of L for each strategy
        result = fn(np.copy(L_copy))
        # Determine the number of PMs used
        if isinstance(result, tuple):
            npms_val = result[0]
        elif isinstance(result, list):
            npms_val = len(result)
        else:
            npms_val = result
        # Update best result
        if best_npms is None or npms_val < best_npms:
            best_npms = npms_val
            best_result = result
    return best_result


def VMPack_SafeMix(L):
    """
    Full VMPack pipeline using SafeMix at the bottleneck stage.

    This function reuses the general VMPackPro framework but substitutes the bottleneck-stage
    heuristic with SafeMix, thereby inheriting the 10/9 approximation guarantee of MixVM301
    while benefiting from the empirical gains of MixVM201Pro and MixPack. See SafeMix for
    details on the selection logic.
    """
    return VMPackPro(L, SafeMix)

def VMPack(L):
    L = np.copy(L)
    T,C,ZERO = gv.T, gv.C,gv.ZERO
    vm0,vm1,vm2 = L[0],L[1],L[2]
    # calcute the parameters
    C0,C1,C2 = CpuSize(vm0),CpuSize(vm1),CpuSize(vm2)
    n0,n1,n2 = np.floor(C0/2**T), np.floor(C1/2**(T-1)), np.floor(C2/2**(T-1))

    # enlarge the vms
    R = []
    for t in range(T-3,-1,-1):
        # enlarge (t,1) type vm
        isbreak = False
        while vm1[t] > 0:
            if n1 >= min(n0,n2): 
                isbreak = True
                break
            # enlarge a vm of type (t,1)
            isEnlarge, records = EnlargeVM(t,L)
            if not isEnlarge: break
            if gv.RECORD_ENLARGE: R.extend(records)
            # update vm0,vm1,vm2
            C0,C1,C2 = CpuSize(vm0),CpuSize(vm1),CpuSize(vm2)
            n0,n1,n2 = np.floor(C0/2**T), np.floor(C1/2**(T-1)), np.floor(C2/2**(T-1))
        if isbreak: break
    npms,PMs = 0,[]
    # pack vms by 211 strategy
    while True:
        Q0 = CollectVMs(C//2, 0, L)
        Q1 = CollectVMs(C//4, 1, L)
        Q2 = CollectVMs(C//4, 2, L)
        if Q0 is None or Q1 is None or Q2 is None: break
        vm0 -= Q0; vm1 -= Q1; vm2 -= Q2
        if gv.RETURN_PMS:
            PMs.append([Q0,Q1,Q2]) 
        else:
            npms += 1
    
    # pack vms by 301 strategy
    while True:
        Q0 = CollectVMs(3*C//4, 0, L)
        Q2 = CollectVMs(C//4, 2, L)
        if Q0 is None or Q2 is None: break
        vm0 -= Q0; vm2 -= Q2
        if gv.RETURN_PMS:
            PMs.append([Q0,ZERO,Q2]) 
        else:
            npms += 1
    
    # pack vms by NF strategy
    PM = [np.zeros(T),np.zeros(T),np.zeros(T)]; cpu_k = 0; mem_k = 0
    for s in range(2,-1,-1):
        for t in range(T-1,-1,-1):
            # the cpu and mem of a type L[s][t]
            c_i, m_i = 2**t, 2**(t+s)
            while L[s][t] > 0:
                # if the remainning cpu or mem size of the last PM is not enough, then crate a new PM
                if c_i + cpu_k > C or m_i + mem_k > 2*C: 
                    if gv.RETURN_PMS:
                        PMs.append(PM) 
                    else:
                        npms += 1
                    PM = [np.zeros(T),np.zeros(T),np.zeros(T)]; cpu_k = 0; mem_k = 0
                PM[s][t] += 1; cpu_k += c_i; mem_k += m_i
                L[s][t] -= 1
    if cpu_k > 0 or mem_k > 0:
        if gv.RETURN_PMS:
            PMs.append(PM) 
        else:
            npms += 1
    return (PMs, R) if gv.RETURN_PMS else (npms, R)


def VMPackPro(L, mixalgo:callable):
    L = np.copy(L)
    T,C,ZERO = gv.T, gv.C,gv.ZERO
    vm0,vm1,vm2 = L[0],L[1],L[2]
    # calcute the parameters
    C0,C1,C2 = CpuSize(vm0),CpuSize(vm1),CpuSize(vm2)
    n0,n1,n2 = np.floor(C0/2**T), np.floor(C1/2**(T-1)), np.floor(C2/2**(T-1))

    # enlarge the vms
    R = []
    for t in range(T-3,-1,-1):
        # enlarge (t,1) type vm
        isbreak = False
        while vm1[t] > 0:
            if n1 >= min(n0,n2): 
                isbreak = True
                break
            # enlarge a vm of type (t,1)
            isEnlarge, records = EnlargeVM(t,L)
            if not isEnlarge: break
            if gv.RECORD_ENLARGE: R.extend(records)
            # update vm0,vm1,vm2
            C0,C1,C2 = CpuSize(vm0),CpuSize(vm1),CpuSize(vm2)
            n0,n1,n2 = np.floor(C0/2**T), np.floor(C1/2**(T-1)), np.floor(C2/2**(T-1))
        if isbreak: break
    npms,PMs = 0,[]
    # pack vms by 211 strategy
    x211 = 0
    while True:
        Q0 = CollectVMs(C//2, 0, L)
        Q1 = CollectVMs(C//4, 1, L)
        Q2 = CollectVMs(C//4, 2, L)
        if Q0 is None or Q1 is None or Q2 is None: break
        vm0 -= Q0; vm1 -= Q1; vm2 -= Q2
        x211 += 1
        if gv.RETURN_PMS:
            PMs.append([Q0,Q1,Q2]) 
        else:
            npms += 1
    
    # pack vms by 301 and Mixing VMs strategy
    if x211 == int(n1) and n1 < min(n0,n2) and np.floor((C0-2**T*n1)/(3*2**(T-1))) < np.floor((C2-2**(T-1)*n1)/2**(T-1)):
        if gv.RETURN_PMS:
            PMs.extend(mixalgo(L)) 
        else:
            npms += mixalgo(L)
        C1 = CpuSize(vm1)
        if C1 > 0:
            if gv.RETURN_PMS:
                # cpu0,cpu2 = CpuSize(PMs[-1][0]),CpuSize(PMs[-1][2])
                #     if cpu0 + C1 + cpu2 > C or cpu0 + 2*C1 + 4*cpu2 > 2*C:
                #         PMs.append([ZERO,L[1],ZERO])
                #     else:
                #         PMs[-1][1] = L[1]
                PMs.append([ZERO,L[1],ZERO]) 
            else:
                npms += 1
        return (PMs, R) if gv.RETURN_PMS else (npms, R)
        
    # pack vms by 301 strategy
    while True:
        Q0 = CollectVMs(3*C//4, 0, L)
        Q2 = CollectVMs(C//4, 2, L)
        if Q0 is None or Q2 is None: break
        vm0 -= Q0; vm2 -= Q2
        if gv.RETURN_PMS:
            PMs.append([Q0,ZERO,Q2]) 
        else:
            npms += 1
    
    # pack vms by NF strategy
    PM = [np.zeros(T),np.zeros(T),np.zeros(T)]; cpu_k = 0; mem_k = 0
    for s in range(2,-1,-1):
        for t in range(T-1,-1,-1):
            # the cpu and mem of a type L[s][t]
            c_i, m_i = 2**t, 2**(t+s)
            while L[s][t] > 0:
                # if the remainning cpu or mem size of the last PM is not enough, then crate a new PM
                if c_i + cpu_k > C or m_i + mem_k > 2*C: 
                    if gv.RETURN_PMS:
                        PMs.append(PM) 
                    else:
                        npms += 1
                    PM = [np.zeros(T),np.zeros(T),np.zeros(T)]; cpu_k = 0; mem_k = 0
                PM[s][t] += 1; cpu_k += c_i; mem_k += m_i
                L[s][t] -= 1
    if cpu_k > 0 or mem_k > 0:
        if gv.RETURN_PMS:
            PMs.append(PM) 
        else:
            npms += 1
    return (PMs, R) if gv.RETURN_PMS else (npms, R)

def VMPack_MixVM201Pro(L):
    return VMPackPro(L, MixVM201Pro)

def VMPack_MixVM201(L):
    return VMPackPro(L, MixVM201)

def VMPack_NoMixPack(L):
    return VMPackPro(L, NoMixPack)

def VMPack_MixPack(L):
    return VMPackPro(L, MixPack)

def VMPack_MixVM301(L):
    return VMPackPro(L, MixVM301)

def ExpandPMsToOriginal(pms, R, T):
    """
    Expand aggregated PMs back to original VM types.
    
    Args:
        pms: PMs returned by VMPackPro(), each PM is [Q0, Q1, Q2]
             where Q1 is the aggregated type-1 VM distribution
        R: Enlarge records list returned by VMPackPro() (in chronological order)
        T: Number of VM size types
    
    Returns:
        expanded_pms: Expanded PMs with original VM types in [Q0, Q1, Q2]
    
    Expansion principle:
        Each record in R has format [-Q0, vm1_changes, -Q2]:
        - vm1_changes[t] = -1 means reducing 1 original size-t type-1 VM
        - vm1_changes[t+2] = 1 means adding 1 aggregated size-(t+2) type-1 VM
        - Q0 and Q2 record other VM types consumed by this enlarge operation
        
        Traverse R in reverse order, splitting each aggregated VM back to original VMs.
        Since VMs of the same type are unordered, it doesn't matter which PM or VM to split.
    """
    # Deep copy PMs to avoid modifying original data
    # Use explicit array copies instead of copy.deepcopy for performance
    expanded_pms = [[Q0.copy(), Q1.copy(), Q2.copy()] for Q0, Q1, Q2 in pms]
    
    # Traverse R in reverse order, split aggregated VMs back to original VMs
    for record in reversed(R):
        Q0_removed, vm1_changes, Q2_removed = record
        
        # Find the original VM size t (vm1_changes[t] = -1)
        # Find the aggregated VM size t_new (vm1_changes[t_new] = 1)
        t_indices = np.where(vm1_changes == -1)[0]
        t_new_indices = np.where(vm1_changes == 1)[0]
        
        if len(t_indices) != 1 or len(t_new_indices) != 1:
            raise RuntimeError(f"Invalid enlarge record: {record}")
        
        t = int(t_indices[0])
        t_new = int(t_new_indices[0])
        
        # Find a PM that contains a (t_new, 1) VM
        found = False
        for pm in expanded_pms:
            Q0, Q1, Q2 = pm
            if Q1[t_new] > 0:
                # Split this (t_new, 1) VM back to original VM
                Q1[t_new] -= 1  # Remove aggregated VM
                Q1[t] += 1      # Add original VM
                
                # Add back the consumed Q0 and Q2 VMs
                # Q0_removed and Q2_removed are negative (stored as -Q0 and -Q2), so negate them
                Q0 = Q0 + (-Q0_removed)
                Q2 = Q2 + (-Q2_removed)
                
                pm[0] = Q0
                pm[1] = Q1
                pm[2] = Q2
                found = True
                break
        
        if not found:
            raise RuntimeError(
                f"Cannot expand enlarged VM: no PM contains type-1 VM of size {t_new}."
            )
    
    return expanded_pms

def BFD(L):
    """
    Best Fit Decreasing (BFD) heuristic for VM packing by the capacity for the L_s.

    """
    L = np.copy(L)
    T,C,S,ZERO = gv.T, gv.C, gv.S, gv.ZERO
    PMs = []
    C_R, M_R = [],[]
    # key = lambda idx:min(C_R[idx], M_R[idx]
    bins_index = [SortedList(), SortedList(), SortedList()]
    for t in reversed(range(T)):
        for s in reversed(range(S)):
            for _ in range(int(L[s][t])):
                best_idx = -1
                pos = bins_index[s].bisect_left((2**t, -float('inf')))
                if pos < len(bins_index[s]):
                    _,best_idx = bins_index[s][pos]
                    c,m = C_R[best_idx], M_R[best_idx]
                    bins_index[0].remove((min(c,m),best_idx))
                    bins_index[1].remove((min(c,m/2),best_idx))
                    bins_index[2].remove((min(c,m/4),best_idx))
                else:
                    PMs.append([np.zeros(T),np.zeros(T),np.zeros(T)])
                    C_R.append(C)
                    M_R.append(2*C)
                    best_idx = len(C_R)-1
                # pack
                PMs[best_idx][s][t] += 1
                C_R[best_idx] -= 2**t
                M_R[best_idx] -= 2**(s+t) 
                # add to sortedlist
                c, m = C_R[best_idx], M_R[best_idx]
                if c > 0 and m > 0:
                    bins_index[0].add((min(c,m),best_idx))
                    bins_index[1].add((min(c,m/2),best_idx))
                    bins_index[2].add((min(c,m/4),best_idx))
    
    return (PMs, []) if gv.RETURN_PMS else (len(PMs), [])


class _FFD_SegTree:
    """Segment tree that accelerates the first-fit search in FFD.

    Each node stores the maximum of max_keys[s] across all PMs in its interval.
    key_s = min(cpu_rem, mem_rem // 2^s) represents the remaining capacity
    for VM class L_s.
    """

    def __init__(self, l, r, C, M):
        self.l, self.r = l, r
        self.C, self.M = C, M
        self.max_keys = [-1, -1, -1]  # unused leaf

        if l == r:
            self.left = self.right = None
        else:
            mid = (l + r) // 2
            self.left = _FFD_SegTree(l, mid, C, M)
            self.right = _FFD_SegTree(mid + 1, r, C, M)

    def update(self, idx, cpu_rem, mem_rem):
        """Update leaf idx with new (cpu_rem, mem_rem) and propagate upward."""
        if self.l == self.r:
            if cpu_rem < 0 or mem_rem < 0:
                self.max_keys = [-1, -1, -1]
            else:
                self.max_keys = [
                    cpu_rem if cpu_rem < mem_rem else mem_rem,            # s=0: min(cpu, mem)
                    cpu_rem if cpu_rem < mem_rem // 2 else mem_rem // 2,    # s=1
                    cpu_rem if cpu_rem < mem_rem // 4 else mem_rem // 4,    # s=2
                ]
            return

        if idx <= self.left.r:
            self.left.update(idx, cpu_rem, mem_rem)
        else:
            self.right.update(idx, cpu_rem, mem_rem)

        self.max_keys = [
            self.left.max_keys[0] if self.left.max_keys[0] > self.right.max_keys[0] else self.right.max_keys[0],
            self.left.max_keys[1] if self.left.max_keys[1] > self.right.max_keys[1] else self.right.max_keys[1],
            self.left.max_keys[2] if self.left.max_keys[2] > self.right.max_keys[2] else self.right.max_keys[2],
        ]

    def find_first_fit(self, s, cpu_size):
        """Return the smallest index with max_keys[s] >= cpu_size, or None."""
        if self.max_keys[s] < cpu_size:
            return None
        if self.l == self.r:
            return self.l
        if self.left.max_keys[s] >= cpu_size:
            return self.left.find_first_fit(s, cpu_size)
        return self.right.find_first_fit(s, cpu_size)


def FFD(L):
    """
    First-Fit Decreasing heuristic for 2-D VM packing.

    Uses a segment tree to reduce the first-fit search from O(n) to O(log n).
    """
    T, S = gv.T, gv.S
    C, M = gv.C, gv.M

    # Total VMs = worst-case number of PMs
    total_vms = int(sum(sum(L[s][t] for t in range(T)) for s in range(S)))
    seg = _FFD_SegTree(0, max(total_vms - 1, 0), C, M) if total_vms > 0 else None

    PMs = []
    cpu_rem = []
    mem_rem = []

    for t in range(T - 1, -1, -1):  # CPU descending
        vm_cpu = 2 ** t
        for s in reversed(range(S)):
            vm_mem = (2 ** s) * vm_cpu
            cnt = int(L[s][t])
            for _ in range(cnt):
                idx = seg.find_first_fit(s, vm_cpu)
                if idx is None:
                    # Open a new PM
                    idx = len(PMs)
                    pm = [np.zeros(T, dtype=int) for _ in range(S)]
                    pm[s][t] = 1
                    PMs.append(pm)
                    cpu_rem.append(C - vm_cpu)
                    mem_rem.append(M - vm_mem)
                    seg.update(idx, C - vm_cpu, M - vm_mem)
                else:
                    # Place into an existing PM
                    PMs[idx][s][t] += 1
                    cpu_rem[idx] -= vm_cpu
                    mem_rem[idx] -= vm_mem
                    seg.update(idx, cpu_rem[idx], mem_rem[idx])

    return (PMs, []) if gv.RETURN_PMS else (len(PMs), [])