import gurobipy as gp
from gurobipy import GRB
import numpy as np
from heuristics import *
import sys,time
import io

def _solve_pricing_subproblem(pi, vm_types, S, T, C, M, time_limit):
    """Solve pricing subproblem with given dual values. Returns (obj_val, pattern, success)."""
    sub = gp.Model("Sub")
    sub.Params.TimeLimit = time_limit
    sub.Params.Threads = 1
    sub.Params.Seed = 42
    sub.Params.MemLimit = 24
    sub.Params.OutputFlag = 0
    x = sub.addVars(S, T, vtype=GRB.INTEGER, name="x")
    sub.setObjective(gp.quicksum(pi[s][t] * x[s,t] for s in range(S) for t in range(T)), GRB.MAXIMIZE)
    sub.addConstr(gp.quicksum(vm_types[s][t][0] * x[s,t] for s in range(S) for t in range(T)) <= C)
    sub.addConstr(gp.quicksum(vm_types[s][t][1] * x[s,t] for s in range(S) for t in range(T)) <= M)
    sub.optimize()
    
    if sub.status != GRB.OPTIMAL:
        return None, None, False
    
    current_a_val = sub.getAttr('X', x)
    pattern = [[int(round(current_a_val[s, t])) for t in range(T)] for s in range(S)]
    return sub.ObjVal, pattern, True

def _pattern_key(pattern):
    return tuple(tuple(row) for row in pattern)

def _add_pattern_to_master(m, patterns, cons, S, T, pattern, seen_patterns):
    """Add a new pattern (column) to the master problem. Returns True if added, False if duplicate."""
    key = _pattern_key(pattern)
    if key in seen_patterns:
        return False

    patterns.append(pattern)
    seen_patterns.add(key)

    new_pattern_coeffs = sum(pattern, [])
    new_col = gp.Column(new_pattern_coeffs, [cons[s, t] for s in range(S) for t in range(T)])
    m.addVar(obj=1.0, vtype=GRB.CONTINUOUS, name=f"z_{m.NumVars}", column=new_col)
    return True

def _patterns_from_heuristic(ub_heuristic_fn, vm_demands, S, T, seen_patterns):
    """Run a full-pipeline heuristic, expand its PMs back to original VM types,
    and absorb every distinct feasible pattern as an initial column.

    Used for BOTH the NoMix baseline (VMPack_NoMixPack) and the Mix warm start
    (VMPack_MixVM201Pro), so that the two initializations differ only in the
    bottleneck-stage strategy — a clean ablation matching Table 18.

    Returns (patterns, heuristic_ub, heuristic_pms).
    """
    old_return = gv.RETURN_PMS
    old_record = gv.RECORD_ENLARGE

    try:
        gv.RETURN_PMS = True
        gv.RECORD_ENLARGE = True
        result = ub_heuristic_fn(vm_demands)
    finally:
        gv.RETURN_PMS = old_return
        gv.RECORD_ENLARGE = old_record

    # result is (PMs, R); expand aggregated type-1 VMs back to original types
    if isinstance(result, tuple) and len(result) >= 2:
        pms, R = result[0], result[1]
        pms = ExpandPMsToOriginal(pms, R, T)
    else:
        pms = result

    patterns = []
    for pm in pms:
        pattern = [[int(round(pm[s][t])) for t in range(T)] for s in range(S)]
        if not any(pattern[s][t] > 0 for s in range(S) for t in range(T)):
            continue
        key = _pattern_key(pattern)
        if key not in seen_patterns:
            seen_patterns.add(key)
            patterns.append(pattern)

    return patterns, len(pms), pms


# using bin packing mip model
def PriceBranch(vm_demands:np.array, ub_heuristic_fn=None, timelimit:int = 100, verbose: bool = True):
    start = time.time()
    S,T,C,M,CPU = gv.S, gv.T, gv.C, gv.M, gv.CPU
    vm_types = []
    for s in range(S):
        vm_types.append([([CPU[t],(2**s) * CPU[t]]) for t in range(T)])

    # Dual smoothing parameters
    dual_alpha = 0.5
    smoothed_duals = None

    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    try:
        # vm packing patterns
        m = gp.Model('virtual_machine_price_solve')
        m.modelSense = GRB.MINIMIZE
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = timelimit
        m.Params.Threads = 1
        m.Params.Seed = 42
        m.Params.MemLimit = 24 # memory limit

        patterns = []
        seen_patterns = set()
        heuristic_ub = None
        heuristic_pms = None

        # Absorb initial patterns from a full VMPack pipeline. When
        # ub_heuristic_fn is None we use VMPack_NoMixPack (the NoMix baseline);
        # when it is VMPack_MixVM201Pro we use the Mix warm start. Both paths
        # therefore share the same full-pipeline structure and differ only in
        # the bottleneck-stage strategy, keeping the NoMix-vs-Mix ablation clean.
        init_fn = ub_heuristic_fn if ub_heuristic_fn is not None else VMPack_NoMixPack
        patterns, heuristic_ub, heuristic_pms = _patterns_from_heuristic(
            init_fn, vm_demands, S, T, seen_patterns)

        z = m.addVars(len(patterns), obj=1.0, vtype=GRB.CONTINUOUS, name="z")
        cons = {}
        for s in range(S):
            for t in range(T):
                cons[s, t] = m.addConstr(gp.quicksum(patterns[j][s][t] * z[j] for j in range(len(patterns))) >= vm_demands[s][t], name=f"Demand_{s}_{t}")
        
        def remaining_time():
            return max(0.0, timelimit - (time.time() - start))

        # column generation iteration
        EPS = 1e-6
        cg_certified_complete = False
        while remaining_time() > 1e-3:
            m.Params.TimeLimit = remaining_time()
            m.optimize()
            if m.status != GRB.OPTIMAL:
                break

            # get Dual Values with exponential smoothing
            raw_pi = [[cons[s,t].Pi  for t in range(T) ]  for s in range(S) ]
            if smoothed_duals is None:
                smoothed_duals = [row[:] for row in raw_pi]
            else:
                alpha_val = dual_alpha
                for s in range(S):
                    for t in range(T):
                        smoothed_duals[s][t] = alpha_val * raw_pi[s][t] + (1 - alpha_val) * smoothed_duals[s][t]
            pi = smoothed_duals

            # sub model solving Pricing with dual smoothing verification
            obj_val, pattern, success = _solve_pricing_subproblem(pi, vm_types, S, T, C, M, remaining_time())
            
            if not success:
                break
            
            # check reduced cost with smoothed duals
            if obj_val <= 1 + EPS:
                # Smoothed duals yield no improvement, verify with raw duals
                obj_val_raw, pattern_raw, success_raw = _solve_pricing_subproblem(raw_pi, vm_types, S, T, C, M, remaining_time())
                if not success_raw:
                    break
                if obj_val_raw <= 1 + EPS:
                    # Truly no improvement, column generation is certified complete
                    cg_certified_complete = True
                    break
                # Raw duals still have improvement, use that column instead
                added = _add_pattern_to_master(m, patterns, cons, S, T, pattern_raw, seen_patterns)
                if not added:
                    break
                continue
            
            # add new pattern to main problem
            added = _add_pattern_to_master(m, patterns, cons, S, T, pattern, seen_patterns)
            if not added:
                # Smoothed dual found a duplicate pattern, try raw duals
                obj_val_raw, pattern_raw, success_raw = _solve_pricing_subproblem(raw_pi, vm_types, S, T, C, M, remaining_time())
                if success_raw and obj_val_raw > 1 + EPS:
                    added = _add_pattern_to_master(m, patterns, cons, S, T, pattern_raw, seen_patterns)
                if not added:
                    break

        # Compute certified lower bound
        total_cpu = sum(vm_demands[s][t] * vm_types[s][t][0] for s in range(S) for t in range(T))
        total_mem = sum(vm_demands[s][t] * vm_types[s][t][1] for s in range(S) for t in range(T))
        resource_lb = int(np.ceil(max(total_cpu / C, total_mem / M) - EPS))

        if m.status == GRB.OPTIMAL and cg_certified_complete:
            lp_lb = int(np.ceil(m.ObjVal - EPS))
            certified_lb = max(resource_lb, lp_lb)
        else:
            certified_lb = resource_lb

        # Branch and Bound (only if time remains)
        ip_solved = False
        if remaining_time() > 1e-3:
            m.Params.TimeLimit = remaining_time()
            for v in m.getVars():
                v.Start = max(0.0, round(v.X))
                v.vtype = GRB.INTEGER
            m.optimize()
            ip_solved = True

        # Compute gap info
        gap_info = {
            'lb': certified_lb,
            'ub': None,
            'gap': None,
            'n_cols': len(patterns),
            'cg_certified': cg_certified_complete,
        }
        if ip_solved and m.SolCount > 0:
            best_obj = int(np.ceil(m.ObjVal - EPS))
            gap_info['ub'] = best_obj
            gap_info['gap'] = best_obj - certified_lb

            if gap_info['gap'] == 0:
                print(f"PriceBranch: certified optimal number of PMs: {best_obj}")
            else:
                print(f"PriceBranch: best incumbent number of PMs: {best_obj}")

            if gv.RETURN_PMS:
                # Generate PMs list from variables and their column coefficients
                PMs = []
                all_vars = m.getVars()
                for i, var in enumerate(all_vars):
                    count = int(round(var.X))
                    if count >= 1:
                        p = patterns[i]
                        Q0 = np.array(p[0])
                        Q1 = np.array(p[1])
                        Q2 = np.array(p[2])
                        PMs.extend([[Q0, Q1, Q2]] * count)
                return (PMs, [], gap_info)
            else:
                return (best_obj, [], gap_info)
        else:
            # No integer solution was obtained from the restricted IP
            if heuristic_ub is not None:
                gap_info['ub'] = heuristic_ub
                gap_info['gap'] = heuristic_ub - certified_lb
                if gv.RETURN_PMS:
                    return (heuristic_pms, [], gap_info)
                else:
                    return (heuristic_ub, [], gap_info)
            else:
                if gv.RETURN_PMS:
                    return ([], [], gap_info)
                else:
                    return (-1, [], gap_info)
    finally:
        if not verbose:
            sys.stdout = old_stdout
