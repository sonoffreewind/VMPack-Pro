"""
vanilla_mip.py
Standard assignment-based MIP formulation for 2-DVBP.
Used to demonstrate that off-the-shelf solvers (Gurobi) cannot solve
this problem efficiently due to symmetry explosion.

Supports checkpoint callbacks to record intermediate solver states
at specified time points (e.g., 1s, 5s, 10s) during a single run.
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import globalvars as gv
import time
import sys
import io
from heuristics import VMPack_NoMixPack


class _CheckpointCallback:
    """
    Gurobi callback that records solver state at specified time points.

    Records at each checkpoint: incumbent (ub), best bound (lb),
    MIP gap, node count, and elapsed wall-clock time.

    Note: uses Gurobi 13 callback API (function-style, constants under
    gp.GRB.Callback.*).
    """
    def __init__(self, checkpoint_times):
        self.checkpoint_times = sorted(checkpoint_times)
        self.recorded = set()
        self.checkpoints = {}  # {time: dict}
        self.t_start = time.time()

    def _get_cb_time(self):
        return time.time() - self.t_start

    def __call__(self, model, where):
        if where == gp.GRB.Callback.MIP:
            rt = self._get_cb_time()
            for ct in self.checkpoint_times:
                if ct not in self.recorded and rt >= ct:
                    self._record(model, ct)
                    self.recorded.add(ct)

    def _record(self, model, ct):
        EPS = 1e-6
        record = {'time': ct, 'real_time': self._get_cb_time()}
        try:
            obj_val = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            if np.isfinite(obj_val):
                record['ub'] = int(np.ceil(obj_val - EPS))
        except (gp.GurobiError, AttributeError, OverflowError):
            pass
        try:
            obj_bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            if np.isfinite(obj_bound):
                record['bestbound'] = float(obj_bound)
                record['lb'] = int(np.ceil(obj_bound - EPS))
        except (gp.GurobiError, AttributeError, OverflowError):
            pass
        try:
            record['nodecount'] = model.cbGet(gp.GRB.Callback.MIP_NODCNT)
        except (gp.GurobiError, AttributeError):
            pass
        if 'ub' in record and 'lb' in record:
            record['gap'] = max(0, int(record['ub'] - record['lb']))
        self.checkpoints[ct] = record

    def final_checkpoints(self):
        """Return the checkpoint dict. Any missing checkpoints (e.g. solver
        finished before reaching that time) are filled with the last available
        state."""
        result = dict(self.checkpoints)
        # If solver finished early, propagate the final state to later times
        final_state = None
        for ct in sorted(self.checkpoint_times):
            if ct in result:
                final_state = result[ct]
            elif final_state is not None:
                result[ct] = dict(final_state)
                result[ct]['time'] = ct
                result[ct]['real_time'] = final_state.get('real_time', ct)
        return result


def VanillaMIP(vm_demands: np.ndarray, timelimit: int = 1500, verbose: bool = True,
                ub_heuristic_fn=None, checkpoint_times=None):
    """
    Solve VM packing using a standard assignment-based MIP formulation.
    This is the naive approach that suffers from symmetry explosion.

    Variables:
        x[s,t,j] : number of VMs of type (s,t) assigned to PM j (integer)
        y[j]     : whether PM j is opened (binary)

    Objective: minimize sum(y[j])

    Args:
        ub_heuristic_fn: Optional heuristic function to provide upper bound.
                         If None, defaults to VMPack_NoMixPack.
        checkpoint_times: Optional list of float time points (seconds) at which
                          to record solver state via callback.
                          E.g., [1, 5, 10] records incumbent at 1s/5s/10s.

    Returns: (npms_or_PMs, [], gap_info)
        gap_info now includes 'checkpoints' dict when checkpoint_times is set.
    """
    S, T, C, M, CPU = gv.S, gv.T, gv.C, gv.M, gv.CPU
    EPS = 1e-6
    t0 = time.time()  # start wall-clock timer (includes heuristic + model build + optimize)

    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    try:
        # Use specified heuristic (or NoMixPack by default) for upper bound
        
        old_return = gv.RETURN_PMS
        old_record = gv.RECORD_ENLARGE

        try:
            gv.RETURN_PMS = True
            gv.RECORD_ENLARGE = True
            if ub_heuristic_fn is None:
                result = VMPack_NoMixPack(vm_demands)
            else:
                result = ub_heuristic_fn(vm_demands)
        finally:
            gv.RETURN_PMS = old_return
            gv.RECORD_ENLARGE = old_record

        # result may be (PMs, R) or just PMs
        if isinstance(result, tuple) and len(result) >= 2:
            PMs, R = result[0], result[1]
            # R may contain enlarge records, expand back to original VM types
            if R is not None and len(R) > 0:
                from heuristics import ExpandPMsToOriginal
                PMs = ExpandPMsToOriginal(PMs, R, T)
        else:
            PMs = result

        # Validate expanded PMs
        from basic import ValidatePMs
        # ValidatePMs(PMs, vm_demands)
        
        ub_pms = len(PMs)

        if ub_pms == 0:
            if gv.RETURN_PMS:
                return ([], [], {'lb': 0, 'ub': 0, 'gap': 0, 'status': 'Optimal'})
            else:
                return (0, [], {'lb': 0, 'ub': 0, 'gap': 0, 'status': 'Optimal'})

        # Build model
        model = gp.Model("VanillaMIP_2DVBP")
        model.Params.OutputFlag = 1 if verbose else 0
        model.Params.TimeLimit = timelimit
        model.Params.Threads = 1  # Fair comparison: single thread
        model.Params.Seed = 42
        model.Params.MemLimit = 24 # memory limit

        # Register checkpoint callback if requested
        # (registered just before optimize() below)

        # Decision variables
        # x[s,t,j] = number of VMs of type (s,t) in PM j
        x = {}
        for s in range(S):
            for t in range(T):
                if vm_demands[s][t] > 0:
                    for j in range(ub_pms):
                        x[s, t, j] = model.addVar(
                            vtype=GRB.INTEGER, lb=0,
                            ub=int(vm_demands[s][t]),
                            name=f"x_{s}_{t}_{j}")
                        x[s, t, j].Start = float(PMs[j][s][t])

        # y[j] = 1 if PM j is used
        y = {}
        for j in range(ub_pms):
            y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
            y[j].Start = 1.0

        # Objective: minimize number of PMs
        model.setObjective(gp.quicksum(y[j] for j in range(ub_pms)), GRB.MINIMIZE)

        # Constraint 1: Demand satisfaction
        for s in range(S):
            for t in range(T):
                if vm_demands[s][t] > 0:
                    model.addConstr(
                        gp.quicksum(x[s, t, j] for j in range(ub_pms))
                        >= vm_demands[s][t],
                        name=f"Demand_{s}_{t}")

        # Constraint 2: CPU capacity per PM
        for j in range(ub_pms):
            cpu_expr = gp.quicksum(
                CPU[t] * x[s, t, j]
                for s in range(S) for t in range(T)
                if (s, t, j) in x)
            model.addConstr(cpu_expr <= C * y[j], name=f"CPU_{j}")

        # Constraint 3: Memory capacity per PM
        for j in range(ub_pms):
            mem_expr = gp.quicksum(
                (2 ** s) * CPU[t] * x[s, t, j]
                for s in range(S) for t in range(T)
                if (s, t, j) in x)
            model.addConstr(mem_expr <= M * y[j], name=f"Mem_{j}")

        # Symmetry-breaking constraints (lexicographic ordering of PMs)
        # This helps but is not sufficient for large instances
        for j in range(ub_pms - 1):
            model.addConstr(y[j] >= y[j + 1], name=f"SymBreak_{j}")

        model.update()

        # Register the callback on the model for checkpoint recording
        if checkpoint_times is not None and len(checkpoint_times) > 0:
            model.Params.LazyConstraints = 1  # required for callbacks
            model._checkpoint_cb = _CheckpointCallback(checkpoint_times)
            model.optimize(model._checkpoint_cb)
        else:
            model.optimize()
        elapsed = time.time() - t0

        # Collect results
        status_map = {
            GRB.OPTIMAL: 'Optimal',
            GRB.TIME_LIMIT: 'TimeLimit',
            GRB.MEM_LIMIT: 'OOM',
            GRB.INFEASIBLE: 'Infeasible',
            GRB.INF_OR_UNBD: 'InfOrUnbd',
            GRB.UNBOUNDED: 'Unbounded',
            GRB.NODE_LIMIT: 'NodeLimit',
        }
        status = status_map.get(model.status, f'Unknown({model.status})')

        gap_info = {
            'lb': None, 'ub': None, 'gap': None,
            'status': status, 'time': elapsed,
            'bestbound': None, 'nodecount': None,
        }

        # Record incumbent upper bound, if any.
        if model.SolCount > 0:
            gap_info['ub'] = int(np.ceil(model.ObjVal - EPS))

        # Record best bound and node count whenever available.
        try:
            obj_bound = model.ObjBound
            if np.isfinite(obj_bound):
                gap_info['bestbound'] = float(obj_bound)
                gap_info['lb'] = int(np.ceil(obj_bound - EPS))
        except (gp.GurobiError, AttributeError, OverflowError):
            pass

        try:
            gap_info['nodecount'] = float(model.NodeCount)
        except (gp.GurobiError, AttributeError):
            pass

        # Certified optimum.
        if model.status == GRB.OPTIMAL:
            gap_info['lb'] = gap_info['ub']
            gap_info['gap'] = 0
        else:
            # For time-limited or memory-limited runs, report an integral
            # absolute gap whenever both a feasible incumbent and a valid
            # lower bound are available.
            if gap_info['ub'] is not None and gap_info['lb'] is not None:
                gap_info['gap'] = max(0, int(gap_info['ub'] - gap_info['lb']))

        # Collect checkpoint data if callback was used
        if checkpoint_times is not None and len(checkpoint_times) > 0:
            cb = model._checkpoint_cb
            if cb is not None:
                cp = cb.final_checkpoints()
                # Build the final solver state for filling missing checkpoints
                final_state = {
                    'time': elapsed, 'real_time': elapsed,
                    'ub': gap_info['ub'],
                    'lb': gap_info['lb'],
                    'gap': gap_info['gap'] if gap_info['gap'] is not None else (
                        0 if gap_info['ub'] is not None and gap_info['lb'] is not None
                        else None),
                    'status': status,
                    'nodecount': gap_info.get('nodecount'),
                    'bestbound': gap_info.get('bestbound'),
                }
                # Fill any checkpoints that were never reached (solver finished
                # too fast) with the final solver state
                for ct in checkpoint_times:
                    if ct not in cp:
                        cp[ct] = dict(final_state)
                        cp[ct]['time'] = ct
                # Add final checkpoint
                cp['final'] = dict(final_state)
                gap_info['checkpoints'] = cp


        result_npms = gap_info['ub'] if gap_info['ub'] is not None else -1
        if gv.RETURN_PMS:
            return ([], [], gap_info)
        else:
            return (result_npms, [], gap_info)

    except gp.GurobiError as e:
        gap_info = {'lb': None, 'ub': None, 'gap': None,
                    'status': f'GurobiError: {e}', 'time': 0,
                    'bestbound': None, 'nodecount': None}
        if gv.RETURN_PMS:
            return ([], [], gap_info)
        else:
            return (-1, [], gap_info)

    finally:
        if not verbose:
            sys.stdout = old_stdout