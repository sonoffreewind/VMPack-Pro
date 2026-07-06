"""
test_heuristics_correctness.py

Validate that every heuristic produces a feasible packing that exactly covers
the original VM demand for both instance families:

  • mixalgos:      two‑class instances (L0/L2 only, with C1=0 and C0<3C2).
  • improvevmpack: three‑class instances (L0/L1/L2) in the VMPack bottleneck stage.

The test includes all proposed heuristics, their VMPack variants, and two
engineering baselines.  For mixalgos we test NoMixPack, MixVM301, MixVM201,
MixVM201Pro, MixPack, SafeMix and the corresponding
VMPack_* wrappers; for improvevmpack we test the VMPack variants only (the
stand‑alone two‑class heuristics do not handle L1 VMs and are skipped).

For each instance we run every applicable heuristic in RETURN_PMS mode and
call basic.ValidatePMs on the resulting PM list, which checks:
  1. no negative VM counts in any PM;
  2. every PM satisfies the CPU and memory capacity constraints [C, 2C];
  3. the union of all PMs exactly covers the original VM demand.

VMPack_* heuristics enlarge (t,1) VMs into (t+2,1) aggregates, so their raw
PMs contain aggregated type‑1 VMs.  We expand them back to original types via
ExpandPMsToOriginal before validation — this step is critical for the warm‑start
and solver initialization flows.

Neither the column‑generation‑based CG‑Benchmark nor the assignment‑based
VanillaMIP solver is tested here; these exact solvers rely on Gurobi to
enforce constraints, so feasibility and coverage are guaranteed by
construction.

Exit code 0 = all checks passed; non‑zero = at least one failure (details
printed).
"""
import sys
import numpy as np

import globalvars as gv
from data import GenExamples, DataTypes
from basic import ValidatePMs
from heuristics import (
    NoMixPack, MixVM301, MixVM201, MixVM201Pro, MixPack, SafeMix,
    VMPack_NoMixPack, VMPack_MixVM301, VMPack_MixVM201,
    VMPack_MixVM201Pro, VMPack_MixPack, VMPack_SafeMix,
    BFD, FFD, ExpandPMsToOriginal,
)

# ── Test matrix ────────────────────────────────────────────────────
# Each entry: (UP, n_instances). Use small/medium scales — enough to exercise
# the algorithms broadly while keeping the test fast. L2 is included to catch
# any large-scale edge cases, but with fewer instances.
TEST_CONFIGS = [
    (10,   50),   # S1
    (50,   50),   # M1
    (500,  20),   # L1  (fewer instances, still large)
]

# Independent heuristics: return PMs directly (no enlarge, no R).
INDEPENDENT = [
    ("NoMixPack",       NoMixPack),
    ("MixVM301",        MixVM301),
    ("MixVM201",        MixVM201),
    ("MixVM201Pro",     MixVM201Pro),
    ("MixPack",         MixPack),
    ("SafeMix",         SafeMix),
]

# VMPack_* heuristics: return (PMs, R) and need ExpandPMsToOriginal.
VMPACK = [
    ("VMPack_NoMixPack",      VMPack_NoMixPack),
    ("VMPack_MixVM301",       VMPack_MixVM301),
    ("VMPack_MixVM201",       VMPack_MixVM201),
    ("VMPack_MixVM201Pro",    VMPack_MixVM201Pro),
    ("VMPack_MixPack",        VMPack_MixPack),
    ("VMPack_SafeMix",        VMPack_SafeMix),
]

# Engineering baselines: return (PMs, []) with empty R.
ENGINEERING = [
    ("BFD", BFD),
    ("FFD", FFD),
]


def _extract_pms(result, T):
    """Normalize a heuristic result to a PM list.

    - Independent heuristics  -> PMs (list)
    - VMPack_* (PMs, R)       -> ExpandPMsToOriginal(PMs, R, T)
    - BFD/FFD (PMs, [])       -> PMs (R is empty, no expansion needed)
    """
    if isinstance(result, tuple) and len(result) >= 2:
        pms, R = result[0], result[1]
        if R is not None and len(R) > 0:
            pms = ExpandPMsToOriginal(pms, R, T)
        return pms
    return result


def _run_one(name, fn, L, T, vm_demands):
    """Run one heuristic in RETURN_PMS mode and validate. Returns error str or None."""
    gv.RETURN_PMS = True
    gv.RECORD_ENLARGE = True
    try:
        result = fn(L)
    except Exception as e:
        return f"EXCEPTION during {name}: {type(e).__name__}: {e}"
    finally:
        gv.RETURN_PMS = False
        gv.RECORD_ENLARGE = False

    try:
        pms = _extract_pms(result, T)
    except Exception as e:
        return f"EXCEPTION in ExpandPMsToOriginal for {name}: {type(e).__name__}: {e}"

    if not isinstance(pms, list):
        return f"{name}: result is not a PM list (got {type(pms).__name__})"

    # ValidatePMs raises RuntimeError on any failure with a detailed message.
    ValidatePMs(pms, vm_demands)
    return None


def test_scenario(scenario, configs):
    """Run all heuristics on all instances of one scenario. Returns (n_ok, n_fail, failures)."""
    # Heuristics applicable to this scenario.
    # mixalgos: independent + engineering (VMPack_* not used in mixalgos experiments,
    #           but they still run on C1=0 instances and should be feasible — test them too).
    # improvevmpack: VMPack_* + engineering + independent (independent heuristics
    #                only handle L0/L2, so they are SKIPPED on three-class instances).
    if scenario == "mixalgos":
        heuristics = INDEPENDENT + VMPACK + ENGINEERING
    else:  # improvevmpack
        heuristics = VMPACK + ENGINEERING
        # Independent heuristics (NoMixPack etc.) ignore L1 VMs entirely, so on
        # three-class instances they would fail the coverage check. They are not
        # meant to run standalone on three-class instances in the paper, so we
        # skip them here.

    n_ok = n_fail = 0
    failures = []

    for up, n_inst in configs:
        gv.InitialGlobalVars(7, up)
        Ls = GenExamples(n_inst, DataTypes.RANDOM, scenario)

        for i, L in enumerate(Ls):
            T = gv.T
            vm_demands = np.array([[int(L[s][t]) for t in range(T)] for s in range(3)])

            for name, fn in heuristics:
                err = _run_one(name, fn, L, T, vm_demands)
                if err is None:
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append(f"  [{scenario} UP={up} inst#{i} {name}] {err}")

    return n_ok, n_fail, failures


def main():
    np.random.seed(42)

    print("=" * 70)
    print("  Heuristic Correctness Test (mixalgos + improvevmpack)")
    print("=" * 70)

    total_ok = total_fail = 0
    all_failures = []

    for scenario in ["mixalgos", "improvevmpack"]:
        print(f"\n--- Scenario: {scenario} ---")
        n_ok, n_fail, failures = test_scenario(scenario, TEST_CONFIGS)
        total_ok += n_ok
        total_fail += n_fail
        all_failures.extend(failures)
        status = "PASS" if n_fail == 0 else "FAIL"
        print(f"  {status}: {n_ok} checks passed, {n_fail} failed")

    print("\n" + "=" * 70)
    if total_fail == 0:
        print(f"  ALL PASSED  ({total_ok} checks across both scenarios)")
        print("=" * 70)
        return 0
    else:
        print(f"  {total_fail} FAILURES (out of {total_ok + total_fail} checks):")
        for f in all_failures:
            print(f)
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
