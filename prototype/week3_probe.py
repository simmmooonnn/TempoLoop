#!/usr/bin/env python3
"""
Week-3 go/no-go probe for GrainRoute (see FRAMING2.md / ROUTER.md).

ONE question: holding the failure fixed, does the best repair granularity
argmax_g SHIFT between a STRONG and a WEAK executor?

  shift (weak picks coarser)  -> H2 supported            -> GO, build the router
  no shift + flat curves      -> H2 absent                -> NO-GO, pivot/stop

This is a SIMULATION HARNESS. The executor is mocked: the only behavioural
assumption is the one from "Harness Updating Is Not Harness Benefit"
(arXiv:2605.30621) -- weaker executors fail to *activate/follow* more
specific (finer-grained) repairs. That single mechanism lives in
`utilization()` and is the thing the real experiment must confirm or refute.
Swap MockExecutor for real model rollouts to run the real probe; everything
else (attribution levels, operators, reward, verdict) stays.

Run:
    python week3_probe.py                 # default: H2 mechanism ON
    python week3_probe.py --null          # mechanism OFF -> should report NO-GO
    python week3_probe.py --instances 400 --seed 7
"""
from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Granularity axis (FRAMING2.md §7).  specificity in [0,1]: coarse -> fine.
# --------------------------------------------------------------------------- #
GRANULARITIES = ["global", "layer", "phase", "step"]
SPEC = {"global": 0.0, "layer": 1 / 3, "phase": 2 / 3, "step": 1.0}

# Per-granularity priors (independent of executor).
P_ATTR = {"global": 0.95, "layer": 0.85, "phase": 0.72, "step": 0.55}   # finer = noisier attribution
GEN    = {"global": 0.70, "layer": 0.85, "phase": 0.85, "step": 0.60}   # step overfits the single trace
REGR   = {"global": 0.25, "layer": 0.08, "phase": 0.07, "step": 0.05}   # coarse patches break more siblings

# Does the operator the granularity selects actually address the fault?
# Phase-sensitive faults (F3/F4/F6 with early/late variants) need the (component, phase)
# level to pick the right operator; layer alone gets the component but the wrong phase op.
P_OP_PHASE_SENSITIVE   = {"global": 0.45, "layer": 0.45, "phase": 0.92, "step": 0.88}
P_OP_PHASE_INSENSITIVE = {"global": 0.55, "layer": 0.92, "phase": 0.85, "step": 0.82}

MU = 0.6  # regression penalty weight in the reward


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Executor:
    name: str
    cap: float          # capability in [0,1]; channels into utilization()


@dataclass(frozen=True)
class Fault:
    fid: str            # F3 / F4 / F6
    component: str      # recovery / memory / termination
    phase: str          # early / late
    phase_sensitive: bool


# The load-bearing time-dependent faults (GRAINBENCH.md §3, §7): same component,
# label changes with loop position -> exactly where phase(G2) must beat layer(G1).
FAULT_TEMPLATES = [
    Fault("F3", "recovery",    "early", True),
    Fault("F3", "recovery",    "late",  True),
    Fault("F4", "memory",      "early", True),
    Fault("F4", "memory",      "late",  True),
    Fault("F6", "termination", "early", True),
    Fault("F6", "termination", "late",  True),
]


def utilization(g: str, cap: float, K: float) -> float:
    """P(executor activates & follows a repair of specificity SPEC[g]).

    The H2 mechanism. K>0: weaker executors (low cap) follow finer repairs
    worse. K=0 (--null): utilization is flat -> no capability effect ->
    argmax_g cannot shift -> the probe should report NO-GO.
    """
    return max(0.02, 1.0 - SPEC[g] * (1.0 - cap) * K)


def p_fix(g: str, fault: Fault, ex: Executor, K: float) -> float:
    p_op = P_OP_PHASE_SENSITIVE if fault.phase_sensitive else P_OP_PHASE_INSENSITIVE
    return P_ATTR[g] * p_op[g] * utilization(g, ex.cap, K) * GEN[g]


def sample_reward(rng: random.Random, g: str, fault: Fault, ex: Executor, K: float) -> float:
    """Validated, regression-penalized held-out improvement for applying g (ROUTER.md §2)."""
    success = 1.0 if rng.random() < p_fix(g, fault, ex, K) else 0.0
    regressed = 1.0 if rng.random() < REGR[g] else 0.0
    return success - MU * regressed


# --------------------------------------------------------------------------- #
def run_executor(rng: random.Random, ex: Executor, n: int, K: float):
    """Return {g: [per-instance rewards]} for one executor over n fault instances."""
    rewards = {g: [] for g in GRANULARITIES}
    for _ in range(n):
        fault = rng.choice(FAULT_TEMPLATES)
        for g in GRANULARITIES:
            rewards[g].append(sample_reward(rng, g, fault, ex, K))
    return rewards


def bootstrap_ci(rng: random.Random, xs, iters=2000, alpha=0.05):
    means = []
    n = len(xs)
    for _ in range(iters):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return lo, hi


def argmax_g(mean_by_g):
    return max(GRANULARITIES, key=lambda g: mean_by_g[g])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=300, help="fault instances per executor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=float, default=1.6, help="strength of the H2 utilization effect")
    ap.add_argument("--null", action="store_true", help="disable H2 mechanism (K=0); expect NO-GO")
    ap.add_argument("--strong-cap", type=float, default=0.90)
    ap.add_argument("--weak-cap", type=float, default=0.35)
    args = ap.parse_args()

    K = 0.0 if args.null else args.K
    executors = [Executor("strong", args.strong_cap), Executor("weak", args.weak_cap)]

    print(f"\nWeek-3 probe  | instances/executor={args.instances}  seed={args.seed}  "
          f"K={K}{'  (NULL: H2 off)' if args.null else ''}")
    print("=" * 68)

    chosen = {}
    for ex in executors:
        rng = random.Random(args.seed + hash(ex.name) % 1000)
        rewards = run_executor(rng, ex, args.instances, K)
        means = {g: statistics.fmean(rewards[g]) for g in GRANULARITIES}
        g_star = argmax_g(means)
        chosen[ex.name] = (g_star, means, rewards)

        print(f"\nexecutor={ex.name:<6} cap={ex.cap}")
        for g in GRANULARITIES:
            lo, hi = bootstrap_ci(rng, rewards[g])
            mark = "  <== argmax_g" if g == g_star else ""
            print(f"   {g:<7} reward={means[g]:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]{mark}")

    # ----------------------------------------------------------------- verdict
    g_strong = chosen["strong"][0]
    g_weak = chosen["weak"][0]
    shift = g_strong != g_weak
    coarser = SPEC[g_weak] < SPEC[g_strong]   # weak picked a coarser granularity

    print("\n" + "=" * 68)
    print(f"argmax_g(strong) = {g_strong}")
    print(f"argmax_g(weak)   = {g_weak}")
    print("-" * 68)
    if shift and coarser:
        print("VERDICT: GO  -- argmax_g shifts COARSER as the executor weakens.")
        print("         H2 supported in this harness. Proceed to the router + AppWorld.")
    elif shift:
        print("VERDICT: GO? -- argmax_g shifts but NOT monotonically coarser; inspect.")
    else:
        print("VERDICT: NO-GO -- argmax_g identical across executors.")
        print("         H2 absent here. With a REAL executor this would mean pivot/stop")
        print("         (FRAMING2.md sec.14 go/no-go).")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
