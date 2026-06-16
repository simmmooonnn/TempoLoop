#!/usr/bin/env python3
"""
Week-3 go/no-go probe, run through the REAL apparatus pipeline (apparatus.py):

    failed trace --A_g--> unit --O--> patch --apply--> config'
        --validate on held-out siblings--> reward

Same single question as week3_probe.py, but now every reward flows through
attribution -> operator -> patch -> re-rollout on held-out siblings -> gate.
The only mock is MockExecutor.rollout. Swap it for real models to run for real.

    python run_probe.py
    python run_probe.py --null              # H2 off -> expect NO-GO
    python run_probe.py --instances 120 --seed 3
"""
from __future__ import annotations

import argparse
import random
import statistics

import apparatus as A


# Load-bearing time-dependent faults (GRAINBENCH.md sec.3, sec.7): same component,
# label changes with loop position -> where phase(G2) must beat layer(G1).
FAULT_TEMPLATES = [
    A.Fault("F3", "recovery",    "early", True),
    A.Fault("F3", "recovery",    "late",  True),
    A.Fault("F4", "memory",      "early", True),
    A.Fault("F4", "memory",      "late",  True),
    A.Fault("F6", "termination", "early", True),
    A.Fault("F6", "termination", "late",  True),
]

N_SIBLINGS = 24     # held-out faulty tasks for validated improvement (more = less noisy)
N_CLEAN = 24        # previously-passing clean tasks for the regression check


def make_instance(rng: random.Random):
    """A failed task plus its held-out siblings of the same fault family."""
    fault = rng.choice(FAULT_TEMPLATES)
    failed = A.Task("fail", fault, rng.randrange(A.N_TRIGGER_POSITIONS))
    siblings = [A.Task(f"sib{i}", fault, rng.randrange(A.N_TRIGGER_POSITIONS))
                for i in range(N_SIBLINGS)]
    return failed, siblings


def run_executor(name: str, cap: float, K: float, n: int, seed: int):
    rng = random.Random(seed + hash(name) % 1000)
    ex = A.MockExecutor(name, cap, K, rng)
    rewards = {g: [] for g in A.GRANULARITIES}
    accepts = {g: 0 for g in A.GRANULARITIES}
    for _ in range(n):
        failed, siblings = make_instance(rng)
        for g in A.GRANULARITIES:
            v = A.repair_and_validate(ex, failed, g, siblings, N_CLEAN, rng)
            rewards[g].append(v.reward)
            accepts[g] += int(v.accepted)
    return rewards, accepts


def bootstrap_ci(rng, xs, iters=2000, alpha=0.05):
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=float, default=2.2, help="strength of the H2 utilization effect")
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--strong-cap", type=float, default=0.90)
    ap.add_argument("--weak-cap", type=float, default=0.35)
    args = ap.parse_args()

    K = 0.0 if args.null else args.K
    print(f"\nWeek-3 probe (wired apparatus) | instances={args.instances} seed={args.seed} "
          f"K={K}{'  (NULL: H2 off)' if args.null else ''}")
    print("=" * 70)

    chosen = {}
    for name, cap in [("strong", args.strong_cap), ("weak", args.weak_cap)]:
        rewards, accepts = run_executor(name, cap, K, args.instances, args.seed)
        means = {g: statistics.fmean(rewards[g]) for g in A.GRANULARITIES}
        g_star = max(A.GRANULARITIES, key=lambda g: means[g])
        chosen[name] = g_star
        rng = random.Random(args.seed + 99)
        print(f"\nexecutor={name:<6} cap={cap}")
        for g in A.GRANULARITIES:
            lo, hi = bootstrap_ci(rng, rewards[g])
            acc = accepts[g] / args.instances
            mark = "  <== argmax_g" if g == g_star else ""
            print(f"   {g:<7} reward={means[g]:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] "
                  f"accept={acc:4.0%}{mark}")

    g_s, g_w = chosen["strong"], chosen["weak"]
    shift = g_s != g_w
    coarser = A.SPEC[g_w] < A.SPEC[g_s]
    print("\n" + "=" * 70)
    print(f"argmax_g(strong) = {g_s}    argmax_g(weak) = {g_w}")
    print("-" * 70)
    if shift and coarser:
        print("VERDICT: GO  -- argmax_g shifts COARSER as the executor weakens.")
        print("         H2 supported. Proceed to the router (ROUTER.md) + AppWorld.")
    elif shift:
        print("VERDICT: GO? -- shifts but not monotonically coarser; inspect.")
    else:
        print("VERDICT: NO-GO -- argmax_g identical across executors.")
        print("         With a REAL executor this means pivot/stop (FRAMING2.md sec.14).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
