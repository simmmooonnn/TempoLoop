#!/usr/bin/env python3
"""
Week-3 go/no-go probe driven by REAL Claude executors (llm_executor.py) through
the same apparatus pipeline as run_probe.py. This is the real experiment: the
only change from run_probe.py is MockExecutor -> ClaudeExecutor.

    export ANTHROPIC_API_KEY=sk-ant-...        # never hardcode; rotate if leaked
    pip install anthropic
    python run_probe_llm.py                     # TINY defaults — validate plumbing first
    python run_probe_llm.py --instances 8 --siblings 6   # scale up once it works

COST WARNING: every rollout is a multi-turn tool loop (2-4 API calls). Cost grows as
instances x granularities x (2*siblings + n_clean). Start TINY. The defaults below make
a few dozen calls; scaling to publishable N makes thousands. You pay for these.

Only F3 (recovery) faults are wired in the real executor for now (see llm_executor.py).
"""
from __future__ import annotations

import argparse
import random
import statistics

import apparatus as A
from llm_executor import make_executor

# Real executor currently implements F3 only; keep the probe to F3 early/late
# (the time-dependent recovery fault — where phase G2 must beat layer G1).
FAULT_TEMPLATES = [
    A.Fault("F3", "recovery", "early", True),
    A.Fault("F3", "recovery", "late", True),
]


def make_instance(rng, n_siblings):
    fault = rng.choice(FAULT_TEMPLATES)
    failed = A.Task("fail", fault, rng.randrange(A.N_TRIGGER_POSITIONS))
    siblings = [A.Task(f"sib{i}", fault, rng.randrange(A.N_TRIGGER_POSITIONS))
                for i in range(n_siblings)]
    return failed, siblings


def run_executor(kind, n, seed, n_siblings, n_clean):
    rng = random.Random(seed + (0 if kind == "strong" else 1))
    ex = make_executor(kind)
    rewards = {g: [] for g in A.GRANULARITIES}
    for _ in range(n):
        failed, siblings = make_instance(rng, n_siblings)
        for g in A.GRANULARITIES:
            v = A.repair_and_validate(ex, failed, g, siblings, n_clean, rng)
            rewards[g].append(v.reward)
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=2, help="failed tasks per executor (keep tiny!)")
    ap.add_argument("--siblings", type=int, default=2, help="held-out faulty siblings for validation")
    ap.add_argument("--clean", type=int, default=0, help="clean tasks for regression (0 = skip; see TODO)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"\nWeek-3 probe (REAL Claude executors) | instances={args.instances} "
          f"siblings={args.siblings} clean={args.clean} seed={args.seed}")
    print("strong=claude-opus-4-8   weak=claude-haiku-4-5")
    print("=" * 70)

    chosen = {}
    for kind in ("strong", "weak"):
        rewards = run_executor(kind, args.instances, args.seed, args.siblings, args.clean)
        means = {g: statistics.fmean(rewards[g]) for g in A.GRANULARITIES}
        g_star = max(A.GRANULARITIES, key=lambda g: means[g])
        chosen[kind] = g_star
        print(f"\nexecutor={kind}")
        for g in A.GRANULARITIES:
            mark = "  <== argmax_g" if g == g_star else ""
            print(f"   {g:<7} reward={means[g]:+.3f}{mark}")

    g_s, g_w = chosen["strong"], chosen["weak"]
    shift = g_s != g_w
    coarser = A.SPEC[g_w] < A.SPEC[g_s]
    print("\n" + "=" * 70)
    print(f"argmax_g(strong) = {g_s}    argmax_g(weak) = {g_w}")
    print("-" * 70)
    if shift and coarser:
        print("VERDICT: GO  -- optimal granularity shifts coarser as the executor weakens.")
        print("         H2 holds for REAL models. Proceed to the router + AppWorld.")
    elif shift:
        print("VERDICT: GO? -- shifts but not monotonically coarser; inspect.")
    else:
        print("VERDICT: NO-GO -- same argmax_g across executors. H2 not supported on real")
        print("         models at this scale. Scale N up to confirm, then pivot (FRAMING2 sec.14).")
    print("=" * 70)
    print("NOTE: tiny N is for plumbing validation only — argmax at N=2 is noise. Scale up")
    print("      (more instances/siblings, all of F3/F4/F6) before trusting the verdict.\n")


if __name__ == "__main__":
    main()
