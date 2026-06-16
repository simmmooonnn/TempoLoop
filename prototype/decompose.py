#!/usr/bin/env python3
"""
Mechanism decomposition for GrainRoute (FRAMING2.md sec.11, claim C3 / hyp H4):
once the Week-3 probe says GO, explain *why* the optimal granularity shifts.

Two channels per (executor, granularity):

  attribution precision : oracle_reward - realized_reward
        how much is lost to landing on the wrong unit (executor-INDEPENDENT in
        the mock: ATTR_ACC does not depend on capability).

  executor utilization  : measure_utilization()  (intervention: force g with
        oracle attribution, measure activate-&-follow rate on the same task;
        executor-DEPENDENT -- this is the H2/H4 channel).

H4 read: if the strong->fine vs weak->coarse shift is driven by *utilization*
(differs sharply by executor) and NOT by *attribution precision* (similar by
executor), then the router wins by matching repair specificity to what the
executor can actually use -- not by attributing more accurately.

    python decompose.py
    python decompose.py --null     # H2 off -> utilization flat -> no shift to explain
"""
from __future__ import annotations

import argparse
import random
import statistics

import apparatus as A
from run_probe import FAULT_TEMPLATES, N_SIBLINGS, N_CLEAN, make_instance


def channel_table(name, cap, K, n, seed):
    rng = random.Random(seed + hash(name) % 1000)
    ex = A.MockExecutor(name, cap, K, rng)
    realized = {g: [] for g in A.GRANULARITIES}
    oracle = {g: [] for g in A.GRANULARITIES}
    util = {g: [] for g in A.GRANULARITIES}
    for _ in range(n):
        failed, siblings = make_instance(rng)
        for g in A.GRANULARITIES:
            realized[g].append(A.repair_and_validate(ex, failed, g, siblings, N_CLEAN, rng).reward)
            oracle[g].append(A.repair_and_validate(ex, failed, g, siblings, N_CLEAN, rng, oracle=True).reward)
            util[g].append(A.measure_utilization(ex, failed, g, m=120))
    rows = {}
    for g in A.GRANULARITIES:
        r = statistics.fmean(realized[g])
        o = statistics.fmean(oracle[g])
        rows[g] = dict(realized=r, oracle=o, attr_gap=o - r, util=statistics.fmean(util[g]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=float, default=2.2)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--strong-cap", type=float, default=0.90)
    ap.add_argument("--weak-cap", type=float, default=0.35)
    args = ap.parse_args()
    K = 0.0 if args.null else args.K

    print(f"\nMechanism decomposition | instances={args.instances} seed={args.seed} "
          f"K={K}{'  (NULL)' if args.null else ''}")
    print("=" * 74)

    tables, best = {}, {}
    for name, cap in [("strong", args.strong_cap), ("weak", args.weak_cap)]:
        rows = channel_table(name, cap, K, args.instances, args.seed)
        tables[name] = rows
        best[name] = max(A.GRANULARITIES, key=lambda g: rows[g]["realized"])
        print(f"\nexecutor={name:<6} cap={cap}      (argmax_g = {best[name]})")
        print(f"   {'g':<7}{'realized':>10}{'oracle':>10}{'attr_gap':>10}{'utilization':>13}")
        for g in A.GRANULARITIES:
            x = rows[g]
            mark = "  <==" if g == best[name] else ""
            print(f"   {g:<7}{x['realized']:>+10.3f}{x['oracle']:>+10.3f}"
                  f"{x['attr_gap']:>+10.3f}{x['util']:>13.2f}{mark}")

    # ---- H4 read on the contested granularities -------------------------------
    gs, gw = best["strong"], best["weak"]
    print("\n" + "=" * 74)
    if gs == gw:
        print(f"No shift (both pick {gs}). Nothing to decompose; see --null vs default.")
        print("=" * 74 + "\n")
        return

    # Why does weak abandon the strong's choice gs in favour of gw?
    s_attr = tables["strong"][gs]["attr_gap"]
    w_attr = tables["weak"][gs]["attr_gap"]
    s_util = tables["strong"][gs]["util"]
    w_util = tables["weak"][gs]["util"]
    print(f"Shift: strong->{gs}, weak->{gw}.  Decomposing weak's retreat from '{gs}':")
    print(f"   attribution gap @ {gs}:  strong {s_attr:+.3f}  vs  weak {w_attr:+.3f}   "
          f"(d={abs(w_attr - s_attr):.3f})")
    print(f"   utilization   @ {gs}:  strong {s_util:.2f}   vs  weak {w_util:.2f}    "
          f"(d={abs(s_util - w_util):.2f})")
    attr_delta = abs(w_attr - s_attr)
    util_delta = abs(s_util - w_util)
    print("-" * 74)
    if util_delta > 2 * max(attr_delta, 1e-6):
        print("H4 SUPPORTED: the shift is UTILIZATION-mediated -- weak executors")
        print("cannot activate/follow the finer repair, while attribution precision")
        print("is comparable across executors. The router wins by matching repair")
        print("specificity to the executor, not by attributing more accurately.")
    else:
        print("H4 NOT clearly supported here: attribution-precision differences are")
        print("comparable to utilization differences; inspect before claiming C3.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
