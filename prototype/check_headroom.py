#!/usr/bin/env python3
"""
Cheap construct-validity check for the redesigned F3 (stale-session recovery),
per FAULT_DESIGN.md. Run this BEFORE any full run_probe_llm.py — it tells you
whether the fault has headroom and whether phase(G2) beats layer(G1) on `late`.

For each phase (early/late) x model (strong/weak), measure success rate under:
  C0     : no repair               (should be LOW)
  layer  : "refresh once up front" (G1; phase-blind)
  phase  : phase-matched op        (G2; up-front for early, per-call for late)

Expected shape if the design works:
  early: C0 low;  layer ~= phase high
  late : C0 low;  layer LOW, phase HIGH   <-- the load-bearing G1<G2 gap

    export ANTHROPIC_API_KEY=sk-ant-...     # never hardcode; rotate if leaked
    python check_headroom.py                # n=4 per cell (~48 rollouts)
    python check_headroom.py --n 6
"""
from __future__ import annotations

import argparse

import apparatus as A
from llm_executor import make_executor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="rollouts per cell")
    args = ap.parse_args()

    print(f"\nF3 stale-session headroom check | n={args.n} per cell")
    print("strong=claude-opus-4-8   weak=claude-haiku-4-5")
    print("=" * 60)

    executors = {k: make_executor(k) for k in ("strong", "weak")}

    for phase in ("early", "late"):
        task = A.Task("t", A.Fault("F3", "recovery", phase, True), 0)
        cfg_layer = A.C0.apply(A.Patch("layer", "recovery", None, None, "retry_bound_fallback"))
        cfg_phase = A.C0.apply(A.Patch("phase", "recovery", phase, None, "retry_bound_fallback"))
        print(f"\nphase = {phase}")
        for kind, ex in executors.items():
            c0 = sum(ex.rollout(task, A.C0).success for _ in range(args.n))
            g1 = sum(ex.rollout(task, cfg_layer).success for _ in range(args.n))
            g2 = sum(ex.rollout(task, cfg_phase).success for _ in range(args.n))
            n = args.n
            flag = ""
            if phase == "late" and g2 > g1:
                flag = "  <-- G2>G1 (good)"
            print(f"   {kind:6}  C0 {c0}/{n}   layer {g1}/{n}   phase {g2}/{n}{flag}")

    print("\n" + "=" * 60)
    print("Read: want C0 LOW everywhere; early layer~=phase high; late layer<phase.")
    print("If early C0 high -> make tool descriptions vaguer. If late layer~=phase")
    print("-> lower TTL_LATE in llm_executor.py. Only then is the full probe meaningful.\n")


if __name__ == "__main__":
    main()
