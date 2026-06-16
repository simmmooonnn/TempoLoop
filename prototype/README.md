# prototype/ — Week-3 go/no-go probe

Minimal, stdlib-only harness that answers the single decision in `FRAMING2.md` sec.14:

> Holding the failure fixed, does the best repair granularity `argmax_g`
> **shift** between a strong and a weak executor?

- **shift, weak picks coarser** → H2 supported → **GO** (build the router, scale to AppWorld)
- **no shift, flat curves** → H2 absent → **NO-GO** (pivot or stop)

## Two probes

| File | What | Use |
|---|---|---|
| `week3_probe.py` | **abstract** sanity probe — reward sampled directly from per-granularity priors | fastest check that the verdict logic + H2 mechanism behave |
| `apparatus.py` + `run_probe.py` | **wired** probe — reward flows through the *real* pipeline: `trace --A_g--> unit --O--> patch --apply--> config' --validate on held-out siblings--> gate` | the one to grow into the real experiment; swap `MockExecutor.rollout` for real models |
| `decompose.py` | **mechanism** decomposition (C3/H4) — once the probe says GO, explains *why* via two channels: attribution precision (`oracle - realized`) vs executor utilization (force `g`, measure activate-&-follow) | the "why it wins" intervention table for the paper |

All report consistent verdicts and flip to NO-GO under `--null`.

## Run

```bash
# wired pipeline (recommended)
python run_probe.py              # H2 ON  -> GO    (strong=phase, weak=layer)
python run_probe.py --null       # H2 OFF -> NO-GO (both=phase)
python run_probe.py --instances 300 --seed 7 --K 2.2

# mechanism decomposition (run after a GO)
python decompose.py              # H2 ON  -> shift is utilization-mediated (H4)
python decompose.py --null       # H2 OFF -> utilization flat, no shift to explain

# abstract sanity probe
python week3_probe.py
python week3_probe.py --null
```

No dependencies (Python 3.8+).

## What is real vs mocked

| Piece | Status | Replace with |
|---|---|---|
| Granularity axis, attribution priors, reward, bootstrap, verdict | **real / reusable** | keep |
| `MockExecutor` (capability → behaviour) | **mocked** | real model rollouts on GrainBench instances |
| `utilization()` | **the H2 assumption** (2605.30621): weak executors follow finer repairs worse | measured patch acceptance-and-follow rate |
| Fault templates F3/F4/F6 | stubs | `GRAINBENCH.md` fault-injection hooks |

The mock bakes in *exactly one* behavioural assumption — that weaker executors
fail to activate/follow more specific repairs — isolated in `utilization()`.
That assumption is the thing the real Week-3 run must confirm or refute. The
probe deliberately reports **NO-GO** when that assumption is switched off
(`--null`), so the verdict logic is not rigged toward GO.

## Path to the real probe (from the wired version)

The wired probe (`apparatus.py`) already implements attribution → operator →
patch → apply → validate-on-siblings → gate → reward. Only one thing is mocked:

1. Implement `Executor.rollout(task, config) -> Trace` with a **real model** over
   the controlled harness — two instances (one strong, one weak). This is the
   single swap; `MockExecutor` shows the required interface.
2. Replace `FAULT_TEMPLATES` + `make_instance` with real GrainBench v0 fault
   injection (`GRAINBENCH.md` F3/F4/F6, ≥15 time-dependent instances) and the
   real held-out sibling tasks.
3. Keep `attribute`, `make_patch`, `validate`, and the gate as-is (or refine the
   operators). Read the same verdict. If real `argmax_g` shifts across executors → GO.
