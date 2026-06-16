# prototype/ — Week-3 go/no-go probe

Minimal, stdlib-only harness that answers the single decision in `FRAMING2.md` sec.14:

> Holding the failure fixed, does the best repair granularity `argmax_g`
> **shift** between a strong and a weak executor?

- **shift, weak picks coarser** → H2 supported → **GO** (build the router, scale to AppWorld)
- **no shift, flat curves** → H2 absent → **NO-GO** (pivot or stop)

## Run

```bash
python week3_probe.py            # H2 mechanism ON  -> expect GO  (strong=phase, weak=layer)
python week3_probe.py --null     # H2 mechanism OFF -> expect NO-GO (both=phase)
python week3_probe.py --instances 400 --seed 7 --K 1.6
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

## Path to the real probe

1. Replace `MockExecutor` with two real executors (one strong, one weak) over the
   controlled harness (shared operator library + regression-safe gate).
2. Replace `FAULT_TEMPLATES` sampling with real GrainBench v0 fault injection
   (`GRAINBENCH.md` F3/F4/F6, ≥15 time-dependent instances).
3. Replace `sample_reward` with: run all 4 attribution levels → apply operator →
   validate on held-out siblings → record validated improvement.
4. Read the same verdict. If real `argmax_g` shifts across executors → GO.
