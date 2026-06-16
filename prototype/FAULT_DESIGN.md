# Fault design: why the first F3 failed, and the construct-valid redesign

This note records a real lesson from the 2026-06-16 real-model probe run and the
redesign it forced. It is the construct-validity work flagged in `GRAINBENCH.md`
(R3) and `FRAMING2.md` (Risk: GrainBench label validity), made concrete.

## What the first F3 got wrong

v1 fault: `get_code` returns `rate_limited (temporary, please retry)` on the first
call; recovery = retry. Real-model result (`claude-opus-4-8`, `claude-haiku-4-5`):

```
strong C0 success 4/4   recovery-patched 4/4
weak   C0 success 4/4   recovery-patched 4/4
```

Both models recover **unaided** — "retry" is the obvious move and any model does it.
So `C0` already succeeds, there is **no headroom**, every granularity's validated
improvement is ~0, and the probe returns a degenerate all-zeros NO-GO that tests
nothing. Hardening it (fail N times) then exposed a second bug: the naive operator
"retry up to 3 times" acts as a **ceiling** and *hurt* capable models that would
otherwise persist. Two lessons:

1. **A fault with an obvious fix has no construct validity.** If the correct action
   is something a capable model already does (retry), `C0` succeeds and there is
   nothing for a repair to improve.
2. **A repair can hurt.** Operators must genuinely help; the regression-safe gate
   exists precisely to catch ones that don't.

## Three requirements for a usable fault

1. **C0 must fail — even for the strong model.** The correct behavior must be
   *non-obvious* and not something the model does by default.
2. **The fix must genuinely help** (raise success above `C0`), not cap or redirect.
3. **Phase-level (G2) must be able to beat layer-level (G1)** on the time-dependent
   (`late`) variant — otherwise the temporal axis adds nothing over functional layers.

## The redesign: stale-session recovery (implemented in `llm_executor.py`)

Tools: `refresh_session()`, `get_code(widget)`, `submit_codes(...)`. Task: fetch
codes for `alpha`, `beta`, `gamma` and submit all three.

Mechanic — **destructive lock-on-stale**:
- The session starts **expired**. `refresh_session` makes it fresh for `TTL` tool-calls.
- `get_code` on an **expired** session **permanently locks that widget** (`E_LOCKED`
  thereafter; refresh cannot recover it) and returns `E_STALE`. On a **fresh**
  session it returns the code.
- The `E_STALE` message is terse; `refresh_session`'s description is vague
  ("Synchronizes the session token") — the protocol "refresh *before* fetching" is
  **not hinted**.

Why each requirement is met:

1. **C0 fails (even Opus).** The default move is to `get_code` first. That first
   naive fetch is on an expired session → the widget is **irrecoverably locked** →
   task fails. Trial-and-error is punished destructively, so the strong model can't
   stumble into the protocol reliably. Headroom exists for both tiers.
2. **The fix genuinely helps and is non-obvious.** The repair supplies the protocol
   (refresh *before* fetch). Reactive "refresh after E_STALE then retry" **fails** —
   the stale fetch already locked the widget. Only proactive refresh-then-fetch wins,
   which the model does not do by default.
3. **G2 beats G1 on `late`.** Phase sets the freshness window:
   - `early`: `TTL` huge → one up-front refresh covers the whole episode.
   - `late`: `TTL=1` → refresh covers only the next fetch → must refresh per-call.

   Operator content by granularity (see `build_system`):

   | Granularity | Operator installed | early | late |
   |---|---|---|---|
   | global | generic "work carefully" (no protocol) | fail | fail |
   | layer (phase-blind) | "refresh **once up front**" | **pass** | **fail** |
   | phase=early | "refresh once up front" | pass | — |
   | phase=late | "refresh **before every** get_code" | — | **pass** |
   | step | "refresh before tool-call #k" (position-specific) | brittle | brittle/fail |

   Over a mix of early+late instances, **phase > layer**, and the entire gap lives on
   the `late` instances — the load-bearing G1-vs-G2 case from `GRAINBENCH.md`.

## What this still does NOT guarantee (honest scope)

- **H2 itself is not baked in.** Whether the *weak* model follows the more specific
  (per-call) repair less reliably than the strong model — the utilization channel
  that would make optimal granularity shift with capability — is exactly what the
  probe must *measure*. This design makes the measurement *possible* (there is now
  headroom and a real G1/G2 gap); it does not predetermine the verdict.
- **Calibration is empirical.** `TTL_LATE`, the vagueness of the tool descriptions,
  and `MAX_TURNS` likely need tuning so that `C0` is clearly below the patched rate
  without being a floor of 0. Tune on a handful of rollouts before scaling N.
- **One fault family.** F4 (memory) and F6 (termination) still need the same
  treatment — a non-obvious failure + a genuinely-helping, phase-distinguished repair.

## How to validate the redesign (when you next run it)

Cheap headroom check first (a few dozen calls), before any full probe:

```python
# C0 should be LOW and the matched repair HIGH, for both phases and both models.
import apparatus as A
from llm_executor import make_executor
for phase, op_phase in [("early", "early"), ("late", "late")]:
    task = A.Task("t", A.Fault("F3", "recovery", phase, True), 0)
    upfront = A.C0.apply(A.Patch("layer", "recovery", None,   None, "retry_bound_fallback"))  # G1
    matched = A.C0.apply(A.Patch("phase", "recovery", op_phase, None, "retry_bound_fallback"))  # G2
    for kind in ("strong", "weak"):
        ex = make_executor(kind); n = 5
        c0 = sum(ex.rollout(task, A.C0).success for _ in range(n))
        g1 = sum(ex.rollout(task, upfront).success for _ in range(n))
        g2 = sum(ex.rollout(task, matched).success for _ in range(n))
        print(phase, kind, f"C0 {c0}/{n}  layer {g1}/{n}  phase {g2}/{n}")
```

Expected shape if the design works:
- `early`: C0 low; layer ≈ phase high (both = "refresh up front").
- `late`:  C0 low; **layer low, phase high** (layer's up-front refresh wears off).

If `early` C0 is already high, the destructive lock isn't biting — make the tool
descriptions vaguer or remove any retry hint. If `late` layer ≈ phase, `TTL_LATE`
is too forgiving — lower it. Only after this shape holds is a full `run_probe_llm.py`
verdict meaningful.
