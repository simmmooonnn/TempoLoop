# GrainBench: Fault-Injection Specification and Checkers

> Design doc for the construct-valid failure dataset used in `FRAMING2.md`. The central idea: **do not annotate failures, inject them.** When a fault is injected by construction, the ground-truth `(component, phase, responsible step)` is *known*, not guessed — which removes the self-labeling attack on attribution-accuracy metrics. Naturally-occurring failures are admitted only as a secondary, IAA-bounded set.

---

## 1. Design Principles

1. **Solvable-then-broken.** Start from a task the executor *can* solve unaided (verified by a clean rollout). Inject exactly one fault. Any resulting failure is causally attributable to the injection.
2. **Ground truth by construction.** Each injected fault carries its own label: the responsible component `ℓ*`, temporal phase `π*` (loop position + recurrence), and the step index `t*` where the fault is *triggered*.
3. **Environment-checkable outcome.** Success/failure is decided by a deterministic state assertion (AppWorld-style), never by an LLM judge.
4. **Phase vs component separable.** At least one fault family must hold the component fixed while varying the phase, so G1 (functional-layer) and G2 (phase) can diverge.
5. **Repair-measurable.** Each fault class defines (a) a *recurrence checker* (did the same fault class reappear after repair?) and (b) the operator(s) that *should* fix it, enabling localization precision/recall.

---

## 2. Injection Harness

The controlled executor `C₀` exposes hooks the injector wraps:

```text
on_task_load(task)        -> mutate prompt / constraints           (planning faults)
on_tool_schema(tool)      -> mutate advertised schema              (tool-use faults)
on_tool_call(call, step)  -> intercept args / force error / delay  (tool-use, recovery, termination)
on_memory_read(query)     -> inject stale/conflicting entries      (memory faults)
on_memory_write(entry)    -> drop / corrupt write                  (memory faults)
on_verify(answer, state)  -> disable grounding contract            (verification faults)
on_step(step_idx, state)  -> manipulate budget / progress signal   (termination faults)
```

Each hook is gated by an **injection spec**:

```json
{
  "fault_id": "TOOL_SCHEMA_RENAME_REQUIRED_FIELD",
  "component": "tool",
  "phase": {"position": "mid", "recurrence": "persistent"},
  "trigger": {"tool": "calendar.create_event", "from_step": 1},
  "params": {"rename": {"participants": "attendees"}},
  "expected_operators": ["tool_schema_narrowing", "argument_validator"],
  "time_dependent": false
}
```

A GrainBench *instance* = (base task, injection spec, executor). Many instances are generated per fault template by varying the target tool/field/step and the executor.

---

## 3. Fault Catalogue

Eight templates spanning the six phases. `ℓ*` = responsible component, `π*` = phase, `t*` = trigger step. "TD" = time-dependent (same component, label changes with loop position/recurrence).

### F1 — Tool-schema corruption  (`ℓ*=tool`, phase=mid, TD=no)
**Inject:** via `on_tool_schema`, rename/retype a required field, drop a field, or add a spurious required field for one tool.
**Effect:** agent emits a structurally wrong call → tool returns `invalid_argument`.
**Failure checker:** task state assert fails AND ≥1 tool call to the target tool returns `invalid_argument`.
**Repair should be:** `tool_schema_narrowing` or `argument_validator` on the target tool.
**Recurrence checker:** after repair, count `invalid_argument` errors on the target tool on sibling tasks; recurrence = (count_after / count_before).

### F2 — Wrong-tool lure  (`ℓ*=tool`, phase=early, TD=no)
**Inject:** via `on_tool_schema`, make a *distractor* tool's description overlap the correct tool's purpose (semantic collision).
**Effect:** agent selects the wrong tool; correct subtask never executes.
**Failure checker:** state assert fails AND the correct tool is never called while the distractor is.
**Repair should be:** `tool_menu_reranking` or tool-description disambiguation.
**Recurrence checker:** fraction of sibling tasks where distractor is chosen over correct tool.

### F3 — Recoverable tool error, no fallback  (`ℓ*=recovery`, phase=mid, TD=**yes**)
**Inject:** via `on_tool_call`, force the target tool to return a *transient* error (`rate_limited` / `temporary_unavailable`) on its **first** invocation only; succeed on retry.
**Effect:** if the agent blindly retries → recovers; if it gives up or loops → fails.
**Phase split (TD):** error on *first* call labels `(recovery, early)`; the *same* injected error fired only after the k-th call labels `(recovery, late)` / borders termination. Use this family to separate G1 from G2.
**Failure checker:** state assert fails AND a retfrom-able error preceded a give-up (no successful retry) OR an unbounded retry loop.
**Repair should be:** `retry_bound` + fallback rule (recovery) for early; `no_progress_detector` (lifecycle) for the late/loop variant.
**Recurrence checker:** rate of give-up-after-transient-error on siblings.

### F4 — Stale / conflicting memory  (`ℓ*=memory`, phase=variable, TD=**yes**)
**Inject:** via `on_memory_read`, return an outdated value (e.g. an old address/price) that conflicts with current environment state.
**Effect:** agent acts on stale info → wrong final state.
**Phase split (TD):** if the stale entry is read **before** any fresh write → `(memory, early)`; if injected to override a value the agent already wrote correctly at step j → `(memory, late)`. Same component, different phase ⇒ different correct operator (`retrieval_rescoring` vs `write_filter`/recency policy).
**Failure checker:** state assert fails AND the final action used the injected stale value (traced via input-provenance link to the poisoned read).
**Repair should be:** `retrieval_rescoring` (early) / `write_filter` + recency (late).
**Recurrence checker:** fraction of siblings where the stale value reaches a terminal action.

### F5 — Verification bypass  (`ℓ*=verification`, phase=late, TD=no)
**Inject:** via `on_verify`, disable the grounding contract so an *unsupported* final answer is accepted by the loop (the external state assert still judges correctness).
**Effect:** agent finalizes a claim not grounded in tool outputs.
**Failure checker:** state assert fails AND final answer references a value with no supporting tool output in `τ`.
**Repair should be:** `grounding_contract` / `finalization_check`.
**Recurrence checker:** rate of ungrounded finalizations on siblings.

### F6 — Premature termination  (`ℓ*=termination`, phase=late, TD=**yes**)
**Inject:** via `on_step`, emit a spurious "task looks done" progress signal at step j (before the goal state is reached).
**Effect:** agent stops early; goal unmet.
**Phase split (TD):** early-j vs late-j changes whether the right fix is a `readiness_check` (verification-gated finalization) vs a `no_progress`/goal-state model. Pairs with F7 to test the loop-position axis.
**Failure checker:** agent terminates AND goal state assert is unmet AND remaining required subtasks were reachable.
**Repair should be:** `verification_gated_finalization` / goal-state model.
**Recurrence checker:** premature-stop rate on siblings.

### F7 — Non-termination / retry loop  (`ℓ*=termination`, phase=late, TD=no)
**Inject:** via `on_tool_call`, make a needed tool return an *unactionable* error persistently (e.g. `permission_denied` with no remedy).
**Effect:** agent retries indefinitely → hits budget, no progress.
**Failure checker:** ≥N identical (tool, args) calls in a row OR step/token budget exhausted with zero state progress.
**Repair should be:** `loop_guard` + `retry_bound` + `no_progress_detector`.
**Recurrence checker:** repeated-identical-call rate on siblings.

### F8 — Planning under-specification  (`ℓ*=planning`, phase=early, TD=no)
**Inject:** via `on_task_load`, delete one explicit constraint from the prompt that the agent must infer (e.g. "only invite people from team X").
**Effect:** decomposition misses a constraint → wrong scope of actions.
**Failure checker:** state assert fails AND the violated constraint corresponds to the deleted one (checked against the unmutated task).
**Repair should be:** `constraint_extraction_checklist` / `decomposition_template`.
**Recurrence checker:** constraint-omission rate on siblings.

---

## 4. Ground-Truth Label Schema

Every instance ships with:

```json
{
  "instance_id": "F4_memory_stale_late_appworld_023_qwen7b",
  "base_task": "appworld_023",
  "executor": {"family": "qwen", "tier": "small", "base_pass": 0.41},
  "label": {
    "component": "memory",
    "phase": {"position": "late", "recurrence": "first"},
    "responsible_step_rule": "input-provenance link to injected read",
    "expected_operators": ["write_filter", "recency_policy"]
  },
  "time_dependent": true,
  "clean_rollout_passes": true
}
```

`responsible_step` is resolved at scoring time by the provenance link the injector planted (the step whose input traces to the injected hook), so `t*` is exact for F1/F3/F4/F5/F7 and rule-based for F2/F6/F8.

---

## 5. Checkers (unified interface)

```python
class FaultChecker:
    def failed(self, trace) -> bool: ...          # did the injected failure occur?
    def attributes(self, pred) -> dict:           # score a granularity's attribution
        # returns {component_correct, phase_correct, step_in_topk,
        #          loc_precision, loc_recall, over_edit}
        ...
    def recurrence(self, sibling_traces_before, sibling_traces_after) -> float:
        # P-FRR for this fault class: after/before occurrence ratio on siblings
        ...
    def repaired(self, patch, sibling_traces_after, regression_set) -> dict:
        # {target_improvement: Δ, regression: R, accepted: bool}
        ...
```

- **`failed`** uses only environment state asserts + structural trace predicates (no LLM judge).
- **`attributes`** compares a granularity's predicted unit to the injected ground truth; `loc_precision/recall` compare the *patched components* to `expected_operators`' target component(s); `over_edit` counts components touched beyond `ℓ*`.
- **`recurrence`** is the P-FRR estimator, computed on held-out sibling tasks of the same fault class.
- **`repaired`** runs the regression-safe gate (`ΔTarget ≥ δ_min`, `R ≤ r_max`) identically for every granularity.

---

## 6. Generation & Splits

- **Per template:** sweep target tool/field/step/constraint × executor panel → ~20–40 instances each.
- **v0 (Week 2):** 60–80 instances, F1/F3/F4/F6/F8 only, ≥15 time-dependent (F3 late, F4 late, F6 early-vs-late). Two executors (one strong, one weak) → first H2 probe.
- **Full:** 150–300 instances, all 8 templates, full executor panel.
- **Splits:** held-in (router training) / validation (gate tuning) / held-out tasks **and** held-out executors. Time-dependent instances are stratified across splits so G1-vs-G2 can be measured on held-out.

---

## 7. What Each Fault Buys the Paper

| Fault | Tests | Why it matters |
|---|---|---|
| F1, F2 | tool-use, fine granularity helps | step/layer should beat global |
| F3 | recovery, **G1 vs G2** | early-vs-late error ⇒ different operator at same component |
| F4 | memory, **G1 vs G2** | stale-before-write vs override-after-write |
| F5 | verification | late-phase grounding |
| F6 | termination, **phase position** | early-vs-late premature stop |
| F7 | termination, loops | loop-guard localization |
| F8 | planning, **coarse helps** | upstream diffuse cause ⇒ global/coarse may win (H1) |

F3/F4/F6 are the load-bearing families: they are where the *temporal-phase* level (G2) must beat *functional-layer* (G1), i.e. where the original TempoLoop thesis lives or dies.
