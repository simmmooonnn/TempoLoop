#!/usr/bin/env python3
"""
Controlled apparatus for GrainRoute (FRAMING2.md sec.6, ROUTER.md sec.1-2).

Real data flow, single mocked piece (the executor):

    failed trace --A_g--> repair unit --O--> patch --apply--> config'
        --validate on held-out siblings--> (improvement, regression) --gate--> reward

Everything except `MockExecutor.rollout` is the real pipeline. To run the
real Week-3 experiment, implement `Executor.rollout` with two real models
(one strong, one weak) and feed real GrainBench faults; nothing else changes.

The ONLY behavioural assumption in the mock is from arXiv:2605.30621
(weaker executors activate/follow finer repairs worse), isolated in
`utilization()`. Set K=0 to switch it off.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional

# --------------------------------------------------------------------------- #
# Granularity axis (FRAMING2.md sec.7). specificity in [0,1]: coarse -> fine.
# --------------------------------------------------------------------------- #
GRANULARITIES = ["global", "layer", "phase", "step"]
SPEC = {"global": 0.0, "layer": 1 / 3, "phase": 2 / 3, "step": 1.0}

# Attribution accuracy: does A_g land on the true repair unit at its level?
# Finer = noisier (component is easy; exact phase/step is hard).
ATTR_ACC = {"global": 1.00, "layer": 0.90, "phase": 0.72, "step": 0.55}

# When a patch is phase-AGNOSTIC but the fault is phase-sensitive, the operator
# only partially matches (right component, wrong/unspecified phase).
PHASE_AGNOSTIC_MATCH = 0.45

# Regression: P(a previously-passing clean task breaks) under a patch of this
# granularity. Broad (global) patches touch many components -> break more.
REGR = {"global": 0.25, "layer": 0.08, "phase": 0.07, "step": 0.05}

P_C0 = 0.05            # success prob on a faulty task with the unpatched config
N_TRIGGER_POSITIONS = 4  # step-scoped patches must match the sibling's trigger step

# Reward / gate (ROUTER.md sec.2).
MU = 0.3
DELTA_MIN = 0.02
R_MAX = 0.15


# --------------------------------------------------------------------------- #
# Core objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fault:
    fid: str            # F3 / F4 / F6
    component: str      # recovery / memory / termination
    phase: str          # early / late
    phase_sensitive: bool


@dataclass(frozen=True)
class Task:
    tid: str
    fault: Fault
    trigger_step: int   # where the fault fires in this particular task


@dataclass(frozen=True)
class TraceStep:
    idx: int
    component: str
    phase: str
    error: Optional[str]


@dataclass(frozen=True)
class Trace:
    task: Task
    steps: tuple
    success: bool
    error_step: Optional[TraceStep]   # the step that carries the injected fault


@dataclass(frozen=True)
class RepairUnit:
    granularity: str
    component: Optional[str]   # None or "*" for global (covers all)
    phase: Optional[str]
    step: Optional[int]


@dataclass(frozen=True)
class Patch:
    granularity: str
    component: Optional[str]   # "*" = all components
    phase: Optional[str]
    step: Optional[int]
    operator: str


@dataclass(frozen=True)
class HarnessConfig:
    """C0 + installed patches. A component is 'patched' if any patch targets it."""
    patches: tuple = field(default_factory=tuple)

    def apply(self, patch: Patch) -> "HarnessConfig":
        return replace(self, patches=self.patches + (patch,))


C0 = HarnessConfig()


# --------------------------------------------------------------------------- #
# Executor (the ONLY mocked component)
# --------------------------------------------------------------------------- #
def utilization(spec: float, cap: float, K: float) -> float:
    """P(executor activates & follows a patch of given specificity). H2 mechanism."""
    return max(0.02, 1.0 - spec * (1.0 - cap) * K)


class Executor:
    def rollout(self, task: Task, config: HarnessConfig) -> Trace:
        raise NotImplementedError


class MockExecutor(Executor):
    def __init__(self, name: str, cap: float, K: float, rng: random.Random):
        self.name, self.cap, self.K, self.rng = name, cap, K, rng

    def _fix_strength(self, task: Task, config: HarnessConfig) -> float:
        """How well the installed patches neutralize this task's fault, in [0,1]."""
        f = task.fault
        best = 0.0
        for p in config.patches:
            comp_ok = 1.0 if (p.component == "*" or p.component == f.component) else 0.0
            if comp_ok == 0.0:
                continue
            if not f.phase_sensitive:
                phase_ok = 1.0
            elif p.phase == f.phase:
                phase_ok = 1.0
            elif p.phase is None:
                phase_ok = PHASE_AGNOSTIC_MATCH
            else:
                phase_ok = 0.0
            step_ok = 1.0 if (p.step is None or p.step == task.trigger_step) else 0.0
            spec = SPEC[p.granularity]
            best = max(best, comp_ok * phase_ok * step_ok * utilization(spec, self.cap, self.K))
        return best

    def rollout(self, task: Task, config: HarnessConfig) -> Trace:
        fix = self._fix_strength(task, config)
        p_success = P_C0 + (self.cap - P_C0) * fix if fix > 0 else P_C0
        success = self.rng.random() < p_success
        f = task.fault
        err_step = TraceStep(task.trigger_step, f.component, f.phase,
                             None if success else f"{f.fid}_error")
        steps = tuple(
            TraceStep(i, f.component if i == task.trigger_step else "plan",
                      f.phase if i == task.trigger_step else "early",
                      err_step.error if i == task.trigger_step else None)
            for i in range(N_TRIGGER_POSITIONS + 1)
        )
        return Trace(task, steps, success, None if success else err_step)

    def rollout_clean(self, granularity: str) -> bool:
        """A previously-passing clean task: does the patch of this granularity break it?"""
        return self.rng.random() < REGR[granularity]   # True == regressed


# --------------------------------------------------------------------------- #
# Attribution A_g : trace -> repair unit  (real logic over the trace)
# --------------------------------------------------------------------------- #
def attribute(trace: Trace, g: str, rng: random.Random) -> RepairUnit:
    if g == "global":
        return RepairUnit("global", "*", None, None)

    es = trace.error_step
    correct = rng.random() < ATTR_ACC[g]
    if es is None:                       # no visible error -> attribution misses
        correct = False
    true_comp = es.component if es else None
    true_phase = es.phase if es else None
    true_step = es.idx if es else None

    if not correct:                      # land on the wrong component -> patch won't match
        comp = "_wrong_"
        phase = None
        step = None
    else:
        comp = true_comp
        phase = true_phase if g in ("phase", "step") else None
        step = true_step if g == "step" else None
    return RepairUnit(g, comp, phase, step)


# --------------------------------------------------------------------------- #
# Shared operator library O : repair unit -> patch
# --------------------------------------------------------------------------- #
OPERATOR_FOR = {
    "recovery":    "retry_bound_fallback",
    "memory":      "retrieval_rescoring",
    "termination": "verification_gated_finalization",
    "*":           "broad_reflection_rewrite",
    "_wrong_":     "noop",
}


def make_patch(unit: RepairUnit) -> Patch:
    op = OPERATOR_FOR.get(unit.component, "noop")
    return Patch(unit.granularity, unit.component, unit.phase, unit.step, op)


# --------------------------------------------------------------------------- #
# Validation + reward (regression-safe gate, ROUTER.md sec.2)
# --------------------------------------------------------------------------- #
@dataclass
class Validation:
    improvement: float
    regression: float
    accepted: bool
    reward: float


def validate(ex: MockExecutor, patch: Patch, siblings_faulty, n_clean: int) -> Validation:
    cfg = C0.apply(patch)
    patched = sum(ex.rollout(t, cfg).success for t in siblings_faulty) / len(siblings_faulty)
    base = sum(ex.rollout(t, C0).success for t in siblings_faulty) / len(siblings_faulty)
    improvement = patched - base
    regression = sum(ex.rollout_clean(patch.granularity) for _ in range(n_clean)) / n_clean
    accepted = improvement >= DELTA_MIN and regression <= R_MAX
    reward = (improvement - MU * regression) if accepted else (-MU * regression)
    return Validation(improvement, regression, accepted, reward)


def repair_and_validate(ex: MockExecutor, failed_task: Task, g: str,
                        siblings_faulty, n_clean: int, rng: random.Random) -> Validation:
    """Full pipeline for one (failure, granularity): attribute -> operator -> validate."""
    trace = ex.rollout(failed_task, C0)          # the failed trace fed to attribution
    unit = attribute(trace, g, rng)
    patch = make_patch(unit)
    return validate(ex, patch, siblings_faulty, n_clean)
