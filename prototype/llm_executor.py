#!/usr/bin/env python3
"""
Real-LLM executor for the GrainRoute apparatus (apparatus.py) — the last mock
swap from FRAMING2.md sec.6. Implements `Executor.rollout(task, config)` against
the Anthropic Messages API with a real tool-use loop and an injected, CONSTRUCT-
VALID fault, so it is a drop-in for `MockExecutor` in `repair_and_validate` /
`validate`.

Strong vs weak executor pair (the H2 axis):
    strong = claude-opus-4-8   (most capable Opus-tier)
    weak   = claude-haiku-4-5  (fast, lower-capability)

SECURITY: the API key is read from the ANTHROPIC_API_KEY environment variable by
the SDK. NEVER hardcode it. Set it before running:
    export ANTHROPIC_API_KEY=sk-ant-...        # bash
    $env:ANTHROPIC_API_KEY = "sk-ant-..."      # PowerShell
If a key was ever pasted into a chat/file/commit, rotate it at console.anthropic.com.

COST: each rollout is a multi-turn tool loop. The probe runs many rollouts.
Keep N tiny while validating, then scale up. See run_probe_llm.py and FAULT_DESIGN.md.

Requires: pip install anthropic

----------------------------------------------------------------------------
THE FAULT (F3, redesigned): STALE-SESSION RECOVERY — see FAULT_DESIGN.md.
An earlier "transient error, please retry" fault had NO construct validity
(capable models retry unaided -> C0 already 4/4 -> zero headroom). This redesign
fixes all three requirements:

  (1) C0 fails even for Opus: the first naive get_code on a stale session
      *permanently locks* that widget (irrecoverable) -> task fails. The fix
      (refresh BEFORE fetching) is non-obvious and not hinted.
  (2) Correct behavior is non-obvious, NOT "retry harder" (which Opus does
      anyway and which does not help here — reactive refresh is too late).
  (3) Phase-level repair (G2) beats layer-level (G1) on `late` instances:
      layer installs a phase-blind "refresh once up front" op (works early,
      FAILS late); phase installs the phase-matched op (per-call refresh on late).

F4 (memory) and F6 (termination) remain TODO — follow this pattern + GRAINBENCH.md.
"""
from __future__ import annotations

import os

import apparatus as A

STRONG_MODEL = "claude-opus-4-8"
WEAK_MODEL = "claude-haiku-4-5"

# Session freshness window (in tool-calls) after a refresh, by failure phase.
TTL_EARLY = 999   # one up-front refresh covers the whole episode
TTL_LATE = 1      # refresh covers only the immediately following fetch -> must refresh per-call

# Number of widgets to fetch. Longer horizon stresses the per-call refresh discipline:
# a weak executor is likelier to DRIFT (forget to refresh before a later fetch) than a
# strong one -> this is the H2 utilization lever (set via env F3_NUM_WIDGETS).
NUM_WIDGETS = int(os.environ.get("F3_NUM_WIDGETS", "3"))
WIDGETS = [f"w{i+1}" for i in range(NUM_WIDGETS)]
CODES = {w: f"K{i+1}-{(1000 + (i + 1) * 137) % 9000:04d}" for i, w in enumerate(WIDGETS)}
MAX_TURNS = 2 * NUM_WIDGETS + 8   # refresh+fetch per widget, plus submit and slack

USER_PROMPT = (
    "Retrieve the access codes for these widgets: " + ", ".join(WIDGETS) + ". "
    "Use get_code for each, then submit all of them with submit_codes to finish."
)

TOOLS = [
    {
        "name": "refresh_session",
        "description": "Synchronizes the session token.",  # deliberately vague — protocol not hinted
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_code",
        "description": "Fetch the access code for a widget.",
        "input_schema": {
            "type": "object",
            "properties": {"widget": {"type": "string"}},
            "required": ["widget"],
        },
    },
    {
        "name": "submit_codes",
        "description": "Submit all retrieved codes to complete the task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "object",
                    "description": "Map of widget name -> code, e.g. {\"alpha\": \"...\"}",
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["codes"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Patch guidance: a HarnessConfig patch becomes a system-prompt snippet. WHICH
# snippet (and how phase-specific) is decided by the granularity's attribution;
# WHETHER the executor follows it is utilization (the H2/H4 channel).
# --------------------------------------------------------------------------- #
_PROTOCOL_UPFRONT = (
    "SESSION PROTOCOL: The session starts expired. Call refresh_session ONCE at the very "
    "start, before any get_code. Calling get_code on an expired session permanently locks "
    "that widget, so never fetch before refreshing."
)
_PROTOCOL_PER_CALL = (
    "SESSION PROTOCOL: The session token expires after a single use. Call refresh_session "
    "IMMEDIATELY BEFORE EVERY get_code call, every single time. Calling get_code on an "
    "expired session permanently locks that widget, so always refresh first."
)
_GENERIC = "GENERAL GUIDANCE: Work carefully, read each tool error, and handle failures before finishing."


def build_system(config: A.HarnessConfig) -> str:
    base = (
        "You are an agent that completes a task using the provided tools. "
        "Call tools to do the work; when the task is complete, stop."
    )
    snippets = []
    for p in config.patches:
        if p.operator == "noop":
            continue
        if p.operator == "broad_reflection_rewrite":      # global: unscoped, no protocol -> fails
            s = _GENERIC
        elif p.operator == "retry_bound_fallback":        # recovery component
            if p.step is not None:                        # step granularity: brittle, position-specific (overfits)
                s = (f"SESSION PROTOCOL: Call refresh_session before tool-call #{p.step + 1}, "
                     f"then fetch.")
            elif p.phase == "late":                       # phase granularity, late -> correct per-call op
                s = _PROTOCOL_PER_CALL
            else:                                         # layer (phase=None) or phase=early -> up-front op
                s = _PROTOCOL_UPFRONT
        else:
            continue
        if s not in snippets:
            snippets.append(s)
    return base + ("\n\n" + "\n".join(snippets) if snippets else "")


# --------------------------------------------------------------------------- #
class _Env:
    """Per-rollout stale-session env with destructive lock-on-stale."""
    def __init__(self, task: A.Task):
        self.ttl = TTL_EARLY if task.fault.phase == "early" else TTL_LATE
        self.calls = 0            # tool-call counter (monotonic)
        self.fresh_until = -1     # session fresh while calls <= fresh_until
        self.locked = set()       # widgets bricked by a stale fetch
        self.got = {}             # widget -> code successfully retrieved
        self.first_fault = None   # (call_idx) of first E_STALE/E_LOCKED

    def step(self, name, tool_input):
        """Return (content_str, is_error, is_fault)."""
        self.calls += 1
        if name == "refresh_session":
            self.fresh_until = self.calls + self.ttl
            return ("session refreshed", False, False)
        if name == "get_code":
            w = str(tool_input.get("widget", "")).strip()
            if w in self.locked:
                self._mark_fault()
                return (f"error: E_LOCKED ({w} is locked)", True, True)
            if self.calls <= self.fresh_until:            # fresh -> success
                self.got[w] = CODES.get(w, "??")
                return (f"code={self.got[w]}", False, False)
            self.locked.add(w)                            # stale -> destructive lock
            self._mark_fault()
            return (f"error: E_STALE ({w} now locked)", True, True)
        if name == "submit_codes":
            codes = tool_input.get("codes", {}) or {}
            ok = all(str(codes.get(w, "")).strip() == CODES[w] for w in WIDGETS)
            return (("accepted" if ok else "rejected: missing or wrong codes"), not ok, False)
        return (f"unknown tool {name}", True, False)

    def _mark_fault(self):
        if self.first_fault is None:
            self.first_fault = self.calls

    def succeeded(self):
        return all(w in self.got for w in WIDGETS)


# --------------------------------------------------------------------------- #
class ClaudeExecutor(A.Executor):
    def __init__(self, name: str, model: str):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running (see module docstring). "
                "Do not hardcode the key."
            )
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic") from e
        self.name = name
        self.model = model
        self.cap = None                       # capability is MEASURED by the probe, not a knob here
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def rollout(self, task: A.Task, config: A.HarnessConfig) -> A.Trace:
        if task.fault.fid != "F3":
            raise NotImplementedError(f"fault {task.fault.fid} not yet wired (see FAULT_DESIGN.md / GRAINBENCH.md)")

        system = build_system(config)
        env = _Env(task)
        messages = [{"role": "user", "content": USER_PROMPT}]
        steps: list[A.TraceStep] = []
        error_step = None

        for _ in range(MAX_TURNS):
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=2048,            # small; no effort/thinking (portable across Opus & Haiku)
                system=system,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                out, is_err, is_fault = env.step(block.name, block.input)
                idx = len(steps)
                step = A.TraceStep(
                    idx,
                    task.fault.component if is_fault else "tool",
                    task.fault.phase if is_fault else "early",
                    f"{task.fault.fid}_error" if is_fault else None,
                )
                steps.append(step)
                if is_fault and error_step is None:
                    error_step = step
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id,
                     "content": out, "is_error": is_err}
                )
            if not tool_results:
                break
            messages.append({"role": "user", "content": tool_results})

        success = env.succeeded()
        if success:
            error_step = None              # recovered cleanly -> no unresolved fault in the trace
        return A.Trace(task, tuple(steps), success, error_step)

    def rollout_clean(self, granularity: str) -> bool:
        """Regression probe: does this patch break a previously-passing clean task?

        TODO: thread the actual Patch through validate() and run a real no-fault
        rollout under the patched config, returning whether it now fails. For the
        first Week-3 run (whose goal is the argmax SHIFT, not the regression term),
        we conservatively report no regression. Document this when reading results.
        """
        return False


def make_executor(kind: str) -> ClaudeExecutor:
    if kind == "strong":
        return ClaudeExecutor("strong", STRONG_MODEL)
    if kind == "weak":
        return ClaudeExecutor("weak", WEAK_MODEL)
    raise ValueError(kind)
