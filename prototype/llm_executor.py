#!/usr/bin/env python3
"""
Real-LLM executor for the GrainRoute apparatus (apparatus.py) — the last mock
swap from FRAMING2.md sec.6. Implements `Executor.rollout(task, config)` against
the Anthropic Messages API with a real tool-use loop and injected faults, so it
is a drop-in for `MockExecutor` in `repair_and_validate` / `validate`.

Strong vs weak executor pair (the H2 axis):
    strong = claude-opus-4-8   (most capable Opus-tier)
    weak   = claude-haiku-4-5  (fast, lower-capability)

SECURITY: the API key is read from the ANTHROPIC_API_KEY environment variable
by the SDK. NEVER hardcode it. Set it before running:
    export ANTHROPIC_API_KEY=sk-ant-...        # bash
    $env:ANTHROPIC_API_KEY = "sk-ant-..."      # PowerShell
If you ever pasted a key into a chat or a file, rotate it at console.anthropic.com.

COST: each rollout is a multi-turn tool loop (2-4 API calls). The probe runs many
rollouts per instance — keep N tiny while validating, then scale up. See run_probe_llm.py.

Requires: pip install anthropic   (https://github.com/anthropics/anthropic-sdk-python)

SCOPE: this skeleton fully implements ONE fault family end-to-end — F3 (recovery:
transient tool error, no fallback) — as a worked example. F4 (memory) and F6
(termination) are marked TODO; add their tools + injection following the F3 pattern
and the GRAINBENCH.md specs.
"""
from __future__ import annotations

import os

import apparatus as A

STRONG_MODEL = "claude-opus-4-8"
WEAK_MODEL = "claude-haiku-4-5"
MAX_TURNS = 6
MAX_RETRY_BUDGET = 4   # how many times the env will keep returning the transient error if asked


# --------------------------------------------------------------------------- #
# Patch guidance: a HarnessConfig patch becomes a system-prompt snippet. WHICH
# snippet is added depends on the granularity's attribution (the operator);
# WHETHER the executor follows it is utilization (the H2/H4 channel).
# --------------------------------------------------------------------------- #
PATCH_GUIDANCE = {
    "retry_bound_fallback": (
        "RECOVERY POLICY: If a tool returns a transient error (e.g. 'rate_limited', "
        "'temporary_unavailable'), do NOT give up and do NOT change approach. Retry the "
        "exact same tool call up to 3 times; the error is temporary and will clear."
    ),
    "broad_reflection_rewrite": (  # global patch: unscoped, generic
        "GENERAL GUIDANCE: Work carefully and think step by step before finishing."
    ),
    # TODO: retrieval_rescoring (F4/memory), verification_gated_finalization (F6/termination)
    "noop": "",
}


def build_system(config: A.HarnessConfig) -> str:
    base = (
        "You are an agent that completes a small task using the provided tools. "
        "Call tools to do the work; when the task is done, stop."
    )
    snippets = []
    for p in config.patches:
        g = PATCH_GUIDANCE.get(p.operator, "")
        if g and g not in snippets:
            snippets.append(g)
    return base + ("\n\n" + "\n".join(snippets) if snippets else "")


# --------------------------------------------------------------------------- #
# F3 tool environment: fetch an access code, then submit it. The injected fault
# makes `get_code` return a transient error on the trigger call; recovery = retry.
# --------------------------------------------------------------------------- #
CORRECT_CODE = "ZK-4417"

F3_TOOLS = [
    {
        "name": "get_code",
        "description": "Fetch the access code for the requested widget.",
        "input_schema": {
            "type": "object",
            "properties": {"widget": {"type": "string"}},
            "required": ["widget"],
        },
    },
    {
        "name": "submit_code",
        "description": "Submit the access code to complete the task.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
]

F3_USER_PROMPT = (
    "Fetch the access code for widget 'alpha' using get_code, then submit it with "
    "submit_code to finish."
)


class _RolloutState:
    """Per-rollout mutable env state (fault injection counters)."""
    def __init__(self, task: A.Task):
        self.task = task
        self.get_code_calls = 0
        self.submitted_correct = False
        # early -> fault fires on the 1st get_code call; late -> on the 2nd (after progress)
        self.trigger_call = 1 if task.fault.phase == "early" else 2


def _dispatch(name, tool_input, state: _RolloutState):
    """Return (content_str, is_error, meta). meta flags fault firing / success."""
    f = state.task.fault
    if name == "get_code":
        state.get_code_calls += 1
        if f.fid == "F3" and state.get_code_calls == state.trigger_call:
            # inject transient error (recoverable by retry)
            return ("error: rate_limited (temporary, please retry the same call)",
                    True, {"is_fault": True})
        return (f"code={CORRECT_CODE}", False, {})
    if name == "submit_code":
        ok = str(tool_input.get("code", "")).strip() == CORRECT_CODE
        state.submitted_correct = state.submitted_correct or ok
        return ("accepted" if ok else "rejected: wrong code", not ok,
                {"submitted_correct": ok})
    return (f"unknown tool {name}", True, {})


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
        self.cap = None                     # capability is what the probe MEASURES, not a knob here
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def rollout(self, task: A.Task, config: A.HarnessConfig) -> A.Trace:
        if task.fault.fid != "F3":
            # TODO: implement F4 (memory) and F6 (termination) tool envs.
            raise NotImplementedError(f"fault {task.fault.fid} not yet wired for the real executor")

        system = build_system(config)
        messages = [{"role": "user", "content": F3_USER_PROMPT}]
        state = _RolloutState(task)
        steps: list[A.TraceStep] = []
        error_step = None

        for _ in range(MAX_TURNS):
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=2048,           # small; no `effort`/`thinking` (portable across Opus & Haiku)
                system=system,
                tools=F3_TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                out, is_err, meta = _dispatch(block.name, block.input, state)
                idx = len(steps)
                is_fault = meta.get("is_fault", False)
                step = A.TraceStep(
                    idx,
                    task.fault.component if is_fault else "tool",
                    task.fault.phase if is_fault else "early",
                    f"{task.fault.fid}_error" if is_fault else None,
                )
                steps.append(step)
                if is_fault:
                    error_step = step
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id,
                     "content": out, "is_error": is_err}
                )
            if not tool_results:
                break
            messages.append({"role": "user", "content": tool_results})

        success = state.submitted_correct
        if success:
            error_step = None              # recovered: no unresolved fault in the trace
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
