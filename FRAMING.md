# Project Framing: Phase-Conditioned Loop Repair for Self-Evolving LLM Agents

## 0. Working Title

**TempoLoop: Phase-Conditioned Loop Repair for Self-Evolving LLM Agents**

Alternative titles:

- **Phase-Conditioned Loop Repair for Self-Evolving LLM Agents**
- **Temporal Failure-Aware Loop Optimization for LLM Agents**
- **From Failed Traces to Targeted Loop Repair in LLM Agents**

---

## 1. Motivation

Modern LLM agents are no longer just single-turn language models. They are systems composed of a base model plus an external execution loop. This loop usually includes planning, tool selection, tool execution, recovery, memory access, verification, and termination. Recent work on agent harness engineering argues that these external components can strongly affect agent performance, reliability, and cost.

At the same time, recent self-improvement and harness-evolution methods already use execution trajectories to improve prompts, tools, memory, or harness components. Therefore, the contribution of this project should not be framed as "we are the first to make agents self-evolve." That would be too broad and already overlaps with existing work.

The more precise gap is this:

> Existing trace-based self-improvement methods often treat a failed trajectory as a global signal. They may reflect on the failure and update a prompt, context, memory, or harness, but they do not explicitly use the temporal phase of failure as the central unit for targeted loop repair.

This project proposes that for agent loop engineering, the most useful repair unit is not necessarily the exact failed step, and not the whole trajectory either. Instead, it is the **failure phase**: planning, tool-use, recovery, memory, verification, or termination. Once a failure is attributed to a phase, the system can modify the corresponding loop component instead of globally editing the entire prompt or harness.

---

## 2. Core Research Question

Given a failed or inefficient LLM agent execution trace, can we identify the temporal loop phase where the failure originated and use this attribution to perform targeted, regression-safe updates to the corresponding loop component?

In short:

> Can phase-level failure attribution provide a more actionable repair signal than global reflection or exact-step attribution for self-evolving LLM agents?

---

## 3. Key Hypothesis

The main hypothesis is:

> Phase-conditioned loop repair will reduce repeated failures, tool-call loops, token waste, and termination errors more effectively than generic reflection or global prompt/harness updates, while maintaining or improving final task success.

This hypothesis is based on a structural observation: different failure phases correspond to different editable components of the agent loop.

For example:

| Failure Phase | Likely Cause | Repair Target |
|---|---|---|
| Planning | Wrong decomposition, missing constraints | Planner prompt, planning checklist |
| Tool-use | Wrong tool, invalid argument, poor schema understanding | Tool policy, tool schema, argument validator |
| Recovery | Tool error but no fallback, repeated blind retry | Recovery policy, fallback rules |
| Memory | Irrelevant memory retrieval, wrong memory write | Memory retrieval/write policy |
| Verification | Unsupported final answer, missing evidence check | Verifier, evidence contract |
| Termination | Infinite loop, premature stop, no-progress behavior | Stopping rule, budget controller |

The novelty is the explicit mapping:

```text
failed trace → failure phase → loop component → targeted repair → validation
```

---

## 4. Relation to Existing Work

This project sits at the intersection of four existing research directions.

### 4.1 Agent Harness / Loop Evolution

Recent work such as **Agentic Harness Engineering (AHE)** and **Self-Harness** shows that the harness around an agent can be automatically improved from execution traces. AHE emphasizes component observability, experience observability, and decision observability; Self-Harness mines weaknesses from execution traces and proposes minimal harness modifications.

This project differs by focusing on **phase-conditioned repair**. Instead of mining generic weaknesses and editing the harness globally, the system first attributes the failure to a loop phase and then repairs only the corresponding component.

### 4.2 Prompt and Context Evolution

Methods such as **GEPA** show that natural-language reflection over trajectories can improve prompts efficiently. However, prompt evolution is still a relatively global update mechanism. It does not necessarily distinguish whether a failure originated from planning, tool-use, recovery, memory, verification, or termination.

This project treats prompts as only one editable component among many. The goal is not simply to optimize text prompts, but to optimize the modular execution loop.

### 4.3 Trace-Based Assurance and Contracts

Trace-based assurance frameworks record agent executions as message-action traces and use contracts to detect failures such as non-termination, unsupported claims, or unsafe actions. This project adopts the idea that traces and contracts are useful for process-level evaluation.

The difference is that this project uses trace-level signals not only for testing or governance, but also as feedback for loop repair.

### 4.4 Failure Attribution

Recent failure-attribution work shows that identifying the exact responsible agent and exact failed step is difficult. This project uses that observation as motivation: exact-step attribution may be too fine-grained and noisy for practical repair. Phase-level attribution is coarser, but more aligned with editable loop components.

---

## 5. Formal Problem Setup

Let an LLM agent be defined by a loop configuration:

```text
C = {P, T, R, M, V, S}
```

where:

- `P`: planner
- `T`: tool-use policy, tool wrapper, argument validator
- `R`: recovery policy
- `M`: memory retrieval and memory write policy
- `V`: verifier or evidence checker
- `S`: stopping and termination policy

Given a task `x ~ D`, the agent interacts with an environment and produces an execution trace:

```text
τ = {(m_t, a_t, o_t, c_t)} for t = 1 ... T
```

where:

- `m_t`: model message or internal state summary
- `a_t`: action or tool call
- `o_t`: observation or tool output
- `c_t`: metadata such as token cost, latency, error, or contract verdict

Define a phase function:

```text
φ(t) ∈ {planning, tool-use, recovery, memory, verification, termination}
```

For a failed or inefficient trace, exact-step attribution tries to identify:

```text
t* ∈ {1, ..., T}
```

This project instead predicts the origin failure phase:

```text
y = φ(t*)
```

The predicted phase `y` determines the repair operator:

```text
C' = R_y(C, τ)
```

where `R_y` modifies only the component associated with phase `y`.

---

## 6. Optimization Objective

The goal is not only to maximize final task success. The loop should also reduce cost, repeated failure, unsafe behavior, and non-termination.

A possible objective is:

```text
J(C) = E_x[ Success(τ)
           - λ1 Cost(τ)
           - λ2 ToolLoop(τ)
           - λ3 ContractViolation(τ)
           - λ4 FailureRecurrence(τ) ]
```

where:

- `Success(τ)`: whether the task is completed correctly
- `Cost(τ)`: token cost and/or latency
- `ToolLoop(τ)`: redundant or repeated tool calls
- `ContractViolation(τ)`: non-termination, unsupported answer, unsafe action, or invalid state change
- `FailureRecurrence(τ)`: whether the same type of failure reappears after repair

A repaired loop `C'` is accepted only if it improves validation performance and does not cause regression:

```text
J_val(C') > J_val(C) + ε
```

with additional constraints:

```text
ΔContractViolation ≤ δ1
ΔCost ≤ δ2
```

This prevents the system from overfitting a single failed trajectory while making the overall agent less reliable.

---

## 7. Proposed Method

The proposed framework has five modules.

### 7.1 Trace Instrumentation

Each agent run is recorded as a structured trace. The trace should include:

- task id
- step id
- loop phase
- model message
- tool call
- tool arguments
- tool output
- error message
- memory read/write operation
- token cost
- latency
- contract verdict
- final task result

Example trace event:

```json
{
  "task_id": "appworld_023",
  "step_id": 7,
  "phase": "tool-use",
  "message": "I will call the calendar API...",
  "tool_name": "calendar.create_event",
  "tool_args": {"date": "...", "participants": [...]},
  "tool_output": "error: missing required field",
  "error": "missing_required_field",
  "token_cost": 512,
  "latency": 1.2,
  "contract_verdict": "fail"
}
```

### 7.2 Temporal Failure Attribution

The system predicts the failure phase and failure type from the trace.

Possible failure phases:

- planning failure
- tool-use failure
- recovery failure
- memory failure
- verification failure
- termination failure

Possible implementation:

1. Rule-based detectors for obvious cases:
   - repeated same tool call → termination/tool-loop failure
   - tool error without fallback → recovery failure
   - wrong API or invalid arguments → tool-use failure
   - unsupported final answer → verification failure
   - irrelevant retrieved memory → memory failure

2. LLM-as-judge for ambiguous cases.

3. Optional: a small classifier trained on automatically labeled traces plus a small human-annotated subset.

### 7.3 Phase-to-Component Mapping

The predicted failure phase determines which loop component is eligible for repair.

| Failure Phase | Editable Component |
|---|---|
| Planning | planner prompt, decomposition template, planning checklist |
| Tool-use | tool description, tool selection prompt, argument validator |
| Recovery | retry rule, fallback strategy, error-handling policy |
| Memory | retrieval scoring, write filter, memory summarizer |
| Verification | evidence checker, final answer validator |
| Termination | max step rule, no-progress detector, stop verifier |

### 7.4 Targeted Loop Repair

The repair operator generates a minimal update to the corresponding component.

Examples:

- Planning failure → add a planning checklist requiring constraint extraction before tool use.
- Tool-use failure → clarify tool schema or add an argument validator.
- Recovery failure → add fallback rules for common tool errors.
- Verification failure → require final claims to be grounded in tool outputs.
- Termination failure → add no-progress detection and repeated-action limits.

The key design principle is minimality: the repair should change the smallest component necessary.

### 7.5 Regression-Safe Validation

After a repair is proposed, the system evaluates it on:

1. the original failed task,
2. nearby failure cases,
3. held-out validation tasks,
4. cost and termination checks,
5. contract compliance checks.

Only repairs that improve validation performance without unacceptable regression are accepted.

---

## 8. Evaluation Plan

### 8.1 Main Benchmark: AppWorld

AppWorld is a good main benchmark because it contains realistic multi-app API tasks, supports state-based evaluation, and naturally requires planning, tool-use, recovery, verification, and termination.

Suggested split:

- held-in evolution set: 100 tasks
- validation set: 50 tasks
- held-out test set: 100 tasks

The agent should only evolve on the held-in set. Final results should be reported on held-out test tasks.

### 8.2 Controlled Benchmark: LoopFailBench

To isolate the mechanism, build a small controlled benchmark with 100-300 tasks. Each task is designed to trigger one or more loop failure phases.

Failure categories:

- planning ambiguity
- wrong tool choice
- invalid tool arguments
- recoverable API error
- stale or conflicting memory
- unsupported final answer
- infinite retry loop
- premature termination

This benchmark is useful for measuring phase attribution accuracy and phase-specific repair success.

### 8.3 Transfer Benchmark: Terminal-Bench or Coding-Agent Subset

A smaller transfer experiment can be run on Terminal-Bench or a coding-agent benchmark subset. The purpose is not necessarily to beat the leaderboard, but to show that phase-conditioned repair can transfer to a more complex agent setting.

---

## 9. Baselines

The minimum baseline set should include:

1. **Vanilla ReAct**
2. **ReAct + Reflection**
3. **ReAct + Memory**
4. **Prompt evolution / GEPA-style update**
5. **Generic harness repair without phase attribution**
6. **Ours: phase-conditioned loop repair**

The most important comparison is:

```text
Generic repair: failed trace → global reflection/update
Ours: failed trace → phase attribution → targeted component repair
```

---

## 10. Metrics

Final-answer success alone is not enough. The evaluation should report both task-level and process-level metrics.

### Task-Level Metrics

- Task success rate
- State-based unit test pass rate
- Collateral damage rate, if supported by the benchmark

### Process-Level Metrics

- Token cost
- Number of tool calls
- Latency
- Termination failure rate
- Contract compliance rate
- Repeated tool-call loop rate
- Recovery success rate

### Proposed New Metric: Phase-Specific Failure Recurrence Rate

Define:

```text
P-FRR_p = (# failures in phase p after repair) / (# failures in phase p before repair)
```

where:

```text
p ∈ {planning, tool-use, recovery, memory, verification, termination}
```

The goal is to show that phase-conditioned repair reduces repeated failures more effectively than generic repair.

---

## 11. Ablation Studies

Required ablations:

1. **No phase attribution**: all failures use global repair.
2. **No targeted repair**: phase is predicted, but repair can edit any component.
3. **No regression validation**: accept every proposed repair.
4. **Rule-based attribution only** vs **LLM-based attribution**.
5. **Exact-step attribution** vs **phase-level attribution**.
6. **No termination repair**.
7. **No memory repair**.

The most important ablation is exact-step vs phase-level attribution. The project should test whether phase-level attribution is less noisy and more actionable than exact-step attribution.

---

## 12. Expected Contributions

The paper can claim the following contributions if the experiments support them:

1. A formulation of phase-conditioned loop repair for self-evolving LLM agents.
2. A phase-level failure taxonomy aligned with editable agent loop components.
3. A trace-based method for temporal failure attribution and targeted component repair.
4. A regression-safe validation protocol for accepting loop updates.
5. A process-level evaluation metric: phase-specific failure recurrence rate.
6. Empirical evidence that phase-conditioned repair improves reliability, token efficiency, and repeated-failure reduction compared with generic reflection or global harness repair.

---

## 13. Minimum Viable Prototype

The first prototype should be small and executable.

### Week 1: Agent Loop and Trace Logger

- Implement a simple ReAct or LangGraph-style agent.
- Add planner, tool executor, verifier, recovery policy, and stopping controller.
- Save traces in JSONL.

### Week 2: LoopFailBench v0

- Create 50 controlled tasks.
- Define automatic checkers.
- Label intended failure phases.

### Week 3: Attribution and Repair

- Implement rule-based phase detectors.
- Add LLM-as-judge attribution for ambiguous failures.
- Implement repair templates for each phase.

### Week 4: Initial Experiments

- Compare ReAct, Reflection, Generic Repair, and Ours.
- Report success, token cost, tool calls, termination failure, and P-FRR.

If the initial signal is strong, scale to AppWorld and then a coding-agent subset.

---

## 14. Risks and Mitigations

### Risk 1: The idea overlaps with harness evolution papers.

Mitigation: frame the contribution around phase-level failure attribution and phase-conditioned repair, not general harness evolution.

### Risk 2: LLM-as-judge attribution may be noisy.

Mitigation: combine rule-based detectors, automatic environment feedback, LLM judge, and a small human-labeled validation subset.

### Risk 3: Repairs may overfit to held-in tasks.

Mitigation: use regression-safe validation and report held-out test performance.

### Risk 4: Success rate gains may be small.

Mitigation: emphasize process-level improvements: reduced repeated failures, lower token cost, fewer tool loops, and better termination reliability.

---

## 15. One-Sentence Summary

This project proposes a trace-based self-evolving agent framework that attributes failures to temporal loop phases and performs targeted, regression-safe repairs to the corresponding loop components, improving not only task success but also process-level reliability, token efficiency, and termination stability.

---

## 16. Short Version for Advisor Update

I am considering a project called **Phase-Conditioned Loop Repair for Self-Evolving LLM Agents**. The core idea is to move beyond global reflection over failed trajectories. Instead, each failed execution trace is attributed to a temporal loop phase, such as planning, tool-use, recovery, memory, verification, or termination. The predicted failure phase then determines which loop component should be repaired. For example, planning failures update the planner, tool-use failures update tool schemas or argument validators, recovery failures update fallback policies, and termination failures update stopping rules.

The hypothesis is that phase-level attribution is more actionable than exact-step attribution or global reflection because it is aligned with the modular structure of the agent loop. The system would use structured traces, phase-level failure attribution, targeted repair operators, and regression-safe validation. Evaluation would report not only task success, but also token cost, tool-call count, termination failure rate, contract compliance, and phase-specific failure recurrence rate.

A feasible first prototype would use a ReAct or LangGraph-style agent on controlled tool-use tasks and AppWorld. Baselines would include ReAct, ReAct with reflection, ReAct with memory, prompt evolution, and generic harness repair. The main claim would be that phase-conditioned repair reduces repeated failures and improves process-level reliability more effectively than generic self-improvement.

---

## 17. References to Track

1. Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses. arXiv:2604.25850. https://arxiv.org/abs/2604.25850
2. Self-Harness: Harnesses That Improve Themselves. arXiv:2606.09498. https://arxiv.org/abs/2606.09498
3. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv:2507.19457. https://arxiv.org/abs/2507.19457
4. A Trace-Based Assurance Framework for Agentic AI Orchestration: Contracts, Testing, and Governance. arXiv:2603.18096. https://arxiv.org/abs/2603.18096
5. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems. arXiv:2505.00212. https://arxiv.org/abs/2505.00212
6. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. arXiv:2407.18901. https://arxiv.org/abs/2407.18901
7. From Failed Trajectories to Reliable LLM Agents / HarnessFix. arXiv:2606.06324. https://arxiv.org/abs/2606.06324
8. From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents. arXiv:2606.04990. https://arxiv.org/abs/2606.04990

