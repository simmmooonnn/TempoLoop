# Project Framing v2: How Fine Should Failure Attribution Be? A Controlled Study of Repair-Signal Granularity for Self-Evolving LLM Agents

> This is an alternative framing to `FRAMING.md`. It abandons the "we propose another self-evolving harness pipeline" positioning, which is now largely subsumed by HarnessFix (arXiv:2606.06324), AHE (2604.25850), and Self-Harness (2606.09498). Instead it reframes the *granularity of the repair signal* as the object of study, and answers a question none of those papers answer: **at what granularity should a failed trace be attributed for the most actionable, regression-safe repair?**

---

## 0. Working Title

**GrainProbe: How Fine Should Failure Attribution Be for Repairing LLM Agents?**

Alternatives:

- **The Granularity of Repair: A Controlled Study of Failure Attribution for Self-Evolving Agents**
- **Step, Phase, Layer, or Global? Locating the Repair-Signal Sweet Spot for LLM Agents**

---

## 1. Motivation and the Precise Gap

The field already has strong methods that turn failed execution traces into harness/prompt updates. They differ mainly in **how finely they localize the failure before repairing it**:

- **Global** — reflect over the whole trajectory and update a prompt or harness wholesale (e.g. GEPA-style reflective prompt evolution, ReAct+Reflection).
- **Functional-layer** — attribute a responsible step to a *component layer* and apply a layer-scoped repair operator. **HarnessFix** does exactly this with its seven ETCLOVG layers (Execution/Sandbox, Tool, Context-Memory, Lifecycle-Orchestration, Observability, Verification, Governance) plus scoped operators and regression-bounded validation. **AHE** and **Self-Harness** are close relatives.
- **Exact-step** — attribute to a single responsible step via counterfactual intervention and repair with minimal behavioral drift. **CausalFlow** (2605.25338) computes step-level Causal Responsibility Scores and argues *finer, causal* attribution is necessary for reliable improvement.

So the literature implicitly disagrees about the right granularity: HarnessFix repairs at the **layer**, CausalFlow argues for the **step**, GEPA stays **global**. **No one has held everything else constant and measured granularity as an independent variable.** That is the gap.

There is also a confound the field has only just surfaced. **"Harness Updating Is Not Harness Benefit"** (2605.30621) shows that *producing* a good harness update is nearly flat in base-model capability, and that most realized gain depends on the *executing* agent actually activating and following the update (a non-monotonic effect). This means any "method A beats method B" result on self-evolving agents may be confounded by the executor rather than the repair signal. A clean granularity study must control for this.

The contribution of this project is therefore **not** a new pipeline. It is a **measurement**: a controlled harness that varies only the attribution granularity, holding the executing agent, the repair-operator library, and the validation protocol fixed, and reports how repair quality, regression risk, and cost change with granularity — and how much of any difference is attribution-quality versus executor-utilization.

---

## 2. Core Research Question

> Holding the executing agent, the repair-operator library, and the regression-safe validation protocol constant, how does the **granularity** at which a failed trace is attributed affect repair effectiveness, regression risk, and cost — and is there a granularity sweet spot that depends on the failure type?

Sub-questions:

- **RQ1 (Sweet spot).** Is repair quality non-monotonic in granularity — i.e. is there a "too coarse → imprecise repair" vs "too fine → noisy/overfit attribution" trade-off, with an interior optimum?
- **RQ2 (Type dependence).** Does the optimal granularity differ by failure type (e.g. tool-argument errors reward fine attribution; planning/decomposition errors reward coarse)?
- **RQ3 (Confound decomposition).** Of the performance difference between granularities, how much comes from *attribution precision* versus the *executing agent's ability to use* the resulting repair (the 2605.30621 confound)?

---

## 3. Hypotheses

- **H1 (Goldilocks).** Repair effectiveness is non-monotonic in granularity. Layer/phase-level attribution outperforms both global (too coarse to localize) and exact-step (attribution noise dominates, repairs overfit the single trace).
- **H2 (Type-conditioned optimum).** The best granularity is not global across failure types. Mechanistic, locally-caused failures (bad tool args, invalid schema use) favor finer attribution; diffuse, upstream-caused failures (wrong plan, premature termination) favor coarser attribution.
- **H3 (Attribution ≠ benefit).** A non-trivial share of the apparent advantage of any granularity is mediated by the executing agent's utilization, not by attribution precision alone; finer attribution can *fail to pay off* when the executor cannot act on the more specific repair.

If H1/H2 hold, "pick the granularity that fits the failure" becomes a concrete, defensible design rule. If H3 holds, it explains why prior pipelines at different granularities report comparable gains, and reframes how the field should report results.

---

## 4. Relation to Existing Work (sharpened)

The prior methods are no longer competitors — they become **named points on the granularity axis** that this study measures against each other under controlled conditions.

| Granularity level | Representative prior work | Repair unit |
|---|---|---|
| Global | GEPA (2507.19457), ReAct+Reflection | whole prompt / whole harness |
| Functional-layer | **HarnessFix (2606.06324)**, AHE (2604.25850), Self-Harness (2606.09498) | ETCLOVG-style component layer |
| Temporal-phase | *this study* (a level being tested, not the contribution per se) | loop phase / iteration position |
| Exact-step | **CausalFlow (2605.25338)**, AgenTracer | single responsible step |

Explicit differences from the closest neighbor, **HarnessFix**:

1. HarnessFix fixes one granularity (step→layer) and optimizes the pipeline. GrainProbe **varies granularity as the independent variable** and fixes the pipeline.
2. HarnessFix reports end-to-end held-out gains. GrainProbe additionally reports **attribution accuracy, repair-localization precision, and a gain-decomposition** that HarnessFix does not isolate.
3. HarnessFix's layers are static *functional roles*. GrainProbe adds a **temporal-phase** level (same component, different loop position / retry index) as a distinct, testable point — directly probing whether HarnessFix's functional-only attribution mislocates time-dependent failures.

Difference from **CausalFlow**: CausalFlow asserts finer/causal is necessary; GrainProbe **tests that assertion** against coarser levels under a shared executor and operator set, rather than assuming it.

Difference from **"Harness Updating Is Not Harness Benefit"**: that paper disentangles *update-production* from *update-utilization* across model tiers. GrainProbe borrows its decomposition logic and applies it to the *granularity* axis, controlling the executor so that granularity, not capability, is what varies.

---

## 5. Formal Setup

An agent loop is the configuration `C = {P, T, R, M, V, S}` (planner, tool policy, recovery, memory, verifier, stopping) — as in `FRAMING.md`. A run produces a trace `τ = {(m_t, a_t, o_t, c_t)}`.

Define an **attribution function at granularity `g`**:

```text
A_g : τ  ->  u_g        (the repair unit at granularity g)
g ∈ { global, layer, phase, step }
```

- `A_global(τ) = τ`            (the whole trace)
- `A_layer(τ)  = ℓ ∈ ETCLOVG`  (a functional component layer)
- `A_phase(τ)  = (ℓ, π)`       (component layer × temporal phase/iteration bucket π)
- `A_step(τ)   = t* ∈ {1..T}`  (a single responsible step)

A **shared repair operator library** `O` maps a repair unit to a candidate patch:

```text
ρ : (u_g, τ, O)  ->  C'      (a minimal patch to C)
```

Crucially, `O`, the executing agent, and the validation rule are **identical across all `g`**. Only `A_g` — the granularity of what gets fed to `ρ` — changes.

A patch `C'` is accepted iff it passes the **same** regression-safe gate for every `g`:

```text
J_val(C') > J_val(C) + ε      and      ΔRegression ≤ δ
```

where `J` is the process-aware objective from `FRAMING.md` §6 (success minus weighted cost, tool-loops, contract violations, failure recurrence).

The study measures, for each `g`: accepted-patch quality, attribution accuracy, localization precision, regression rate, cost, and the H3 decomposition.

---

## 6. The Controlled Harness (what is held constant)

This is the heart of the paper — the apparatus, not the algorithm.

| Held constant across all granularities | Why |
|---|---|
| **Executing agent** (same base model, same loop `C₀`) | isolates granularity from executor capability (the 2605.30621 confound) |
| **Repair-operator library `O`** | a difference must come from *which* operator the granularity selects, not from a richer operator set |
| **Validation protocol** (`ε`, `δ`, validation/test splits) | acceptance bar identical, so regression behavior is comparable |
| **Trace instrumentation & contracts** | all granularities see the same logged evidence |
| **Repair LLM and its budget** | finer attribution cannot win by simply spending more |

Only the attribution stage `A_g` is swapped. This is the single design decision that makes the result a measurement rather than another leaderboard entry.

---

## 7. Granularity Levels Under Test

1. **Global (G0)** — reflect over the full trace; repair may touch any component (GEPA-like).
2. **Layer (G1)** — attribute to one ETCLOVG-style functional component; repair scoped to that component (HarnessFix-like).
3. **Phase (G2)** — attribute to `(component, temporal-phase)`, where phase buckets the loop by position/iteration (early-plan / mid-exec / late-finalize) and by recurrence (first occurrence vs k-th retry). Same component, different time → different operator.
4. **Step (G3)** — attribute to a single responsible step via counterfactual / LLM-judge; repair targets that step's component with maximal locality (CausalFlow-like).

G2 is the level the original `FRAMING.md` cared about; here it is *one contestant*, not the thesis. The thesis is the **curve over G0–G3 and its dependence on failure type**.

---

## 8. Shared Repair-Operator Library

A fixed catalogue of minimal operators, reused verbatim by every granularity (examples):

- Tool: schema narrowing, argument validator insertion, tool-menu re-ranking, error-message rewrite.
- Context/Memory: failure-tail evidence preservation, retrieval re-scoring, write filter.
- Lifecycle: loop guard, retry bound, no-progress detector, verification-gated finalization.
- Verification: readiness check, intermediate validation gate, grounding/evidence contract.
- Planning: constraint-extraction checklist, decomposition template.

The attribution granularity determines **which operator(s) become eligible** and **how tightly scoped** the patch is — nothing else.

---

## 9. Evaluation Plan

- **Main benchmark: AppWorld** (90:45:90 split, matching HarnessFix so numbers are comparable). Stateful multi-app API tasks exercise all components.
- **Diagnostic benchmark: a granularity-labeled failure set (GrainBench).** 150–300 controlled failures, each annotated with (a) the *true* responsible component, (b) the *true* temporal phase, (c) the *true* responsible step. This is what makes attribution-accuracy and localization-precision measurable per granularity. Designed to include **time-dependent failures** (same component, different loop position) to test whether G1 (functional-only) mislocates them relative to G2.
- **Transfer check: GAIA or a Terminal-Bench subset.** Confirm the granularity ranking is not AppWorld-specific.

The agent only evolves on the held-in split; all headline numbers are held-out test.

---

## 10. Baselines / Conditions

The conditions *are* the granularity levels (G0–G3 from §7), all running through the shared harness of §6. Two extra references:

- **No-repair control** (`C₀` frozen) — the floor.
- **Oracle attribution** — feed the ground-truth label from GrainBench at each granularity, to separate *attribution error* from *operator/executor ceiling*.

The oracle condition is what cleanly answers RQ3/H3: if oracle-G3 ≫ realized-G3 but oracle-G1 ≈ realized-G1, the bottleneck at fine granularity is attribution noise, not the operator.

---

## 11. Metrics

**Per-granularity attribution quality** (needs GrainBench labels):

- Attribution accuracy (does `A_g` hit the true unit at its granularity?)
- Repair-localization precision/recall (does the patch touch the right component and only it?)
- Over-edit rate (lines/components changed beyond the true cause)

**Repair outcome:**

- Held-out task success
- Process-level: token cost, tool-call count, termination-failure rate, contract compliance, tool-loop rate, recovery success
- **Phase-Specific Failure Recurrence Rate (P-FRR)** carried over from `FRAMING.md`
- Regression rate (newly-broken previously-passing tasks)

**Decomposition (the H3 deliverable):**

- Gain attributable to *attribution precision* = (oracle-`g` − realized-`g`)
- Gain attributable to *executor utilization* = patch-acceptance-and-follow rate under fixed executor
- A simple mediation/ablation table separating the two

**Headline figure:** repair quality vs granularity, one curve per failure type — the visual answer to H1/H2.

---

## 12. Ablations

1. **Oracle vs realized attribution** at each granularity (isolates attribution noise).
2. **Operator-set held vs enriched** (confirms gains aren't from a bigger toolbox at finer levels).
3. **Executor swapped** across two capability tiers (probes H3 / the 2605.30621 effect directly).
4. **Phase definition variants** for G2 (position-only vs recurrence-only vs both).
5. **Validation gate on/off** (does finer granularity overfit single traces more when unguarded?).
6. **Cost-matched** comparison (equalize repair-LLM tokens across granularities).

The most important is #1 + #3 together: they convert "which granularity wins" into "*why* it wins."

---

## 13. Expected Contributions

1. A **formalization of repair-signal granularity** and a controlled harness that varies it while holding executor, operator library, and validation constant.
2. **GrainBench**: a granularity-labeled failure dataset (component / phase / step ground truth) enabling per-granularity attribution and localization metrics — including time-dependent failures.
3. An **empirical granularity–performance characterization** across AppWorld and a transfer set, with the per-failure-type sweet-spot analysis (H1/H2).
4. A **decomposition of gains into attribution-precision vs executor-utilization** (H3), directly controlling the "harness-updating ≠ harness-benefit" confound — a result no prior pipeline isolates.
5. Practical guidance: **a failure-type → granularity selection rule**, and evidence on whether the temporal-phase level adds anything over functional-layer attribution.

These are measurement/diagnostic contributions, which are much harder to scoop than "another pipeline," and which the nearest neighbors (HarnessFix, CausalFlow) structurally cannot claim because each fixes a single granularity.

---

## 14. Minimum Viable Prototype

- **Week 1.** Build the controlled harness: one ReAct/LangGraph executor `C₀`, the shared operator library `O`, the regression-safe validation gate, JSONL tracing. The four attribution functions `A_global/layer/phase/step` as swappable modules.
- **Week 2.** GrainBench v0: 60–80 controlled failures with component/phase/step labels, including ≥20 time-dependent cases. Automatic checkers.
- **Week 3.** Run G0–G3 + oracle on GrainBench v0. Produce the first granularity curve and the oracle-vs-realized gap.
- **Week 4.** If the curve shows structure (non-monotonic or type-dependent), scale to AppWorld held-in/held-out and add the executor-swap ablation.

Kill criterion: if G0–G3 are statistically indistinguishable on GrainBench *and* the oracle gap is flat, the granularity axis carries no signal — report that as a negative result and stop before AppWorld.

---

## 15. Risks and Mitigations

- **Risk: granularities collapse to similar performance.** This is itself a publishable finding (it would corroborate "harness-updating ≈ flat" 2605.30621 and tell the field to stop competing on granularity). Mitigation: pre-register the oracle conditions so a null is interpretable, not just a failed method.
- **Risk: GrainBench labels are subjective** (phase/step ground truth is hard — exactly what the failure-attribution literature warns). Mitigation: inter-annotator agreement study; report attribution metrics with agreement bounds; lean on environment-checkable failures where possible.
- **Risk: the temporal-phase level (G2) is just functional-layer in disguise.** Mitigation: deliberately construct time-dependent failures where the *same* component fails differently by loop position; if G2 = G1 there too, honestly conclude the temporal axis is redundant.
- **Risk: results are AppWorld-specific.** Mitigation: the transfer set (GAIA / Terminal-Bench subset); report the ranking, not just absolute numbers.
- **Risk: executor confound leaks back in.** Mitigation: fixed executor by default, with the swap only as a controlled ablation for H3.

---

## 16. One-Sentence Summary

Rather than proposing another self-evolving harness, this project builds a controlled apparatus that varies only the **granularity** of failure attribution — global, functional-layer, temporal-phase, or exact-step — over a fixed executor, operator library, and validation gate, to measure where the repair-signal sweet spot lies, whether it depends on failure type, and how much of any advantage is attribution precision versus the executing agent's ability to use the repair.

---

## 17. References to Track

1. From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws (HarnessFix). arXiv:2606.06324. https://arxiv.org/abs/2606.06324
2. CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures. arXiv:2605.25338. https://arxiv.org/abs/2605.25338
3. Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents. arXiv:2605.30621. https://arxiv.org/abs/2605.30621
4. Agentic Harness Engineering (AHE). arXiv:2604.25850. https://arxiv.org/abs/2604.25850
5. Self-Harness: Harnesses That Improve Themselves. arXiv:2606.09498. https://arxiv.org/abs/2606.09498
6. Retrospective Harness Optimization. arXiv:2606.05922. https://arxiv.org/abs/2606.05922
7. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv:2507.19457. https://arxiv.org/abs/2507.19457
8. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. arXiv:2407.18901. https://arxiv.org/abs/2407.18901
9. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems. arXiv:2505.00212. https://arxiv.org/abs/2505.00212
