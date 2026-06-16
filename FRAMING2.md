# Project Framing v2: Adaptive-Granularity Loop Repair for Self-Evolving LLM Agents

> Alternative to `FRAMING.md`. The original "we propose another self-evolving harness pipeline" framing is now largely subsumed by HarnessFix (2606.06324), AHE (2604.25850), and Self-Harness (2606.09498). This version is built around two things those papers structurally cannot claim, because each fixes a single repair granularity:
> 1. **A method** — a failure-type-conditioned router that *chooses* repair granularity per failure and beats every fixed-granularity baseline, including HarnessFix-style layer repair.
> 2. **A generality finding** — the optimal repair granularity *shifts with the executing agent's capability*, which operationalizes "Harness Updating Is Not Harness Benefit" (2605.30621) instead of merely citing it.
>
> The controlled granularity study from the earlier draft is retained, but demoted to the **control condition** that the method and the generality result are built on top of.

---

## 0. Working Title

**GrainRoute: Adaptive-Granularity Loop Repair for Self-Evolving LLM Agents**

Alternatives:

- **The Right Repair Is the Right Size: Failure-Conditioned Granularity Selection for LLM Agents**
- **Routing Repairs by Granularity: Beyond Fixed-Layer Harness Evolution**

---

## 1. Motivation and the Precise Gap

Methods that turn failed traces into harness/prompt updates already exist; they differ mainly in **how finely they localize the failure before repairing it**, and each commits to *one* fixed granularity:

- **Global** — reflect over the whole trajectory, update a prompt/harness wholesale (GEPA, ReAct+Reflection).
- **Functional-layer** — attribute a step to a component layer, apply a layer-scoped operator. **HarnessFix** (seven ETCLOVG layers + scoped operators + regression-bounded validation), **AHE**, **Self-Harness**.
- **Exact-step** — attribute to a single responsible step via counterfactual intervention, repair with minimal drift. **CausalFlow** (2605.25338) argues *finer/causal* attribution is necessary.

Two unexploited facts follow:

1. **The field implicitly disagrees on the right granularity** — HarnessFix says *layer*, CausalFlow says *step*, GEPA stays *global* — and **no method adapts granularity to the failure**. If different failures are best repaired at different granularities, every fixed-granularity method is leaving performance on the table by construction.

2. **"Harness Updating Is Not Harness Benefit"** (2605.30621) shows the realized benefit of an update depends non-monotonically on the *executing* agent's ability to activate and follow it. This implies the *best granularity may move with the executor* — a finer, more specific repair only pays off if the executor can act on it. No prior work tests this.

**The gap, precisely:** there is no method that selects repair granularity per failure, and no evidence on how the optimal granularity depends on the executor. This project delivers both, on top of a controlled apparatus that holds the executor, operator library, and validation gate fixed so that granularity is the only thing that varies.

---

## 2. Core Research Question and Claims

> Can a repair system that **selects attribution granularity per failure** outperform any fixed-granularity method, and how does the optimal granularity depend on the failure type and on the executing agent's capability?

Three claims, in priority order:

- **C1 (Method).** A failure-type-conditioned granularity **router** beats the best *fixed* granularity — including HarnessFix-style layer repair — on held-out tasks, under identical executor, operator library, and validation gate.
- **C2 (Generality).** The optimal fixed granularity is **not constant across executors**: it shifts with model capability, and finer granularity can *hurt* weaker executors that cannot act on specific repairs. This is the headline scientific finding.
- **C3 (Mechanism).** The advantage decomposes into *attribution precision* vs *executor utilization*; the router wins primarily by avoiding granularities the executor cannot exploit, not merely by attributing more accurately.

---

## 3. Hypotheses

- **H1 (Type-conditioned optimum).** No single granularity is best across failure types. Mechanistic, locally-caused failures (bad tool args, invalid schema) favor finer attribution; diffuse, upstream-caused failures (wrong plan, premature termination) favor coarser attribution.
- **H2 (Capability-dependent optimum — the surprise).** The optimal granularity shifts with executor capability. Strong executors benefit from fine, specific repairs; **weak executors do worse with fine repairs** because they fail to activate/follow specific patches — so a coarser, more prescriptive repair helps them more. The granularity ranking therefore *inverts* across capability tiers.
- **H3 (Router dominance).** A router conditioned on (failure type, executor) beats every fixed granularity, and the gap is largest where H1/H2 predict the most disagreement among fixed granularities.
- **H4 (Mechanism).** The router's gain is mediated more by *executor utilization* (acting on the repair) than by raw *attribution accuracy* — i.e. it wins by matching repair specificity to what the executor can use.

If H2 holds it is genuinely non-obvious and citable on its own: *the right repair granularity depends on the model that has to use the repair.*

---

## 4. Relation to Existing Work (sharpened)

Prior methods become **named fixed-granularity baselines** that the router is measured against.

| Granularity | Representative prior work | Repair unit | Role here |
|---|---|---|---|
| Global | GEPA (2507.19457), ReAct+Reflection | whole prompt/harness | fixed baseline |
| Functional-layer | **HarnessFix (2606.06324)**, AHE (2604.25850), Self-Harness (2606.09498) | ETCLOVG component layer | **strongest fixed baseline** |
| Temporal-phase | this work (a contestant *and* a router input) | (component, loop-phase) | fixed baseline + router signal |
| Exact-step | **CausalFlow (2605.25338)**, AgenTracer | single responsible step | fixed baseline |

Differences from the closest neighbor **HarnessFix**:

1. HarnessFix commits to step→layer granularity. GrainRoute **chooses granularity per failure**; HarnessFix is recovered as the router's fixed-layer special case.
2. HarnessFix reports single-executor held-out gains. GrainRoute reports **how the optimal granularity moves across executors** — a result HarnessFix cannot produce without abandoning its fixed granularity.
3. HarnessFix's layers are static functional roles. GrainRoute adds **temporal phase** as a router input and tests whether functional-only attribution mislocates time-dependent failures.

From **CausalFlow**: it asserts finer/causal is necessary; GrainRoute **tests when finer pays off and when it backfires** (H2), rather than assuming it universally.

From **"Harness Updating Is Not Harness Benefit"** (2605.30621): that paper disentangles update-*production* from update-*utilization* across capability tiers. GrainRoute applies the same decomposition to the **granularity** axis and turns the confound into the C2/H2 finding.

---

## 5. Formal Setup

Agent loop `C = {P, T, R, M, V, S}`; run produces trace `τ`. Define an **attribution function at granularity `g`**:

```text
A_g : τ -> u_g          g ∈ { global, layer, phase, step }
A_global(τ) = τ          A_layer(τ) = ℓ ∈ ETCLOVG
A_phase(τ)  = (ℓ, π)     A_step(τ)  = t* ∈ {1..T}
```

A **shared repair-operator library** `O` maps a repair unit to a minimal patch: `ρ(u_g, τ, O) -> C'`. The executor, `O`, and the acceptance gate are identical across all `g`.

**The method — a granularity router.** Given a failure signature `s(τ)` (failure-type features) and an executor descriptor `e` (capability tier / family), learn a policy

```text
π_route : (s(τ), e) -> g* ∈ { global, layer, phase, step }
```

that selects the granularity whose repair maximizes expected validated improvement:

```text
g* = argmax_g  E[ J_val( accept( ρ(A_g(τ), τ, O) ) ) ]
```

Acceptance is the same regression-safe gate for every `g`: `J_val(C') > J_val(C) + ε` and `ΔRegression ≤ δ`. The router is trained on held-in (failure, executor, outcome-per-granularity) tuples and evaluated on held-out tasks/executors. Fixed-granularity baselines are the degenerate policies `π ≡ g`.

---

## 6. The Controlled Apparatus (foundation, held constant)

| Held constant across all granularities & the router | Why |
|---|---|
| **Repair-operator library `O`** | a difference must come from *which* operator the granularity selects, not a richer toolbox |
| **Validation gate** (`ε`, `δ`, splits) | identical acceptance bar → comparable regression behavior |
| **Trace instrumentation & contracts** | every granularity sees the same evidence |
| **Repair-LLM budget** | finer attribution cannot win by spending more tokens |

What **varies on purpose**: (a) the attribution granularity `A_g` (the axis), and (b) the **executor** — varied as a *primary experimental factor*, not an ablation, to establish C2/H2.

---

## 7. Granularity Levels and the Router

Levels under test: **Global (G0)**, **Layer (G1, HarnessFix-like)**, **Phase (G2)**, **Step (G3, CausalFlow-like)** — defined as in the earlier draft (G2 = (component, loop-phase) with position and recurrence buckets).

**Router variants** (increasing ambition):

- **R-type** — conditions on failure type only (tests H1/H3).
- **R-full** — conditions on (failure type, executor descriptor) (tests H2/H3; this is the headline method).
- **R-oracle** — uses ground-truth best granularity per failure (upper bound; isolates router error from ceiling).

The headline claim is **R-full > best fixed granularity (incl. G1/HarnessFix) on held-out tasks and held-out executors.**

---

## 8. Shared Repair-Operator Library

Fixed catalogue reused verbatim by every granularity and the router: tool-schema narrowing / argument validator / tool-menu re-ranking / error-message rewrite; failure-tail evidence preservation / retrieval re-scoring / write filter; loop guard / retry bound / no-progress detector / verification-gated finalization; readiness check / intermediate validation gate / grounding contract; constraint-extraction checklist / decomposition template. Granularity determines **which operators are eligible** and **how tightly scoped** the patch is — nothing else.

---

## 9. Evaluation Plan

- **Main benchmark: AppWorld** (90:45:90 split, matching HarnessFix for direct comparability). Exercises all components.
- **Multiple executors (primary factor): ≥3 model families across ≥2 capability tiers** (e.g. a strong frontier model, a mid-tier, a small open model). This is where C2/H2 lives — single-executor results are explicitly insufficient.
- **GrainBench (diagnostic, construct-valid):** 150–300 failures with ground truth for (component, temporal phase, responsible step). To avoid the self-labeling attack:
  - Prefer **deterministically injected faults** (corrupt a tool schema, force a stale memory, cap retries) so the true responsible unit is *known by construction*, not annotated.
  - For naturally-occurring failures, anchor labels to **environment-checkable state assertions** where possible; report **inter-annotator agreement** on the residual and treat attribution accuracy as *bounded*, not exact.
  - Include ≥20% **time-dependent failures** (same component, different loop position/retry) to test whether G1 mislocates relative to G2.
- **Transfer: GAIA or a Terminal-Bench subset** — confirm the granularity ranking and router advantage are not AppWorld-specific.

Agent evolves only on held-in; all headline numbers are held-out task **and** held-out executor.

---

## 10. Baselines / Conditions

- **No-repair control** (`C₀` frozen) — floor.
- **Fixed G0 / G1 / G2 / G3** — G1 is the HarnessFix-style strongest fixed baseline.
- **R-type, R-full** — the proposed routers.
- **R-oracle** — router upper bound.
- **Oracle attribution at each fixed `g`** — feed GrainBench ground truth to separate *attribution error* from *operator/executor ceiling*.

The decisive comparisons: **R-full vs best fixed `g`** (C1/H3), and **how argmax_g shifts across executors** (C2/H2).

---

## 11. Metrics

**Attribution quality (per granularity, needs GrainBench labels):** attribution accuracy, repair-localization precision/recall, over-edit rate.

**Repair outcome:** held-out task success; process-level (token cost, tool-call count, termination-failure rate, contract compliance, tool-loop rate, recovery success); **P-FRR** (phase-specific failure recurrence); regression rate.

**Router-specific:** router accuracy vs R-oracle; **net gain of R-full over best fixed `g`**; regret per failure type.

**Mechanism decomposition (C3/H4):**

- Gain from *attribution precision* = (oracle-`g` − realized-`g`).
- Gain from *executor utilization* = patch acceptance-and-follow rate under fixed executor.
- A mediation/intervention table attributing the router's win between the two — done via *intervention* (force a granularity, measure utilization) rather than hand-wavy gap subtraction.

**Headline figures:** (1) R-full vs all fixed granularities, per benchmark; (2) **argmax_g vs executor capability** — the curve that shows the optimum shifting (and inverting) across tiers.

---

## 12. Ablations

1. **Oracle vs realized attribution** at each granularity (isolates attribution noise).
2. **Executor swap** across capability tiers — *promoted to main result* (C2/H2), retained here as the formal ablation that the optimum moves.
3. **Router feature ablation** — type-only vs type+executor vs +trace-features (what the router actually needs).
4. **Operator-set held vs enriched** (gains aren't from a bigger toolbox at finer levels).
5. **Phase definition variants** for G2 (position-only / recurrence-only / both).
6. **Validation gate on/off** (does finer granularity overfit single traces when unguarded?).
7. **Cost-matched** comparison (equal repair-LLM tokens across granularities).

Most important: #1 + #2 + #3 together convert "the router wins" into "*why and when* it wins."

---

## 13. Expected Contributions

1. **GrainRoute**, a failure-type- and executor-conditioned granularity **router** for loop repair, shown to beat every fixed-granularity method including HarnessFix-style layer repair under identical executor/operators/validation.
2. The **first evidence that optimal repair granularity shifts with executor capability** (and can invert across tiers) — operationalizing the "harness-updating ≠ harness-benefit" effect for the granularity axis.
3. A **controlled apparatus** that varies only attribution granularity, enabling the above to be measured rather than confounded.
4. **GrainBench**, a construct-valid, fault-injected, partially environment-checked failure dataset with component/phase/step ground truth, including time-dependent failures.
5. A **mechanism decomposition** (attribution precision vs executor utilization) showing *why* the router wins — a result no fixed-granularity pipeline isolates.

### Why this clears a top-venue bar
It is a **method with a non-obvious, generalizable finding**, not a measurement note. The router gives reviewers a tool; the capability-shift finding gives them a surprise; the controlled apparatus + fault-injected benchmark give them rigor. HarnessFix/CausalFlow cannot make claims (1)–(2) without abandoning their fixed granularity.

---

## 14. Minimum Viable Prototype

- **Week 1.** Controlled apparatus: one executor `C₀`, shared operator library `O`, regression-safe gate, JSONL tracing, and the four swappable `A_g`.
- **Week 2.** GrainBench v0: 60–80 **fault-injected** failures (known ground truth) + ≥15 time-dependent cases; automatic checkers.
- **Week 3.** Run fixed G0–G3 + oracle on GrainBench v0 on **two** executors (one strong, one weak). First test of H2: does argmax_g differ between them? *This is the early go/no-go.*
- **Week 4.** Train R-type / R-full; show R-full ≥ best fixed `g`. If H2 signal + router advantage hold, scale to AppWorld (90:45:90), full executor panel, and the mechanism decomposition.

Go/no-go at end of Week 3: **if argmax_g is identical across the strong and weak executor and the fixed-granularity curves are flat, the central novelty (C2) is absent** — pivot to publishing the apparatus + null as a focused finding, or stop.

---

## 15. Risks and Mitigations

- **R1: The router barely beats the best fixed granularity.** Mitigation: report per-failure-type regret; even a *small* mean gain is publishable if it concentrates on an identifiable failure class. Lead with C2 (the capability-shift finding) which does not depend on a large router margin.
- **R2: argmax_g does not move across executors (H2 fails).** Mitigation: this collapses the headline; the Week-3 go/no-go exists precisely to detect it early. Fallback: the controlled apparatus + type-conditioned router (C1/H1) is still a workshop-to-borderline contribution.
- **R3: GrainBench label validity.** Mitigation: fault-injection gives ground truth by construction for the core set; IAA bounds + environment checks for the rest; attribution metrics reported as bounded.
- **R4: Results are AppWorld-specific.** Mitigation: the transfer set, and reporting the *ranking/shift*, not just absolute numbers.
- **R5: Executor confound leaks back in.** Mitigation: executor is varied as a *declared factor*; everything else in the apparatus is frozen.
- **R6: Mechanism decomposition is contested.** Mitigation: do it by *intervention* (force granularity, measure utilization), not regression on observational gaps.

---

## 16. One-Sentence Summary

GrainRoute selects failure repair **granularity** per failure and per executor — global, functional-layer, temporal-phase, or exact-step — over a fixed executor, operator library, and validation gate; it beats every fixed-granularity method including HarnessFix-style layer repair, and shows that the optimal repair granularity *shifts with the executing agent's capability*, turning the "harness-updating ≠ harness-benefit" confound into the paper's central finding.

---

## 17. References to Track

1. From Failed Trajectories to Reliable LLM Agents (HarnessFix). arXiv:2606.06324. https://arxiv.org/abs/2606.06324
2. CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures. arXiv:2605.25338. https://arxiv.org/abs/2605.25338
3. Harness Updating Is Not Harness Benefit. arXiv:2605.30621. https://arxiv.org/abs/2605.30621
4. Agentic Harness Engineering (AHE). arXiv:2604.25850. https://arxiv.org/abs/2604.25850
5. Self-Harness: Harnesses That Improve Themselves. arXiv:2606.09498. https://arxiv.org/abs/2606.09498
6. Retrospective Harness Optimization. arXiv:2606.05922. https://arxiv.org/abs/2606.05922
7. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv:2507.19457. https://arxiv.org/abs/2507.19457
8. AppWorld: A Controllable World of Apps and People. arXiv:2407.18901. https://arxiv.org/abs/2407.18901
9. Which Agent Causes Task Failures and When? arXiv:2505.00212. https://arxiv.org/abs/2505.00212
