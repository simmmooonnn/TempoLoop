# GrainRoute: Router Training Objective and Pseudocode

> Design doc for the granularity router in `FRAMING2.md`. The router selects, per failure and per executor, which attribution granularity `g ∈ {global, layer, phase, step}` to use for repair. Training is **offline with full feedback**: during data collection we run *all four* granularities through the identical apparatus, so for every failure we observe the validated outcome of every granularity. That makes this a **cost-sensitive classification / off-policy full-information** problem, not a bandit — we never have to explore at deployment.

---

## 1. Data Collection

For each failure instance `i` (a GrainBench instance or an AppWorld held-in failure) and its executor `e_i`, run the full repair-and-validate pipeline once per granularity:

```python
for i in failures:
    s_i = featurize(i.trace)          # failure signature (Section 3)
    e_i = describe(i.executor)        # executor descriptor (Section 3)
    for g in [GLOBAL, LAYER, PHASE, STEP]:
        patch   = rho(A[g](i.trace), i.trace, OPERATORS)   # shared O, fixed budget
        outcome = validate(patch, i, gate)                 # same regression-safe gate ∀ g
        r_i[g]  = reward(outcome)                           # Section 2
    dataset.append((s_i, e_i, r_i))   # r_i ∈ R^4 : full feedback over granularities
```

Everything except `A[g]` is frozen across the inner loop (executor, `OPERATORS`, repair-LLM budget, `gate`) — the apparatus guarantee from `FRAMING2.md` §6. Result: a dataset of `(features, executor, reward-vector)` triples.

---

## 2. Reward

The reward of applying granularity `g` to failure `i` is the **validated, regression-penalized improvement** the gate would accept:

```text
r_i[g] =  1[accepted_g] · ( ΔSuccess_g
                            − λ1·ΔCost_g
                            − λ2·ΔToolLoop_g
                            − λ3·ΔContractViol_g )
          − μ · Regression_g
```

- `ΔSuccess_g`: held-out task-success delta from the patch (primary).
- process terms mirror `J` in `FRAMING.md` §6 (cost, tool-loops, contract violations).
- `Regression_g`: newly-broken previously-passing tasks; `μ` large so regressive patches score below "do nothing".
- `1[accepted_g]`: if the gate rejects the patch, only the regression penalty applies (a rejected repair ≈ no-op with wasted budget).

`r_i` is a length-4 vector; we know it **fully** (all four entries) at training time. The per-instance **oracle** is `g*_i = argmax_g r_i[g]`; the **regret** of choosing `ĝ` is `r_i[g*_i] − r_i[ĝ]`.

---

## 3. Features

**Failure signature `s(τ)`** — cheap, trace-derived, no LLM call:

| Group | Features |
|---|---|
| Error profile | dominant error code, #distinct error codes, has `invalid_argument`/`permission_denied`/`rate_limited` |
| Repetition | max identical-(tool,args) run length, #retries, retry-success ratio |
| Locality | step index of first error / `T`, span between first and last error, #components touched |
| Tool | #distinct tools, wrong-tool indicator (tool called but never re-used) |
| Memory | #memory reads, provenance-link-to-final flag, read/write conflict flag |
| Verification | final-answer-grounded flag, #unsupported claims |
| Termination | budget-hit flag, no-progress-tail length, premature-stop flag |
| Cost | tokens, #steps, #tool calls |

**Executor descriptor `e`** — the input that makes H2 testable:

```text
e = [ family_onehot,  tier_ordinal,  base_pass_rate,  mean_instruction_follow_score ]
```

`base_pass_rate` (clean rollout pass rate) and an instruction-following probe are the features expected to carry the capability-shift signal: weaker executors should route away from fine granularities.

---

## 4. Objective

Because feedback is full (we know `r_i[g]` for all `g`), train by **directly maximizing expected validated reward** under the router's distribution — no REINFORCE/importance weighting needed.

Let `f_θ(s,e) ∈ R^4` be logits, `p_θ = softmax(f_θ / T)`.

### Primary loss — expected-reward (cost-sensitive)

```text
L_reward(θ) = − (1/N) Σ_i  Σ_g  p_θ(g | s_i, e_i) · r_i[g]
              − β · H(p_θ(·|s_i,e_i))          # entropy bonus, avoids early collapse
```

Minimizing `L_reward` pushes mass onto high-reward granularities; the entropy term `H` keeps exploration during training. At deployment, act greedily: `ĝ = argmax_g f_θ(s,e)`.

### Auxiliary loss — reward regression (calibration + interpretability)

Predict the reward vector directly with a head `r̂_θ(s,e) ∈ R^4`:

```text
L_reg(θ) = (1/N) Σ_i  || r̂_θ(s_i,e_i) − r_i ||²
```

This gives a calibrated estimate of each granularity's payoff (useful for the R-oracle gap and for abstaining when all granularities look bad), and a second way to select: `ĝ = argmax_g r̂_θ`.

### Total

```text
L(θ) = L_reward(θ) + α · L_reg(θ)
```

### Regret-weighting (optional, focuses capacity where choice matters)

Weight each instance by how much the granularity choice matters, so the router spends capacity on decisive failures instead of ties:

```text
w_i = max_g r_i[g] − mean_g r_i[g]
L_reward(θ) = − (1/Σw) Σ_i w_i Σ_g p_θ(g|·) r_i[g]
```

---

## 5. Pseudocode (end to end)

```python
# ---- training ----
def train_router(dataset, epochs, T=1.0, alpha=0.5, beta=0.01, lr=1e-3):
    theta = init_model(in_dim=feat_dim, out_heads={"logits":4, "reward":4})
    opt = Adam(theta, lr)
    for _ in range(epochs):
        for (s, e, r) in batches(dataset):          # r: [B,4] full-feedback rewards
            x = concat(s, e)                          # [B, feat_dim]
            logits, r_hat = theta(x)                  # [B,4], [B,4]
            p = softmax(logits / T)                   # [B,4]
            w = (r.max(1) - r.mean(1)).detach()       # regret weight (optional)
            L_reward = -(w[:,None] * p * r).sum(1).mean() - beta * entropy(p).mean()
            L_reg    = ((r_hat - r) ** 2).mean()
            (L_reward + alpha * L_reg).backward()
            opt.step(); opt.zero_grad()
    return theta

# ---- deployment ----
def route_and_repair(trace, executor, theta, A, rho, OPERATORS, gate):
    s, e = featurize(trace), describe(executor)
    logits, _ = theta(concat(s, e))
    g = argmax(logits)                                # chosen granularity
    patch = rho(A[g](trace), trace, OPERATORS)
    return validate(patch, gate), g                   # same gate as every baseline
```

---

## 6. Variants Evaluated (maps to `FRAMING2.md` §7, §10)

| Variant | Input | Tests |
|---|---|---|
| **Fixed-`g`** | none (constant policy) | granularity baselines incl. HarnessFix≈Fixed-Layer |
| **R-type** | `s(τ)` only | H1 / H3 — does failure type alone suffice? |
| **R-full** | `s(τ)` + `e` | **headline**; H2 — does adding executor flip choices across tiers? |
| **R-oracle** | ground-truth `g*_i` | upper bound; router-error vs ceiling |

**Decisive read for H2:** train R-full, then inspect `argmax_g f_θ(s, e)` as `e` sweeps from strong→weak with `s` held fixed. If the chosen granularity **shifts coarser as the executor weakens**, H2 is supported directly in the learned policy, independent of end-task numbers.

---

## 7. Avoiding Leakage & Overfitting

- **Held-out executors.** The executor panel is split; R-full is evaluated on executors unseen in training so it cannot memorize per-model quirks — it must learn a *capability→granularity* mapping. This is what makes C2 a generalization claim, not a lookup table.
- **Held-out tasks.** Standard task split; the failure signature must generalize across tasks of the same fault family.
- **No reward leakage into features.** `s(τ)` is computed from the *failed* trace only, before any repair; `r_i` (the label) never appears in features.
- **Class balance.** Stratify by fault family and by `g*_i` so no single granularity dominates the label distribution (else the router degenerates to a constant ≈ best fixed-`g`, which is exactly the baseline it must beat).

---

## 8. Success Criteria

1. **C1:** R-full's mean validated reward (held-out task + held-out executor) exceeds the best Fixed-`g`, incl. Fixed-Layer (HarnessFix proxy), by a margin that survives bootstrap CIs.
2. **C2/H2:** the learned `argmax_g` shifts monotonically coarser as `base_pass_rate` drops; fixed-fine granularity underperforms fixed-coarse on the weak-executor slice.
3. **C3/H4:** the router's advantage correlates with the *utilization* gap (acceptance-and-follow rate), not with raw attribution accuracy — established by the intervention table (force `g`, measure utilization) from `FRAMING2.md` §11.
4. **Router quality:** R-full closes ≥X% of the R-oracle − best-Fixed gap.

If (1) holds but (2) fails, the paper is the type-conditioned router (still a contribution). If (2) holds even when (1) is marginal, the capability-shift finding carries the paper. Both failing at the Week-3 go/no-go triggers the pivot in `FRAMING2.md` §14.
