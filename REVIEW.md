# Code Review: REVI Distributionally Robust RL

Reviewed files:
- `revi.py` — REVI algorithm + plain VI.
- `utils.py` — `find_inf_market_dist` (SLSQP), `find_inf_P` (unused), `empirical_nominal_kernel`, Bellman operators.
- `supply_chain.py` — Liu et al. (2022) supply chain MDP.

## Strengths

- Clean separation: env / algorithm / helpers.
- Parameterising adversary over `md` (length `n+1`) rather than over the full `P` (length `S`) is a real dimensionality win and is mathematically clean.
- Defining the ambiguity ball on the induced kernel `P(md)` rather than on `md` itself (`utils.py:60-62`) is the right modelling choice.
- Convergence check with infinity-norm tolerance + mask for `-inf` entries is sensible (`revi.py:56-57`).
- Analytic `true_nominal_kernel` (`supply_chain.py:169-191`) provides a clean baseline for comparison.

## Issues

### Critical (Must Fix)

#### 1. `utils.py:77` — KL formula adds `eps` to both numerator AND denominator

```python
kl = np.sum(p_ * np.log((p_ + 1e-12) / (nom_P + 1e-12)))
```

When `nom_P[i] = 0` and `p_[i] > 0`, true KL is `+inf` (forbidden move). Current formula gives finite `p * log(p/eps)`, large but bounded. Adversary can move mass into transitions with zero nominal probability. Because supply-chain `nom_P` is sparse, this loosens the ambiguity ball substantially. REVI is less robust than intended.

Fix:
```python
mask = p_ > 0
kl = np.sum(p_[mask] * np.log(p_[mask] / np.maximum(nom_P[mask], 1e-12)))
```

#### 2. `supply_chain.py:59` — `reward()` multiplies by `np.power(self.gamma, self.t)`

VI/REVI already discount inside the Bellman operator via `gamma * P^T V`. Multiplying the reward by `gamma**t` would double-discount.

Not currently biting because `expected_reward_sa` (line 152) and `nominal_expected_reward` (line 137) call `self.reset()` at the end (setting `t=0`), and `t` is never incremented except in `step()`. So `self.t == 0` during VI/REVI and `gamma**0 = 1`.

Future-bug magnet: any flow that calls `step()` then `expected_reward_sa()` without an explicit reset will silently corrupt rewards. Drop the `gamma**t` factor; discounting is the planner's job.

#### 3. `supply_chain.py:152` (also `:137`) — helpers mutate `self.S` and `self.t` mid-computation

`expected_reward_sa` calls `self.reset()` at the end, clobbering `self.S`. Called S·A·K times from the SLSQP objective. Survives now because `legal_actions()` is read once per `(s,a)` *before* the SLSQP loop starts, but fragile: any future `legal_actions()` check inside the inner loop will be wrong.

Fix: don't mutate state from helpers that take `s` as an argument. Make `reward` pure (`reward(s, a, dt)`), drop `self.reset()` from both helpers.

### Important (Should Fix)

#### 4. SLSQP is wrong tool — inner problem is convex with linear objective

`r(s,a,md)` and `P(s,a,md)` are both linear in `md`. So:

```
objective(md) = r_sa @ md + gamma * V @ (M_sa @ md)
              = (r_sa + gamma * M_sa.T @ V) @ md       # linear in md
```

Constraints:
- `sum(md) = 1` (linear)
- `md ∈ [0,1]^(n+1)` (linear)
- KL or L2 ball on `M_sa @ md` (convex)

This is an LP plus one convex constraint. Three replacement options:

**(a) Precompute linear maps** once per `(s,a)`:

```python
M_sa = np.zeros((S, n+1))  # M_sa[:, dt] = e_{max(s+a-dt, 0)}
r_sa = np.zeros(n+1)       # r_sa[dt]    = reward(s, a, dt)
```

Then objective collapses to a single inner product. ~10× speedup with no other change. Currently `expected_reward_sa` and `trans_prob_kernel_sa` rebuild from scratch on every SLSQP objective evaluation.

**(b) CVXPY + Clarabel/ECOS** — native exponential-cone solver:

```python
md   = cp.Variable(n+1, nonneg=True)
P    = M_sa @ md
prob = cp.Problem(
    cp.Minimize((r_sa + gamma * M_sa.T @ V_param.value) @ md),
    [cp.sum(md) == 1, cp.sum(cp.kl_div(P, nom_P)) <= sigma],
)
```

Use `cp.Parameter` for `V` so each `(s,a)` problem is built once and re-solved cheaply across Bellman iterations. Exact `0 log 0`, no `eps` hacks, automatic simplex projection.

**(c) Stay in scipy → `trust-constr`** instead of SLSQP. Interior-point-ish, handles nonlinear constraints far better than SLSQP active-set near the KL boundary. No new dep but still inferior to a true conic solver.

#### 5. `utils.py:102` — `return result.x` doesn't project onto simplex despite the comment

SLSQP routinely violates constraints by 1e-6 to 1e-4. `inf_P = trans_kernel_func(s, a, inf_md)` (`revi.py:50`) then doesn't sum to 1. Bias propagates into `Q_k`.

```python
x = np.clip(result.x, 0, 1)
return x / x.sum()
```

#### 6. `utils.py:105-141` — `find_inf_P` is dead code and broken

Not imported anywhere. `x0=P_hat` would receive the full `(S,A,S)` tensor at the only conceivable call site, not a vector. Delete.

#### 7. `utils.py:29-33` — `bellman_operator` and `robust_bellman_operator` are byte-identical

Drop `robust_bellman_operator`; "robust" lives in the choice of `inf_P`, not in the operator.

#### 8. `revi.py:33-54` — hot loop has avoidable overhead

- `env.reset(s=s)` inside the `(s,a)` loop is only needed because `legal_actions()` reads `self.S`. Replace with `if s + a > env.n` and drop the reset.
- Each `find_inf_market_dist` call rebuilds `P(md)` and `r(md)` via Python loops over `dt` on every SLSQP iteration. See fix (a) above.

#### 9. `supply_chain.py:78` TODO — peaks at `m`, `m+1`

Math sums to 1 (verified for `b=0` and `b=1`, `n=10`). Whether peaks at `m, m+1` is the right modelling choice is a separate question, not a bug. Probabilities are valid.

### Minor (Nice to Have)

#### 10. `supply_chain.py:46` — `self.dt` doubles as live demand and scratch variable

The pattern `self.dt = n; self.reward(s, a)` in `expected_reward_sa` / `nominal_expected_reward` is the root cause of the `self.t` / `self.reset()` mess (issues 2 & 3). Replace with a pure `reward(self, s, a, dt) -> float`.

#### 11. `revi.py:38` — `legal_actions()` returns max-legal-**index**, not count

`action_size()` returns count `n+1`. `legal_actions()` returns max legal action `n - S`. Naming is inconsistent. Rename to `max_legal_action()` or change semantics.

#### 12. `supply_chain.py:103` — `legal_actions` docstring says "size of action space" but returns max legal action

#### 13. `revi.py:54` — `np.max(Q_k, axis=1)` over rows with `-inf` entries

Safe because every state has at least one legal action (`a=0` is always legal: `0 > n-s` is false for `s ≤ n`). Worth a comment.

#### 14. `utils.py:6-27` — `empirical_nominal_kernel` reuse is safe

`P_hat[s, a] = running_counts` is slice-assignment (copy), so the subsequent `running_counts.fill(0)` doesn't leak. Correct.

## Recommendations

Priority order:

1. **Fix KL `eps`** (`utils.py:77`) — only real math bug.
2. **Project to simplex** on return (`utils.py:102`).
3. **Precompute `M_sa`, `r_sa`** per `(s,a)` — biggest performance win, ~10×.
4. **Drop `gamma**t`** from `reward()` and **make `reward(s, a, dt)` pure** — kills issues 2, 3, 10 in one move.
5. **Delete `find_inf_P`**, dedupe Bellman operator.
6. **CVXPY + Clarabel** rewrite of the inner adversary for exactness and further speed. Use `cp.Parameter` for `V` and memoise the `Problem` per `(s,a)`.

## Suggested Adversary Reformulation (CVXPY)

```python
import cvxpy as cp
import numpy as np

def build_adversary(s, a, env, sigma, nom_md, metric="KL"):
    n = env.n
    S = env.state_size()
    M = np.zeros((S, n+1))
    r_vec = np.zeros(n+1)
    for dt in range(n+1):
        sp = max(s + a - dt, 0)
        M[sp, dt] += 1.0
        r_vec[dt] = reward_pure(env, s, a, dt)
    nom_P = M @ nom_md

    md = cp.Variable(n+1, nonneg=True)
    V_param = cp.Parameter(S)
    P  = M @ md
    obj = cp.Minimize(r_vec @ md + env.gamma * V_param @ P)
    constraints = [cp.sum(md) == 1]
    if metric == "KL":
        constraints.append(cp.sum(cp.kl_div(P, nom_P)) <= sigma)
    elif metric == "L2":
        constraints.append(cp.norm(P - nom_P, 2) <= sigma)
    return cp.Problem(obj, constraints), md, V_param
```

Build once per `(s,a)` outside the Bellman loop. Inside, update `V_param.value = V_k` and call `prob.solve(solver=cp.CLARABEL)`. Read `md.value`.

## Why Not scipy for the CVXPY Replacement

scipy.optimize has no exponential-cone solver. The exact KL formulation requires exp-cone support (Clarabel, ECOS, SCS, MOSEK). Within scipy, `trust-constr` is the best available upgrade over SLSQP — handles nonlinear constraints better near the KL boundary but still inferior to a true conic solver.

## Assessment

**Ready to merge?** With fixes.

**Reasoning:** KL `eps` (issue 1) is the only thing that meaningfully distorts the math. Everything else is performance, dead code, or footguns. Biggest research-relevant win is exploiting linearity of the inner problem (precompute `M_sa`, `r_sa`), which gives an order-of-magnitude speedup for negligible code complexity.
