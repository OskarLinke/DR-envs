# DR-envs

A small testbed of MDPs and reference algorithms for **Distributionally
Robust Reinforcement Learning**. Currently ships one environment
(`SupplyChain`, after Liu et al. 2022) and Distributionally Robust Value
Iteration (`DRVI`) against KL, TV, and chi-square ambiguity sets.

The goal is to make it cheap to reproduce the standard convergence,
sample-complexity, and out-of-distribution robustness diagnostics on a
shared environment, and to extend the suite with new environments and
algorithms without reworking the experiment harness.

## Repository layout

```
algos.py                  DRVI and (non-robust) VI.
supply_chain.py           SupplyChain environment.
utils.py                  Inner-adversary solver, empirical model
                          estimator, KL/TV/CHI_SQ helpers, radius
                          calibration.
const.py                  Experiment knobs (sigmas, sample counts,
                          parallelism, output paths).
my_typing.py              Shared type aliases.

get_optimal_per_config.py Solve V* / Q* / Pi* for every config.
run_convergence_exp.py    ||V_k - V*||_2 vs. iteration k.
run_num_samples_exp.py    ||V_K - V*||_2 vs. sample budget N.
run_robustness_exp.py     Realised cost of each policy under perturbed
                          market-ask distributions.
plotting.py               Plot the parquet outputs.

data/                     Parquet caches (auto-created).
plots/                    PNGs (auto-created by plotting.py).
main.tex                  Notes on the closed-form constraint gradients
                          for the DRVI inner adversary.
```

## Install

Requires Python >= 3.12. Dependencies are pinned in `pyproject.toml`
and `uv.lock`.

```bash
uv sync
source .venv/bin/activate
```

(Or use any other PEP 621-aware installer.)

## Run the experiments

The runners are caching: each one skips configs already in its parquet
output, so reruns only compute missing cells. Order matters because the
convergence and sample-complexity runners need `V*` produced by
`get_optimal_per_config.py`.

```bash
# 1. Reference solutions (V*, Q*, Pi*) per (metric, sigma) and the
#    non-robust baseline.
python get_optimal_per_config.py

# 2. Convergence: ||V_k - V*||_2 over iterations k.
python run_convergence_exp.py

# 3. Sample complexity: ||V_K - V*||_2 over sample budget N.
python run_num_samples_exp.py

# 4. Robustness: realised cost of each saved policy under perturbed
#    market-ask distributions.
python run_robustness_exp.py

# 5. Plots into plots/.
python plotting.py
```

Tweak `const.py` to change the swept distance metrics, sigmas, sample
budgets, number of repeats, or joblib worker count.

## Algorithm

`DRVI` (in `algos.py`) applies the robust Bellman operator

```
Q_{k+1}(s, a) = inf_{(P, r) in C_{s, a}^{sigma}} ( r + gamma * <P, V_k> )
V_{k+1}(s)   = max_a Q_{k+1}(s, a)
```

until `||Q_{k+1} - Q_k||_inf < tolerance` or `K` steps elapse. The
inner adversary `find_inf_kernel_reward` (in `utils.py`) is solved
either in closed form (TV, via water-filling, `worst_case_tv`) or with
SLSQP (KL, chi-square). When `learn_model=True` the nominal `(P, r)` is
replaced by an `N`-sample empirical estimate from
`empirical_nominal_kernels`. Extension to coupled (transition, reward)
distributions follows Panaganti & Kalathil (2023) and Shi et al.
(2025).
