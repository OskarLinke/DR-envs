"""Robustness experiment.

For each solved policy (loaded from ``STAR_SAVE_NAME``) and each
perturbation ``(b, m)`` of the market ask, roll out the policy in the
perturbed environment for ``NUM_ROB_EXPERIMENTS`` episodes of length
``T`` and record cost mean / std. Output: ``ROBUSTNESS_SAVE_NAME``.
"""

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from supply_chain import SupplyChain
from tqdm import tqdm
from const import (
    T,
    BS,
    MS,
    N_JOBS,
    DATA_FOLDER,
    TRUE_FOLDER,
    STAR_SAVE_NAME,
    NUM_ROB_EXPERIMENTS,
    ROBUSTNESS_SAVE_NAME,
)


def run_cell(pi: tuple, b: float, m: int) -> dict:
    env = SupplyChain(b=b, m=m)
    running_costs = np.zeros(NUM_ROB_EXPERIMENTS, dtype=np.float64)
    for round_idx in range(NUM_ROB_EXPERIMENTS):
        env.reset()
        acc_cost = 0
        for _ in range(T):
            action = pi[env.s]
            _, reward = env.step(action)
            acc_cost -= reward  # cost is -reward
        running_costs[round_idx] = acc_cost
    return {
        "policy": [pi],
        "running_cost": running_costs,
        "mean_cost": running_costs.mean(),
        "std_cost": running_costs.std(),
        "b": b,
        "m": m,
    }


if __name__ == "__main__":
    solved_save_path = TRUE_FOLDER / STAR_SAVE_NAME
    robustness_save_path = DATA_FOLDER / ROBUSTNESS_SAVE_NAME
    solved_models = pl.read_parquet(solved_save_path)
    # Find uniques, to avoid extra computations
    policies = list(set(tuple(p) for p in solved_models["Pi_star"].to_list()))

    existing_exps = None
    existing_exps_names = None
    if robustness_save_path.exists():
        existing_exps = pl.read_parquet(robustness_save_path)
        existing_exps_names = []
        for row in existing_exps.rows(named=True):
            existing_exps_names.append(
                str(row["policy"]) + "-" + str(row["b"]) + "-" + str(row["m"])
            )
        print(f"Found {len(existing_exps_names)} existing configs")

    tasks = []
    for pi in policies:
        for b in BS:
            for m in MS:
                name = str([list(pi)]) + "-" + str(b) + "-" + str(m)
                if existing_exps_names is not None and name in existing_exps_names:
                    print(f"Config (policy, b={b}, m={m}) already exists, skipping...")
                    continue
                tasks.append((pi, b, m))

    rows = list(tqdm(
        Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(run_cell)(pi, b, m) for pi, b, m in tasks
        ),
        total=len(tasks),
        desc="cells",
    ))

    all_exps_df = pl.DataFrame(rows) if rows else None

    if existing_exps is not None:
        if all_exps_df is not None:
            all_exps_df = pl.concat([all_exps_df, existing_exps], how="vertical")
        else:
            all_exps_df = existing_exps

    if all_exps_df is not None:
        all_exps_df = all_exps_df.sort(by=["b", "m"])
        all_exps_df.write_parquet(robustness_save_path)
    else:
        raise ValueError("No existing data and no computed results")
