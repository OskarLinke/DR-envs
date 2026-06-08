import numpy as np
import polars as pl
from joblib import Parallel, delayed
from supply_chain import SupplyChain
from tqdm import tqdm
from const import (
    T,
    BS,
    MS,
    DATA_FOLDER,
    STAR_SAVE_NAME,
    NUM_ROB_EXPERIMENTS,
    ROBUSTNESS_SAVE_NAME,
    N_JOBS,
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
    save_path = DATA_FOLDER / STAR_SAVE_NAME
    solved_models = pl.read_parquet(save_path)
    # Find uniques, to avoid extra computations
    policies = list(set(tuple(p) for p in solved_models["Pi_star"].to_list()))

    tasks = [(pi, b, m) for pi in policies for b in BS for m in MS]

    rows = list(tqdm(
        Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(run_cell)(pi, b, m) for pi, b, m in tasks
        ),
        total=len(tasks),
        desc="cells",
    ))

    all_exps = pl.DataFrame(rows)
    all_exps = all_exps.sort(by=["b", "m"])
    all_exps.write_parquet(DATA_FOLDER / ROBUSTNESS_SAVE_NAME)
