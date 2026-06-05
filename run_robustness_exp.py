import numpy as np
import polars as pl
from algos import REVI, VI
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
)

save_path = DATA_FOLDER / STAR_SAVE_NAME
solved_models = pl.read_parquet(save_path)
# Find uniques, to avoid extra computations
policies = list(set(tuple(p) for p in solved_models["Pi_star"].to_list()))

# PERTUBE ENNVIRONMENTS AND DO STUFF
rows = []
running_costs = np.zeros(NUM_ROB_EXPERIMENTS, dtype=np.float64)
for pi in tqdm(policies, desc="Policy Pi...", position=0, leave=True):
    for b in tqdm(BS, desc="b...", position=1, leave=False):
        for m in tqdm(MS, desc="m...", position=2, leave=False):
            env = SupplyChain(b=b, m=m)
            for round in range(NUM_ROB_EXPERIMENTS):
                env.reset()
                acc_cost = 0
                for t in range(T):
                    action = pi[env.s]
                    _, reward = env.step(action)
                    acc_cost -= reward # cost is -reward
                running_costs[round] = acc_cost
            rows.append({
                "policy": [pi],
                "running_cost": running_costs.copy(),
                "mean_cost": running_costs.mean(),
                "std_cost": running_costs.std(),
                "b": b,
                "m": m,
            })
            running_costs.fill(np.float64(0))

breakpoint()
all_exps = pl.DataFrame(rows)
all_exps = all_exps.sort(by=["b", "m"])
all_exps.write_parquet(DATA_FOLDER / ROBUSTNESS_SAVE_NAME)
