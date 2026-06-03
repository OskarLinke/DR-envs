import numpy as np
import polars as pl
import time
from supply_chain import SupplyChain
from algos import REVI, VI
from const import (
    SIGMAS,
    DISTANCE_METRICS,
    MAX_ITER_K,
    DATA_FOLDER,
)

nominal_env = SupplyChain(b=0)
nom_md = nominal_env.market_ask_distribution()

rows = []

start_time = time.time()
current_time = time.time()
for sigma in SIGMAS:
    for dm in DISTANCE_METRICS:
        Q_K, V_K, _, evals = REVI(
            env=nominal_env, md_nom=nom_md, sigma=sigma,
            K=MAX_ITER_K, V_star=None, dist_metric=dm,
        )
        Pi_K = np.argmax(Q_K, axis=1)
        config_name = dm + "_" + str(sigma)
        rows.append({
            "config": config_name,
            "Pi_star": [Pi_K.tolist()],   # wrap in list to make it one row
            "V_star": [V_K.tolist()],
            "Q_star": [Q_K.tolist()],
            "Evaluations": evals,
        })
        print(f"Run with sigma: {sigma} and distance metric: {dm} took {time.time() - current_time} seconds") 
        current_time = time.time()


# Vanilla VI
P = nominal_env.nominal_kernel()
R_exp = nominal_env.nominal_expected_reward()
Q_K, V_K, evals = VI(nominal_env, P, R_exp, MAX_ITER_K)
Pi_K = np.argmax(Q_K, axis=1)
rows.append({
    "config": "non-robust",
    "Pi_star": [Pi_K.tolist()],
    "V_star": [V_K.tolist()],
    "Q_star": [Q_K.tolist()],
    "Evaluations": evals,
})

# Build one DataFrame per row and extend
all_configs_df = pl.DataFrame(rows[0])
for row in rows[1:]:
    all_configs_df = all_configs_df.extend(pl.DataFrame(row))

# Ensure data folder exists
DATA_FOLDER.mkdir(parents=False, exist_ok=True)
all_configs_df.write_parquet(DATA_FOLDER / "all_configs.parquet") 

print(f"Finished! Entire run took {time.time() - start_time} seconds") 
