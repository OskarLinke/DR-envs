from genericpath import exists
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
    STAR_SAVE_NAME
)

nominal_env = SupplyChain(b=0)
nom_md = nominal_env.market_ask_distribution()
save_path = DATA_FOLDER / STAR_SAVE_NAME

rows = []

ex_config_names = None
existing_configs = None
all_configs_df = None

if save_path.exists():
    existing_configs = pl.read_parquet(save_path)
    ex_config_names = existing_configs["config"].to_list()
    print(f"Found existing configs: {ex_config_names}")

start_time = time.time()
current_time = time.time()
for sigma in SIGMAS:
    for dm in DISTANCE_METRICS:
        config_name = dm + "_" + str(sigma)
        if ex_config_names is None or config_name not in ex_config_names:
            Q_K, V_K, _, evals = REVI(
                env=nominal_env, md_nom=nom_md, sigma=sigma,
                K=MAX_ITER_K, V_star=None, dist_metric=dm,
            )
            Pi_K = np.argmax(Q_K, axis=1)
            # Check if this config already 
            rows.append({
                "config": config_name,
                "Pi_star": [Pi_K.tolist()],   # wrap in list to make it one row
                "V_star": [V_K.tolist()],
                "Q_star": [Q_K.tolist()],
                "Evaluations": evals,
            })
            print(
                f"Run with sigma: {sigma} and distance metric: {dm} took "
                f"{time.time() - current_time:.2f} seconds"
            ) 
        else:
            print(f"Config {config_name} already exists in the data, skipping...")
        current_time = time.time()


# Vanilla VI
if ex_config_names is not None and "non-robust" not in ex_config_names:
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
if rows != []:
    all_configs_df = pl.DataFrame(rows[0])
    for row in rows[1:]:
        all_configs_df = all_configs_df.extend(pl.DataFrame(row))

# Extend with existing data if it exists
if existing_configs is not None:
    all_configs_df = (
        pl.concat([all_configs_df, existing_configs], how="vertical")
        .unique(subset="config", keep="last")
        # Add rows, drop 'config' duplicates.
        # Keep last gives 'existing_configs' rows priority
    ) if all_configs_df is not None else existing_configs

# Ensure data folder exists
DATA_FOLDER.mkdir(parents=False, exist_ok=True)
if all_configs_df is not None:
    all_configs_df.write_parquet(save_path)
    print(f"Finished! Entire run took {time.time() - start_time:.2f} seconds") 
else:
    raise ValueError("Both non-existing data path and no computed results")
