import polars as pl
import numpy as np
from algos import REVI
from supply_chain import SupplyChain
from const import(
        CONVERGENCE_SAVE_NAME,
        SIGMAS, 
        DISTANCE_METRICS,
        MAX_ITER_K, 
        NUM_EXPERIMENTS, 
        DATA_FOLDER, 
        STAR_SAVE_NAME,
        )
from time import time

### SupplyChain and init vars
nominal_env = SupplyChain(b=0) # With b=0 uniform
nom_md = nominal_env.market_ask_distribution()
conv_save_path = DATA_FOLDER / CONVERGENCE_SAVE_NAME
existing_exps = None
existing_exps_names = None
all_exps_df = None
start_time = time()

try: 
    data = pl.read_parquet(DATA_FOLDER / STAR_SAVE_NAME)
except FileNotFoundError:
    print(f"Make sure to run `get_optimal_per_config.py` before {__name__}")
    raise  

if conv_save_path.exists():
    existing_exps = pl.read_parquet(conv_save_path)
    existing_exps_names = existing_exps["config"].to_list()
    print(f"Found existing configs: {existing_exps_names}")

# Run REVI with uniform as nominal transition
rows: list[dict] = [] # FIGURE OUT WHAT'S BEST
for metric in DISTANCE_METRICS:
    for sigma in SIGMAS:
        config_name = metric + "_" + str(sigma)
        if existing_exps_names is None or config_name not in existing_exps_names:
            row = data.row(by_predicate=pl.col("config") == config_name, named=True)
            V_robust_star = np.array(row["V_star"])
            results = []
            for n in range(NUM_EXPERIMENTS):
                Q_K, V_K, V_dists, _ = REVI(
                    env=nominal_env, md_nom=nom_md, sigma=sigma,
                    dist_metric=metric, K=MAX_ITER_K, V_star=V_robust_star,
                )
                assert V_dists is not None, "Must enter valid V_star to REVI"
                results.append(V_dists)
            results = np.array(results)
            rows.append({
                "config": config_name, 
                "convergence": results, 
                "means": results.mean(axis=0),
                "stds": results.std(axis=0),
                })
        else:
            print(f"Config {config_name} already exists in the data, skipping...")


# Build one DataFrame per row and extend

if rows != []:
    all_exps_df = pl.DataFrame(rows)

if existing_exps is not None: 
    if all_exps_df is not None: 
        all_exps_df = ( 
        pl.concat([all_exps_df, existing_exps], how="vertical")
                       .unique(subset="config", keep="last") 
                       )
    else: 
        all_exps_df = existing_exps

# Ensure data folder exists
DATA_FOLDER.mkdir(parents=False, exist_ok=True)



if all_exps_df is not None:
    all_exps_df.sort(by="config")
    all_exps_df.write_parquet(conv_save_path)
    print(f"Finished! Entire run took {time() - start_time:.2f} seconds") 

else: 
    raise ValueError("No existing data and no computed results")
