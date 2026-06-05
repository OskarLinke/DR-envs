from numpy.typing import NDArray
from my_typing import DistancesToOptimum
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

### SupplyChain
nominal_env = SupplyChain(b=0) # With b=0 uniform
nom_md = nominal_env.market_ask_distribution()
## TODO: Get this from the data/all_configs.parquet
try: 
    data = pl.read_parquet(DATA_FOLDER / STAR_SAVE_NAME)
except FileNotFoundError:
    print(f"Make sure to run `get_optimal_per_config.py` before {__name__}")
    raise  


# Run REVI with uniform as nominal transition
rows: list[dict] = [] # FIGURE OUT WHAT'S BEST
for metric in DISTANCE_METRICS:
    for sigma in SIGMAS:
        config_name = metric + "_" + str(sigma)
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
            "mean": results.mean(axis=0),
            "std": results.std(axis=0),
            })

all_configs_df = pl.DataFrame(rows)

# Ensure data folder exists
DATA_FOLDER.mkdir(parents=False, exist_ok=True)
all_configs_df.write_parquet(DATA_FOLDER / CONVERGENCE_SAVE_NAME) 
