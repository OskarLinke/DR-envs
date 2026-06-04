from my_typing import DistancesToOptimum
import polars as pl
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
        breakpoint()
        V_robust_star = data["config" == config_name]["V_star"][0].to_numpy()
        results: list[DistancesToOptimum] = []
        for n in range(NUM_EXPERIMENTS):
            Q_K, V_K, V_dists, _ = REVI(
                env=nominal_env, md_nom=nom_md, sigma=sigma,
                K=MAX_ITER_K, V_star=V_robust_star,
            )
            assert V_dists is not None, "Must enter valid V_star to REVI"
            results.append(V_dists)
        rows.append({
            "config": config_name, 
            "convergence": results,
            })

# Build one DataFrame per row and extend
all_configs_df = pl.DataFrame({})
for row in rows:
    all_configs_df = all_configs_df.extend(pl.DataFrame(row))

# Ensure data folder exists
DATA_FOLDER.mkdir(parents=False, exist_ok=True)
all_configs_df.write_parquet(DATA_FOLDER / CONVERGENCE_SAVE_NAME) 
