from collections import defaultdict
from time import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from algos import REVI
from const import (
    DATA_FOLDER,
    DISTANCE_METRICS,
    MAX_ITER_K,
    N_JOBS,
    NS,
    NUM_SAM_EXPERIMENTS,
    SAMPLES_SAVE_NAME,
    SIGMAS,
    STAR_SAVE_NAME,
    TRUE_FOLDER,
)
from supply_chain import SupplyChain

print(NS)
def run_one_revi(metric: str, sigma: float, V_star: NDArray[Any], N: int) -> float:
    # Each worker constructs its own env to avoid pickling overhead and shared state.
    env = SupplyChain(b=0)  # b=0 -> uniform nominal
    md = env.market_ask_distribution()
    _, _, V_dists, _ = REVI(
        env=env,
        md_nom=md,
        sigma=sigma,
        learn_model=True,
        dist_metric=metric,
        K=MAX_ITER_K,
        V_star=V_star,
        N=N,
    )
    assert V_dists is not None, "Must enter valid V_star to REVI"
    return V_dists[-1]


if __name__ == "__main__":
    start_time = time()
    samples_save_path = DATA_FOLDER / SAMPLES_SAVE_NAME

    try:
        data = pl.read_parquet(TRUE_FOLDER / STAR_SAVE_NAME)
    except FileNotFoundError:
        print(f"Make sure to run `get_optimal_per_config.py` before {__name__}")
        raise

    existing_exps = None
    existing_exps_names = None
    if samples_save_path.exists():
        existing_exps = pl.read_parquet(samples_save_path)
        existing_exps_names = []  
        for row in existing_exps.rows(named=True): 
            existing_exps_names.append(row["config"]+"-"+str(row["N_samples"]))
        print(f"Found existing configs: {existing_exps_names}")

    to_run_config_names = ["non-robust"]
    for metric in DISTANCE_METRICS:
        for sigma in SIGMAS:
            for N in NS:
                if metric == "TV" and sigma >= 1:
                    continue
                else:
                    to_run_config_names.append(metric + "_" + str(sigma)+"-"+str(N)) 

    # Build flat (config_name, metric, sigma, V_star, num_samples, repeat_idx) task list.
    tasks: list[tuple[str, str, float, NDArray[Any], int, int]] = []
    for metric in DISTANCE_METRICS:
        for sigma in SIGMAS:
            if metric == "TV" and sigma >= 1:
                continue
            config_name = metric + "_" + str(sigma)
            
            row = data.row(by_predicate=pl.col("config") == config_name, named=True)
            V_robust_star = np.array(row["V_star"])
            for N in NS:
                if existing_exps_names is not None and config_name+"-"+str(N) in existing_exps_names:
                    print(f"Config {config_name} already exists, skipping...")
                    continue
                for i in range(NUM_SAM_EXPERIMENTS):
                    tasks.append((config_name, metric, sigma, V_robust_star, N, i))

    # Run all REVI invocations in parallel.
    results = list(tqdm(
        Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(run_one_revi)(metric, sigma, V_star, N)
            for _, metric, sigma, V_star, N, _ in tasks
        ),
        total=len(tasks),
        desc="REVI runs",
    ))

    # Group results back by config_name.
    grouped: dict[str, list[float]] = defaultdict(list)
    for (config_name, _, _, _, N, _), V_dists in zip(tasks, results, strict=True):
        hash = config_name + f"-{N}"
        grouped[hash].append(V_dists)

    rows: list[dict] = []
    for hash, V_dists_list in grouped.items():
        arr = np.array(V_dists_list)
        rows.append({
            "config": hash.split("-")[0],
            "N_samples": int(hash.split("-")[1]),
            "results": arr,
            "mean": arr.mean(),
            "std": arr.std(),
        })

    all_exps_df = pl.DataFrame(rows) if rows else None

    if existing_exps is not None:
        if all_exps_df is not None:
            all_exps_df = (
                pl.concat([all_exps_df, existing_exps], how="vertical")
            )
        else:
            all_exps_df = existing_exps

    DATA_FOLDER.mkdir(parents=False, exist_ok=True)

    if all_exps_df is not None:
        all_exps_df = all_exps_df.sort(by="config")
        all_exps_df.write_parquet(samples_save_path)
        print(f"Finished! Entire run took {time() - start_time:.2f} seconds")
    else:
        raise ValueError("No existing data and no computed results")
