from collections import defaultdict
from time import time

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from algos import REVI
from const import (
    CONVERGENCE_SAVE_NAME,
    DATA_FOLDER,
    DISTANCE_METRICS,
    MAX_ITER_K,
    N_JOBS,
    NUM_CONV_EXPERIMENTS,
    SIGMAS,
    STAR_SAVE_NAME,
)
from supply_chain import SupplyChain


def run_one_revi(metric: str, sigma: float, V_star: np.ndarray) -> np.ndarray:
    # Each worker constructs its own env to avoid pickling overhead and shared state.
    env = SupplyChain(b=0)  # b=0 -> uniform nominal
    md = env.market_ask_distribution()
    _, _, V_dists, _ = REVI(
        env=env,
        md_nom=md,
        sigma=sigma,
        dist_metric=metric,
        K=MAX_ITER_K,
        V_star=V_star,
    )
    assert V_dists is not None, "Must enter valid V_star to REVI"
    return V_dists


if __name__ == "__main__":
    start_time = time()
    conv_save_path = DATA_FOLDER / CONVERGENCE_SAVE_NAME

    try:
        data = pl.read_parquet(DATA_FOLDER / STAR_SAVE_NAME)
    except FileNotFoundError:
        print(f"Make sure to run `get_optimal_per_config.py` before {__name__}")
        raise

    existing_exps = None
    existing_exps_names = None
    if conv_save_path.exists():
        existing_exps = pl.read_parquet(conv_save_path)
        existing_exps_names = existing_exps["config"].to_list()
        print(f"Found existing configs: {existing_exps_names}")

    to_run_config_names = ["non-robust"]
    for metric in DISTANCE_METRICS:
        for sigma in SIGMAS:
            if metric == "TV" and sigma > 1:
                continue
            else:
                to_run_config_names.append(metric + "_" + str(sigma))

    assert set(to_run_config_names) == set(data["config"].to_list()), (
        f"Missing configs. Run wants:\n{set(to_run_config_names)}\n"
        f"Existing are:\n{set(data['config'].to_list())}"
    )

    # Build flat (config_name, metric, sigma, V_star, repeat_idx) task list.
    tasks: list[tuple[str, str, float, np.ndarray, int]] = []
    for metric in DISTANCE_METRICS:
        for sigma in SIGMAS:
            if metric == "TV" and sigma > 1:
                continue
            config_name = metric + "_" + str(sigma)
            if existing_exps_names is not None and config_name in existing_exps_names:
                print(f"Config {config_name} already exists, skipping...")
                continue
            row = data.row(by_predicate=pl.col("config") == config_name, named=True)
            V_robust_star = np.array(row["V_star"])
            for n in range(NUM_CONV_EXPERIMENTS):
                tasks.append((config_name, metric, sigma, V_robust_star, n))

    # Run all REVI invocations in parallel.
    results = list(tqdm(
        Parallel(n_jobs=N_JOBS, return_as="generator")(
            delayed(run_one_revi)(metric, sigma, V_star)
            for _, metric, sigma, V_star, _ in tasks
        ),
        total=len(tasks),
        desc="REVI runs",
    ))

    # Group results back by config_name.
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for (config_name, _, _, _, _), V_dists in zip(tasks, results):
        grouped[config_name].append(V_dists)

    rows: list[dict] = []
    for config_name, V_dists_list in grouped.items():
        arr = np.array(V_dists_list)
        rows.append({
            "config": config_name,
            "convergence": arr,
            "means": arr.mean(axis=0),
            "stds": arr.std(axis=0),
        })

    all_exps_df = pl.DataFrame(rows) if rows else None

    if existing_exps is not None:
        if all_exps_df is not None:
            all_exps_df = (
                pl.concat([all_exps_df, existing_exps], how="vertical")
                .unique(subset="config", keep="last")
            )
        else:
            all_exps_df = existing_exps

    DATA_FOLDER.mkdir(parents=False, exist_ok=True)

    if all_exps_df is not None:
        all_exps_df = all_exps_df.sort(by="config")
        all_exps_df.write_parquet(conv_save_path)
        print(f"Finished! Entire run took {time() - start_time:.2f} seconds")
    else:
        raise ValueError("No existing data and no computed results")
