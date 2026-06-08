from time import time
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from algos import REVI, VI
from const import (
    DATA_FOLDER,
    DISTANCE_METRICS,
    MAX_ITER_K,
    N_JOBS,
    SIGMAS,
    STAR_SAVE_NAME,
)
from supply_chain import SupplyChain


def run_revi_config(sigma: float, dm: str) -> dict:
    env = SupplyChain(b=0)
    nom_md = env.market_ask_distribution()
    Q_K, V_K, _, evals = REVI(
        env=env, md_nom=nom_md, sigma=sigma,
        K=MAX_ITER_K, V_star=None, dist_metric=dm,
    )
    Pi_K = np.argmax(Q_K, axis=1)
    return {
        "config": dm + "_" + str(sigma),
        "Pi_star": Pi_K.tolist(),
        "V_star": V_K.tolist(),
        "Q_star": Q_K.tolist(),
        "Evaluations": evals,
    }


def run_non_robust() -> dict[str, Any]:
    env = SupplyChain(b=0)
    P = env.nominal_kernel()
    R_exp = env.nominal_expected_reward()
    Q_K, V_K, evals = VI(env, P, R_exp, MAX_ITER_K)
    Pi_K = np.argmax(Q_K, axis=1)
    return {
        "config": "non-robust",
        "Pi_star": Pi_K.tolist(),
        "V_star": V_K.tolist(),
        "Q_star": Q_K.tolist(),
        "Evaluations": evals,
    }


if __name__ == "__main__":
    save_path = DATA_FOLDER / STAR_SAVE_NAME
    ex_config_names = None
    existing_configs = None
    all_configs_df = None

    if save_path.exists():
        existing_configs = pl.read_parquet(save_path)
        ex_config_names = existing_configs["config"].to_list()
        print(f"Found existing configs: {ex_config_names}")

    start_time = time()

    # Build job list for missing configs.
    jobs = []
    for sigma in SIGMAS:
        for dm in DISTANCE_METRICS:
            config_name = dm + "_" + str(sigma)
            if ex_config_names is None or config_name not in ex_config_names:
                jobs.append(delayed(run_revi_config)(sigma, dm))
            else:
                print(f"Config {config_name} already exists, skipping...")

    if ex_config_names is None or "non-robust" not in ex_config_names:
        jobs.append(delayed(run_non_robust)())
    else:
        print("Config non-robust already exists, skipping...")

    rows: list[dict] = []
    if jobs:
        rows = list(tqdm(
            Parallel(n_jobs=N_JOBS, return_as="generator")(jobs),
            total=len(jobs),
            desc="configs",
        ))

    all_configs_df = pl.DataFrame(rows) if rows else None

    if existing_configs is not None:
        if all_configs_df is not None:
            # Match existing parquet schema (Evaluations stored as Int32).
            all_configs_df = all_configs_df.cast({"Evaluations": pl.Int32})
            all_configs_df = (
                pl.concat([all_configs_df, existing_configs], how="vertical")
                .unique(subset="config", keep="last")
            )
        else:
            all_configs_df = existing_configs

    DATA_FOLDER.mkdir(parents=False, exist_ok=True)
    if all_configs_df is not None:
        all_configs_df = all_configs_df.sort(by="config")
        all_configs_df.write_parquet(save_path)
        print(f"Finished! Entire run took {time() - start_time:.2f} seconds")
    else:
        raise ValueError("Both non-existing data path and no computed results")
