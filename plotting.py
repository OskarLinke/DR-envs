import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl


def plot_convergence(
    convergence_df: pl.DataFrame,
    save_path: Path,
    dist_metric: str,
) -> None:
    title = f"Convergence of REVI by {dist_metric} norm of error"
    # Filter for dist metric
    convergence_df = convergence_df.filter(pl.col("config").str.contains(dist_metric))
    labels = (
        convergence_df["config"]
        .str.replace(f"{dist_metric}_", r"$\sigma=$")
        .to_list()
    )
    means = convergence_df["means"].to_list()
    num_plots = len(labels)
    for i in range(num_plots):
        x = range(len(means[i]))
        plt.plot(x, means[i], label=labels[i], alpha=0.6, marker="x", markersize=2)
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|V_k - V^*\|_2$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()


def plot_robustness(
    robustness_df: pl.DataFrame,
    solved_df: pl.DataFrame,
    b: float,
    save_path: Path,
    y_lim: tuple[float, float] | None = None,
) -> None:
    # Get list of policy per config
    for row in robustness_df.sort(by="policy").iter_rows(named=True):
        policy = row["policy"][0]
        matches = solved_df.filter(pl.col("Pi_star") == policy)
        if matches.height == 1:
            label = matches["config"].str.replace("_", " ").item()
        else:
            label = matches["config"].str.replace("_", " ").str.join(" & ").item()
        x = range(len(row["mean_cost"]))
        plt.scatter(x, row["mean_cost"], label=label)
        # TODO: Discuss between errors and not erros.
        # plt.errorbar(x, row["mean_cost"], yerr=row["std_cost"], label=label, fmt="o", capsize=3, alpha=0.6)
        # Do std above and below with fill between
        # import numpy as np
        # plt.fill_between(
        #     x,
        #     np.array(row["mean_cost"]) - np.array(row["std_cost"]),
        #     np.array(row["mean_cost"]) + np.array(row["std_cost"]),
        #     alpha=0.2,
        # )

        
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.title(fr"Cost over pertubed market ask b={b} over m")
    plt.xlabel(r"Pertubed market ask $m$")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.clf()

if __name__ == "__main__":
    from const import (
        DATA_FOLDER,
        PLOTS_FOLDER,
        STAR_SAVE_NAME,
        ROBUSTNESS_SAVE_NAME,
        CONVERGENCE_SAVE_NAME,
    )

    conv_df = pl.read_parquet(DATA_FOLDER / CONVERGENCE_SAVE_NAME)
    rob_df = pl.read_parquet(DATA_FOLDER / ROBUSTNESS_SAVE_NAME)
    solved_df = pl.read_parquet(DATA_FOLDER / STAR_SAVE_NAME)
    bs = rob_df["b"].unique().to_list()

    PLOTS_FOLDER.mkdir(parents=False, exist_ok=True)
    # Plot for all distance metric
    # for metric in DISTANCE_METRICS:
    #     plot_convergence(
    #         convergence_df=conv_df,
    #         save_path=(PLOTS_FOLDER / f"convergence_w_{metric}_dmetric"),
    #         dist_metric=metric,
    #     )
    # Plot for robustness
    all_costs = rob_df["mean_cost"].explode()
    y_min, y_max = all_costs.min(), all_costs.max()
    padding = (y_max - y_min) * 0.05
    y_lim = (y_min - padding, y_max + padding)
    for b in bs:
        b_df = rob_df.filter(pl.col("b") == b)
        grouped_df = b_df.group_by('policy').agg(
            pl.col("mean_cost"),
            pl.col("std_cost"),
        )
        plot_robustness(
            robustness_df=grouped_df,
            solved_df=solved_df,
            b=b,
            save_path=(PLOTS_FOLDER / f"robustness_b_{b}".replace(".", ",")),
            y_lim=y_lim,
        )
