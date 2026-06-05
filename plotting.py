import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
import numpy as np

def plot_convergence(
    convergence_df: pl.DataFrame,
    save_path: Path,
    dist_metric: str,
) -> None:
    title = f"Convergence of REVI by {dist_metric} norm of error"
    # Filter for dist metric
    convergence_df = convergence_df.filter(pl.col("config").str.contains(dist_metric))
    labels = convergence_df["config"].str.replace(f"{dist_metric}_", r"$\sigma=$").to_list()
    means = convergence_df["means"].to_list()
    num_plots = len(labels)
    for i in range(num_plots):
        x = range(len(means[i]))
        plt.plot(x, means[i], label=labels[i])
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|V_k - V^*\|_2$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.savefig(save_path)


def plot_robustness(robustness_df: pl.DataFrame) -> None:
    raise NotImplementedError

if __name__ == "__main__":
    from const import DATA_FOLDER, PLOTS_FOLDER, CONVERGENCE_SAVE_NAME

    conv_df = pl.read_parquet(DATA_FOLDER / CONVERGENCE_SAVE_NAME)

    PLOTS_FOLDER.mkdir(parents=False, exist_ok=True)
    # Plot for L2 distance metric
    plot_convergence(
        convergence_df=conv_df, save_path=(PLOTS_FOLDER / "test"), dist_metric="KL",
    )
