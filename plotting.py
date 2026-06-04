import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
import numpy as np

def plot_convergence(
    convergence_df: pl.DataFrame,
    save_path: Path,
    configs: list[str] | None = None,
    title: str = "Convergence of REVI by L2 norm of error",
) -> None:
    # Filter the configs we want to plot
    if configs is not None:
        convergence_df = convergence_df.filter(pl.col("config").is_in(configs))
    labels = convergence_df["config"].str.replace("_", " ").to_list()
    means = convergence_df["mean"].to_list()
    num_plots = len(labels)
    x = range(len(means[0]))
    for i in range(num_plots):
        plt.plot(x, means[i], label=labels[i])
    plt.xlabel(r"Iteration $k$")
    plt.ylabel(r"$\|V_k - V^*\|_2$")
    plt.yscale("log")
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)


def plot_robustness(robustness_df: pl.DataFrame) -> None:
    raise NotImplementedError

if __name__ == "__main__":
    from const import DATA_FOLDER, PLOTS_FOLDER, CONVERGENCE_SAVE_NAME

    conv_df = pl.read_parquet(DATA_FOLDER / CONVERGENCE_SAVE_NAME)

    PLOTS_FOLDER.mkdir(parents=False, exist_ok=True)
    plot_convergence(conv_df, (PLOTS_FOLDER / "test"), configs=["L2_1.0"])
