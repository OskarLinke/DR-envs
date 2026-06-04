import matplotlib.pyplot as plt
import polars as pl

def plot_convergence(
    convergence_df: pl.DataFrame, configs: list[str] | None = None,
) -> None:
    # Filter the configs we want to plot
    if configs is not None:
        breakpoint()
        convergence_df = convergence_df.filter(pl.col("config").is_in(configs))


def plot_robustness(robustness_df: pl.DataFrame) -> None:
    raise NotImplementedError

if __name__ == "__main__":
    from const import DATA_FOLDER, CONVERGENCE_SAVE_NAME

    conv_df = pl.read_parquet(DATA_FOLDER / CONVERGENCE_SAVE_NAME)
    plot_convergence(conv_df, ["non-robus"])
