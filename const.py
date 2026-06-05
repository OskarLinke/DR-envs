from pathlib import Path

SIGMAS: list[float] = [0.5, 1., 2.]
DISTANCE_METRICS: list[str] = ["KL", "L2"]
MAX_ITER_K: int = 200
NUM_EXPERIMENTS: int = 10

DATA_FOLDER = Path(__name__).parent / "data"
PLOTS_FOLDER = Path(__name__).parent / "plots"
STAR_SAVE_NAME = "all_configs.parquet"
CONVERGENCE_SAVE_NAME = "convergence_results.parquet"
