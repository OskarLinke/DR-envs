from pathlib import Path

DISTANCE_METRICS: list[str] = ["KL", "TV", "CHI_SQ"]
SIGMAS: list[float] = [0.5, 1., 2.]
BS: list[float] = [1., 1.5, 2., 2.5]
MS: list[int] = [i for i in range(10)]
T: int = 100
NS: list[int] = [10**n for n in range(2, 7)]
MAX_ITER_K: int = 200
NUM_CONV_EXPERIMENTS: int = 10
NUM_SAM_EXPERIMENTS: int = 10
NUM_ROB_EXPERIMENTS: int = 2000
N_JOBS: int = -1  # joblib worker count: -1 = all cores, -2 = all minus one

DATA_FOLDER = Path(__name__).parent / "data"
TRUE_FOLDER = DATA_FOLDER / "true_model"
PLOTS_FOLDER = Path(__name__).parent / "plots"
STAR_SAVE_NAME = "all_configs.parquet"
CONVERGENCE_SAVE_NAME = "convergence_results.parquet"
ROBUSTNESS_SAVE_NAME = "robustness_results.parquet"
SAMPLES_SAVE_NAME = "samples_results.parquet"
