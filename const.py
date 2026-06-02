from pathlib import Path

SIGMAS: list[float] = [1.]
DISTANCE_METRICS: list[str] = ["KL"]#, "L2"]
MAX_ITER_K: int = 1
NUM_EXPERIMENTS: int = 1

DATA_FOLDER = Path(__name__).parent / "data"
