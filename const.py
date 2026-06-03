from pathlib import Path

SIGMAS: list[float] = [0.5, 1., 2.]
DISTANCE_METRICS: list[str] = ["KL", "L2"]
MAX_ITER_K: int = 200
NUM_EXPERIMENTS: int = 1

DATA_FOLDER = Path(__name__).parent / "data"
