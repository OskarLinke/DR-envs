import numpy as np
from algos import REVI, VI
from const import DISTANCE_METRICS, MAX_ITER_K, NUM_EXPERIMENTS, SIGMAS
from supply_chain import SupplyChain

### SupplyChain
nominal_env = SupplyChain(b=0) # With b=0 uniform
uncertainty_lvl = 1
max_iter = 4
distance_metrics = ["KL", "L2"]
num_rounds = 10

## TODO: Get this from the data/all_configs.parquet
try:
    V_robust_star = np.load("V_K_star-dr.npy")
except FileNotFoundError:
    print("V_K_star-dr.npy not found, running REVI with b=0 to get V_robust_star...")
    V_robust_star, _, _ = REVI(
        env=nominal_env, md_nom=nominal_env.market_ask_distribution(),
        sigma=uncertainty_lvl, K=max_iter, V_star=None,
    )
    np.save("V_K_star-dr.npy", V_robust_star)

nom_md = nominal_env.market_ask_distribution()

# Run REVI with uniform as nominal transition
results = [] # FIGURE OUT WHAT"S BEST
for metric in DISTANCE_METRICS:
    for sigma in SIGMAS:
        for n in range(NUM_EXPERIMENTS):
            Q_K, V_K, V_dists = REVI(
                env=nominal_env, md_nom=nom_md, sigma=uncertainty_lvl,
                K=MAX_ITER_K, V_star=V_robust_star,
            )
            Pi = np.argmax(Q_K, axis=1)
            results.append((Q_K, V_K, V_dists)) # FIGURE OUT WHAT"S BEST
