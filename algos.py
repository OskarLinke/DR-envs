from my_typing import (
    DistancesToOptimum,
    QFunction,
    VFunction,
)
from typing import Any
from numpy.typing import NDArray
import numpy as np
from utils import (
    empirical_nominal_kernels,
    find_inf_kernel_reward,
)

def _bellman_operator(P, V, r_val, gamma):
    return r_val + gamma*np.dot(P, V)

def REVI(
    env,
    md_nom,
    sigma: float,
    K: int, 
    learn_model: bool = True,
    dist_metric: str = "KL",
    tolerance: float = 0.05, 
    N: int = 1000,
    V_star: NDArray[Any] | None = None,
) -> tuple[QFunction, VFunction, DistancesToOptimum | None, int]:
    """
    Robust Empirical Value Iteration

    NOTE
    ----
        Assumes tabular learning, thus states and actions are given as indices. 
        Assumes deterministic rewards. 
    """

    # Init Q_0 and V_0
    S, A = env.state_size(), env.action_size()
    gamma = env.gamma
    Q_k = np.zeros((S, A))
    V_k = np.zeros(S)
    V_dist_to_star = np.zeros(K) if V_star is not None else None
    evaluations = K  
    nom_model = None
    nom_model_sa = None
    if learn_model:
        nom_model = empirical_nominal_kernels(env, N)

    # for steps k up to K apply robust bellman operator
    # update both Q_k and V_k
    for k in range(K):
        if (k + 1) % 50 == 0: 
            print(
                f"Iteration {k} of setup with sigma: {sigma} and "
                f"distance_metric {dist_metric}"
            )
        Q_prev = Q_k.copy()
        for s in range(S):
            for a in range(A):
                env.reset(s = s)
                if a > env.legal_actions(): 
                    Q_k[s, a] = -np.inf
                    continue

                if learn_model and nom_model is not None:
                    nom_model_sa = (nom_model[0][s, a], nom_model[1][s, a])
                inf_P, inf_r = find_inf_kernel_reward(
                    state=s, action=a, nom_md=md_nom,
                    sigma_p=sigma, sigma_r=sigma,
                    V=V_k, gamma=gamma,
                    env=env, dist_metric=dist_metric,
                    nom_model_sa=nom_model_sa if learn_model else None,
                )
                # Bellman operator becomes robust with inf (p,r)
                Q_k[s, a] = _bellman_operator(inf_P, V_k, inf_r, gamma)

        V_k = np.max(Q_k, axis=1)
        if V_dist_to_star is not None:
            V_dist_to_star[k] = np.linalg.norm(V_k - V_star)

        mask = np.isfinite(Q_k) & np.isfinite(Q_prev) # We can't subtract np.inf values
        if np.linalg.norm(Q_k[mask] - Q_prev[mask], ord = np.inf) < tolerance: 
            if V_dist_to_star is not None:
                V_dist_to_star = V_dist_to_star[:k+1]
            evaluations = k
            break

    # return Q_K, V_K, and Distance to Optimum per k arrays
    return Q_k, V_k, V_dist_to_star, evaluations

def VI(
    env, P, R_exp, K: int, tolerance: float = 0.05
) -> tuple[QFunction, VFunction, int]:
    """
    Value Iteration

    NOTE
    ----
        Assumes tabular learning, thus states and actions are given as indices. 
        Assumes deterministic rewards. 
    """

    # Init Q_0 and V_0
    S, A = env.state_size(), env.action_size()
    gamma = env.gamma
    Q_k = np.zeros((S, A))
    V_k = np.zeros(S)
    evaluations = K
    # for steps k up to K apply robust bellman operator
    # update both Q_k and V_k
    for k in range(K):
        Q_prev = Q_k.copy()
        for s in range(env.state_size()):
            for a in range(env.action_size()):
                env.reset(s = s)
                if a > env.legal_actions(): 
                    Q_k[s, a] = -np.inf
                    continue
                r_val = R_exp[s, a]
                Q_k[s, a] = _bellman_operator(P[s,a], V_k, r_val, gamma)
        V_k = np.max(Q_k, axis=1)  

        mask = np.isfinite(Q_k) & np.isfinite(Q_prev) # We can't subtract np.inf values

        if np.linalg.norm(Q_k[mask] - Q_prev[mask], ord = np.inf) < tolerance: 
            print(f"VI terminated at step {k}") 
            evaluations = k
            break

    # return Q_K and V_K
    return Q_k, V_k, evaluations
