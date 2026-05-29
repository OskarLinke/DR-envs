from typing import Any
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray
from utils import (
    bellman_operator,
    find_inf_market_dist,
    robust_bellman_operator,
)

QAndValueFunctions: TypeAlias = tuple[NDArray[Any], NDArray[Any]]

def REVI(
    env, md_nom, sigma: float, K: int, dist_metric: str = "KL", tolerance = 0.05
) -> QAndValueFunctions:
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

    # for steps k up to K apply robust bellman operator
    # update both Q_k and V_k
    for k in range(K):
        Q_prev = Q_k.copy()
        for s in range(S):
            for a in range(A):
                env.reset(s = s)
                if a > env.legal_actions(): 
                    Q_k[s, a] = -np.inf
                    continue

                inf_md = find_inf_market_dist(
                    state=s, action=a, nom_md=md_nom,
                    V=V_k, gamma=gamma, sigma=sigma, 
                    dist_metric=dist_metric,
                    exp_rw_func=env.expected_reward_sa,
                    trans_kernel_func=env.trans_prob_kernel_sa,
                )
                
                inf_P = env.trans_prob_kernel_sa(s, a, inf_md) 
                inf_r = env.expected_reward_sa(s, a, inf_md)
                Q_k[s, a] = robust_bellman_operator(inf_P, V_k, inf_r, gamma)

        V_k = np.max(Q_k, axis=1)

        mask = np.isfinite(Q_k) & np.isfinite(Q_prev) # We can't subtract np.inf values
        if np.linalg.norm(Q_k[mask] - Q_prev[mask], ord = np.inf) < tolerance: 
            print(f"REVI terminated at step {k}") 
            break

    # return Q_K and V_K
    return Q_k, V_k

def VI(
    env, P, R_exp, K: int
) -> QAndValueFunctions:
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

    # for steps k up to K apply robust bellman operator
    # update both Q_k and V_k
    for k in range(K):
        for s in range(env.state_size()):
            for a in range(env.action_size()):
                env.reset(s = s)
                if a > env.legal_actions(): 
                    Q_k[s, a] = -np.inf
                    continue
                r_val = R_exp[s, a]
                Q_k[s, a] = bellman_operator(P[s,a], V_k, r_val, gamma)
        V_k = np.max(Q_k, axis=1)             

    # return Q_K and V_K
    return Q_k, V_k

if __name__ == "__main__":
    from supply_chain import SupplyChain

    ### SupplyChain
    nominal_env = SupplyChain(b=0) # With b=0 uniform
    uncertainty_lvl = 1
    max_iter = 400
    
    nom_md = nominal_env.market_ask_distribution()


    # Run REVI with uniform as nominal transition
    Q_K, V_K = REVI(nominal_env, nom_md, uncertainty_lvl, max_iter)
    print("Q_K:", Q_K)
    print("V_K:", V_K)
    print("Pi:\n", np.argmax(Q_K, axis=1))


    # Run VI with uniform market ask on true nominal transition and exp R
    P = nominal_env.true_nominal_kernel()
    R_exp = nominal_env.nominal_expected_reward()
    Q_K, V_K = VI(nominal_env, P, R_exp, max_iter)
    print("Q_K:\n", Q_K)
    print()
    print("V_K:\n", V_K)
    print("Pi:\n", np.argmax(Q_K, axis=1))
