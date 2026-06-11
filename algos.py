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

def DRVI(
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
    """Distributionally Robust Value Iteration.

    Applies the robust Bellman operator to (Q_k, V_k) until either K
    iterations elapse or the sup-norm change in Q falls below `tolerance`.
    At each (s, a) the inner adversary is solved by `find_inf_kernel_reward`
    over an ambiguity ball of radius `sigma` in the chosen `dist_metric`.

    Follows [1]_ and [2]_, extended to reward distributions whose
    parametrisation is coupled to the transition kernel (as in the
    SupplyChain environment).

    Parameters
    ----------
    env :
        Tabular environment exposing state_size, action_size, gamma,
        legal_actions, and the kernel/reward helpers consumed by
        `find_inf_kernel_reward`.
    md_nom :
        Nominal market-ask distribution shared across all (s, a).
    sigma : float
        Ambiguity-ball radius. Used for both the transition kernel and
        the reward distribution.
    K : int
        Maximum number of value-iteration steps.
    learn_model : bool, default True
        If True, replace the analytic nominal (P, r) with an N-sample
        empirical estimate via `empirical_nominal_kernels`.
    dist_metric : str, default "KL"
        One of "KL", "TV", "CHI_SQ".
    tolerance : float, default 0.05
        Stop when ||Q_k - Q_{k-1}||_inf < tolerance.
    N : int, default 1000
        Sample budget used when `learn_model` is True.
    V_star : ndarray or None
        Optional reference V used to record ||V_k - V_star||_2 at every
        iteration.

    Returns
    -------
    Q_k : ndarray, shape (S, A)
        Final action-value function. Illegal actions are masked with -inf.
    V_k : ndarray, shape (S,)
        Final value function.
    V_dist_to_star : ndarray or None
        ||V_k - V_star||_2 per iteration, truncated to the number of
        iterations actually run. None when `V_star` is None.
    evaluations : int
        Number of iterations executed before exit.

    Notes
    -----
    Assumes tabular learning; states and actions are integer indices.

    References
    ----------
    .. [1] Panaganti, K. and Kalathil, D. (2023). Sample Complexity of
       Robust Reinforcement Learning with a Generative Model.
    .. [2] Shi, L. et al. (2025). Curriculum-style robust value iteration
       for distributionally robust MDPs.
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
    
    hot_mds = np.full((S, A, len(md_nom)), md_nom[0])
    
    if learn_model:
        nom_model = empirical_nominal_kernels(env, N)

    # for steps k up to K apply robust bellman operator
    # update both Q_k and V_k
    for k in range(K):
        if (k + 1) % 56 == 0: 
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
                inf_P, inf_r, hot_md = find_inf_kernel_reward(
                    state=s, action=a, nom_md=md_nom,
                    sigma_p=sigma, sigma_r=sigma,
                    V=V_k, gamma=gamma,
                    x0 = hot_mds[s,a],
                    env=env, dist_metric=dist_metric,
                    nom_model_sa=nom_model_sa if learn_model else None,
                )
                # Bellman operator becomes robust with inf (p,r)
                Q_k[s, a] = _bellman_operator(inf_P, V_k, inf_r, gamma)

                n = len(hot_md)
                hot_mds[s, a] = 0.99 * hot_md + 0.01 * np.full(n, 1.0 / n)

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
    """Tabular value iteration on a known model.

    Parameters
    ----------
    env :
        Tabular environment (provides state_size, action_size, gamma,
        and the legal-action mask).
    P : ndarray, shape (S, A, S)
        Transition kernel.
    R_exp : ndarray, shape (S, A)
        Expected reward.
    K : int
        Maximum number of iterations.
    tolerance : float, default 0.05
        Stop when ||Q_k - Q_{k-1}||_inf < tolerance.

    Returns
    -------
    Q_k : ndarray, shape (S, A)
    V_k : ndarray, shape (S,)
    evaluations : int

    Notes
    -----
    States and actions are integer indices. Rewards are treated as
    deterministic given (s, a).
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
