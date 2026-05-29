import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable
from scipy.optimize import minimize

def empirical_nominal_kernel(env, N: int = 1000):
    # Estimate the nominal model for Supply Chain
    S, A = env.state_size(), env.action_size()
    running_counts = np.zeros(S)
    P_hat = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            env.reset(s = s)
            if a > env.legal_actions():
                # Illegal action - set uniform distribution as placeholder
                P_hat[s, a] = np.ones(S) / S
                continue
            for _ in range(N):
                env.reset(s = s)
                s_prime, r = env.step(a)
                running_counts[s_prime] += 1

            running_counts /= N
            P_hat[s, a] = running_counts
            running_counts.fill(0)

    return P_hat

def robust_bellman_operator(inf_P, V, r_val, gamma):
    return r_val + gamma*np.dot(inf_P, V)

def bellman_operator(P, V, r_val, gamma):
    return r_val + gamma*np.dot(P, V)

def find_inf_market_dist(
    state: int, 
    action: int, 
    nom_md: NDArray[Any], 
    V: NDArray[Any], 
    gamma: float,
    sigma: float,
    exp_rw_func: Callable[[int, int, NDArray[Any]], float],
    trans_kernel_func: Callable[[int, int, NDArray[Any]], NDArray[Any]],
    dist_metric: str = "KL",
) -> NDArray[Any]: 
    """
    state is state
    action is action
    nom_md is the nomrinal market ask distribution (md = market distribution) 
    exp_rw_func is a function which given a market distribution and state-action
        pair gives an expected reward 
    trans_kernel_func is a function which takes a market distribution and state-action
        pair, and returns a transition probability vector 
    V is the value-fucntion 
    gamma is the discount factor
    sigma is the robustness level
    dist_metric is the distance metric
    """ 

    # Nominal transition kernel induced by the nominal market distribution.
    # The ambiguity ball is defined on the transition kernel, not on md_.
    nom_P = trans_kernel_func(state, action, nom_md)

    def objective(md_):
        r_val = exp_rw_func(state, action, md_)
        p_ = trans_kernel_func(state, action, md_)
        return r_val + gamma*np.dot(p_, V)

    def distance_constraint(md_):
        # Distance is measured between the induced transition kernel P_ and the
        # nominal kernel nom_P, not between md_ and nom_md.
        p_ = trans_kernel_func(state, action, md_)
        if dist_metric == "L2":
            return sigma - np.sqrt(np.sum(np.square(p_ - nom_P)))
        elif dist_metric == "KL":
            # KL(P_ || nom_P). p_*log(p_/nom_P) -> 0 where p_ == 0.
            kl = np.sum(p_ * np.log((p_ + 1e-12) / (nom_P + 1e-12)))
            return sigma - kl
        else:
            raise NotImplementedError("Implemented distance metrics include: L2, KL")

    def sum_constraint(md_): 
        return 1.0 - np.sum(md_) 

    constraints = [
        {"type": "ineq", "fun": distance_constraint}, 
        {"type": "eq", "fun": sum_constraint}, 
    ]

    result = minimize(
        objective,
        x0=nom_md,
        constraints= constraints,
        bounds=[(0, 1) for _ in range(len(nom_md))],
        method="SLSQP", # TODO: Think on this together
        options = {"maxiter": 500}, # TODO: Why this??
    )
    if not result.success:
        print(f"Warning: find_inf_market_dist optimization failed: {result.message}")

    # Clamp to [0,1] to handle numerical errors
    return result.x


def find_inf_P(P_hat, V, sigma, dist_metric: str = "TV"):
    """
    P_hat is P^hat_{s,a}
    # Constrained minimization.
    # We have to find a P that minimizes the dot product P^TV
    # where P is within distance sigma according to the distance metric
    """

    def objective(p_):
        return np.dot(p_, V)

    def distance_constraint(p_):  # distance metric
        if dist_metric == "TV":
            return sigma - np.max(np.abs(p_ - P_hat))  # change norm type here
        else:
            raise ValueError("Implemented distance metrics are: TV")

    def sum_constraint(p_): 
        return 1 - np.sum(p_)

    constraints = [
        {"type": "ineq", "fun": distance_constraint}, 
        {"type": "eq", "fun": sum_constraint}, 
    ]

    result = minimize(
      objective,
      x0=P_hat,
      constraints= constraints,
      bounds=[(0, 1) for _ in range(len(P_hat))],
      method="SLSQP",
    )
    if not result.success:
        print(f"Warning: find_inf_P optimization failed: {result.message}")

    # Clamp to [0,1] to handle numerical errors
    return result.x
