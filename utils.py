import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing import Any

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
        s: int, 
        a: int, 
        nom_md: NDArray[Any], 
        exp_rw_func, 
        trans_kernel_func, 
        V: NDArray[Any], 
        gamma: float,
        sigma: float,
        dist_metric: str = "KL"): 
    """
    s is the state
    a is the action
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

    def objective(md_): 
        r_val = exp_rw_func(s, a, md_) 
        p_ = trans_kernel_func(s, a, md_) 
        return r_val + gamma*np.dot(p_, V) 

    def distance_constraint(md_): 
        if dist_metric == "L2": 
            return sigma - np.sqrt(np.sum(np.square(md_ - nom_md))) 
        elif dist_metric == "KL": 
            return sigma - np.sum(md_ * np.log(md_ / nom_md + 1e-12))
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
      method="SLSQP",
      options = {'maxiter': 500}, 
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
