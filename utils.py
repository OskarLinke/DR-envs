import numpy as np
from numpy.typing import NDArray
from typing import Any
from scipy.optimize import minimize

from my_typing import ProbVector

MAX_OPTIM_ITER = 1000

def bellman_operator(P, V, r_val, gamma):
    return r_val + gamma*np.dot(P, V)

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

def find_inf_market_dist(
    state: int, 
    action: int, 
    nom_md: ProbVector,
    V: NDArray[Any], 
    gamma: float,
    sigma_p: float,
    sigma_r: float,
    env,
    dist_metric: str = "KL",
) -> ProbVector:
    """
    state is state
    action is action
    nom_md is the nomrinal market ask distribution (md = market distribution) 
    V is the value-fucntion 
    gamma is the discount factor
    sigma_p is the robustness level
    exp_rw_func is a function which given a market distribution and state-action
        pair gives an expected reward 
    trans_kernel_func is a function which takes a market distribution and state-action
        pair, and returns a transition probability vector 
    env is the MDP. Currently tied to the supply-chain env
    dist_metric is the distance metric
    """ 

    # Nominal transition kernel induced by the nominal market distribution.
    # The ambiguity ball is defined on the transition kernel, not on md_.
    nom_P = env.transition_kernel_sa(state, action, nom_md)
    nom_r = env.reward_probabilities_sa(state, action, nom_md)

    def objective(md_):
        r_val = env.expected_reward_sa(state, action, md_)
        p_ = env.transition_kernel_sa(state, action, md_)

        return r_val + gamma*np.dot(p_, V)

    def r_distance_constraint(md_):
        # Distance is measured between the induced transition kernel P_ and the
        # nominal kernel nom_P, not between md_ and nom_md.
        r_ = env.reward_probabilities_sa(state, action, md_)
        if dist_metric == "L2":
            return sigma_r - np.sqrt(np.sum(np.square(r_ - nom_r)))
        elif dist_metric == "KL":
            # KL(P_ || nom_P). p_*log(p_/nom_P) -> 0 where p_ == 0.
            mask = r_ > 0
            kl = np.sum(r_[mask] * np.log(r_[mask] / np.maximum(nom_r[mask], 1e-12)))
            return sigma_r - kl
        else:
            raise NotImplementedError("Implemented distance metrics include: L2, KL")

    def p_distance_constraint(md_):
        # Distance is measured between the induced transition kernel P_ and the
        # nominal kernel nom_P, not between md_ and nom_md.
        p_ = env.transition_kernel_sa(state, action, md_)
        if dist_metric == "L2":
            return sigma_p - np.sqrt(np.sum(np.square(p_ - nom_P)))
        elif dist_metric == "KL":
            # KL(P_ || nom_P). p_*log(p_/nom_P) -> 0 where p_ == 0.
            mask = p_ > 0
            kl = np.sum(p_[mask] * np.log(p_[mask] / np.maximum(nom_P[mask], 1e-12)))
            return sigma_p - kl
        else:
            raise NotImplementedError("Implemented distance metrics include: L2, KL")

    def sum_constraint(md_): 
        return 1.0 - np.sum(md_) 

    constraints = [
    {"type": "ineq", "fun": p_distance_constraint}, 
        {"type": "ineq", "fun": r_distance_constraint}, 
        {"type": "eq", "fun": sum_constraint}, 
        ]

    result = minimize(
        objective,
        x0=nom_md,
        constraints= constraints,
        bounds=[(0, 1) for _ in range(len(nom_md))],
        method="SLSQP",
        options = {"maxiter": MAX_OPTIM_ITER, "ftol": 1e-8},
    )
    if not result.success:
        print(f"Warning: find_inf_market_dist optimization failed: {result.message}")

    # Clamp to [0,1] to handle numerical errors
    return result.x


def better_find_inf_market_dist(
    nom_md: NDArray[Any],
    V: NDArray[Any],
    gamma: float,
    sigma: float,
    M_sa: NDArray[Any],
    r_sa: NDArray[Any],
    dist_metric: str = "KL",
) -> NDArray[Any]:
    """
    Find the adversarial market-ask distribution for one (s, a).

    Uses precomputed linear maps so that P(s, a, md) = M_sa @ md and
    E_md[r(s, a)] = r_sa @ md. The objective is then linear in md:

        objective(md) = (r_sa + gamma * M_sa.T @ V) @ md

    Parameters
    ----------
    nom_md : (n+1,) nominal market-ask distribution. Used as warm start
        and to compute nom_P for the ambiguity ball.
    V : (S,) current value function.
    gamma : discount factor.
    sigma : ambiguity ball radius.
    M_sa : (S, n+1) linear map md -> transition probability vector.
    r_sa : (n+1,) linear map md -> expected reward.
    dist_metric : "KL" or "L2". Distance is measured between the induced
        transition kernel and nom_P, not between md and nom_md.
    """

    nom_P = M_sa @ nom_md
    # Constant coefficient vector: objective(md) = c @ md.
    c = r_sa + gamma * (M_sa.T @ V)

    def objective(md_):
        return c @ md_

    def objective_jac(md_):
        return c

    def distance_constraint(md_):
        p_ = M_sa @ md_
        if dist_metric == "L2":
            return sigma - np.sqrt(np.sum(np.square(p_ - nom_P)))
        elif dist_metric == "KL":
            # KL(P_ || nom_P). True KL is +inf when p_[i] > 0 but
            # nom_P[i] == 0 (support violation). SLSQP cannot consume
            # np.inf in a constraint, so return a large finite penalty
            # to push the iterate back into the feasible region.
            mask = p_ > 0
            if np.any(nom_P[mask] <= 0):
                kl = 1e6
            else:
                kl = np.sum(
                    p_[mask] * np.log(p_[mask] / np.maximum(nom_P[mask], 1e-12))
                )
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
        jac=objective_jac,
        constraints=constraints,
        bounds=[(0, 1) for _ in range(len(nom_md))],
        method="SLSQP",
        options={"maxiter": 500},
    )
    if not result.success:
        print(f"Warning: find_inf_market_dist optimization failed: {result.message}")

    return result.x
