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

def _kl_with_support_check(p: NDArray[Any], q: NDArray[Any]) -> float:
    """
    KL(p || q). True KL is +inf when p_[i] > 0 but q[i] == 0
    (support violation). SLSQP cannot consume np.inf in a constraint,
    so we return a large finite penalty to push the iterate back
    into the feasible region.
    """
    mask = p > 0
    if np.any(q[mask] <= 0):
        return 1e6
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))


def find_inf_market_dist(
    nom_md: ProbVector,
    V: NDArray[Any],
    gamma: float,
    sigma_p: float,
    sigma_r: float,
    M_sa: NDArray[Any],
    r_sa: NDArray[Any],
    R_sa: NDArray[Any],
    dist_metric: str = "KL",
) -> ProbVector:
    """
    Find the adversarial market-ask distribution for one (s, a) under
    joint ambiguity on the transition kernel P and the reward
    distribution.

    Uses precomputed linear maps so that all three quantities are
    matrix-vector products against md:

        P(s, a, md)              = M_sa @ md         shape (S,)
        E_md[r(s, a)]            = r_sa @ md         scalar
        reward_distribution(md)  = R_sa @ md         shape (R,)

    The objective is linear in md:

        objective(md) = (r_sa + gamma * M_sa.T @ V) @ md

    The two ambiguity balls (KL or L2) are evaluated on the induced
    transition kernel P(md) and reward distribution R(md), with separate
    radii sigma_p and sigma_r.

    Parameters
    ----------
    nom_md : (n+1,) nominal market-ask distribution. Used as warm start
        and to compute nominal P and R for the ambiguity balls.
    V : (S,) current value function.
    gamma : discount factor.
    sigma_p : ambiguity radius for the transition kernel.
    sigma_r : ambiguity radius for the reward distribution.
    M_sa : (S, n+1) linear map md -> P.
    r_sa : (n+1,) linear map md -> expected reward.
    R_sa : (R, n+1) linear map md -> reward distribution.
    dist_metric : "KL" or "L2".
    """

    nom_P = M_sa @ nom_md
    nom_R = R_sa @ nom_md
    # Constant coefficient vector: objective(md) = c @ md.
    c = r_sa + gamma * (M_sa.T @ V)

    def objective(md_):
        return c @ md_

    def objective_jac(md_):
        return c

    def p_distance_constraint(md_):
        p_ = M_sa @ md_
        if dist_metric == "L2":
            return sigma_p - np.sqrt(np.sum(np.square(p_ - nom_P)))
        elif dist_metric == "KL":
            return sigma_p - _kl_with_support_check(p_, nom_P)
        else:
            raise NotImplementedError("Implemented distance metrics: L2, KL")

    def r_distance_constraint(md_):
        r_ = R_sa @ md_
        if dist_metric == "L2":
            return sigma_r - np.sqrt(np.sum(np.square(r_ - nom_R)))
        elif dist_metric == "KL":
            return sigma_r - _kl_with_support_check(r_, nom_R)
        else:
            raise NotImplementedError("Implemented distance metrics: L2, KL")

    def sum_constraint(md_):
        return 1.0 - np.sum(md_)

    constraints = [
        {"type": "ineq", "fun": p_distance_constraint},
        {"type": "ineq", "fun": r_distance_constraint},
        {"type": "eq",   "fun": sum_constraint},
    ]

    result = minimize(
        objective,
        x0=nom_md,
        jac=objective_jac,
        constraints=constraints,
        bounds=[(0, 1) for _ in range(len(nom_md))],
        method="SLSQP",
        options={"maxiter": MAX_OPTIM_ITER, "ftol": 1e-8},
    )
    if not result.success:
        print(f"Warning: find_inf_market_dist optimization failed: {result.message}")

    return result.x
