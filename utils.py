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
        jac=objective_jac,
        constraints=constraints,
        bounds=[(0, 1) for _ in range(len(nom_md))],
        method="SLSQP",
        options={"maxiter": 500},
    )
    if not result.success:
        print(f"Warning: find_inf_market_dist optimization failed: {result.message}")

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
