import numpy as np
from numpy.typing import NDArray
from typing import Any
from scipy.optimize import minimize
from scipy.stats import chi2

from my_typing import ProbVector

MAX_OPTIM_ITER = 150

_KL_EPS = 1e-12
_TV_TOL = 1e-15


### Find infimum model of the system functions
# distance metric helper functions
def _kl(p: ProbVector, q: ProbVector) -> float:
    """KL(p || q). Treats 0*log(0) = 0. Floors q at _KL_EPS for solver stability."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], _KL_EPS))))


def _tv(p: ProbVector, q: ProbVector) -> float:
    """Total variation distance: (1/2) * ||p - q||_1."""
    return 0.5 * float(np.linalg.norm(p - q, ord=1))


def _chi_sq(p: ProbVector, q: ProbVector) -> float:
    """Chi-square divergence chi^2(p || q) = sum (p_i - q_i)^2 / q_i.

    Returns +inf when q has zero mass where p > 0 (absolute continuity violated).
    """
    mask = q > 0
    if np.any((p > 0) & ~mask):
        return float("inf")
    return float(np.sum((p[mask] - q[mask]) ** 2 / q[mask]))


def _distance(p: ProbVector, q: ProbVector, dist_metric: str) -> float:
    if dist_metric == "KL":
        return _kl(p, q)
    if dist_metric == "TV":
        return _tv(p, q)
    if dist_metric == "CHI_SQ":
        return _chi_sq(p, q)
    raise NotImplementedError("Implemented distance metrics: KL, TV, CHI_SQ")


def find_inf_kernel_reward(
    state: int,
    action: int,
    nom_md: ProbVector,
    V: NDArray[Any],
    gamma: float,
    sigma_p: float,
    sigma_r: float,
    env,
    nom_model_sa: tuple[ProbVector, ProbVector] | None = None,
    dist_metric: str = "KL",
) -> tuple[ProbVector, float]:
    """Worst-case (transition kernel, expected reward) over an ambiguity ball.

    Returns (inf_P, inf_r) achieving
        min  inf_r + gamma * <inf_P, V>
        s.t. d(inf_P, nom_P) <= sigma_p,  inf_P in simplex(S)
             d(inf_r_dist, nom_r_dist) <= sigma_r,  inf_r_dist in simplex(R)
    where d is the requested ambiguity metric. nom_P and nom_r are the
    transition kernel and reward distribution induced by nom_md.

    Dispatch
    --------
    TV: closed-form via worst_case_tv (Iyengar 2005, Nilim & El Ghaoui 2005,
        Ho-Petrik-Wiesemann 2018). Exact and robust.
    KL / CHI_SQ: SLSQP minimize over the market distribution md, then map
        the optimizer back to (P, r) via env.
    """

    if nom_model_sa is None:
        nom_P: ProbVector = env.transition_kernel_sa(state, action, nom_md)
        nom_r: ProbVector = env.reward_probabilities_sa(state, action, nom_md)
    else:
        nom_P, nom_r = nom_model_sa
    # Position i in nom_r holds prob mass for reward value -i (env stores at -reward).
    r_values = -np.arange(len(nom_r), dtype=np.float64)

    if dist_metric == "TV":
        # Independent rectangular inner problems: kernel uses V as cost,
        # reward distribution uses reward values as cost.
        inf_P = worst_case_tv(nom_P, V, sigma_p)
        inf_r_dist = worst_case_tv(nom_r, r_values, sigma_r)
        inf_r = float(np.dot(inf_r_dist, r_values))
        return inf_P, inf_r

    # KL / CHI_SQ: minimize over md and recover (P, r) afterwards.
    def objective(md_: ProbVector) -> float:
        r_val = env.expected_reward_sa(state, action, md_)
        p_ = env.transition_kernel_sa(state, action, md_)
        return r_val + gamma * np.dot(p_, V)

    def r_distance_constraint(md_: ProbVector) -> float:
        # Distance measured between induced reward distribution r_ and nominal nom_r.
        r_: ProbVector = env.reward_probabilities_sa(state, action, md_)
        return sigma_r - _distance(r_, nom_r, dist_metric)

    def p_distance_constraint(md_: ProbVector) -> float:
        # Distance measured between induced transition kernel p_ and nominal nom_P.
        p_: ProbVector = env.transition_kernel_sa(state, action, md_)
        return sigma_p - _distance(p_, nom_P, dist_metric)

    def sum_constraint(md_: ProbVector) -> float:
        return 1.0 - np.sum(md_)

    constraints = [
        {"type": "ineq", "fun": p_distance_constraint},
        {"type": "ineq", "fun": r_distance_constraint},
        {"type": "eq", "fun": sum_constraint},
    ]

    result = minimize(
        objective,
        x0=nom_md,
        constraints=constraints,
        bounds=[(0, 1) for _ in range(len(nom_md))],
        method="SLSQP",
        options={"maxiter": MAX_OPTIM_ITER, "ftol": 1e-8},
    )
    if not result.success:
        print(f"Warning: find_inf_kernel_reward optimization failed: {result.message}")

    inf_md = result.x
    inf_P = env.transition_kernel_sa(state, action, inf_md)
    inf_r = float(env.expected_reward_sa(state, action, inf_md))
    return inf_P, inf_r

def worst_case_tv(
    p_nom: ProbVector, costs: NDArray[Any], rho: float
) -> ProbVector:
    """Closed-form worst-case distribution over a TV ball.

    Solves
        min_p  <p, costs>
        s.t.   (1/2) * ||p - p_nom||_1 <= rho
               p in simplex.

    Algorithm: water-filling. Shift probability mass from highest-cost
    coordinate to lowest-cost coordinate, limited by the per-coordinate
    bounds [0, 1] and the remaining transport budget rho. O(S log S).

    Constraint satisfied by construction: total mass moved is at most rho,
    which equals the TV distance between the returned p and p_nom.

    References
    ----------
    Iyengar (2005), "Robust Dynamic Programming", Math. Oper. Res. 30(2),
        Section 4 (rectangular ambiguity, TV/L1 specialization).
    Nilim & El Ghaoui (2005), "Robust Control of Markov Decision Processes
        with Uncertain Transition Matrices", Oper. Res. 53(5).
    Ho, Petrik, Wiesemann (2018), "Fast Bellman Updates for Robust MDPs",
        ICML — explicit O(S log S) algorithm.
    """
    p = p_nom.astype(np.float64).copy()
    budget = float(rho)
    order = np.argsort(costs)  # ascending: cheapest first
    lo, hi = 0, len(costs) - 1
    while budget > _TV_TOL and lo < hi:
        i_lo, i_hi = order[lo], order[hi]
        give = min(p[i_hi], budget, 1.0 - p[i_lo])
        if give <= _TV_TOL:
            # Endpoints saturated; advance whichever side is stuck.
            if p[i_hi] <= _TV_TOL:
                hi -= 1
            elif p[i_lo] >= 1.0 - _TV_TOL:
                lo += 1
            else:
                break
            continue
        p[i_hi] -= give
        p[i_lo] += give
        budget -= give
        if p[i_hi] <= _TV_TOL:
            hi -= 1
        if p[i_lo] >= 1.0 - _TV_TOL:
            lo += 1
    return p


#### Estimate the model of the system functions
def empirical_nominal_kernels(env, N: int):
    # Estimate the nominal model for Supply Chain
    S, A = env.state_size(), env.action_size()
    M = env.find_num_possible_rewards()
    running_counts = np.zeros(S)
    running_rewards = np.zeros(M)
    P_hat = np.zeros((S, A, S))
    r_hat = np.zeros((S, A, M))
    for s in range(S):
        for a in range(A):
            env.reset(s = s)
            if a > env.legal_actions():
                # Illegal action - set uniform distribution as placeholder
                P_hat[s, a] = np.ones(S) / S
                r_hat[s, a] = np.ones(M) / M
                continue
            for _ in range(N):
                env.reset(s = s)
                s_prime, r = env.step(a)
                r_idx = int(-r) # no gamma discount, reset every time i.e. int
                running_counts[s_prime] += 1
                running_rewards[r_idx] += 1

            running_counts /= N
            running_rewards /= N
            P_hat[s, a] = running_counts
            r_hat[s, a] = running_rewards
            running_counts.fill(0)
            running_rewards.fill(0)

    return P_hat, r_hat


############################# NOT CURRENTLY USED #################################
def calibrate_radius(metric: str, N: int, alpha: float, S: int) -> float:
    """Ambiguity-ball radius encoding (1 - alpha) statistical confidence.

    Assumes the nominal distribution is an empirical estimate from N i.i.d.
    samples over S atoms. Returns the radius rho such that the true distribution
    lies within the metric-ball around the empirical with probability >= 1 - alpha
    (asymptotically for phi-divergences; finite-sample for TV via DKW).

    Use to compare ambiguity sets across metrics on the same statistical footing:
    sigma_KL = calibrate_radius("KL", N, alpha, S) and
    sigma_TV = calibrate_radius("TV", N, alpha, S) encode the same confidence.

    References
    ----------
    Ben-Tal, den Hertog, De Waegenaere, Melenberg, Rennen (2013),
        "Robust Solutions of Optimization Problems Affected by Uncertain
        Probabilities", Management Science 59(2) -- KL and CHI_SQ via LR test.
    Dvoretzky-Kiefer-Wolfowitz inequality (Massart 1990 tight constant) -- TV
        finite-sample bound on empirical distributions.
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if S < 2:
        raise ValueError("S must be >= 2.")

    if metric == "KL":
        # Asymptotic LR / Wilks: 2N * KL ~ chi^2_{S-1} under nominal.
        return float(chi2.ppf(1.0 - alpha, df=S - 1) / (2 * N))
    if metric == "CHI_SQ":
        # Pearson chi^2 test: N * chi_sq ~ chi^2_{S-1}.
        return float(chi2.ppf(1.0 - alpha, df=S - 1) / N)
    if metric == "TV":
        # DKW with Massart constant: P(TV > eps) <= 2 * exp(-2 N eps^2).
        # Solve 2 * exp(-2 N eps^2) = alpha for eps.
        return float(np.sqrt(np.log(2.0 / alpha) / (2 * N)))
    raise NotImplementedError("Implemented distance metrics: KL, TV, CHI_SQ")
