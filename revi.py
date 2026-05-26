import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

def empirical_nominal_kernel(env, N: int = 1000):
    # Estimate the nominal model for Supply Chain
    S, A = env.state_size(), env.action_size()
    running_counts = np.zeros(S)
    P_hat = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for _ in range(N):
                env.S = s
                s_prime, _ = env.step(a)
                running_counts[s_prime] += 1 
            running_counts /= N
            P_hat[s, a] = running_counts
            running_counts.fill(0)

    return P_hat

def robust_bellman_operator(inf_P, V, r_val, gamma):
    pass

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
   
    def value_constraint_lower(p_):
        return np.min(p_)

    def value_constraint_upper(p_): 
        return 1 - np.max(p_)

    constraints = [
        {"type": "ineq", "fun": distance_constraint}, 
        {"type": "eq", "fun": sum_constraint}, 
        {"type": "ineq", "fun": value_constraint_upper}, 
        {"type": "ineq", "fun": value_constraint_lower}, 
    ]

    result = minimize(
      objective,
      x0=P_hat,
      constraints= constraints, 
      method="SLSQP",
    )
    return result.x


def REVI(env, P_hat, sigma: float, K: int, distance_metric: str = "TV"):
    """
    Robust Empirical Value Iteration
    NOTE
    ----
        Assumes tabular learning, thus states and actions are given as indices
    """

    # Init Q_0 and V_0
    S, A = env.state_size(), env.action_size()
    gamma = env.gamma
    Q_k = np.zeros((S, A))
    V_k = np.zeros(A)

    # for steps k up to K apply robust bellman operator
    # update both Q_k and V_k
    for k in range(K):
        for s in env.state_space:
            max_a = 0
            for a in env.action_space:
                r_val = env.reward(s, a)
                inf_P = find_inf_P(
                    P_hat[s, a], V_k, sigma, dist_metric=distance_metric
                ) # find_inf_P needs to be tested and debugged
                Q_k[s, a] = robust_bellman_operator(inf_P, V_k, r_val, gamma) # fill args
                V_k = np.max(Q_k, axis=1) # Maybe baby
            

    # return Q_K and V_K
    return Q_k, V_k


if __name__ == "__main__":
    from supply_chain import SupplyChain

    ### SupplyChain
    nominal_env = SupplyChain(b=0) # With b=0 uniform
    uncertainty_lvl = 0.5
    max_iter = 100
    #P_hat = empirical_nominal_kernel(nominal_env, N=1000)


    # Run REVI with uniform as nominal transition
    #Q_K, V_K = REVI(nominal_env, P_hat, uncertainty_lvl, max_iter)
    #print("Q_K:", Q_K)
    #print()
    #print("V_K:", V_K)
    p_hat = np.array([0,0.2,0.8]) 
    V = np.array([0,1,2]) 
    sigma = 0.5 

    print(find_inf_P(p_hat, V, sigma))
