import numpy as np
from numpy.typing import NDArray
from typing import Any
from my_typing import ProbVector


class SupplyChain:
    """Supply-chain inventory environment from Liu et al. (2022).

    State is the current inventory level in [0, n]. Each step the agent
    orders `a` units; demand `dt` is drawn from a (possibly perturbed)
    market-ask distribution; cost combines shipping (if a > 0), holding,
    and lost-sales penalties.

    References
    ----------
    .. [1] Liu, Z. et al. (2022). https://proceedings.mlr.press/v162/liu22a/liu22a.pdf
    """
    def __init__(
        self,
        n: int = 10,
        h: int = 1,
        l: int = 2,
        k: int = 3,
        gamma: float = 0.9,
        b: float = 0.0,
        m: int = 5,
    ) -> None:
        """Initialise the environment.

        Parameters
        ----------
        n : int, default 10
            Maximum order size; also sets state and action space cardinality.
        h : int, default 1
            Holding cost weight.
        l : int, default 2
            Lost-sales cost weight.
        k : int, default 3
            Shipping cost weight (charged when a > 0).
        gamma : float, default 0.9
            Discount factor.
        b : float, default 0.0
            Extra weight placed on market-ask outcomes m and m+1.
            b = 0 yields a uniform distribution.
        m : int, default 5
            Index of the market-ask outcome to up-weight. Must lie in
            [0, n-1].
        """
        self.n = n
        self.h = h
        self.l = l
        self.k = k
        self.gamma = gamma
        self.t = 0
        self.s = 0
        self.b = b
        assert 0 <= m < n ,"m must be in [0, n-1]"
        self.m = m
        self.dt: int = 0

    def reward(self, s: int, a: int) -> int:
        """Compute the (non-positive) reward at inventory `s`, order `a`.

        Uses the current market ask `self.dt` to determine holding vs
        lost-sales costs.
        """
        cost = self.k if a > 0 else 0
        holding_cost = self.h*(s + a - self.dt)
        lost_sales_cost = self.l*(self.dt - s - a)

        cost += holding_cost if holding_cost > 0 else 0
        cost += lost_sales_cost if lost_sales_cost > 0 else 0

        return -cost


    def market_ask(self) -> int:
        """Sample the demand for the current step.

        Draw from the market-ask distribution (uniform when ``b = 0``,
        otherwise extra mass on indices m and m+1).
        """
        probs = self.market_ask_distribution()        
        return np.random.choice(np.arange(self.n + 1),  p=probs)

    def market_ask_distribution(self) -> NDArray[Any]:
        """Return the (length-``n+1``) market-ask probability vector."""
        fill_val = (self.n - 1 - 2*self.b)/(np.square(self.n) - 1)  # (n-1-2b)/(sqr(n)-1)
        probs = np.full(self.n + 1, fill_val)                       # n+1 long vector
        probs[self.m] = (self.b + 1)/(self.n + 1)                   # at m and m + 1 get higher prob
        probs[self.m + 1] = probs[self.m]
        return probs


    def step(self, a: int, verbose: bool = False) -> tuple[int, float]:
        """Advance the environment by one step.

        Samples a market ask, returns the discounted reward, and updates
        the inventory state (clipped at zero).

        Returns
        -------
        s_next : int
            New inventory level.
        reward : float
            Discounted reward for this step.
        """
        self.dt = self.market_ask() 
        reward = (self.gamma**self.t) * self.reward(self.s, a)
        self.s = self.s + a - self.dt 
        if self.s < 0: 
            self.s = 0 
        self.t += 1 
        if verbose:
            print(f"{a} items are ordered")
            print(f"{self.dt} goods are requested by the market")
            print(f"The reward is {reward}")
            print(f"New state is {self.s}")
        return (self.s, reward)

    def legal_actions(self) -> int:
        """Return the largest legal order at the current state.

        Inventory is capped at ``n``, so the maximum order is
        ``self.n - self.s``.
        """
        return self.n - self.s

    def action_size(self) -> int:
        """Cardinality of the action space."""
        return self.n + 1

    def state_size(self) -> int:
        """Cardinality of the state space."""
        return self.n + 1

    def reset(self, s: int = 0) -> None:
        """Reset time to zero and inventory to ``s``."""
        self.t = 0 
        self.s = s 

    
    ### Get nominal Distributions #############################################
    def nominal_expected_reward(self) -> NDArray[Any]:
        """Expected reward ``R(s, a)`` under the nominal market ask.

        Only implemented for the uniform case (``b = 0``).
        """
        R_exp = np.zeros((self.state_size(), self.action_size()))
        for s in range(self.state_size()):
            for a in range(self.action_size()):
                for n in range(self.n + 1):
                    self.dt = n
                    R_exp[s, a] += self.reward(s, a)
        R_exp /= (self.n + 1)

        return R_exp

    def nominal_kernel(self) -> NDArray[Any]:
        """Analytic transition kernel ``P(s, a, s')`` under uniform market ask.

        Only implemented for ``b = 0``.

        Raises
        ------
        NotImplementedError
            If ``self.b != 0``.
        """
        if self.b != 0: 
            raise NotImplementedError(
                "True nominal kernel only implemented for uniform market ask"
                f"i.e. b=0 not b={self.b}"
            )
        P_0 = np.zeros((self.state_size(), self.action_size(), self.state_size())) 
        for s in range(self.state_size()): 
            for a in range(self.action_size()): 
                if s + a > self.n: 
                    continue
                for s_ in range(self.state_size()): 
                    if s_ > s + a: 
                        P_0[s, a, s_] = 0 
                    elif 0 < s_ <= s + a: 
                        P_0[s, a, s_] = 1 / (self.n + 1) 
                    elif s_ == 0: 
                        P_0[s, a, s_] = 1 - ((s + a) / (self.n + 1))

        return P_0

    ### For finding inf (p,r) in C_s,a^theta ##################################
    def expected_reward_sa(
        self, s: int, a: int, market_ask_dist: ProbVector
    ) -> float:
        """Expected reward at ``(s, a)`` under the given market-ask distribution."""
        old_dt = self.dt
        exp_r = 0
        for n in range(self.n + 1):
            self.dt = n
            exp_r += self.reward(s, a) * market_ask_dist[n]
        # Reset market ask
        self.dt = old_dt

        return exp_r

    def transition_kernel_sa(
        self, s: int, a: int, market_ask_dist: ProbVector
    ) -> ProbVector:
        """Next-state distribution at ``(s, a)`` under the given market-ask distribution.

        Raises
        ------
        ValueError
            If ``s + a > self.n`` (illegal action).
        """
        if s + a > self.n:
            raise ValueError("Illegal action taken. s + a must be less than self.n")

        P_ = np.zeros(self.state_size())
        for dt in range(self.n + 1):
            s_ = s + a - dt
            if s_ < 0:
                s_ = 0
            P_[s_] += market_ask_dist[dt]
        return P_

    def find_num_possible_rewards(self) -> int:
        """Upper bound on the number of distinct reward values.

        Returns ``|worst_case_reward| + 1`` so that every reward value
        ``-r`` indexes into a length-this array. The worst case depends
        on which cost weight dominates.
        """
        old_dt = self.dt
        r = None
        # If k dominates
        if self.k > max(self.l, self.h):
            if self.h > self.l:
                # worst case is buy 10, inventory 0, market ask 0
                # high shipping and holding cost
                self.dt = 0
                r = self.reward(s=0, a=self.n)
            else:
                # high shipping and lost sales cost
                self.dt = 10
                r = self.reward(s=0, a=1)
        # Else  if lost sales dominates
        elif self.l > max(self.h, self.k):
            # High ask, no buy, no inventory 
            self.dt = 10
            r = self.reward(s=0, a=0)
        else:
            # Either holding dominates or all equal. Worst case either way
            self.dt = 0
            r = self.reward(s=0, a=self.n)

        self.dt = old_dt
        if r is None:
            raise ValueError("r was supposed to be assigned")
        return abs(r) + 1


    def reward_probabilities_sa(
        self, s: int, a: int, market_ask_dist: ProbVector
    ) -> ProbVector:
        """Reward distribution at ``(s, a)`` under the given market-ask distribution.

        Position ``i`` of the returned vector holds the probability mass
        for reward value ``-i`` (rewards are non-positive).
        """
        R = self.find_num_possible_rewards()
        r_probs = np.zeros(R)

        old_dt = self.dt
        for n in range(self.n + 1):
            self.dt = n
            # Reward is non-positive (-cost); store at index -reward so position i
            # holds probability mass for reward value -i.
            r_probs[-self.reward(s, a)] += market_ask_dist[n]
        # Reset market ask
        self.dt = old_dt

        return r_probs


if __name__ == "__main__": 
    def random_action(env: SupplyChain): 
        """
        Takes a random action from the set of allowed actions. 
        """
        return np.random.randint(env.legal_actions())

    env = SupplyChain(m = 5, b = 2)
    
    market_dist = env.market_ask_distribution()
    print(f"Transition probability function from s = 0, a = 1: {env.transition_kernel_sa(0, 1, market_dist)}")
