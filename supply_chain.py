import numpy as np
from numpy.typing import NDArray
from typing import Any


class SupplyChain: 
    """ 
    Class of the supply chain environment. Described in Liu et al (2022):
    https://proceedings.mlr.press/v162/liu22a/liu22a.pdf
    """ 
    def __init__(
        self,
        n: int = 10,
        h: int = 1,
        p: int = 2,
        k: int = 3,
        gamma: float = 0.9,
        b: float = 0.0,
        m: int = 5,
    ) -> None:
        """
        Initializes the environment.

        Args:
            n: Amount of goods we can buy. Defines both action and state space.
            h: Holding cost weight.
            p: Lost sales cost weight.
            k: Shipping cost weight.
            gamma: Discount factor.
            t: Current time step.
            S: Current state of the environment. Amount of goods in stock.
            b: Extra weight for outcomes m and m+1 (0.0 = uniform distribution).
            m: Which value to make more likely. Must be in [0, n-1].
            dt: Current market ask (pertubations), gets updated each step.
        """
        self.n = n
        self.h = h
        self.p = p
        self.k = k
        self.gamma = gamma
        self.t = 0
        self.S = 0
        self.b = b
        assert 0 <= m < n ,"m must be in [0, n-1]"
        self.m = m
        self.dt: int = 0

    def reward(self, s: int, a: int) -> float:
        """
        Calculates the reward a time t
        """
        cost = self.k if a > 0 else 0
        holding_cost = self.h*(s + a - self.dt)
        lost_sales_cost = self.p*(self.dt - s - a)

        cost += holding_cost if holding_cost > 0 else 0
        cost += lost_sales_cost if lost_sales_cost > 0 else 0

        return -cost


    def market_ask(self) -> int:
        """
        The amount of goods requested by the market. A perturbed normal
        distribution. When b = 0, it is a regular uniform distribution. 
        """
        probs = self.market_ask_distribution()        
        return np.random.choice(np.arange(self.n + 1),  p=probs)

    def market_ask_distribution(self) -> NDArray[Any]:
        """
        The distribution of market ask probabilities
        """
        fill_val = (self.n - 1 - 2*self.b)/(np.square(self.n) - 1)  # (n-1-2b)/(sqr(n)-1)
        probs = np.full(self.n + 1, fill_val)                       # n+1 long vector
        probs[self.m] = (self.b + 1)/(self.n + 1)                   # at m and m + 1 get higher prob
        probs[self.m + 1] = probs[self.m]
        # TODO: Shouldn't the two m indices be + 1? We have n+1 elements.
        return probs

    def step(self, a: int, verbose: bool = False) -> tuple[int, float]: 
        """
        Takes a step in the environment given an action. 
        """
        self.dt = self.market_ask() 
        reward = self.reward(self.S, a) * np.power(self.gamma, self.t)
        self.S = self.S + a - self.dt 
        if self.S < 0: 
            self.S = 0 
        self.t += 1 
        if verbose:
            print(f"{a} items are ordered")
            print(f"{self.dt} goods are requested by the market")
            print(f"The reward is {reward}")
            print(f"New state is {self.S}")
        return (self.S, reward)

    def legal_actions(self) -> int: 
        """
        Returns all legal actions as size of action space. The agent cannot 
        posses more than n goods at any time, i.e. more than self.n - self.S
        """
        return self.n - self.S

    def action_size(self) -> int:
        """
        Size of the action space (total num possible actions)
        """
        return self.n + 1

    def state_size(self) -> int:
        """
        Size of the state space (total num possible states)
        """
        return self.n + 1

    def reset(self, s: int = 0) -> None: 
        """
        Resets the environment. 
        """
        self.t = 0 
        self.S = s 

    def nominal_expected_reward(self) -> NDArray[Any]:
        """
        Get the expected reward for each state and action.
        Only implemented for uniform market ask (b=0)
        """
        R_exp = np.zeros((self.state_size(), self.action_size()))
        for s in range(self.state_size()):
            for a in range(self.action_size()):
                for n in range(self.n + 1):
                    self.dt = n
                    R_exp[s, a] += self.reward(s, a)
        R_exp /= (self.n + 1)

        self.reset()
        return R_exp

    def expected_reward_sa(
        self, s: int, a: int, market_ask_dist: NDArray[Any]
    ) -> float:
        """ 
        Get the expected reward for a single state-action pair 
        given some market ask distribution 
        """
        exp_r = 0
        for n in range(self.n + 1):
            self.dt = n
            exp_r += self.reward(s, a) * market_ask_dist[n]

        self.reset() # TODO: What about the reset state here? s=s?
        return exp_r

    def trans_prob_kernel_sa(
        self, s: int, a: int, market_ask_dist: NDArray[Any]
    ) -> NDArray[Any]:
        if s + a > self.n:
            raise ValueError("Illegal action taken. s + a must be less than self.n")

        P_ = np.zeros(self.state_size())
        for dt in range(self.n + 1):
            s_ = s + a - dt
            if s_ < 0:
                s_ = 0
            P_[s_] += market_ask_dist[dt]
        return P_

    def true_nominal_kernel(self) -> NDArray[Any]: 
        """
        Get the true nominal kernel for each state and action.
        Only implemented for uniform market ask (b=0)
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



if __name__ == "__main__": 
    def random_action(env: SupplyChain): 
        """
        Takes a random action from the set of allowed actions. 
        """
        return np.random.randint(env.legal_actions())

    env = SupplyChain(m = 5, b = 2)
    
    market_dist = env.market_ask_distribution()
    print(f"Transition probability function from s = 0, a = 1: {env.trans_prob_kernel_sa(0, 1, market_dist)}")
