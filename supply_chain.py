from typing import Sequence
import numpy as np


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

        return -cost*np.power(self.gamma, self.t)

    def pertubed_distribution(self) -> int: 
        """
        The amount of goods requested by the market. A perturbed normal
        distribution. When b = 0, it is a regular uniform distribution. 
        """
        fill_val = (self.n - 1 - 2*self.b)/(np.square(self.n) - 1)
        probs = np.full(self.n + 1, fill_val)
        probs[self.m] = (self.b + 1)/(self.n + 1)
        probs[self.m + 1] = probs[self.m] 
        
        return np.random.choice(np.arange(self.n + 1),  p=probs)


    def step(self, a: int, verbose: bool = False) -> tuple[int, float]: 
        """
        Takes a step in the environment given an action. 
        """
        self.dt = self.pertubed_distribution() 
        reward = self.reward(self.S, a)
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

    def state_space(self) -> Sequence[int]:
        return range(self.n)

    def action_space(self) -> Sequence[int]:
        return range(self.legal_actions())

    def action_size(self) -> int:
        """
        Size of the action space (total num possible actions)
        """
        return self.n

    def state_size(self) -> int:
        """
        Size of the state space (total num possible states)
        """
        return self.n

    def reset(self) -> None: 
        """
        Resets the environment. 
        """
        self.t = 0 
        self.S = 0 

def random_action(env: SupplyChain): 
    """
    Takes a random action from the set of allowed actions. 
    """
    return np.random.randint(env.action_space())

if __name__ == "__main__": 
    env = SupplyChain() 
    for i in range(5): 
        env.step(random_action(env), True)
