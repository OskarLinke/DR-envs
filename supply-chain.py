import numpy as np

class SupplyChain: 
    def __init__(self, n = 10, h = 1, p = 2, k = 3, gamma = 0.9) -> None: 
        self.n = n #Amount of goods we can buy. Defines both action and state space 
        self.h = h #Holding cost weight
        self.p = p #Lost sales cost weight 
        self.k = k #Shipping cost weight 
        self.gamma = gamma  
        self.t = 0 #Time
        self.S = 0 #State 

    def cost(self, s, a, dt) -> float:
        cost = self.k if a > 0 else 0 
        holding_cost = self.h*(s+a-dt) 
        lost_sales_cost = self.p*(dt-s-a) 

        cost += holding_cost if holding_cost > 0 else 0
        cost += lost_sales_cost if lost_sales_cost > 0 else 0

        return cost * self.gamma**self.t

    def step(self, a) -> tuple[int, float]: 
        print(f"{a} items are ordered")
        dt = np.random.randint(0,self.n) 
        print(f"{dt} goods are requested by the market")
        cost = self.cost(self.S, a, dt) 
        print(f"The cost is {cost}")
        self.S = self.S + a - dt 
        if self.S < 0: 
            self.S = 0 
        print(f"New state is {self.S}")
        self.t += 1 
        return (self.S, cost)

    def action_space(self) -> int: 
        return self.n - self.S  

    def reset(self) -> None: 
        self.t = 0 
        self.S = 0 

def random_action(env: SupplyChain): 
    return np.random.randint(env.action_space())

if __name__ == "__main__": 
    env = SupplyChain() 
    for i in range(5): 
        env.step(random_action(env))
