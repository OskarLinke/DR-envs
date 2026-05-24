import numpy as np

class SupplyChain: 
    def __init__(self, n: int = 10, h: int = 1, p: int = 2, k: int = 3, gamma: float = 0.9, b: float = 2.5, m: int = 5) -> None: 
        self.n = n #Amount of goods we can buy. Defines both action and state space 
        self.h = h #Holding cost weight
        self.p = p #Lost sales cost weight 
        self.k = k #Shipping cost weight 
        self.gamma = gamma  
        self.t = 0 #Time
        self.S = 0 #State 
        self.b = b #How much extra weight is attributed to outcomes m and m+1. b=0.0 means regular uniform distribution

        assert 0 <= m < n ,"m must be in [0,n-1]"
        self.m = m #Which value to make more likely

    def cost(self, s, a, dt) -> float:
        cost = self.k if a > 0 else 0 
        holding_cost = self.h*(s+a-dt) 
        lost_sales_cost = self.p*(dt-s-a) 

        cost += holding_cost if holding_cost > 0 else 0
        cost += lost_sales_cost if lost_sales_cost > 0 else 0

        return cost * np.power(self.gamma, self.t)

    def pertubed_distribution(self): 
        probs = np.full(self.n + 1, (self.n - 1 - 2*self.b)/(np.square(self.n) -1))
        probs[self.m] = (self.b+1)/(self.n +1)
        probs[self.m +1] = probs[self.m] 
        
        return np.random.choice(np.arange(self.n + 1), size=1, p=probs)[0]


    def step(self, a) -> tuple[int, float]: 
        print(f"{a} items are ordered")
        dt = self.pertubed_distribution() 
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
