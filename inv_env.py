#Gynasium environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from configs import Econ, Demand, Sim

class InventoryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, econ = Econ(), demand = Demand(), sim = Sim(), seed = None):
        super().__init__()
        self.econ = econ
        self.demand = demand
        self.sim = sim
        self.rng = np.random.default_rng(seed)

        # Our observation is : [inventory, forecast, day_of_week]
        high = np.array([self.sim.inv_max, 10 * self.demand.lam, 6], dtype = np.float32)
        low = np.array([0, 0, 0], dtype = np.float32)

        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        self.action_space = spaces.Discrete(self.sim.action_max + 1)

        self.reset(seed=seed)

    def _sample_demand(self, day):
        lam = self.demand.lam
        if self.demand.seasonal:
            lam = lam * self.demand.season_mult[day]
        return self.rng.poisson(lam = lam)

    def _get_obs(self):
        return np.array([self.inventory, self.forecast, self.day], dtype = np.float32)
    

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.day = 0
        self.inventory = int(self.sim.inv_max * 0.1)
        self.forecast = float(self.demand.lam)
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        #order quantity today
        action = int(action)
        #oders arrive immediately in the base version
        stock = self.inventory + action

        #demand realization
        d = self._sample_demand(self.day)
        sales = min(stock, d)
        unmet = max(0, d - stock)
        self.inventory = stock - sales

        #Economics
        revenue = self.econ.price * sales
        hold = self.econ.holding_cost * self.inventory
        stockout = self.econ.stockout_cost * unmet
        order_cost = self.econ.order_cost_var * action + (self.econ.order_cost_fix if action > 0 else 0)
        reward = revenue - hold - stockout - order_cost

        #Forecase update (EMA)
        self.forecast = self.sim.ema_alpha * d + (1 - self.sim.ema_alpha) * self.forecast

        #advance time
        self.t += 1
        self.day = (self.day + 1) % 7

        terminated = self.t >= self.sim.horizon
        truncated = False
        obs = self._get_obs()
        info = {"sales": sales, "demand": d, "unmet": unmet, "revenue": revenue, "profit": reward} 
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"t={self.t} day={self.day} inv={self.inventory:.0f} forecast={self.forecast:.1f}")