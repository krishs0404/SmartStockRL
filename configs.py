from dataclasses import dataclass

@dataclass
class Econ:
    price: float = 10.0
    holding_cost: float = 0.2
    stockout_cost: float = 3.0
    order_cost_var: float = 0.5
    order_cost_fix: float = 1.0

@dataclass
class Demand:
    lam: float = 8.0
    seasonal: bool = True
    season_mult = [0.8,0.9,1.0,1.1,1.2,1.0,0.7]

@dataclass
class Sim:
    horizon: int = 60
    inv_max: int = 200
    action_max : int = 20
    ema_alpha: float = 0.3