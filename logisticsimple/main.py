"""
Example setups:

no_bw = InventorySystem(delay=5, alpha=0.01, gamma=0.25, target_inventory=80, init_inventory=80)
small_bw = InventorySystem(delay=3, alpha=0.05, gamma=0.25, target_inventory=80, init_inventory=80)
mild_bw = InventorySystem(delay=6, init_inventory=50, target_inventory=100, alpha=0.5, gamma=2.0)
strong_bw = InventorySystem(delay=10, init_inventory=50, target_inventory=100, alpha=0.9, gamma=5.0)

"""
import math
import random
from pprint import pprint

from InventorySystem import *

FIELDS = ["inventory", "backlog", "demand", "order", "arrivals"]
METRICS = ["bullwhip", "backlog_mean", "backlog_std", "backlog_spikes", "inv_mad", "inv_std", "inv_in_band"]

RANDOM_SEED = 42
T = 1000

DELAY = 5
INIT_INV = 100
CLEARANCE_FACTOR = 1.1

INVENTORY_HORIZON = 100
K = 10
BAND_FRACTION = .05

ALPHA = .05
GAMMA = .25

BURN_IN = 50   # discard first steps for analysis
PLOT = True

rng = random.Random(RANDOM_SEED)

# Simple sim of daily demand
def demand_fn(t, *, base_level=20, daily_amp=6, weekly_amp=4,
              noise_std=2.0, trend=0.1, shock_prob=0.02, shock_scale=15,
              seed=42):
    """
    Realistic store demand generator.
    All parameters are environmental constants.
    """


    # deterministic structure
    daily_cycle = daily_amp * math.sin(2 * math.pi * t / 24)
    weekly_cycle = weekly_amp * math.sin(2 * math.pi * t / (24 * 7))
    trend_component = trend * t

    # stochastic noise
    noise = rng.gauss(0, noise_std)

    # rare demand shocks (promotions, weather, events)
    shock = shock_scale if rng.random() < shock_prob else 0

    demand = base_level + daily_cycle + weekly_cycle + trend_component + noise + shock

    return max(0.0, demand)


inv_sys = InventorySystem(alpha=ALPHA, gamma=GAMMA,
                          delay=DELAY, inventory_horizon=INVENTORY_HORIZON,
                          clearance_factor=CLEARANCE_FACTOR,
                          init_inventory=INIT_INV,
                          band_fraction=BAND_FRACTION)


for t in range(T):

    demand = demand_fn(t)
    inv_sys.step(demand)

    print_str = ''
    for field in FIELDS:
        print_str += f"{field}: {round(inv_sys.history[field][-1], 2)} "
    #print(print_str)

results = inv_sys.analyze(burn_in=BURN_IN, plot=PLOT)

pprint(results)