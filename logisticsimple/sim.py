"""
Example setups:

CLEARANCE_FACTOR = 1.1

# Low bullwhip, high inv_in_band
INVENTORY_HORIZON = 1.3
ALPHA = .05
GAMMA = .25


# Bullwhip
CLEARANCE_FACTOR = np.inf
INVENTORY_HORIZON = 3
ALPHA = 1
GAMMA = .25

# Congestion (backlog creep)
CLEARANCE_FACTOR = .9
INVENTORY_HORIZON = 2
ALPHA = 0.2
GAMMA = 0.4


# Congestion + Bullwhip:
CLEARANCE_FACTOR = .9
INVENTORY_HORIZON = 3
ALPHA = 1
GAMMA = 0.8


# Backlog spikes:
CLEARANCE_FACTOR = 1.05
INVENTORY_HORIZON = 2
ALPHA = 0.1
GAMMA = 0.4


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

CLEARANCE_FACTOR = .99
INVENTORY_HORIZON = 3
ALPHA = 1
GAMMA = 0.7

BURN_IN = 50   # discard first steps for analysis
PLOT = True

rng = random.Random(RANDOM_SEED)

# Simple sim of daily demand
def demand_fn(t, *, base_level=INIT_INV, daily_amp=6, weekly_amp=4,
              noise_std=2.0, trend=0.1, shock_prob=0.04, shock_scale=15,
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
    sigma_est = noise_std + 0.5 * daily_amp + 0.3 * weekly_amp
    shock = sigma_est * shock_scale if rng.random() < shock_prob else 0

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