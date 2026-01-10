import math
import random
import itertools
import numpy as np

from InventorySystem import InventorySystem

# -----------------------
# CONFIG
# -----------------------

FIELDS = ["inventory", "backlog", "demand", "order", "arrivals"]
METRICS = [
    "bullwhip",
    "backlog_mean",
    "backlog_std",
    "backlog_spikes",
    "inv_mad",
    "inv_std",
    "inv_in_band",
]

T = 500
BURN_IN = 20
DELAY = 5
INIT_INV = 100
RANDOM_SEED = 42
CLEARANCE_FACTOR = 1.1

ALPHAS = np.logspace(-3, -1.3, 12)     # ~0.001 → ~0.05
GAMMAS = np.logspace(-2, 0, 12)        # 0.01 → 1.0
INVENTORY_HORIZONS = [1.3]



rng = random.Random(RANDOM_SEED)

# -----------------------
# DEMAND FUNCTION
# -----------------------

def demand_fn(
    t,
    base_level=20,
    daily_amp=6,
    weekly_amp=4,
    noise_std=2.0,
    trend=0.01,
    shock_prob=0.02,
    shock_scale=15,
):
    daily_cycle = daily_amp * math.sin(2 * math.pi * t / 24)
    weekly_cycle = weekly_amp * math.sin(2 * math.pi * t / (24 * 7))
    trend_component = trend * t

    noise = rng.gauss(0, noise_std)
    shock = shock_scale if rng.random() < shock_prob else 0

    demand = (
        base_level
        + daily_cycle
        + weekly_cycle
        + trend_component
        + noise
        + shock
    )

    return max(0.0, demand)

# -----------------------
# METRIC TARGETS
# -----------------------

METRIC_TARGETS = {
    "bullwhip": 1.0,
    "backlog_mean": 5.0,
    "backlog_std": 5.0,
    "backlog_spikes": 3.0,
    "inv_mad": 10.0,
    "inv_std": 15.0,
    "inv_in_band": 0.9,  # higher is better
}

LOWER_IS_BETTER = {
    "bullwhip",
    "backlog_mean",
    "backlog_std",
    "backlog_spikes",
    "inv_mad",
    "inv_std",
}

def metric_penalty(name, value):
    target = METRIC_TARGETS[name]

    if name in LOWER_IS_BETTER:
        return max(0.0, value / target - 1.0)
    else:
        return max(0.0, (target - value) / target)

# -----------------------
# SINGLE SIM RUN
# -----------------------

def run_sim(alpha, gamma, inventory_horizon):
    inv_sys = InventorySystem(
        alpha=alpha,
        gamma=gamma,
        delay=DELAY,
        inventory_horizon=inventory_horizon,
        init_inventory=INIT_INV,
        clearance_factor=CLEARANCE_FACTOR,
        band_fraction=0.05,
    )

    for t in range(T):
        inv_sys.step(demand_fn(t))

    metrics = inv_sys.analyze(burn_in=BURN_IN, plot=False)

    if metrics is None:
        raise RuntimeError("analyze() returned None")

    return metrics

# -----------------------
# GRID SEARCH
# -----------------------

results = []

for alpha, gamma, inventory_horizon in itertools.product(ALPHAS, GAMMAS, INVENTORY_HORIZONS):
    metrics = run_sim(alpha, gamma, inventory_horizon)

    penalties = {
        m: metric_penalty(m, metrics[m])
        for m in METRICS
    }

    score = max(penalties.values())  # minimax objective

    results.append({
        "alpha": alpha,
        "gamma": gamma,
        "ih": inventory_horizon,
        "score": score,
        "metrics": metrics,
        "penalties": penalties,
    })

# sort by worst-case metric deviation
results.sort(key=lambda r: r["score"])

# -----------------------
# OUTPUT
# -----------------------

print("\nTOP PARAMETER SETTINGS (most stable overall):\n")

for r in results[:5]:
    print(f"alpha={r['alpha']}, gamma={r['gamma']}, ih={r['ih']}, score={round(r['score'], 3)}")
    for m in ["bullwhip", "inv_in_band"]:
        print(f"  {m}: {round(r['metrics'][m], 3)}")
    print()


# -----------------------
# BULLWHIP-ONLY RANKING
# -----------------------

bullwhip_ranked = sorted(results, key=lambda r: r["metrics"]["bullwhip"])

print("\nTOP PARAMETER SETTINGS (lowest bullwhip):\n")

for r in bullwhip_ranked[-5:]:
    print(
        f"alpha={round(r['alpha'], 5)}, "
        f"gamma={round(r['gamma'], 5)}, "
        f"ih={r['ih']}, "
        f"bullwhip={round(r['metrics']['bullwhip'], 3)}, "
        f"score={round(r['score'], 3)}"
    )
