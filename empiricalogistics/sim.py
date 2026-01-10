"""
Core difference from logisticsimple:

C_{t+1} doesn't go down by D_t. rather, C_t is just a limiter of how much of D_t can be processed today and how much goes to backlog


Best config so far:

delay = 4
inventory_horizon = 8
alpha = 0.1
gamma = 0.047863
clearance_factor = 12624.299878
score = 0.708072

"""

import os
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from CapacitySystem import CapacitySystem


K = 0 # It doesn't seem to make a difference

# --------------------------------------------------
# LOAD PATHS
# --------------------------------------------------

load_dotenv()

TRAIN_PATH = Path(os.environ["TRAIN_PATH"])
EVAL_PATH  = Path(os.environ["EVAL_PATH"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
#eval_df  = pd.read_csv(EVAL_PATH,  parse_dates=["date"])

# ensure ordering
train_df = train_df.sort_values(["portid", "date"])
#eval_df  = eval_df.sort_values(["portid", "date"])

# --------------------------------------------------
# HYPERPARAMETER GRID
# --------------------------------------------------

DELAYS = [2, 4, 6]
HORIZONS = [4, 8]
ALPHAS = np.logspace(-3, -1, 6)
GAMMAS = np.logspace(-2, -0.3, 6)

tpc = (train_df.loc[train_df["demand"] > 0, "processed"] /
       train_df.loc[train_df["demand"] > 0, "demand"]).clip(lower=0)

base = tpc.quantile([0.50, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99]).to_numpy()

CLEARANCE_FACTORS = base[[1, 3, 5]]  # pick 3 quantiles




# --------------------------------------------------
# Helpers
# --------------------------------------------------


def score_smape_01(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    smape = np.mean(
        2.0 * np.abs(y_pred - y_true)
        / (np.abs(y_pred) + np.abs(y_true) + eps)
    )
    return float(1.0 - 0.5 * smape)  # in [0,1]

def per_port_scores_for_config(
    train_df,
    *,
    config,
    K,
):
    """
    Returns dict: {portid: score}
    """
    delay = int(config["delay"])
    horizon = int(config["inventory_horizon"])
    alpha = config["alpha"]
    gamma = config["gamma"]
    clearance_factor = config["clearance_factor"]

    port_scores = {}

    for portid, port_df in train_df.groupby("portid"):
        if len(port_df) <= K + 1:
            continue

        mean_processed = port_df["processed"].mean()
        init_inventory = (delay + horizon) * mean_processed

        inv_sys = CapacitySystem(
            delay=delay,
            init_inventory=init_inventory,
            inventory_horizon=horizon,
            alpha=alpha,
            gamma=gamma,
            clearance_factor=clearance_factor,
        )

        # warmup
        for r in port_df.iloc[:K].itertuples():
            demand_tons = r.demand * clearance_factor
            inv_sys.step(demand=demand_tons, processed=r.processed)

        # predict
        y_true = []
        y_pred = []
        for r in port_df.iloc[K:].itertuples():
            demand_tons = r.demand * clearance_factor
            y_pred.append(inv_sys.step(demand=demand_tons))
            y_true.append(r.processed)

        score = score_smape_01(y_true, y_pred)
        port_scores[portid] = score

    return port_scores


# --------------------------------------------------
# GRID SEARCH
# --------------------------------------------------

results = []

combs = list(itertools.product(
    DELAYS, HORIZONS, ALPHAS, GAMMAS, CLEARANCE_FACTORS
))

combs = random.sample(combs, 50)

for delay, horizon, alpha, gamma, clearance_factor in combs:
    port_scores = []
    total_abs_error = 0.0
    total_true = 0.0

    for portid, port_df in train_df.groupby("portid"):

        if len(port_df) <= K + 1:
            continue

        # -----------------------------
        # INIT INVENTORY (KEY POINT)
        # -----------------------------
        mean_processed = port_df["processed"].mean()
        init_inventory = (delay + horizon) * mean_processed

        inv_sys = CapacitySystem(
            delay=delay,
            init_inventory=init_inventory,
            inventory_horizon=horizon,
            alpha=alpha,
            gamma=gamma,
            clearance_factor=clearance_factor,
        )

        for r in port_df.iloc[:K].itertuples():
            demand_tons = r.demand * clearance_factor
            inv_sys.step(demand=demand_tons, processed=r.processed)

        y_true = []
        y_pred = []
        for r in port_df.iloc[K:].itertuples():
            demand_tons = r.demand * clearance_factor
            pred = inv_sys.step(demand=demand_tons)  # processed=None → predicted
            y_pred.append(pred)
            y_true.append(r.processed)

        eps = 1e-6
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + eps))
        score = 1.0 - 0.5 * smape  # in [0,1]

        port_scores.append(score)

    mean_score = float(np.mean(port_scores))

    result = {
        "delay": delay,
        "inventory_horizon": horizon,
        "alpha": alpha,
        "gamma": gamma,
        "clearance_factor": clearance_factor,
        "score": mean_score,
    }

    results.append(result)

    print(
        f"delay={delay:2d} | "
        f"horizon={horizon:2d} | "
        f"alpha={alpha:.4f} | "
        f"gamma={gamma:.4f} | "
        f"clearance={clearance_factor:.1f} | "
        f"WAPE={mean_score:.4f}"
    )
    print('-----------------')

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values("score")
results_df.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)

best = results_df.iloc[-1]

print("\nBEST CONFIG:")
print(best)

# Stat checks

print(results_df.score.describe(percentiles=[0.9, 0.95, 0.99]))

# p-value
p = np.mean(results_df.score >= best.score)
print(p)

# Effect size
effect = best.score - results_df.score.mean()
print(effect)

# Robustness

best_config = best.to_dict()
random_config = results_df.sample(1, random_state=42).iloc[0].to_dict()

scores_best = per_port_scores_for_config(train_df, config=best_config, K=K)
scores_rand = per_port_scores_for_config(train_df, config=random_config, K=K)

common_ports = set(scores_best) & set(scores_rand)

dominance = np.mean(
    [scores_best[p] > scores_rand[p] for p in common_ports]
)

print("Fraction of ports where BEST beats RANDOM:", dominance)
