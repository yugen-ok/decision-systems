import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from empiricalogistics.CapacitySystem import CapacitySystem

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------

K = 30
WINDOW = 7        # days before / after congestion
MAX_PLOTS = 5     # how many episodes to visualize

best_config = {
    "delay": 4,
    "inventory_horizon": 8,
    "alpha": 0.1,
    "gamma": 0.047863,
    "clearance_factor": 12624.299878,
}


TRAIN_PATH = os.getenv("TRAIN_PATH")
train_df = pd.read_csv(TRAIN_PATH)

# ----------------------------
# HELPERS
# ----------------------------

def find_congestion_episodes(backlog):
    """Return list of (start, end) indices where backlog > 0."""
    episodes = []
    in_ep = False
    for i, b in enumerate(backlog):
        if b > 0 and not in_ep:
            start = i
            in_ep = True
        elif b == 0 and in_ep:
            episodes.append((start, i))
            in_ep = False
    if in_ep:
        episodes.append((start, len(backlog)))
    return episodes


def backlog_half_life(backlog, start):
    """Days until backlog halves."""
    b0 = backlog[start]
    if b0 <= 0:
        return 0
    for i in range(start + 1, len(backlog)):
        if backlog[i] <= 0.5 * b0:
            return i - start
    return np.inf


# ----------------------------
# MAIN ANALYSIS
# ----------------------------

episode_stats = []
plot_count = 0

for portid, port_df in train_df.groupby("portid"):
    if len(port_df) <= K + 5:
        continue

    port_df = port_df.reset_index(drop=True)

    mean_processed = port_df["processed"].mean()
    init_inventory = (best_config["delay"] + best_config["inventory_horizon"]) * mean_processed

    inv = CapacitySystem(
        delay=int(best_config["delay"]),
        init_inventory=init_inventory,
        inventory_horizon=int(best_config["inventory_horizon"]),
        alpha=best_config["alpha"],
        gamma=best_config["gamma"],
        clearance_factor=best_config["clearance_factor"],
    )

    # warmup
    for r in port_df.iloc[:K].itertuples():
        inv.step(demand=r.demand * best_config["clearance_factor"],
                 processed=r.processed)

    # simulate
    for r in port_df.iloc[K:].itertuples():
        inv.step(demand=r.demand * best_config["clearance_factor"])

    h = inv.history
    backlog = np.array(h["backlog"])
    demand = np.array(h["demand"])
    inventory = np.array(h["inventory"])
    order = np.array(h["order"])
    target = np.array(h["target"])

    episodes = find_congestion_episodes(backlog)

    for (start, end) in episodes:
        duration = end - start
        peak_backlog = backlog[start:end].max()
        hl = backlog_half_life(backlog, start)

        episode_stats.append({
            "portid": portid,
            "start": start,
            "end": end,
            "duration": duration,
            "peak_backlog": peak_backlog,
            "half_life": hl,
        })

        # ----------------------------
        # OPTIONAL PLOTTING
        # ----------------------------
        if plot_count < MAX_PLOTS:
            lo = max(0, start - WINDOW)
            hi = min(len(backlog), end + WINDOW)
            t = np.arange(lo, hi)

            plt.figure(figsize=(12, 8))

            plt.subplot(3, 1, 1)
            plt.plot(t, demand[lo:hi], label="Demand")
            plt.plot(t, inventory[lo:hi], label="Capacity")
            plt.axvspan(start, end, color="red", alpha=0.2)
            plt.legend()
            plt.title(f"Port {portid} – Demand vs Capacity")

            plt.subplot(3, 1, 2)
            plt.plot(t, backlog[lo:hi], label="Backlog")
            plt.axvspan(start, end, color="red", alpha=0.2)
            plt.legend()
            plt.title("Backlog")

            plt.subplot(3, 1, 3)
            plt.plot(t, order[lo:hi], label="Order")
            plt.plot(t, target[lo:hi], linestyle="--", label="Target")
            plt.axvspan(start, end, color="red", alpha=0.2)
            plt.legend()
            plt.title("Control Response")

            plt.tight_layout()
            plt.show()

            plot_count += 1

# ----------------------------
# SUMMARY STATISTICS
# ----------------------------

episodes_df = pd.DataFrame(episode_stats)

print("\n=== CONGESTION SUMMARY ===")
print("Number of congestion episodes:", len(episodes_df))
print("\nDuration (days):")
print(episodes_df["duration"].describe())

print("\nPeak backlog:")
print(episodes_df["peak_backlog"].describe())

print("\nBacklog half-life (days):")
print(episodes_df["half_life"].replace(np.inf, np.nan).describe())
