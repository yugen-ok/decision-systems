import numpy as np
from collections import deque
from matplotlib import pyplot as plt

class InventorySystem:
    def __init__(
        self,
        *,
        delay,
        init_inventory,
        clearance_factor,
        inventory_horizon,
        alpha,
        gamma,
        band_fraction
    ):
        self.clearance_factor = clearance_factor
        self.delay = delay
        self.inventory_horizon = inventory_horizon

        self.alpha = alpha
        self.gamma = gamma

        # state
        self.inventory = init_inventory
        self.backlog = 0.0
        self.demand_hat = 0.0
        self.lost_sales = 0.0

        # pipeline (fixed delay)
        self.pipeline = deque([0.0] * delay, maxlen=delay)
        self.prev_inventory_position = init_inventory + sum(self.pipeline) - self.backlog

        # history
        self.history = {
            "inventory": [],
            "backlog": [],
            "demand": [],
            "order": [],
            "arrivals": [],
            "target": [],
            "lost_sales": [],
        }

        self.band_fraction = band_fraction

    # -----------------------
    # ONE STEP
    # -----------------------

    def step(self, demand):
        # --- update demand forecast (EWMA) ---
        # --- update demand forecast (EWMA), with warm-start ---
        if len(self.history["demand"]) == 0:
            self.demand_hat = demand
        else:
            self.demand_hat = (1 - self.alpha) * self.demand_hat + self.alpha * demand

        target_inventory_position = (
                self.delay * self.demand_hat
                + self.inventory_horizon * self.demand_hat
        )

        # --- inventory position BEFORE ordering ---
        inventory_position = (
                self.inventory
                + sum(self.pipeline)
                - self.backlog
        )

        # --- order decision (P + damping) ---
        order = self.demand_hat + self.gamma * (target_inventory_position - inventory_position)
        order = max(0.0, order)

        # --- pipeline update ---
        arrivals = self.pipeline.popleft()
        self.pipeline.append(order)

        # --- fulfill demand ---
        max_clear = self.clearance_factor * max(self.demand_hat, 1e-6)
        available = self.inventory + arrivals
        fulfilled = min(available, demand, max_clear)

        self.inventory = available - fulfilled
        self.backlog = demand + self.backlog - fulfilled

        # --- record history ---
        self.history["inventory"].append(self.inventory)
        self.history["backlog"].append(self.backlog)
        self.history["demand"].append(demand)
        self.history["order"].append(order)
        self.history["arrivals"].append(arrivals)
        self.history["target"].append(target_inventory_position)

    # -----------------------
    # ANALYSIS
    # -----------------------

    def analyze(self, *, burn_in=0, plot=False):
        demand = np.array(self.history["demand"][burn_in:])
        orders = np.array(self.history["order"][burn_in:])
        backlog = np.array(self.history["backlog"][burn_in:])
        inventory = np.array(self.history["inventory"][burn_in:])
        target = np.array(self.history["target"][burn_in:])
        arrivals = np.array(self.history["arrivals"][burn_in:])
        print(self.history)
        # bullwhip
        demand_var = np.var(demand)
        orders_var = np.var(orders)
        bullwhip = orders_var / demand_var if demand_var > 0 else np.nan

        # backlog metrics
        backlog_mean = backlog.mean()
        backlog_std = backlog.std()
        backlog_spike_threshold = backlog_mean + 2.5 * backlog_std
        backlog_spikes = np.sum(
            backlog > backlog_spike_threshold
        )


        # inventory deviation
        inv_error = inventory - target
        inv_mad = np.mean(np.abs(inv_error))
        inv_std = np.std(inv_error)

        # dynamic in-band fraction
        band = self.band_fraction * target
        in_band = np.abs(inv_error) <= band
        inv_in_band = in_band.mean()



        if plot:

            # ---- plotting ----
            t = np.arange(len(demand))

            plt.figure(figsize=(14, 12))

            plt.subplot(4, 1, 1)
            plt.plot(t, demand, label="Demand")
            plt.plot(t, orders, label="Order")
            plt.axvline(burn_in, color="gray", linestyle="--", alpha=0.6)
            plt.title("Demand vs Orders")
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(t, inventory, label="Inventory")
            plt.plot(t, target, label="Target", linestyle="--")

            upper_band = target * (1 + self.band_fraction)
            lower_band = target * (1 - self.band_fraction)

            plt.fill_between(
                t,
                lower_band,
                upper_band,
                alpha=0.2,
                label="Target Band"
            )

            plt.axvline(burn_in, color="gray", linestyle="--", alpha=0.6)
            plt.title("Inventory Tracking")
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(t, backlog, label="Backlog")
            plt.axhline(backlog_spike_threshold, color="red", linestyle="--", alpha=0.4, label="Backlog spike threshold")
            plt.axvline(burn_in, color="gray", linestyle="--", alpha=0.6)
            plt.title("Backlog")
            plt.legend()

            plt.subplot(4, 1, 4)
            plt.plot(t, arrivals, label="Arrivals")
            plt.axvline(burn_in, color="gray", linestyle="--", alpha=0.6)
            plt.title("Arrivals")
            plt.legend()

            plt.xlabel("Time step")
            plt.tight_layout()
            plt.show()

        return {
            "bullwhip": bullwhip,
            "backlog_mean": backlog_mean,
            "backlog_std": backlog_std,
            "backlog_spikes": int(backlog_spikes),
            "inv_mad": inv_mad,
            "inv_std": inv_std,
            "inv_in_band": inv_in_band,
        }
