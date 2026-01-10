import numpy as np
from collections import deque
from matplotlib import pyplot as plt

class CapacitySystem:
    def __init__(
        self,
        *,
        delay,
        init_inventory,
        inventory_horizon,
        alpha,
        gamma,
        clearance_factor,
    ):
        self.delay = delay
        self.inventory_horizon = inventory_horizon

        self.alpha = alpha
        self.gamma = gamma

        self.clearance_factor = clearance_factor

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
            "inventory_position": [],
        }


    # -----------------------
    # ONE STEP
    # -----------------------

    def step(self, demand, processed=None):
        """
        demand    : observed demand (always required)
        processed : observed processed volume (optional)
                    if None → model predicts processed internally
        Returns: processed_used
        """

        # --- update demand forecast (EWMA) ---
        if len(self.history["demand"]) == 0:
            self.demand_hat = demand
        else:
            self.demand_hat = (1 - self.alpha) * self.demand_hat + self.alpha * demand

        # --- target capacity buffer ---
        target_inventory_position = (
                (self.delay + self.inventory_horizon) * self.demand_hat
        )

        # --- inventory position BEFORE ordering ---
        inventory_position = (
                self.inventory
                + sum(self.pipeline)
                - self.backlog
        )

        # --- capacity adjustment decision ---
        order = self.demand_hat + self.gamma * (target_inventory_position - inventory_position)
        order = max(0.0, order)

        # --- delayed capacity adjustment ---
        arrivals = self.pipeline.popleft()
        self.pipeline.append(order)

        # --- flow constraints ---
        max_clear = self.clearance_factor * max(self.demand_hat, 1e-6)

        total_demand = demand + self.backlog
        total_supply = self.inventory + arrivals

        # --------------------------------------------------
        # PROCESSED: observed if given, else predicted
        # --------------------------------------------------
        if processed is None:
            processed_used = min(total_supply, total_demand, max_clear)
        else:
            processed_used = min(processed, total_supply, total_demand)

        # --- state update ---
        self.backlog = total_demand - processed_used

        # This is the core line that differentiates work capacity with daily regeneration (which is how it works for ports)
        # from physical stock inventory
        # (which is what we have in the vanilla InventorySystem class):
        #self.inventory = total_supply - processed_used

        # --- record history ---
        self.history["inventory"].append(self.inventory)
        self.history["backlog"].append(self.backlog)
        self.history["demand"].append(demand)
        self.history["order"].append(order)
        self.history["arrivals"].append(arrivals)
        self.history["target"].append(target_inventory_position)
        self.history["inventory_position"].append(inventory_position)

        return processed_used

    # -----------------------
    # ANALYSIS
    # -----------------------

    def analyze(self, *, burn_in=0, band_fraction=.05, plot=False):
        demand = np.array(self.history["demand"][burn_in:])
        orders = np.array(self.history["order"][burn_in:])
        backlog = np.array(self.history["backlog"][burn_in:])
        inventory = np.array(self.history["inventory"][burn_in:])
        target = np.array(self.history["target"][burn_in:])
        arrivals = np.array(self.history["arrivals"][burn_in:])
        inventory_position = np.array(self.history["inventory_position"][burn_in:])

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
        inv_error = inventory_position - target
        inv_mad = np.mean(np.abs(inv_error))
        inv_std = np.std(inv_error)

        # dynamic in-band fraction
        band = band_fraction * target
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
            plt.plot(t, inventory_position, label="Inv. Pos.")
            plt.plot(t, target, label="Target", linestyle="--")

            upper_band = target * (1 + band_fraction)
            lower_band = target * (1 - band_fraction)

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
