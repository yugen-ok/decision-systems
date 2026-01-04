"""
Empirical Logistics Simulation

Extracts capacity dynamics from actual data.
Each node is modeled as an isolated unit that:
- Receives arriving workload (arriving_*_tons)
- Manages capacity to process workload
- Accumulates backlog when capacity insufficient
- Makes mobilization decisions to adjust capacity

(Note: mobilization corresponds to "orders" in many logistics systems)

Key dynamics:
- Capacity = C_base + mobilized - γ * backlog (congestion reduces capacity)
- Processed = min(backlog + arriving, capacity)
- Backlog carries over when processing insufficient
- Mobilization has delay (L days)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class CapacityCalibrator:
    """Calibrate capacity parameters from observed data"""

    def __init__(self, timeseries: pd.DataFrame):
        self.ts = timeseries
        self.params = {}

    def calibrate(self) -> Dict:
        """
        Calibrate capacity parameters from historical data only.
        Assumes the provided time series is strictly pre-evaluation.
        """

        # Clean data
        arriving = self.ts['arriving'].fillna(0)
        processed = self.ts['processed'].fillna(0)
        backlog = self.ts['backlog'].fillna(0)

        # Filter out zero periods for more accurate estimates
        active = (arriving > 0) | (backlog > 0)

        # 1) Baseline capacity: median processing when backlog is low
        low_backlog_threshold = backlog.quantile(0.25)
        low_backlog_mask = (backlog <= low_backlog_threshold) & active

        if low_backlog_mask.sum() > 10:
            C_base = processed[low_backlog_mask].median()
        else:
            C_base = processed[active].median() if active.sum() > 0 else 100.0

        # 2) Max capacity: 95th percentile of processing
        C_max = processed[active].quantile(0.95) if active.sum() > 0 else C_base * 2

        # 3) Estimate congestion penalty (gamma)
        # How much does processing drop per unit backlog increase?
        gamma = self._estimate_congestion_penalty(arriving, processed, backlog, active)

        # 4) Mobilization delay (L): estimate from backlog response time
        L = self._estimate_mobilization_delay(backlog)

        # 5) Max mobilization order
        O_max = max(C_max * 0.5, C_base * 0.3)

        self.params = {
            'C_base': float(C_base),
            'C_max': float(C_max),
            'gamma': float(gamma),
            'L': int(L),
            'O_max': float(O_max),
            'mean_arriving': float(arriving.mean()),
            'std_arriving': float(arriving.std()),
        }

        return self.params

    def _estimate_congestion_penalty(self, arriving, processed, backlog, active) -> float:
        """Estimate how much backlog reduces processing capacity"""

        # Group by similar demand levels, compare processing vs backlog
        df = pd.DataFrame({
            'arriving': arriving,
            'processed': processed,
            'backlog': backlog,
        })
        df = df[active].copy()

        if len(df) < 20:
            return 0.01  # Default small penalty

        # Compute processing shortfall
        df['shortfall'] = df['arriving'] - df['processed']
        df['shortfall'] = df['shortfall'].clip(lower=0)

        # Estimate gamma via simple regression concept
        # When backlog increases, how much does processing capacity drop?
        if df['backlog'].std() > 0 and df['shortfall'].std() > 0:
            cov = df[['backlog', 'shortfall']].cov().iloc[0, 1]
            var_backlog = df['backlog'].var()
            gamma = max(0, cov / var_backlog) if var_backlog > 0 else 0.01

            # Scale to reasonable range
            gamma = np.clip(gamma, 0.0001, 0.1)
        else:
            gamma = 0.01

        return gamma

    def _estimate_mobilization_delay(self, backlog) -> int:
        """Estimate delay from backlog autocorrelation structure"""

        # Simple heuristic: look at backlog response time
        # If backlog spikes, how many days until it starts declining?

        if len(backlog) < 20:
            return 3  # Default

        # Compute differences
        backlog_diff = backlog.diff().fillna(0)

        # Find spikes (large increases)
        spikes = backlog_diff > backlog_diff.quantile(0.8)

        if spikes.sum() < 3:
            return 3

        # After spike, how long until decrease?
        delays = []
        spike_indices = np.where(spikes)[0]

        for idx in spike_indices[:10]:  # Check first 10 spikes
            if idx + 10 < len(backlog):
                # Look ahead for when backlog starts decreasing
                future = backlog.iloc[idx:idx+10]
                decreases = (future.diff() < 0)

                if decreases.sum() > 0:
                    first_decrease = decreases.idxmax()
                    delay = first_decrease - idx
                    delays.append(delay)

        if delays:
            L = int(np.median(delays))
            L = np.clip(L, 1, 14)  # Reasonable range
        else:
            L = 3

        return L


class NodeSimulator:
    """
    Single-node capacity flow simulator

    State: backlog, capacity, pipeline (delayed mobilization orders)
    Action: mobilization order (how much capacity to add)
    Dynamics: congestion reduces capacity, mobilization arrives with delay
    """

    def __init__(self, params: Dict):
        self.params = params
        self.reset()

    def reset(self, backlog_init: float = 0.0, capacity_init: float | None = None):
        """
        Initialize simulator state.

        Parameters
        ----------
        backlog_init : float
            Backlog at t0-1 (conditioning on history)
        capacity_init : float | None
            Initial capacity. Defaults to C_base if None.
        """
        self.backlog = float(backlog_init)
        self.capacity = (
            float(capacity_init)
            if capacity_init is not None
            else float(self.params['C_base'])
        )

        self.pipeline = []
        self.day = 0

        self.history = {
            'day': [],
            'arriving': [],
            'processed': [],
            'backlog': [],
            'capacity': [],
            'mobilization_order': [],
        }

    def step(self, arriving_tons: float, mobilization_order: float = 0.0) -> Dict:
        """Execute one day of simulation"""

        # Cap mobilization
        mobilization_order = np.clip(mobilization_order, 0, self.params['O_max'])

        # Add mobilization to pipeline (arrives after L days)
        if mobilization_order > 0:
            arrival_day = self.day + self.params['L']
            self.pipeline.append((mobilization_order, arrival_day))

        # Check if any mobilization arrives today
        mobilized_today = 0.0
        self.pipeline = [
            (qty, day) for (qty, day) in self.pipeline
            if day != self.day or (mobilized_today := mobilized_today + qty, False)[1]
        ]

        # Update capacity with congestion penalty
        C_base = self.params['C_base']
        gamma = self.params['gamma']
        C_max = self.params['C_max']

        self.capacity = C_base + mobilized_today - gamma * self.backlog
        self.capacity = np.clip(self.capacity, 0, C_max)

        # Process workload
        available_work = self.backlog + arriving_tons
        processed = min(available_work, self.capacity)

        # Update backlog
        self.backlog = max(0, available_work - processed)

        # Record history
        self.history['day'].append(self.day)
        self.history['arriving'].append(arriving_tons)
        self.history['processed'].append(processed)
        self.history['backlog'].append(self.backlog)
        self.history['capacity'].append(self.capacity)
        self.history['mobilization_order'].append(mobilization_order)

        self.day += 1

        return {
            'processed': processed,
            'backlog': self.backlog,
            'capacity': self.capacity,
        }

    def get_history_df(self) -> pd.DataFrame:
        """Return history as DataFrame"""
        return pd.DataFrame(self.history)


class SimplePolicy:
    """Simple reactive mobilization policy"""

    def __init__(self, backlog_threshold: float = 1000, mobilization_rate: float = 0.3):
        self.backlog_threshold = backlog_threshold
        self.mobilization_rate = mobilization_rate

    def decide(self, backlog: float, capacity: float, params: Dict) -> float:
        """Decide mobilization based on backlog"""

        if backlog > self.backlog_threshold:
            # Mobilize proportional to excess backlog
            excess = backlog - self.backlog_threshold
            order = self.mobilization_rate * excess
        else:
            order = 0.0

        return order


def compute_validation_metrics(
    actual: pd.DataFrame,
    simulated: pd.DataFrame
) -> Dict:
    """Compute metrics comparing simulation to actual data"""

    df = pd.DataFrame({
        'processed_actual': actual['processed'].values,
        'processed_sim': simulated['processed'].values,
        'backlog_actual': actual['backlog'].values,
        'backlog_sim': simulated['backlog'].values,
    })

    valid = df.dropna()

    if len(valid) == 0:
        return {}

    metrics = {}

    pa = valid['processed_actual'].values
    ps = valid['processed_sim'].values

    ba = valid['backlog_actual'].values
    bs = valid['backlog_sim'].values

    metrics['processed_mae'] = np.mean(np.abs(pa - ps))
    metrics['processed_rmse'] = np.sqrt(np.mean((pa - ps) ** 2))
    metrics['processed_mape'] = np.mean(np.abs((pa - ps) / (pa + 1e-6))) * 100

    metrics['backlog_mae'] = np.mean(np.abs(ba - bs))
    metrics['backlog_rmse'] = np.sqrt(np.mean((ba - bs) ** 2))

    if np.std(pa) > 0 and np.std(ps) > 0:
        metrics['processed_correlation'] = np.corrcoef(pa, ps)[0, 1]
    else:
        metrics['processed_correlation'] = 0.0

    if np.std(ba) > 0 and np.std(bs) > 0:
        metrics['backlog_correlation'] = np.corrcoef(ba, bs)[0, 1]
    else:
        metrics['backlog_correlation'] = 0.0

    return metrics

