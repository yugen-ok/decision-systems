"""
Run empirical port logistics experiment with proper temporal validation.

Pipeline:
1. Load train and eval data
2. Extract port/cargo time series
3. Calibrate parameters on train only
4. Initialize simulator from last train state
5. Roll forward on eval arrivals
6. Evaluate and print results

Obtain data here:
https://portwatch.imf.org/datasets/959214444157458aad969389b3ebe1a0/about

Split to train and eval and place it in TRAIN_PATH and EVAL_PATH using split.py.

"""

import matplotlib.pyplot as plt
from pathlib import Path

import os

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from dotenv import load_dotenv


from EmpiricalLogistics import (
    CapacityCalibrator,
    NodeSimulator,
    SimplePolicy,
    compute_validation_metrics,
)



# --------------------------------------------------
# CONFIG
# --------------------------------------------------

load_dotenv()


TRAIN_PATH = Path(os.getenv('TRAIN_PATH'))
EVAL_PATH = Path(os.getenv('EVAL_PATH'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR'))
SAMPLE_FRAC = os.getenv('SAMPLE_FRAC') or .1


TRAIN_SAMPLE_FRAC = 1.0
EVAL_SAMPLE_FRAC  = 1.0

MIN_TRAIN_DAYS = 80
MIN_EVAL_DAYS  = 30

MAX_PORTS = 100      # cap for runtime control

plt.style.use('seaborn-v0_8-whitegrid')


class PortDataLoader:
    """Load and preprocess port training data"""

    def __init__(self, data_path):
        self.train_path = data_path
        self.df = None
        self.cargo_types = ['container', 'dry_bulk', 'general_cargo', 'roro', 'tanker']

    def load(self, sample_frac: float = 0.1) -> pd.DataFrame:
        """Load training data with sampling for speed"""
        print(f"Loading data from {self.train_path}...")

        # Read with date parsing
        self.df = pd.read_csv(
            self.train_path,
            parse_dates=['date'],
            low_memory=False
        )

        # Sample for faster iteration
        if sample_frac < 1.0:
            self.df = self.df.sample(frac=sample_frac, random_state=42)
            print(f"Sampled {sample_frac*100:.0f}% of data: {len(self.df):,} rows")

        # Sort by port and date
        self.df = self.df.sort_values(['portid', 'date']).reset_index(drop=True)

        print(f"Loaded {len(self.df):,} rows")
        print(f"Unique ports: {self.df['portid'].nunique()}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")

        return self.df

    def get_port_timeseries(self, portid: str, cargo_type: str) -> pd.DataFrame:
        """Extract time series for specific port and cargo type"""
        port_data = self.df[self.df['portid'] == portid].copy()

        # Extract relevant columns
        cols = {
            'date': 'date',
            'arriving': f'arriving_{cargo_type}_tons',
            'processed': f'processed_{cargo_type}_tons',
            'backlog': f'backlog_{cargo_type}_tons',
        }

        ts = port_data[list(cols.values())].copy()
        ts.columns = list(cols.keys())
        ts = ts.sort_values('date').reset_index(drop=True)

        return ts

    def find_active_ports(self, min_days: int = 100, min_avg_arriving: float = 100) -> List[Tuple[str, str]]:
        """Find ports with sufficient activity for calibration"""
        active = []

        for port in self.df['portid'].unique():
            port_data = self.df[self.df['portid'] == port]

            for cargo in self.cargo_types:
                arriving_col = f'arriving_{cargo}_tons'

                if arriving_col in port_data.columns:
                    arriving = port_data[arriving_col].fillna(0)

                    # Check activity criteria
                    n_days = len(arriving)
                    avg_arriving = arriving.mean()
                    pct_nonzero = (arriving > 0).mean()

                    if n_days >= min_days and avg_arriving >= min_avg_arriving and pct_nonzero > 0.1:
                        active.append((port, cargo, n_days, avg_arriving))

        # Sort by activity
        active.sort(key=lambda x: x[3], reverse=True)

        print(f"\nFound {len(active)} active port-cargo combinations")
        return [(p, c) for p, c, _, _ in active]


# -----------------------------
# MAIN
# -----------------------------

def main():

    print("=" * 90)
    print("EMPIRICAL VALIDATION: SAME-DAY PROCESSING PREDICTION")
    print("=" * 90)

    # 1. Load data
    train_loader = PortDataLoader(TRAIN_PATH)
    eval_loader  = PortDataLoader(EVAL_PATH)

    train_loader.load(sample_frac=SAMPLE_FRAC)
    eval_loader.load(sample_frac=SAMPLE_FRAC)

    active_ports = train_loader.find_active_ports(
        min_days=MIN_TRAIN_DAYS,
        min_avg_arriving=100,
    )

    results = []

    # 2. Loop ports
    for i, (port, cargo) in enumerate(active_ports[:MAX_PORTS], 1):

        ts_train = train_loader.get_port_timeseries(port, cargo)
        ts_eval  = eval_loader.get_port_timeseries(port, cargo)

        if len(ts_eval) < MIN_EVAL_DAYS:
            continue

        # 3. Calibrate on TRAIN only
        calibrator = CapacityCalibrator(ts_train)
        params = calibrator.calibrate()

        # 4. Initialize simulator at eval boundary
        sim = NodeSimulator(params)
        sim.reset(backlog_init=ts_train.iloc[-1]["backlog"])

        predicted = []
        actual = []

        # 5. Predict processing day-by-day (NO POLICY)
        for _, row in ts_eval.iterrows():

            arriving = row["arriving"]

            # capacity prediction uses backlog(t-1)
            C_base = params["C_base"]
            gamma  = params["gamma"]
            C_max  = params["C_max"]

            capacity_pred = np.clip(
                C_base - gamma * sim.backlog,
                0,
                C_max
            )

            processed_pred = min(sim.backlog + arriving, capacity_pred)

            # record
            predicted.append(processed_pred)
            actual.append(row["processed"])

            # update state using REAL processed
            # (important: we are testing prediction, not sim dynamics)
            sim.backlog = max(
                0,
                sim.backlog + arriving - row["processed"]
            )

        # 6. Evaluate
        df_pred = pd.DataFrame({
            "processed": actual,
            "backlog": ts_eval["backlog"].values,
        })

        df_sim = pd.DataFrame({
            "processed": predicted,
            "backlog": ts_eval["backlog"].values,
        })

        metrics = compute_validation_metrics(df_pred, df_sim)

        if not metrics:
            continue

        results.append({
            "port": port,
            "cargo": cargo,
            **metrics,
            "gamma": params["gamma"],
        })

    # 7. Aggregate answer
    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("FINAL ANSWER")
    print("=" * 90)

    print(f"Ports evaluated: {len(df)}")

    print(
        f"\nProcessed prediction correlation:\n"
        f"  median = {df['processed_correlation'].median():.3f}\n"
        f"  p25    = {df['processed_correlation'].quantile(0.25):.3f}\n"
        f"  p75    = {df['processed_correlation'].quantile(0.75):.3f}"
    )

    print(
        f"\nInterpretation:\n"
        f"- Correlation measures same-day capacity prediction\n"
        f"- This directly answers: can the port know today if it is overloaded?\n"
    )

    print(
        "Rule of thumb:\n"
        "- median corr ≥ 0.5 → actionable operational signal\n"
        "- corr rises with gamma → congestion feedback is the driver"
    )

    print("\nDone.")
    print("=" * 90)


if __name__ == "__main__":
    main()
