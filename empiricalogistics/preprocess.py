

from dotenv import load_dotenv

import os
import pandas as pd
from pathlib import Path
import numpy as np

rng = np.random.default_rng(seed=42)

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
LATENCY_PORT_FRACTION = 0.1   # e.g. 10% of ports

def subsample_ports(df, fraction, rng):
    ports = df["portid"].unique()
    k = max(1, int(len(ports) * fraction))
    rng.shuffle(ports)
    keep = set(ports[:k])
    return df[df["portid"].isin(keep)]

df = pd.read_csv(DATA_PATH)

# copy + ensure datetime
df = df.copy()
df["date"] = pd.to_datetime(df["date"], utc=True)

# -----------------------------
# CORE DERIVED COLUMNS
# -----------------------------

df["demand"] = df["portcalls"]
df["processed"] = df["import"] + df["export"]

# ensure proper ordering inside each port
df = df.sort_values(["portid", "date"]).reset_index(drop=True)


# -----------------------------
# OPTIONAL: drop tiny ports
# -----------------------------

def filter_short_ports(df, min_len=30):
    return df.groupby("portid").filter(lambda g: len(g) >= min_len)

df = filter_short_ports(df)


# -----------------------------
# SPLIT BY PORT IDS (SHUFFLED)
# -----------------------------

port_ids = df["portid"].unique()


rng.shuffle(port_ids)

n_ports = len(port_ids)
n_train = int(0.7 * n_ports)
n_eval  = int(0.15 * n_ports)

train_ports = set(port_ids[:n_train])
eval_ports  = set(port_ids[n_train:n_train + n_eval])
test_ports  = set(port_ids[n_train + n_eval:])

train_df = df[df["portid"].isin(train_ports)]
eval_df  = df[df["portid"].isin(eval_ports)]
test_df  = df[df["portid"].isin(test_ports)]

train_df_small = subsample_ports(train_df, LATENCY_PORT_FRACTION, rng)
eval_df_small  = subsample_ports(eval_df,  LATENCY_PORT_FRACTION, rng)
test_df_small  = subsample_ports(test_df,  LATENCY_PORT_FRACTION, rng)


# -----------------------------
# SAVE
# -----------------------------

out_dir = Path("data/Daily_Port_Activity_Data_and_Trade_Estimates/splits")
out_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(out_dir / "train.csv", index=False)
eval_df.to_csv(out_dir / "eval.csv", index=False)
test_df.to_csv(out_dir / "test.csv", index=False)

pct = LATENCY_PORT_FRACTION*100
pct_marker = int(pct) if pct.is_integer() else pct

train_df_small.to_csv(out_dir / f"train_{pct_marker}%.csv", index=False)
eval_df_small.to_csv(out_dir / f"eval_{pct_marker}%.csv", index=False)
test_df_small.to_csv(out_dir / f"test_{pct_marker}%.csv", index=False)
