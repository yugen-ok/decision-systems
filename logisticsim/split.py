from pathlib import Path

import pandas as pd
import numpy as np
import os

def zscore(series):
    mu = series.mean()
    sigma = series.std()
    if sigma == 0:
        return series * 0.0
    return (series - mu) / sigma

# ---------- PARAMETERS ----------
TRAIN_FRAC = 0.6
EVAL_FRAC  = 0.2
MIN_DAYS_PER_PORT = 90   # drop ports shorter than this
OUT_DIR = "../data/Daily_Port_Activity_Data_and_Trade_Estimates/splits_logistcsim"
SHIP_TYPES = ["container", "dry_bulk", "general_cargo", "roro", "tanker"]
DATA_PATH = Path(os.getenv('DATA_PATH'))

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(
    DATA_PATH)


window = 30

# ensure datetime + sort
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df = df.sort_values(["portid", "date"])

for k in SHIP_TYPES:
    A = f"portcalls_{k}"
    imp = f"import_{k}"
    exp = f"export_{k}"

    # processed tons
    P = f"processed_{k}_tons"
    df[P] = df[imp].fillna(0.0) + df[exp].fillna(0.0)

    # tons-per-call ratio (only valid when A>0)
    ratio = np.where(df[A].fillna(0).values > 0,
                     df[P].values / df[A].values,
                     np.nan)
    df[f"tons_per_call_{k}"] = ratio

    # rolling median tons-per-call per port (robust to spikes)
    df[f"alpha_{k}"] = (
        df.groupby("portid")[f"tons_per_call_{k}"]
        .transform(lambda s: s.rolling(window, min_periods=5).median())
        .fillna(method="ffill")
        .fillna(0.0)
    )

    # estimated arriving work (tons)
    df[f"arriving_{k}_tons"] = df[f"alpha_{k}"] * df[A].fillna(0.0)


    # backlog recursion per port
    def compute_backlog(group: pd.DataFrame) -> pd.Series:
        b = 0.0
        out = []
        for w, p in zip(group[f"arriving_{k}_tons"].values, group[P].values):
            b = max(0.0, b + float(w) - float(p))
            out.append(b)
        return pd.Series(out, index=group.index)


    df[f"backlog_{k}_tons"] = df.groupby("portid", group_keys=False).apply(compute_backlog)

# total backlog across ship types (tons)
df["backlog_total_tons"] = df[[f"backlog_{k}_tons" for k in SHIP_TYPES]].sum(axis=1)


df2 = df.copy()
df2["date"] = pd.to_datetime(df2["date"], utc=True, errors="coerce")
df2 = df2.sort_values(["portid", "date"])

splits = []

for portid, g in df2.groupby("portid"):
    n = len(g)
    if n < MIN_DAYS_PER_PORT:
        continue

    n_train = int(n * TRAIN_FRAC)
    n_eval  = int(n * (TRAIN_FRAC + EVAL_FRAC))

    g = g.copy()
    g["split"] = "test"
    g.iloc[:n_train, g.columns.get_loc("split")] = "train"
    g.iloc[n_train:n_eval, g.columns.get_loc("split")] = "eval"

    splits.append(g)

df_split = pd.concat(splits, ignore_index=True)

# Save
df_split[df_split["split"] == "train"].to_csv(f"{OUT_DIR}/train.csv", index=False)
df_split[df_split["split"] == "eval"].to_csv(f"{OUT_DIR}/eval.csv", index=False)
df_split[df_split["split"] == "test"].to_csv(f"{OUT_DIR}/test.csv", index=False)

# Quick sanity check
df_split["split"].value_counts(), df_split["portid"].nunique()
