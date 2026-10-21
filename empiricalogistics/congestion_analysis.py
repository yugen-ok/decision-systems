import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import mannwhitneyu

# ----------------------------
# LOAD
# ----------------------------
load_dotenv()
TRAIN_PATH = os.getenv("TRAIN_PATH")
df = pd.read_csv(TRAIN_PATH)

df = df.sort_values(["portid"]).reset_index(drop=True)

# ----------------------------
# DEMAND UNIT (one-time scaling)
# ----------------------------
# Convert demand (portcalls) into processed-units using a robust global factor.
mask = df["demand"] > 0
DEMAND_UNIT = (df.loc[mask, "processed"] / df.loc[mask, "demand"]).median()

print("DEMAND_UNIT =", DEMAND_UNIT)

# ----------------------------
# HELPERS
# ----------------------------
def find_overload_runs(demand_scaled, processed):
    """
    Returns list of (start, length) where demand_scaled > processed
    on consecutive days.
    """
    runs = []
    i = 0
    n = len(demand_scaled)
    while i < n:
        if demand_scaled[i] > processed[i]:
            start = i
            while i < n and demand_scaled[i] > processed[i]:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1
    return runs

def backlog_recovery_time(backlog, start, eps=0.1):
    """
    Time until backlog recovers to within eps of its pre-overload baseline.
    """
    if start == 0:
        base = 0.0
    else:
        base = backlog[start - 1]

    peak = backlog[start:].max()

    # recovery threshold
    thresh = base + eps * max(1.0, peak - base)

    for t in range(start + 1, len(backlog)):
        if backlog[t] <= thresh:
            return t - start

    return np.inf

def compute_backlog(demand_scaled, processed):
    """
    backlog[t] = max(0, backlog[t-1] + demand_scaled[t] - processed[t])
    """
    n = len(demand_scaled)
    backlog = np.zeros(n, dtype=float)
    for t in range(n):
        prev = backlog[t-1] if t > 0 else 0.0
        backlog[t] = max(0.0, prev + demand_scaled[t] - processed[t])
    return backlog

# ----------------------------
# MAIN SCAN
# ----------------------------
records = []

for portid, port_df in df.groupby("portid"):
    port_df = port_df.reset_index(drop=True)

    demand_scaled = (port_df["demand"].to_numpy(dtype=float) * DEMAND_UNIT)
    processed     = port_df["processed"].to_numpy(dtype=float)

    backlog = compute_backlog(demand_scaled, processed)

    runs = find_overload_runs(demand_scaled, processed)

    for start, length in runs:
        # peak backlog AFTER the run starts (you can change window if you want)
        peak_backlog = float(backlog[start:].max())
        clear_time = backlog_recovery_time(backlog, start, eps=0.1)

        records.append({
            "portid": portid,
            "start": int(start),  # <-- ADD THIS
            "overload_run_length": int(length),
            "peak_backlog": peak_backlog,
            "clear_time": clear_time,
        })

runs_df = pd.DataFrame(records)


###################################

# Last exp:
# Is backlog already high before a never-recovered event

# ============================================================
# PRE-SPIKE BACKLOG TEST (THE RIGHT 1-D EXPERIMENT)
# ============================================================

K = 7  # days before run start

rows = []

for portid, port_df in df.groupby("portid"):
    port_df = port_df.reset_index(drop=True)

    demand_scaled = port_df["demand"].to_numpy(float) * DEMAND_UNIT
    processed     = port_df["processed"].to_numpy(float)

    backlog = compute_backlog(demand_scaled, processed)

    # runs for this port
    port_runs = runs_df[runs_df.portid == portid]

    for _, r in port_runs.iterrows():
        t0 = int(r.start)
        if t0 <= 0:
            continue

        # backlog just before / shortly before the run
        B_pre = float(backlog[max(0, t0-K):t0].max())

        rows.append({
            "B_pre": B_pre,
            "never": ~np.isfinite(r.clear_time),
        })


pre_df = pd.DataFrame(rows)

print("Total runs:", len(pre_df))
print("Never runs:", int(pre_df.never.sum()))
print("Never rate:", float(pre_df.never.mean()))

# ----------------------------
# BINNING (P(never | B_pre))
# ----------------------------
pre_df["bin"] = pd.qcut(pre_df["B_pre"], q=20, duplicates="drop")

bin_table = (
    pre_df.groupby("bin")
    .agg(
        n=("never", "size"),
        never_rate=("never", "mean"),
        Bpre_min=("B_pre", "min"),
        Bpre_med=("B_pre", "median"),
        Bpre_max=("B_pre", "max"),
    )
    .reset_index(drop=True)
)

print("\n=== P(never | pre-spike backlog) ===")
print(bin_table)

# monotonicity sanity check
rates = bin_table["never_rate"].to_numpy()
print("\nDownward steps in never_rate:",
      int(np.sum(np.diff(rates) < 0)),
      "out of", len(rates)-1)

# ----------------------------
# 1-D SEPARABILITY (AUC)
# ----------------------------
x = pre_df["B_pre"].to_numpy()
y = pre_df["never"].astype(int).to_numpy()

pos = x[y == 1]
neg = x[y == 0]

# rank-based AUC (no sklearn)
scores = np.concatenate([pos, neg])
labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

order = np.argsort(scores)
ranks = np.empty_like(order, float)
ranks[order] = np.arange(1, len(scores)+1)

i = 0
s = scores[order]
while i < len(s):
    j = i
    while j+1 < len(s) and s[j+1] == s[i]:
        j += 1
    if j > i:
        ranks[order[i:j+1]] = (i+1 + j+1) / 2
    i = j+1

R_pos = ranks[labels == 1].sum()
auc = (R_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

print("\nROC AUC using ONLY pre-spike backlog:", float(auc))
# .68 -> anser is yes
# If you randomly pick one never-recover run and one recovered run,
# 68% of the time the never-recover run had higher pre-backlog.








#####################################


runs_df["never_recovered"] = ~np.isfinite(runs_df["clear_time"])

# First we check whether there's a difference between the ones that recovered and the ones that didnt:

# keep only valid numeric rows
runs_df = runs_df[
    runs_df["overload_run_length"].notna() &
    runs_df["peak_backlog"].notna()
]

rec = runs_df[~runs_df["never_recovered"]]
nev = runs_df[runs_df["never_recovered"]]

print(f"Recovered runs: {len(rec)}")
print(f"Never-recovered runs: {len(nev)}")

# guard against degenerate cases
def safe_mwu(x, y):
    if len(x) == 0 or len(y) == 0:
        return None, None
    return mannwhitneyu(x, y, alternative="two-sided")

# ----------------------------
# 1. RUN LENGTH TEST
# ----------------------------
u_len, p_len = safe_mwu(
    rec["overload_run_length"],
    nev["overload_run_length"]
)

# ----------------------------
# 2. PEAK BACKLOG TEST
# ----------------------------
u_back, p_back = safe_mwu(
    rec["peak_backlog"],
    nev["peak_backlog"]
)

# ----------------------------
# EFFECT SIZE (rank-biserial)
# ----------------------------
def rank_biserial(u, n1, n2):
    return 1 - (2*u) / (n1*n2)

rb_len  = rank_biserial(u_len,  len(rec), len(nev))  if u_len  is not None else None
rb_back = rank_biserial(u_back, len(rec), len(nev))  if u_back is not None else None

# ----------------------------
# RESULTS
# ----------------------------
print("\n=== STATISTICAL DIFFERENCE TEST ===")

print("\nOverload run length:")
print("  p-value:", p_len)
print("  rank-biserial effect:", rb_len)
print("  medians:", rec["overload_run_length"].median(),
      "(recovered) vs", nev["overload_run_length"].median(), "(never)")

print("\nPeak backlog:")
print("  p-value:", p_back)
print("  rank-biserial effect:", rb_back)
print("  medians:", rec["peak_backlog"].median(),
      "(recovered) vs", nev["peak_backlog"].median(), "(never)")



# Now we check recovery trends for rec:


summary = (
    rec
    .groupby("overload_run_length")
    .agg(
        mean_peak_backlog=("peak_backlog", "mean"),
        median_peak_backlog=("peak_backlog", "median"),
        mean_clear_time=("clear_time", "mean"),
        median_clear_time=("clear_time", "median"),
        count=("clear_time", "count"),
    )
    .reset_index()
)

print("\n=== OVERLOAD RUN IMPACT (DATA-ONLY) ===")
print(summary)

# ----------------------------
# NONLINEARITY CHECK
# ----------------------------
x = summary["overload_run_length"].to_numpy()
y_backlog = summary["mean_peak_backlog"].to_numpy()
y_clear   = summary["mean_clear_time"].to_numpy()

print("\nLinear fit (peak backlog):", np.polyfit(x, y_backlog, 1))
print("Quadratic fit (peak backlog):", np.polyfit(x, y_backlog, 2))

print("\nLinear fit (clear time):", np.polyfit(x, y_clear, 1))
print("Quadratic fit (clear time):", np.polyfit(x, y_clear, 2))

# Inspect events that never recovered:


# ============================================================
# NEVER-RECOVER EVENTS: demand vs processed
# (simple, local, fast)
# ============================================================

records = []

for _, r in nev.iterrows():
    port_df = df[df.portid == r.portid].reset_index(drop=True)
    t0 = int(r.start)

    if t0 <= 0 or t0 >= len(port_df):
        continue

    records.append({
        "when": "day_before",
        "demand": port_df.loc[t0 - 1, "demand"],
        "processed": port_df.loc[t0 - 1, "processed"],
    })

    records.append({
        "when": "overload_day",
        "demand": port_df.loc[t0, "demand"],
        "processed": port_df.loc[t0, "processed"],
    })

simple_df = pd.DataFrame(records)

summary = (
    simple_df
    .groupby("when")
    .agg(
        demand_median=("demand", "median"),
        processed_median=("processed", "median"),
        demand_mean=("demand", "mean"),
        processed_mean=("processed", "mean"),
        count=("demand", "count"),
    )
)

print("\n=== NEVER-RECOVER: DEMAND vs PROCESSED ===")
print(summary)

print("\n=== NEVER-RECOVER: DEMAND / PROCESSED RATIO ===")
print(
    (summary["demand_median"] / summary["processed_median"])
)


# ============================================================
# NEVER-RECOVER EVENTS: demand vs processed (MINIMAL)
# ============================================================

rows = []

for _, r in nev.iterrows():
    port_df = df[df.portid == r.portid].reset_index(drop=True)
    t = int(r.start)

    if t < 0 or t >= len(port_df):
        continue

    rows.append({
        "demand": port_df.loc[t, "demand"] * DEMAND_UNIT,
        "processed": port_df.loc[t, "processed"],
        "gap": port_df.loc[t, "demand"] - port_df.loc[t, "processed"],
        "ratio": port_df.loc[t, "demand"] / max(port_df.loc[t, "processed"], 1e-9),
    })

never_event_df = pd.DataFrame(rows)

print("\n=== NEVER-RECOVER EVENT DAY STATS ===")
print(never_event_df.describe(percentiles=[0.5, 0.9, 0.95]))
