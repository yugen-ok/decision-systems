"""
bullwhip_sim_corrected.py

General idea:
d_hat = (1 - ORDER_ALPHA) * d_hat + ORDER_ALPHA * D_t
O_t = Dhat_t + GAMMA(T + L * Dhat_t - (C_t + Pipeline_t - Backlog_t) )

(C_t + Pipeline_t - Backlog_t) is also called Position_t

Corrected + instrumented 2-node "beer-game-ish" simulation built on EmpiricalLogistics.NodeSimulator.

Key fixes vs your previous sim:
- FIXED sign: inventory position should DECREASE with backlog (backorders), not increase.
- Adds OPTIONAL on-hand inventory state (so "position" is not a weird proxy).
- Metrics are made robust against drift/non-stationarity:
  - amplification on raw, detrended, and first-differences
  - periodicity detection ignores lag=1 smoothness trap and reports the best lag >= MIN_CYCLE_LAG
- Heavy diagnostics: prints intermediate states and flags inconsistencies early.

Usage:
  # PowerShell example:
  $env:T="260"
  $env:SEED="42"
  $env:DEMAND_MEAN="100"
  $env:DEMAND_STD="10"
  $env:L_ORDER="10"
  $env:USE_ON_HAND="1"
  python bullwhip_sim_corrected.py

All knobs are env vars (see section "ENV KNOBS").
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

# ---- Import plant model ----
from EmpiricalLogistics import NodeSimulator


# =============================================================================
# ENV KNOBS (every tweakable knob is here)
# =============================================================================

def env_float(name: str, default: float) -> float:
    v = os.getenv(name, None)
    return float(default if v is None or v == "" else v)

def env_int(name: str, default: int) -> int:
    v = os.getenv(name, None)
    return int(default if v is None or v == "" else v)

def env_bool(name: str, default: int) -> bool:
    v = os.getenv(name, None)
    if v is None or v == "":
        return bool(default)
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def env_str(name: str, default: str) -> str:
    v = os.getenv(name, None)
    return default if v is None or v == "" else str(v)

# ---- Horizon / randomness ----
T               = env_int("T", 260)
SEED            = env_int("SEED", 42)

# ---- Demand process ----
DEMAND_MEAN     = env_float("DEMAND_MEAN", 100.0)
DEMAND_STD      = env_float("DEMAND_STD", 10.0)
DEMAND_CLIP_MIN = env_float("DEMAND_CLIP_MIN", 0.0)

# ---- Retailer->Wholesaler lead time (orders in transit) ----
L_ORDER         = env_int("L_ORDER", 10)

# ---- Plant (NodeSimulator) params: retailer ----
R_C_BASE        = env_float("R_C_BASE", 95.0)
R_C_MAX         = env_float("R_C_MAX", 150.0)
R_GAMMA         = env_float("R_GAMMA", 0.002)
R_L_MOB         = env_int("R_L_MOB", 8)
R_O_MAX         = env_float("R_O_MAX", 300.0)

# ---- Plant params: wholesaler ----
W_C_BASE        = env_float("W_C_BASE", 105.0)
W_C_MAX         = env_float("W_C_MAX", 180.0)
W_GAMMA         = env_float("W_GAMMA", 0.0015)
W_L_MOB         = env_int("W_L_MOB", 10)
W_O_MAX         = env_float("W_O_MAX", 80.0)

# ---- Mobilization policies (capacity “ordering”) ----
R_BACKLOG_TARGET = env_float("R_BACKLOG_TARGET", 800.0)
R_MOB_GAIN       = env_float("R_MOB_GAIN", 0.08)

W_BACKLOG_TARGET = env_float("W_BACKLOG_TARGET", 1500.0)
W_MOB_GAIN       = env_float("W_MOB_GAIN", 0.06)

# ---- Ordering policy (retailer ordering to wholesaler) ----
ORDER_POLICY     = env_str("ORDER_POLICY", "order_up_to")  # "order_up_to" or "smoothed_order_up_to"
Kp               = env_float("Kp", 0.15)                   # proportional correction gain
ORDER_ALPHA      = env_float("ORDER_ALPHA", 0.25)          # smoothing for smoothed_order_up_to
ORDER_MIN        = env_float("ORDER_MIN", 0.0)
ORDER_MAX        = env_float("ORDER_MAX", 1e18)            # optionally cap orders
TARGET_PIPELINE_MODE = env_str("TARGET_PIPELINE_MODE", "mean")  # "mean" or "manual"
TARGET_PIPELINE_MANUAL = env_float("TARGET_PIPELINE_MANUAL", DEMAND_MEAN * L_ORDER)

# ---- Inventory / position modeling ----
USE_ON_HAND      = env_bool("USE_ON_HAND", 1)              # 1 recommended; 0 uses a crude proxy
ON_HAND_INIT     = env_float("ON_HAND_INIT", DEMAND_MEAN * 2.0)
BASE_STOCK       = env_float("BASE_STOCK", DEMAND_MEAN * (L_ORDER + 2))  # target position

# ---- Diagnostics / checks ----
PRINT_EVERY      = env_int("PRINT_EVERY", 20)              # 0 disables periodic prints
PRINT_FIRST_N    = env_int("PRINT_FIRST_N", 15)            # always print first N steps (if >0)
DIAG_ASSERTS     = env_bool("DIAG_ASSERTS", 1)
PLOT             = env_bool("PLOT", 1)

# ---- Metric knobs ----
ACF_MAX_LAG      = env_int("ACF_MAX_LAG", 80)
MIN_CYCLE_LAG    = env_int("MIN_CYCLE_LAG", 5)             # ignore lag 1 smoothness trap
DETREND_METRICS  = env_bool("DETREND_METRICS", 1)
DIFF_METRICS     = env_bool("DIFF_METRICS", 1)

# =============================================================================
# Helpers
# =============================================================================

class MobilizeToTargetPolicy:
    """Mobilize when backlog exceeds a target; delay lives in NodeSimulator."""
    def __init__(self, target_backlog: float, gain: float):
        self.target = float(target_backlog)
        self.gain = float(gain)

    def decide(self, backlog: float) -> float:
        err = backlog - self.target
        return max(0.0, self.gain * err)

def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.std(x)) + 1e-12

def detrend_linear(x: np.ndarray) -> np.ndarray:
    """Remove best-fit line: x ~ a*t + b."""
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float)
    # Solve least squares for [a, b]
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (a * t + b)

def slope_linear(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    a, _b = np.linalg.lstsq(A, x, rcond=None)[0]
    return float(a)

def acf_peak_score(x: np.ndarray, max_lag: int, min_cycle_lag: int):
    """
    Returns:
      best_lag, best_acf, acf[0:max_lag+1]
    We ignore lag < min_cycle_lag to avoid confusing smoothness (lag=1) with cycles.
    """
    s = np.asarray(x, dtype=float)
    s = s - s.mean()
    if len(s) < 5 or np.allclose(s, 0.0):
        return 0, 0.0, np.zeros(max_lag + 1)

    full = np.correlate(s, s, mode="full")
    acf = full[len(s)-1:len(s)-1 + (max_lag + 1)]
    if acf[0] <= 0:
        return 0, 0.0, acf

    acf = acf / acf[0]

    start = max(1, min_cycle_lag)
    if start > max_lag:
        start = 1

    best_lag = int(np.argmax(acf[start: max_lag + 1]) + start)
    best_val = float(acf[best_lag])
    return best_lag, best_val, acf

def amp_ratio(x, y) -> float:
    """std(y) / std(x)"""
    return safe_std(y) / safe_std(x)

def amp_suite(x, y, name: str):
    """
    Returns dict of amplification ratios under different stationarity treatments.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    out = {}
    out[f"amp_raw_{name}"] = amp_ratio(x, y)

    if DETREND_METRICS:
        out[f"amp_detrended_{name}"] = amp_ratio(detrend_linear(x), detrend_linear(y))

    if DIFF_METRICS:
        dx = np.diff(x)
        dy = np.diff(y)
        if len(dx) >= 3 and len(dy) >= 3:
            out[f"amp_diff_{name}"] = amp_ratio(dx, dy)
        else:
            out[f"amp_diff_{name}"] = float("nan")

    return out

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# =============================================================================
# Initialize
# =============================================================================

np.random.seed(SEED)

params_retailer = {
    "C_base": R_C_BASE,
    "C_max":  R_C_MAX,
    "gamma":  R_GAMMA,
    "L":      R_L_MOB,
    "O_max":  R_O_MAX,
}
params_wholesaler = {
    "C_base": W_C_BASE,
    "C_max":  W_C_MAX,
    "gamma":  W_GAMMA,
    "L":      W_L_MOB,
    "O_max":  W_O_MAX,
}

retailer   = NodeSimulator(params_retailer)
wholesaler = NodeSimulator(params_wholesaler)

policy_r = MobilizeToTargetPolicy(R_BACKLOG_TARGET, R_MOB_GAIN)
policy_w = MobilizeToTargetPolicy(W_BACKLOG_TARGET, W_MOB_GAIN)

# Order pipeline (retailer -> wholesaler)
order_pipe = [DEMAND_MEAN] * max(0, L_ORDER)

# Optional on-hand inventory at retailer
on_hand = float(ON_HAND_INIT)

# Demand estimate (for smoothed policy)
d_hat = float(DEMAND_MEAN)

# Target pipeline
if TARGET_PIPELINE_MODE.lower() == "manual":
    TARGET_PIPELINE = float(TARGET_PIPELINE_MANUAL)
else:
    TARGET_PIPELINE = float(DEMAND_MEAN * L_ORDER)

# =============================================================================
# Histories
# =============================================================================

demand           = []
orders_upstream  = []
arriving_w_hist  = []

processed_r      = []
processed_w      = []
backlog_r        = []
backlog_w        = []
cap_r_hist       = []
cap_w_hist       = []
on_hand_hist     = []
position_hist    = []
pipe_sum_hist    = []

# =============================================================================
# Diagnostics header
# =============================================================================

print("\n=== CONFIG SUMMARY ===")
print(f"T={T} SEED={SEED}")
print(f"Demand: mean={DEMAND_MEAN} std={DEMAND_STD} clip_min={DEMAND_CLIP_MIN}")
print(f"L_ORDER={L_ORDER} TARGET_PIPELINE={TARGET_PIPELINE} BASE_STOCK={BASE_STOCK} USE_ON_HAND={int(USE_ON_HAND)}")
print(f"Order policy={ORDER_POLICY} Kp={Kp} alpha={ORDER_ALPHA} order_clip=[{ORDER_MIN},{ORDER_MAX}]")
print(f"Retailer plant: {params_retailer}")
print(f"Wholesaler plant: {params_wholesaler}")
print(f"Retailer mob: target={R_BACKLOG_TARGET} gain={R_MOB_GAIN}")
print(f"Wholesaler mob: target={W_BACKLOG_TARGET} gain={W_MOB_GAIN}")
print(f"Metrics: ACF_MAX_LAG={ACF_MAX_LAG} MIN_CYCLE_LAG={MIN_CYCLE_LAG} detrend={int(DETREND_METRICS)} diff={int(DIFF_METRICS)}")
print(f"Diagnostics: PRINT_FIRST_N={PRINT_FIRST_N} PRINT_EVERY={PRINT_EVERY} ASSERTS={int(DIAG_ASSERTS)}\n")


# =============================================================================
# Simulation loop
# =============================================================================
for t in range(T):
    # --- Customer demand arriving to retailer (work) ---
    D_t = max(DEMAND_CLIP_MIN, float(np.random.normal(DEMAND_MEAN, DEMAND_STD)))
    demand.append(D_t)

    # --- Retailer mobilization (capacity) ---
    mob_r = policy_r.decide(retailer.backlog)
    out_r = retailer.step(arriving_tons=D_t, mobilization_order=mob_r)

    processed_r.append(out_r["processed"])
    backlog_r.append(out_r["backlog"])
    cap_r_hist.append(out_r["capacity"])

    # --- Inventory dynamics (optional) ---
    # Interpret processed_r as "units shipped to customers" (or work completed).
    # If you model "on_hand inventory", demand consumes inventory, replenishment arrives later from upstream.
    # Here we use a minimal consistent bookkeeping:
    # - customer demand consumes on_hand first; unmet demand becomes backlog (already tracked in plant)
    # - wholesaler deliveries later add to on_hand
    #
    # NOTE: because your plant backlog is already "work backlog", if you want strict inventory semantics,
    # you'd separate "customer backorders" from "processing backlog". This is a light proxy, but with correct sign.

    if USE_ON_HAND:
        # customer demand draws down inventory; we allow negative to represent backorders,
        # but we separately keep plant backlog anyway. For a clean position, we use:
        # position = on_hand + on_order - backorders
        on_hand = on_hand - D_t  # demand consumes inventory
    on_hand_hist.append(on_hand)

    # --- Demand estimate for smoothed ordering ---
    d_hat = (1.0 - ORDER_ALPHA) * d_hat + ORDER_ALPHA * D_t

    # --- Retailer ordering policy to wholesaler ---
    pipe_sum = float(sum(order_pipe)) if len(order_pipe) else 0.0
    pipe_sum_hist.append(pipe_sum)

    # CORRECT SIGN:
    # Backlog/backorders reduce position.
    # If USE_ON_HAND: position = on_hand + on_order - backlog_like
    # If not: we approximate on_hand as 0, so position ≈ on_order - backlog
    if USE_ON_HAND:
        position = on_hand + pipe_sum - retailer.backlog
    else:
        position = pipe_sum - retailer.backlog

    position_hist.append(position)

    if ORDER_POLICY.lower() == "smoothed_order_up_to":
        # order-up-to: aim position -> BASE_STOCK, include a demand term via d_hat
        order_to_wholesaler = d_hat + Kp * (BASE_STOCK - position)
    else:
        # plain order-up-to using mean demand baseline
        order_to_wholesaler = DEMAND_MEAN + Kp * (BASE_STOCK - position)

    order_to_wholesaler = clamp(order_to_wholesaler, ORDER_MIN, ORDER_MAX)
    orders_upstream.append(order_to_wholesaler)

    # --- Apply lead time between retailer and wholesaler ---
    if L_ORDER > 0:
        order_pipe.append(order_to_wholesaler)
        arriving_w = order_pipe.pop(0)
    else:
        arriving_w = order_to_wholesaler

    arriving_w_hist.append(arriving_w)

    # Wholesaler deliveries replenish retailer on-hand (optional)
    if USE_ON_HAND:
        on_hand = on_hand + arriving_w

    # --- Wholesaler mobilization + step ---
    mob_w = policy_w.decide(wholesaler.backlog)
    out_w = wholesaler.step(arriving_tons=arriving_w, mobilization_order=mob_w)

    processed_w.append(out_w["processed"])
    backlog_w.append(out_w["backlog"])
    cap_w_hist.append(out_w["capacity"])

    # --- Diagnostics prints ---
    do_print = (PRINT_FIRST_N > 0 and t < PRINT_FIRST_N) or (PRINT_EVERY > 0 and (t % PRINT_EVERY == 0))
    if do_print:
        print(
            f"[t={t:03d}] "
            f"D={D_t:8.2f} "
            f"ord={order_to_wholesaler:8.2f} "
            f"arrW={arriving_w:8.2f} "
            f"pos={position:10.2f} pipe={pipe_sum:10.2f} "
            f"R:proc={out_r['processed']:8.2f} cap={out_r['capacity']:8.2f} back={out_r['backlog']:9.2f} "
            f"W:proc={out_w['processed']:8.2f} cap={out_w['capacity']:8.2f} back={out_w['backlog']:9.2f} "
            + (f"on_hand={on_hand:10.2f}" if USE_ON_HAND else "")
        )

    # --- Consistency checks (fail-fast) ---
    if DIAG_ASSERTS:
        # basic non-negativity constraints from plant
        if out_r["processed"] < -1e-6 or out_w["processed"] < -1e-6:
            raise RuntimeError("Processed went negative (should not happen).")
        if out_r["backlog"] < -1e-6 or out_w["backlog"] < -1e-6:
            raise RuntimeError("Backlog went negative (should not happen).")
        if out_r["capacity"] < -1e-6 or out_w["capacity"] < -1e-6:
            raise RuntimeError("Capacity went negative (clip should prevent).")

        # sanity: if there is a lot of available work, processed should often be near capacity
        # (not a strict rule, but we flag extremes)
        if retailer.backlog > 1e6 and out_r["processed"] < 1e-3:
            print("WARNING: Retailer backlog huge but processed ~0. Check gamma/C_max causing collapse.")


# =============================================================================
# Metrics + vetting diagnostics
# =============================================================================
demand_arr = np.asarray(demand)
orders_arr = np.asarray(orders_upstream)
proc_r_arr = np.asarray(processed_r)
proc_w_arr = np.asarray(processed_w)
back_r_arr = np.asarray(backlog_r)
cap_r_arr  = np.asarray(cap_r_hist)

print("\n=== VETTING: STATIONARITY / DRIFT CHECK ===")
print(f"slope(order) per step: {slope_linear(orders_arr): .6f}")
print(f"slope(demand) per step: {slope_linear(demand_arr): .6f}")
print(f"slope(proc_w) per step: {slope_linear(proc_w_arr): .6f}")

print("\n=== AMPLIFICATION (multi-view) ===")
amp1 = amp_suite(demand_arr, orders_arr, "orders_vs_demand")
amp2 = amp_suite(proc_r_arr, proc_w_arr, "procW_vs_procR")

for k, v in {**amp1, **amp2}.items():
    print(f"{k:28s}: {v: .4f}")

print("\n=== 'PERIODICITY' (ACF peak beyond smoothness) ===")
lag_back_r, val_back_r, _acf_back_r = acf_peak_score(back_r_arr, ACF_MAX_LAG, MIN_CYCLE_LAG)
lag_cap_r,  val_cap_r,  _acf_cap_r  = acf_peak_score(cap_r_arr,  ACF_MAX_LAG, MIN_CYCLE_LAG)

print(f"Retailer backlog: best_lag={lag_back_r} best_acf={val_back_r:.3f} (min_cycle_lag={MIN_CYCLE_LAG})")
print(f"Retailer capacity: best_lag={lag_cap_r} best_acf={val_cap_r:.3f} (min_cycle_lag={MIN_CYCLE_LAG})")

# Quick interpretive flags (not proof, just warnings)
def flag_cycle(lag, val, label):
    if lag == 0 or val <= 0.2:
        print(f"NOTE: {label}: no strong cyclic peak detected (val<=0.2).")
    elif lag < MIN_CYCLE_LAG:
        print(f"NOTE: {label}: peak is at very small lag; likely smoothness, not a cycle.")
    else:
        print(f"OK: {label}: candidate cycle lag ~{lag} with ACF peak {val:.3f}.")

flag_cycle(lag_back_r, val_back_r, "Retailer backlog")
flag_cycle(lag_cap_r,  val_cap_r,  "Retailer capacity")

# "Bullwhip-ish" decision heuristic (still heuristic, but now less gameable)
amp_orders_detr = amp1.get("amp_detrended_orders_vs_demand", float("nan"))
amp_orders_diff = amp1.get("amp_diff_orders_vs_demand", float("nan"))

print("\n=== HEURISTIC DECISION (less fragile than before) ===")
print("We only call it bullwhip-like if amplification holds after detrending or on differences,")
print("AND there is a cycle-like ACF peak beyond lag=1 smoothness.\n")

bullwhip_like = False
if (not math.isnan(amp_orders_detr) and amp_orders_detr > 1.0) or (not math.isnan(amp_orders_diff) and amp_orders_diff > 1.0):
    if val_back_r > 0.25 and lag_back_r >= MIN_CYCLE_LAG:
        bullwhip_like = True

print(f"bullwhip_like = {bullwhip_like} "
      f"(amp_detrended={amp_orders_detr:.3f}, amp_diff={amp_orders_diff:.3f}, "
      f"acf_peak_backlog={val_back_r:.3f} at lag={lag_back_r})")

# =============================================================================
# Plot
# =============================================================================
if PLOT:
    fig, axs = plt.subplots(6, 1, figsize=(11, 16), sharex=True)

    axs[0].plot(demand, label="Customer Demand")
    axs[0].legend()
    axs[0].set_ylabel("Demand")

    axs[1].plot(orders_upstream, label="Orders (Retailer→Wholesaler)")
    axs[1].legend()
    axs[1].set_ylabel("Orders")

    axs[2].plot(processed_r, label="Retailer Processed")
    axs[2].plot(processed_w, label="Wholesaler Processed")
    axs[2].legend()
    axs[2].set_ylabel("Processed")

    axs[3].plot(backlog_r, label="Retailer Backlog")
    axs[3].plot(backlog_w, label="Wholesaler Backlog")
    axs[3].legend()
    axs[3].set_ylabel("Backlog")

    axs[4].plot(cap_r_hist, label="Retailer Capacity")
    axs[4].plot(cap_w_hist, label="Wholesaler Capacity")
    axs[4].legend()
    axs[4].set_ylabel("Capacity")

    axs[5].plot(position_hist, label="Retailer Position")
    axs[5].plot(pipe_sum_hist, label="Pipeline Sum")
    if USE_ON_HAND:
        axs[5].plot(on_hand_hist, label="On-hand")
    axs[5].legend()
    axs[5].set_ylabel("Position components")
    axs[5].set_xlabel("Time")

    plt.tight_layout()
    plt.show()
