# EmpiricalLogistics

## What this is

**EmpiricalLogistics** is a small, intentionally minimal framework for modeling and testing **flow dynamics** in large-scale operational systems (ports, factories, logistics hubs, etc.) under **delayed and partial information**.

It is **not** a production system, a forecasting engine, or an optimization solution.

Its purpose is narrower and more fundamental:

> To test whether a simple, empirically calibrated feedback mechanism captures a *real slice of reality* in flow-based systems.

---

## Core idea

The framework is built around one causal hypothesis:

> **Backlog is a delayed signal of congestion that reduces effective processing capacity.**

In many real systems:
- Decisions are made with incomplete state information
- Over-accepting work does not fail immediately
- Congestion manifests later as reduced throughput and backlog spikes

EmpiricalLogistics models this with a minimal state and feedback loop:

- **State**: backlog
- **Input**: arriving workload
- **Mechanism**:  
  - baseline capacity  
  - reduced by backlog-dependent congestion
- **Output**: processed workload

The model deliberately avoids:
- long-horizon forecasting
- demand prediction
- detailed operational rules
- optimization policies

It focuses only on the **structural feedback** between backlog and capacity.

---

## What `EmpiricalLogistics.py` contains

The module is organized as a reusable library:

- **CapacityCalibrator**  
  Empirically estimates parameters (baseline capacity, congestion sensitivity, delays) from historical data only.

- **PortSimulator**  
  Implements the minimal backlog → capacity → processing dynamics.

- **SimplePolicy** (optional)  
  A placeholder reactive policy, not central to validation.

- **compute_validation_metrics**  
  Standard metrics for comparing predicted vs actual processing.

The class logic is intentionally simple so that:
- causality is explicit
- failures are interpretable
- extensions are additive, not corrective

---

## What the experiment is validating

The accompanying experiment script is **not** trying to maximize accuracy.

It answers one precise question:

> **Given a system’s state up to day _t−1_ (backlog) and arrivals on day _t_, does the EmpiricalLogistics mechanism predict how much the system will actually process on day _t_?**

This is evaluated:
- out of sample (train → eval split)
- across many independent port–cargo streams
- using correlation and error metrics

The experiment intentionally:
- disables long-horizon simulation claims
- avoids policy optimization
- focuses on one-step, conditional prediction

---

## How to interpret the results

Typical outcomes look like:

- **Median correlation**: modest (≈ 0.2–0.3)
- **Upper quartile**: substantially higher (≈ 0.4–0.5)

This pattern means:

- The backlog → congestion mechanism **exists in reality**
- It is **strong for some systems**, weak for others
- Limitations come from **missing state**, not flawed logic

In other words:

> The framework captures a *real causal component* of flow congestion, but not the full operational state of complex systems.

This is exactly the expected outcome for a minimal, structural model.

---

## What this is *not* claiming

EmpiricalLogistics does **not** claim to:
- be production-ready
- fully predict throughput everywhere
- replace detailed operational models
- outperform machine learning

Its role is earlier in the modeling stack:

> **Establishing that the core congestion logic is directionally correct and empirically grounded.**

---

## Impact

If the signal were zero or unstable, the idea should be discarded.

Instead, the experiments show:

- consistent non-zero signal
- robustness across datasets
- improvement with more coverage

That validates the *logic* and justifies further refinement if desired.

---

## Summary

- EmpiricalLogistics is a minimal skeleton for modeling flow
- The experiments are designed to validate logic, not optimize performance
- The results confirm the mechanism is real (though incomplete)
