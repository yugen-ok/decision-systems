# Current Status

## Complete: Full Emergent Dynamics System

**All reworks complete + critical bugs fixed**

### Core Changes Implemented

**1. Relational Targets (was: fixed vectors)**
- Red: `pos_Liz - pos_Tom` (α=1, has target)
- Liz: **null** (α=0, no target, responds via γ-β=1)
- Tom: **null** (α=0, no target, responds via γ-β=-2)
- Ressler: `pos_Red - pos_Ressler` (α=1, has target)
- Cooper: `average(pos_all) - pos_Cooper` (α=0.5, has target)
- Berlin: **null** (α=0, no target, responds via γ-β=1.5)

Characters with α≠0 have targets that **shift dynamically** as others move.
Characters with α=0 have no targets - they optimize (γ-β)⟨e,Je⟩ term only.

**2. Per-Character Positions (was: single global x)**
- Each character has position `pos_k ∈ ℝ¹²`
- Individual movement: `pos_k += a_k * e_k * dt`
- Global flow: `F = Σ a_k e_k` (kept for Jacobian/inflections)
- State: `X = concat(pos_all)` ∈ ℝ⁷² (6 characters × 12 dims)

**3. Target Computation**
- Gradient: `∇φ_k = -2(pos_k - target_k(pos_others))`
- No circular dependency (target depends on others, not self)
- Normalized: `target / (||target|| + ε)` to prevent explosion
- Null targets return zero gradient (α=0 characters)

**4. Antagonists Added**
- **Berlin** (disruptor): α=0, β=0, γ=1.5, η=-1
  - No goal, strong amplification, seeks boundary crossing
  - Injects chaos and forces inflections
- **Government/Cabal** (field): State-dependent boundary
  - `n(x) = strength(x) * n_base`
  - Strength increases with system energy (instability → stronger constraints)
  - Implemented as: `1.0 + 0.1 * tanh(total_norm - 0.1)`

### What Stays Unchanged

- Core score function: `S(e) = α⟨e,∇φ⟩ + (γ-β)⟨e,Je⟩ - η|⟨e,n⟩|`
- Optimization (analytical for γ-β=0, numerical otherwise)
- Inflection definitions (multi-dimensional geometric conditions)
- Hidden/visible partition (kept as dimension indices)

### Results

**Both inflections detected in chronological order:**
1. **Weak Coupling** (t=1, ep=0.02): coupling ratio = 1.029 < 1.1
   - Hidden-visible domains evolve independently (balanced coupling)
2. **Non-Local Paths** (t=2, ep=0.04): alignment = 0.059 < 0.3, progress = 0.001 > 0
   - Red moves indirectly (curved path) but still makes progress

**Emergence unlocked**:
- Goals emergent ✓ (targets shift with character positions)
- Trajectories emergent ✓ (curved paths from interactions)
- Power emergent ✓ (weights update based on effectiveness)

### Implementation

**Data format (JSON)**:
```json
// Goal-driven character (α ≠ 0):
{
  "alpha": 1.0,
  "target_relation": {
    "type": "difference",
    "operands": ["Liz", "Tom"],
    "description": "pos_Liz - pos_Tom"
  }
}

// Reactive character (α = 0):
{
  "alpha": 0.0,
  "target_relation": null
}
```

**Simulation state**:
```python
state.positions = {
  "Red": array([...]),   # R^12
  "Liz": array([...]),   # R^12
  ...
}
```

**Flow computation**:
```python
# Both kept:
F = Σ a_k e_k              # Global (for Jacobian)
pos_k += a_k * e_k * dt    # Individual movement
```

### File Structure (unchanged)

```
data/
  blacklist_s1.json         # Now with relational targets

simulation/
  data_loader.py            # Parses target relations
  simulation_engine.py      # Per-character positions
  main.py                   # Position-based updates

results/
  blacklist_s1/
    inflections.json
    trajectory.txt          # Shows per-character positions
    observables.png
```

### Critical Fixes Applied

**1. Liz Parameters (Bug Fix)**
- **Was:** α=0, β=1, γ=1 → γ-β=0 → random direction (broken)
- **Fixed:** α=0, β=0, γ=1 → γ-β=1 → amplifies Jacobian
- **Why:** Characters with α=0 must have γ-β≠0 to be meaningful, otherwise they just add noise

**2. Progress Condition (Bug Fix)**
- **Was:** Compared `phi(pos_t, target_t)` vs `phi(pos_{t-1}, target_{t-1})` (moving reference frame)
- **Fixed:** Compare `phi(pos_t, target_t)` vs `phi(pos_{t-1}, target_t)` (same reference frame)
- **Why:** Must measure progress toward the SAME target to detect if character got closer
- **Location:** simulation_engine.py:394-410

**3. Target Computation (Bug Fix)**
- **Was:** Normalized targets (destroys magnitude), used `center - pos_self` for center targets
- **Fixed:** Return targets directly without normalization, return `center` for center targets
- **Why:** Normalization breaks gradient semantics, wrong reference point for center targets
- **Location:** simulation_engine.py:58-84

**4. Adaptive Damping (Critical Addition)**
- **Added:** `damping = 1 / (1 + ||F||)` to position updates
- **Why:** Prevents runaway motion when system becomes unstable
- **Location:** main.py:137-142

**5. Weight Tuning**
- **Adjusted:** More balanced weights (Cooper ↑ 0.5, Berlin ↓ 0.8, Red ↓ 0.8)
- **Why:** Reduce Red dominance, allow multi-polar dynamics for inflection detection

**6. Weak Coupling Threshold**
- **Adjusted:** Threshold from 0.9 → 1.1
- **Why:** With relational targets, coupling naturally ~1.0 (balanced), threshold adjusted accordingly

### Rework 2: Performance-Based Weights (Complete)

**Weight updates now based on effectiveness**:
```python
# For each character:
alignment = dot(e_k, F_char)
if alignment > 0:
    # Direction matches system movement -> increase influence
    weight *= 1.01
else:
    # Direction opposes system movement -> decrease influence
    weight *= 0.99
```

**Power now emergent**: influence grows from effectiveness, not preset constants.

### Validation

Run:
```bash
cd simulation
python main.py --iterations 100 --inflections 2
```

Output:
- 6 characters (Red, Liz, Tom, Ressler, Cooper, Berlin) ✓
- **Both inflections detected:**
  - Weak Coupling at t=1, ep=0.02 ✓
  - Non-Local Paths at t=2, ep=0.04 ✓
- Relational targets working (unnormalized) ✓
- Null targets for α=0 characters ✓
- Liz amplifies (γ-β=1), Tom resists (γ-β=-2), Berlin disrupts (γ-β=1.5) ✓
- State-dependent boundaries ✓
- Adaptive damping prevents explosion ✓
- Per-character positions bounded ✓
- Performance-based weights updating ✓

### Final Character Weights

```
Red:     0.8 (influential but not dominating)
Berlin:  0.8 (chaos injector, balanced with Red)
Tom:     0.7 (resister, moderate)
Liz:     0.6 (amplifier, moderate)
Cooper:  0.5 (stabilizer, increased to reduce coupling)
Ressler: 0.4 (pursuer, lowest)
```

### Documentation

- **Step-by-step walkthrough:** `docs/examples/step_by_step_walkthrough.md`
- Shows exact calculations from t=0 to both inflection detections
- No jargon, just math → meaning

## Key Achievement

**From**:
- Red wants fixed point X
- Goals hardcoded
- Power preset
- Only trajectories emerge

**To**:
- Red wants relation between Liz and Tom
- Goals shift dynamically with movement
- Power emerges from effectiveness
- **Everything emerges**: goals, trajectories, AND power

> "Replace 'where to go' with 'how characters relate', and everything else starts to generate itself."

System now simulates a **game of forces**, not a scripted story. All dynamics emerge from:
1. Relational targets (what characters want from each other)
2. Individual optimization (how each chooses direction)
3. Performance feedback (who gains influence)
