# Step-by-Step Simulation Walkthrough

This walks through exactly what happens at each step of the simulation. No jargon, no shortcuts - just the actual math and what it means.

---

## Initial Setup (t=0)

### State

Each character starts at a random position in 12-dimensional space:

```
Red:     position = [0.0050, -0.0014, 0.0065, ...] (12 numbers)
         ||position|| = 0.027  (distance from origin)
         weight = 0.8  (influence on system)

Liz:     position = [0.0024, -0.0191, -0.0172, ...]
         ||position|| = 0.039
         weight = 0.6

Tom:     position = [-0.0054, 0.0011, -0.0115, ...]
         ||position|| = 0.030
         weight = 0.7

Ressler: position = [0.0021, -0.0196, -0.0133, ...]
         ||position|| = 0.032
         weight = 0.4

Cooper:  position = [0.0034, -0.0176, 0.0032, ...]
         ||position|| = 0.028
         weight = 0.5

Berlin:  position = [-0.0048, -0.0019, -0.0111, ...]
         ||position|| = 0.031
         weight = 0.8
```

**What this means:** Everyone starts near the origin. Weights are balanced (Red and Berlin at 0.8 are strongest, but not overwhelming).

---

## Step 1: Compute Targets

Some characters have targets (things they want). Others don't.

### Red's target

Red wants: **pos_Liz - pos_Tom** (the vector from Tom to Liz)

```python
target_Red = pos_Liz - pos_Tom
           = [0.0024, ...] - [-0.0054, ...]
           = [0.0079, ...]  (12 numbers)

# NOT normalized (keep magnitude)
||target_Red|| = 0.0455
```

**What this means:** Red wants Liz and Tom to be in a certain relationship (Liz away from Tom). As Liz and Tom move, Red's target moves. The magnitude (0.0455) matters - it tells Red how far apart they should be.

### Ressler's target

Ressler wants: **pos_Red - pos_Ressler** (vector from Ressler to Red)

```python
target_Ressler = pos_Red - pos_Ressler
               = [0.0050, ...] - [0.0021, ...]
               = [0.0029, ...]
```

**What this means:** Ressler wants to be at Red's position (chase him).

### Cooper's target

Cooper wants: **center** (the average position of everyone)

```python
center = mean([pos_Red, pos_Liz, pos_Tom, pos_Ressler, pos_Cooper, pos_Berlin])
       = [0.0006, -0.0157, -0.0085, ...]

target_Cooper = center  # NOT center - pos_Cooper
              = [0.0006, ...]
```

**What this means:** Cooper wants to be at the center of everyone.

### Characters with NO targets

- **Liz:** No target
- **Tom:** No target
- **Berlin:** No target

**What this means:** These three don't want anything specific. They just respond to what's happening.

---

## Step 2: Compute Gradients

For characters with targets, compute the gradient (direction toward target):

```python
# Red
∇φ_Red = -2 * (pos_Red - target_Red)
       = -2 * ([0.0050, ...] - [0.0079, ...])
       = [0.0058, ...]  (points toward target)

# Ressler
∇φ_Ressler = -2 * (pos_Ressler - target_Ressler)
           = [...]

# Cooper
∇φ_Cooper = -2 * (pos_Cooper - target_Cooper)
          = [...]

# Characters with no target
∇φ_Liz = [0, 0, 0, ...] (zero vector)
∇φ_Tom = [0, 0, 0, ...]
∇φ_Berlin = [0, 0, 0, ...]
```

---

## Step 3: Each Character Picks a Direction

Everyone optimizes this score function:

```
S(e) = α⟨e,∇φ⟩ + (γ-β)⟨e,Je⟩ - η|⟨e,n⟩|
```

Where:
- `e` = direction to move (unit vector)
- `∇φ` = gradient toward target
- `J` = Jacobian (system dynamics)
- `n` = boundary normal

### Red (α=1, β=1, γ=1, η=1)

```
γ-β = 0
S(e) = 1·⟨e,∇φ⟩ + 0·⟨e,Je⟩ - 1·|⟨e,n⟩|
     = ⟨e,∇φ⟩ - |⟨e,n⟩|
```

Red picks direction `e` that:
- Aligns with gradient (toward target)
- Avoids boundary

At t=0, J=0, so Red picks: `e_Red ≈ ∇φ_Red / ||∇φ_Red||` (direction toward target)

### Liz (α=0, β=0, γ=1, η=0)

```
γ-β = 1
S(e) = 0 + 1·⟨e,Je⟩ - 0
     = ⟨e,Je⟩
```

Liz picks direction `e` that aligns with `Je` (where Jacobian pushes her).

At t=0, J=0, so she picks a random direction initially. **On subsequent steps**, she responds to the Jacobian (amplifies system dynamics).

### Tom (α=0, β=1, γ=-1, η=1)

```
γ-β = -2
S(e) = 0 + (-2)·⟨e,Je⟩ - 1·|⟨e,n⟩|
     = -2⟨e,Je⟩ - |⟨e,n⟩|
```

Tom picks direction `e` that:
- Opposes where Jacobian pushes (negative coefficient)
- Avoids boundary

**What this means:** Tom resists system dynamics.

### Ressler (α=1, β=0, γ=0, η=-1)

```
γ-β = 0
S(e) = 1·⟨e,∇φ⟩ + 0 - (-1)·|⟨e,n⟩|
     = ⟨e,∇φ⟩ + |⟨e,n⟩|
```

Ressler picks direction that:
- Aligns with gradient (toward Red)
- Favors boundary (willing to cross lines)

### Cooper (α=0.5, β=0.5, γ=-0.5, η=0.5)

```
γ-β = -1
S(e) = 0.5·⟨e,∇φ⟩ + (-1)·⟨e,Je⟩ - 0.5·|⟨e,n⟩|
```

Cooper: moderate target + resist system + avoid boundary (stabilizer)

### Berlin (α=0, β=0, γ=1.5, η=-1)

```
γ-β = 1.5
S(e) = 0 + 1.5·⟨e,Je⟩ + |⟨e,n⟩|
     = 1.5⟨e,Je⟩ + |⟨e,n⟩|
```

Berlin picks direction that:
- Strongly amplifies Jacobian (1.5x)
- Seeks boundary crossing

**Summary:** No categories, just different coefficients. Everyone uses the same formula.

---

## Step 4: Compute Global Flow

Combine everyone's directions, weighted by influence:

```python
F = w_Red·e_Red + w_Liz·e_Liz + w_Tom·e_Tom + w_Ressler·e_Ressler + w_Cooper·e_Cooper + w_Berlin·e_Berlin
  = 0.8·e_Red + 0.6·e_Liz + 0.7·e_Tom + 0.4·e_Ressler + 0.5·e_Cooper + 0.8·e_Berlin
```

This is a 72-dimensional vector (6 characters × 12 dimensions each).

At t=0: `||F|| = 1.594`

**What this means:** The system's overall movement direction. Red and Berlin have equal influence (0.8), system is multi-polar.

---

## Step 5: Update Weights (Who's Effective?)

For each character, check if their chosen direction aligns with where the system is actually moving:

```python
# Extract character k's portion of global flow
F_k = F[k*12 : (k+1)*12]  (their 12 dimensions)

# How aligned is their direction with actual movement?
alignment_k = ⟨e_k, F_k⟩

if alignment_k > 0:
    weight_k *= 1.01  # They're effective, gain influence
else:
    weight_k *= 0.99  # They're ineffective, lose influence
```

**What this means:** Characters who pick directions that match where the system goes gain power. This is emergence - power isn't preset.

---

## Step 6: Move Characters (with Adaptive Damping)

Each character moves in their chosen direction, scaled by their weight, with damping:

```python
# Compute damping factor from system energy
damping = 1 / (1 + ||F||)
        = 1 / (1 + 1.594)
        = 0.385

# Move each character
pos_Red = pos_Red + dt * w_Red * e_Red * damping
        = [0.0050, ...] + 0.1 * 0.8 * [...] * 0.385
        = [new position]

pos_Liz = pos_Liz + dt * w_Liz * e_Liz * damping
...
```

**What this means:** When system is calm (small F), damping ≈1, normal movement. When system is energetic (large F), damping <<1, movement shrinks. This prevents explosion.

After this step:
```
Red: ||pos|| = 0.030 (moved slightly from 0.027)
Liz: ||pos|| = 0.040
Tom: ||pos|| = 0.032
...
Total ||positions|| = 0.186
```

**What this means:** Everyone moves a bit. Because relational targets change too as positions change.

---

## Step 7: Compute Jacobian

The Jacobian J measures: "How does changing one character's position affect everyone's flow?"

```python
J = dF/dX  (72×72 matrix)
```

To compute it, perturb each dimension and see how F changes:

```python
for i in range(72):  # Each dimension
    # Figure out which character and which dimension
    char_idx = i // 12
    dim_idx = i % 12

    # Perturb that character's position
    pos_perturbed = copy(positions)
    pos_perturbed[char_idx][dim_idx] += ε

    # Recompute targets (they depend on positions!)
    # Recompute directions
    # Recompute flow
    F_perturbed = compute_flow(...)

    # Finite difference
    J[:, i] = (F_perturbed - F) / ε
```

**What this means:** Because targets are relational, changing positions changes targets, which changes gradients, which changes directions, which changes flow. Everything couples.

---

## Iteration 1 → First Inflection Detected!

After one step, the system detects **Weak Coupling** inflection.

### Check Condition: Domain Coupling Ratio

```python
# Extract Jacobian submatrices
# J_cross = dFv/dxh (how hidden dims affect visible flow)
# J_self  = dFv/dxv (how visible dims affect visible flow)

coupling_ratio = ||J_cross|| / ||J_self||
               = 1.029

# Check threshold
1.029 < 1.1  ✓ SATISFIED
```

**What this means:** Cross-domain coupling (1.029) is close to self-coupling (1.0). Hidden and visible dimensions are **balanced/weakly coupled** - they're not strongly interdependent. This allows independent evolution in different dimensions.

**Narrative interpretation:** Internal dynamics (hidden) and external actions (visible) can evolve somewhat independently - characters have internal motivations that don't always directly translate to visible actions.

---

## Iteration 2 → Second Inflection Detected!

After two steps, the system detects **Non-Local Paths** inflection.

### Check Condition 1: Low Alignment

```python
# Red's gradient at t=2
grad_phi = target_gradient("Red", positions_t2)
         = -2 * (pos_Red - target_Red)

# Red's portion of global flow
F_Red = F[0:12]

# Compute alignment
alignment = |⟨F_Red, grad_phi⟩| / (||F_Red|| × ||grad_phi||)
          = 0.059

# Check threshold
0.059 < 0.3  ✓ SATISFIED
```

**What this means:** Red's movement direction (F_Red) is nearly perpendicular to the gradient (direct path to target). He's only 5.9% aligned - moving almost sideways relative to his goal.

### Check Condition 2: Positive Progress

```python
# Current target (at t=2)
target_current = compute_target("Red", positions_t2)
               = pos_Liz_t2 - pos_Tom_t2

# Distance from current position to current target
d_current = ||pos_Red_t2 - target_current||

# Distance from previous position to current target
d_previous = ||pos_Red_t1 - target_current||

# Progress (both measured against SAME target)
phi_current = -d_current^2
phi_previous = -d_previous^2
delta_phi = phi_current - phi_previous = 0.0011

# Check
0.0011 > 0  ✓ SATISFIED
```

**What this means:** Even though Red moved away from the gradient direction (sideways), he got **closer to where the target is NOW** than he was before. He moved 0.0011 closer.

### Why This Matters

This is **non-local geometry**: You can't get there by going straight, but you can get there by following a curved path.

**Narrative interpretation:** Red doesn't manipulate Liz and Tom directly. He influences the broader system (other characters, hidden dynamics), which indirectly achieves his goal. The Jacobian coupling creates curved paths through state space - you can make progress without moving directly toward your target.

This is exactly how Red operates in the show - indirect manipulation that achieves objectives without direct action.

---

## Long-Term Evolution (t=0 to t=100)

### Position Norms

```
t=0:   Total=0.187, Red=0.027, Liz=0.039, Tom=0.030
t=20:  Total=0.872, Red=0.177, Liz=0.134, Tom=0.135
t=40:  Total=1.776, Red=0.352, Liz=0.278, Tom=0.279
t=100: Total=6.228, Red=1.271, Liz=0.975, Tom=0.969
```

**What this means:** Positions grow but stay bounded (damping works). Red moves most but not overwhelmingly (balanced weights).

### Weights

```
t=0:   Red=0.8,  Liz=0.6,  Tom=0.7,  Berlin=0.8
t=100: Red=2.71, Liz=2.16, Tom=1.62, Berlin=3.25
```

**What this means:** Everyone gains influence (all are somewhat effective), Berlin gains most (strongest amplifier, γ-β=1.5). Red is influential but not dominating.

### System Energy

```
t=0:   ||F|| = 1.59,  damping = 0.385
t=20:  ||F|| = 2.37,  damping = 0.297
t=100: ||F|| = 5.26,  damping = 0.160
```

**What this means:** As system becomes more energetic, damping automatically reduces movement. This keeps the system bounded and stable.

---

## Character Behaviors (No Bullshit Categories)

### Red (α=1, β=1, γ=1, η=1)

Score: `S(e) = ⟨e,∇φ⟩ - |⟨e,n⟩|`

Picks direction toward target while avoiding boundary.

### Liz (α=0, β=0, γ=1, η=0)

Score: `S(e) = ⟨e,Je⟩`

Picks direction that aligns with Jacobian. **Amplifies system dynamics** (γ-β=1 > 0).

### Tom (α=0, β=1, γ=-1, η=1)

Score: `S(e) = -2⟨e,Je⟩ - |⟨e,n⟩|`

Picks direction that opposes Jacobian. **Resists system dynamics** (γ-β=-2 < 0).

### Ressler (α=1, β=0, γ=0, η=-1)

Score: `S(e) = ⟨e,∇φ⟩ + |⟨e,n⟩|`

Picks direction toward Red, willing to cross boundaries.

### Cooper (α=0.5, β=0.5, γ=-0.5, η=0.5)

Score: `S(e) = 0.5⟨e,∇φ⟩ - ⟨e,Je⟩ - 0.5|⟨e,n⟩|`

Moderate target seeking, resists flow (γ-β=-1), avoids boundary. **Stabilizer**.

### Berlin (α=0, β=0, γ=1.5, η=-1)

Score: `S(e) = 1.5⟨e,Je⟩ + |⟨e,n⟩|`

Strongly amplifies Jacobian (γ-β=1.5), seeks boundary crossing. **Strongest chaos injector**.

**The only difference:** coefficient values in one formula. That's it.

---

## State-Dependent Boundary

The boundary constraint adapts to system energy:

```python
total_energy = sum(||pos_k|| for all k)
             = ||pos_Red|| + ||pos_Liz|| + ... + ||pos_Berlin||

strength = 1.0 + 0.1 * tanh(total_energy - 0.1)

n_boundary = strength * n_base
```

At t=0:   total_energy = 0.187 → strength = 1.0
At t=100: total_energy = 6.228  → strength = 1.09

**What this means:** As the system becomes more unstable (higher energy), the boundary constraint strengthens slightly. This represents institutional pressure that scales with chaos level.

---

## Summary: What's Emergent?

✓ **Goals:** Red's target moves as Liz/Tom move (unnormalized, magnitude matters)
✓ **Trajectories:** Non-local paths detected (curved, not straight)
✓ **Power:** Weights grow based on effectiveness, not preset
✓ **Conflict:** Characters with opposite (γ-β) signs naturally oppose each other
✓ **Constraints:** Boundary strength emerges from system state
✓ **Stability:** Damping emerges from system energy (prevents explosion)

**Nothing is scripted.** Just characters optimizing their individual score functions, and complex dynamics emerge from interaction.

---

## Are We Meeting Our Goals?

### ✓ Goal 1: Emergent Goals

**YES.** Red's target is `pos_Liz - pos_Tom` (unnormalized). As Liz and Tom move, Red's target moves. The magnitude (0.0455) matters - it tells Red how far apart they should be.

**Evidence:** Target recomputed at every step based on current positions, without normalization.

### ✓ Goal 2: Emergent Trajectories

**YES.** Non-Local Paths inflection at t=2 proves Red takes curved paths through state space, not straight lines.

**Evidence:**
- Alignment = 0.059 (nearly perpendicular to gradient, only 5.9% aligned)
- Progress = 0.0011 (still getting closer)
- This is only possible with curved geometry from multi-character coupling via Jacobian

### ✓ Goal 3: Emergent Power

**YES.** Weights update based on effectiveness: `weight *= 1.01 if alignment > 0 else 0.99`

**Evidence:**
- All characters gain influence (all effective)
- Berlin: 0.8 → 3.25 (gains most, strongest amplifier)
- Red: 0.8 → 2.71 (gains but not dominating)
- Power is earned through performance, not preset

### ✓ Goal 4: Emergent Conflict

**YES.** Characters with opposite signs of (γ-β) naturally oppose each other:
- Liz (γ-β=1) amplifies what Tom (γ-β=-2) resists
- Berlin (γ-β=1.5) injects chaos that destabilizes Cooper (γ-β=-1) who tries to stabilize

**Evidence:** Conflict emerges from coefficient differences, not scripted opposition.

### ✓ Goal 5: Unified Framework

**YES.** All characters use the same score function: `S(e) = α⟨e,∇φ⟩ + (γ-β)⟨e,Je⟩ - η|⟨e,n⟩|`

**No artificial categories:**
- Just α=0 vs α≠0 (one parameter)
- All behaviors emerge from coefficient choices
- No "goal-driven" vs "reactive" - just math

### ✓ Goal 6: State-Dependent Constraints

**YES.** Boundary strength adapts to system energy: `strength = 1.0 + 0.1 * tanh(total_energy - 0.1)`

**Evidence:**
- t=0: strength = 1.0
- t=100: strength = 1.09
- Institutional pressure scales with instability

### ✓ Goal 7: System Stability

**YES.** Adaptive damping prevents explosion: `damping = 1 / (1 + ||F||)`

**Evidence:**
- System stays bounded (positions reach ~6, not 2827)
- Damping scales inversely with system energy
- All dynamics preserved, only magnitude controlled

---

## Final Verdict

**System Status:** Complete and correct.

**What emerged:**
1. Goals shift dynamically (relational targets, unnormalized)
2. Trajectories curve through state space (non-local geometry from Jacobian coupling)
3. Power grows from effectiveness (performance-based weights)
4. Conflict arises from opposing coefficients (natural opposition)
5. Constraints adapt to system state (dynamic boundaries)
6. Stability emerges from energy (adaptive damping)

**What's NOT scripted:**
- Who wins (Berlin currently gains most influence, Red is second)
- What paths they take (curved, emergent from coupling)
- What conflicts arise (emerge from interactions)
- When things happen (inflections detected geometrically: Weak Coupling t=1, Non-Local Paths t=2)

This is a **physics engine for narrative**, not a scripted story.
