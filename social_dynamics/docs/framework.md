# Geometric Flow Dynamics Framework

## Motivation

**Central claim**: Complex human dynamics and narrative structure can be modeled as geometric flows in an abstract state space, emerging from a small set of primitive forces.

**Why this approach?**

Traditional models of human interaction and narrative either:
- Hardcode specific behaviors (brittle, non-generalizable)
- Use black-box machine learning (uninterpretable, no clear mechanism)
- Rely on graph/network structures (lose geometric richness)

Our framework instead:
- Defines minimal mathematical primitives
- Lets complex patterns emerge from composition
- Maintains interpretability through geometric intuition
- Generalizes across different domains

**Core insight**: Just as physical systems evolve according to force fields, social and narrative systems can be modeled as flows in state space, where different "characters" or "modes" contribute different directional preferences based on primitive geometric properties.

## Conceptual Foundation

### The State Space

We model a system as a point x in an n-dimensional state space. The system evolves according to:

```
dx/dt = F(x)
```

where F(x) is a vector field describing how the system moves from state x.

### Key Idea: Flow Decomposition

Rather than defining F(x) directly, we decompose it into multiple modes:

```
F(x) = Σ a_k(x) e_k(x)
```

where:
- e_k(x) is the preferred direction for mode k at state x
- a_k(x) is the weight or activity level of mode k at state x

Each mode represents a distinct "character" or behavioral pattern. The total system dynamics emerge from their weighted composition.

## The Four Primitives

Each mode selects its preferred direction by scoring candidate directions according to four primitive geometric properties. These primitives are not arbitrary - they represent fundamental ways a flow can relate to local geometry.

### 1. Love (L): Alignment with Target

**Geometric form**:
```
l[e](x) = <e(x), ∇φ(x)>
```

**Meaning**: How much direction e aligns with the gradient of a target field φ(x).

**Interpretation**:
- φ(x) represents "distance to goal" or "alignment with desired state"
- ∇φ(x) points in the direction of steepest increase
- Positive love means moving toward what matters

**Plain language**: Moving toward an objective.

### 2. Money (M): Conservation

**Geometric form**:
```
m[e](x) = -(∇·e(x))^2
```

**Meaning**: Highest when the flow is locally conserved (divergence near zero).

**Interpretation**:
- ∇·e measures whether flow is expanding (positive) or contracting (negative)
- Money favors directions that preserve structure
- Nothing created or destroyed, only redistributed

**Plain language**: Maintaining balance, avoiding waste or expansion.

### 3. Respect (R): Amplification

**Geometric form**:
```
r[e](x) = max_{|δ|=1} <δ, (∇e(x))δ>
```

**Meaning**: How much the field amplifies small perturbations.

**Interpretation**:
- Measures sensitivity to initial differences
- High respect means small distinctions get magnified
- Hidden structure becomes visible through amplification

**Plain language**: Making small differences matter, exposing inconsistencies.

### 4. Violence (V): Boundary Crossing

**Geometric form**:
```
v[e](x) = |<e(x), n(x)>|
```

**Meaning**: Component of flow perpendicular to a boundary or constraint.

**Interpretation**:
- n(x) is the boundary normal (direction of resistance)
- Violence measures pushing through constraints
- Zero means moving parallel to boundaries

**Plain language**: Breaking through resistance or constraints.

## Character Score Functions

### The Universal Scoring Equation

Each character mode k is defined by how it weights the four primitives:

```
S_k(e) = α_k·l[e](x) + (γ_k - β_k)·<e, Je> - η_k·v[e](x)
```

where:
- α_k is the love coefficient (target orientation)
- β_k is the money coefficient (conservation preference)
- γ_k is the respect coefficient (amplification preference)
- η_k is the violence coefficient (boundary aversion/preference)
- J is the Jacobian matrix ∇F(x)

**Note on the combined term**: The money and respect terms both involve the Jacobian. For implementation, we use <e, Je> which combines their effects through the coefficient (γ - β).

### Interpreting Coefficients

**Zero means indifference, not absence**:
- α = 0: No target, not "no movement toward target"
- β = 0: Indifferent to conservation, not "actively wastes"
- γ = 0: Indifferent to amplification, not "actively dampens"
- η = 0: Boundary-neutral, not "avoids boundaries"

**Sign matters**:
- Positive coefficient: Favor this primitive
- Negative coefficient: Oppose this primitive
- Zero: Ignore this primitive

**Examples**:
- α = 1, β = 1, γ = 1, η = 1: Seeks target, conserves, amplifies, avoids conflict (γ - β = 0)
- α = 0, β = 0, γ = 1, η = 0: No target, amplifies differences, boundary-neutral
- α = 0, β = 1, γ = -1, η = 1: No target, conserves, dampens, avoids conflict (γ - β = -2)
- α = 1, β = 0, γ = 0, η = -1: Has target, seeks boundaries, neutral on conservation/amplification

### Finding Preferred Directions

For each mode, the preferred direction e_k is:

```
e_k = argmax_{|e|=1} S_k(e)
```

This is an optimization problem on the unit sphere. For modes where (γ - β) = 0, this simplifies to:

```
e_k = argmax_{|e|=1} [α_k·<e,∇φ> - η_k·|<e,n>|]
```

which has analytical solutions:
- If η > 0: Project ∇φ perpendicular to boundary normal
- If η < 0: Add boundary normal component to ∇φ
- If η = 0: Use ∇φ direction directly

For modes with (γ - β) ≠ 0, numerical optimization (gradient ascent on the sphere) is used.

## Flow Composition

Once each mode's preferred direction is determined, the total flow is:

```
F(x) = Σ a_k(x) e_k(x)
```

The weights a_k can be:
- Fixed constants (each mode has constant influence)
- State-dependent functions (mode activity changes based on x)
- Dynamically regulated (constrained by stability conditions)

**Emergence**: Complex behavior arises from the interaction of these simple directional preferences, not from explicit programming of behaviors.

## Inflection Points

Inflection points are geometric phase transitions in the flow structure. They are defined as compound conditions on local and global properties, not simple thresholds.

### General Principle

An inflection point occurs when multiple geometric constraints are simultaneously satisfied, indicating a qualitative change in system behavior.

### Generic Inflection Conditions

**Inflection Type 1: Non-Local Paths**

Indicates the system is making progress toward a goal via indirect routes rather than direct approach.

Conditions:
1. Low alignment: <F, ∇φ> / (|F| |∇φ|) < threshold (e.g., 0.3)
2. Positive progress: φ(x_current) > φ(x_previous)

**Geometric meaning**: Flow is nearly orthogonal to target gradient, yet target field value is increasing. The system is "going around" rather than "going toward."

**Inflection Type 2: Domain Decoupling**

Indicates separation between different subspaces of the system.

Condition:
- ||∂F_A/∂x_B|| / ||∂F_A/∂x_A|| < threshold (e.g., 0.9)

where:
- x_A and x_B are disjoint subspaces of the state space
- ∂F_A/∂x_B measures how subspace B influences dynamics in subspace A
- ∂F_A/∂x_A measures internal dynamics of subspace A

**Geometric meaning**: Changes in subspace B have diminishing influence on the flow in subspace A. The domains are becoming independent.

### Why Compound Conditions?

Simple threshold-based detection (e.g., "alignment < 0.3") produces false positives. Inflection points are genuine regime changes, requiring:
- Local conditions (current geometry)
- Trajectory conditions (direction of change)
- Multi-constraint satisfaction (simultaneous conditions)

This ensures detected transitions reflect actual structural changes in system behavior, not momentary fluctuations.

## Implementation Details

### State Space Structure

Typical structure for modeling systems with hidden and visible aspects:

- Total dimension: n
- Hidden subspace: First n_h dimensions (internal states, private information)
- Visible subspace: Remaining n_v dimensions (public states, observable information)

### Target Fields

Each mode can have its own target field φ_k(x). Common choices:
- Distance to a fixed point: φ(x) = -||x - x_target||^2
- Projection onto subspace: φ(x) = ||P x||^2 where P is projection operator
- No target: φ(x) = constant (α = 0)

### Boundary Fields

Boundaries can represent:
- Constraints (social norms, legal limits)
- Interfaces between domains
- Resource limitations

The boundary normal n(x) defines the direction of resistance.

### Numerical Integration

The system evolves via standard ODE integration:
```
x(t+dt) = x(t) + F(x(t))·dt
```

At each timestep:
1. Compute Jacobian J = ∇F(x) via finite differences
2. For each mode k, optimize to find e_k(x)
3. Compute total flow F(x) = Σ a_k e_k
4. Update state x
5. Check inflection conditions

### Computational Complexity

- Number of modes: K
- State dimension: n
- Jacobian computation: O(n^2)
- Direction optimization per mode: O(n) for simple, O(n·iterations) for general
- Total per timestep: O(K·n·iterations + n^2)

For small systems (K ≈ 5, n ≈ 12), this is highly tractable.

## Key Design Principles

**Minimality**: Four primitives, one equation per mode, no special cases.

**Emergence**: Complex patterns arise from composition, not explicit programming.

**Interpretability**: Each coefficient has clear geometric meaning.

**Generalizability**: Same framework applies across domains - only coefficient values change.

**Consistency**: Mathematical formulation matches conceptual description matches implementation.

## Summary

The framework models system dynamics as:

1. **State space**: System lives at point x in n-dimensional space
2. **Primitives**: Four geometric properties (love, money, respect, violence) score directions
3. **Modes**: Different weightings of primitives yield different behavioral patterns
4. **Composition**: Total flow is weighted sum of mode preferences
5. **Inflections**: Geometric phase transitions detected via compound conditions
6. **Emergence**: Complex behavior arises naturally from simple, interpretable rules

This provides a minimal, interpretable foundation for modeling complex adaptive systems where structured behavior emerges from primitive geometric interactions.
