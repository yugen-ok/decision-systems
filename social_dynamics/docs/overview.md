# Overview

**Claim**: Human dynamics emerge from four primitive forces acting on state space.

## The System

**Primitives** (score candidate directions):
- Love: <e,grad_phi> (toward target)
- Money: -<e,Je> (conserve, avoid expansion)
- Respect: <e,Je> (amplify, expose hidden)
- Violence: |<e,n>| (cross boundaries)

**Score function**:
```
S(e) = alpha<e,grad_phi> + (gamma-beta)<e,Je> - eta|<e,n>|
```

**Characters** = different coefficient weightings (alpha, beta, gamma, eta)

**Zero = indifference** (not action): alpha=0 is no target, eta=0 is boundary-neutral

**Flow**: F = sum a_k e_k (weighted composition of character directions)

**Inflections** = geometric phase transitions (multi-dimensional conditions)

## Documentation

- **framework.md** - Generic mathematical framework (no case study references)
- **case_study1.md** - Blacklist Season 1 application

## Implementation

Modular architecture:
- **data/** - Case study specifications (JSON)
- **simulation/data_loader.py** - Load case study data
- **simulation/simulation_engine.py** - Abstract dynamics engine
- **simulation/main.py** - Main script with CLI arguments

Run:
```bash
cd simulation
python main.py --case-study blacklist_s1 --inflections 2 --iterations 1000
```

Results saved to `results/<case_study_id>/`:
- inflections.json - Detected inflection points
- trajectory.txt - Detailed numerical evolution
- observables.png - Visualization

## Results

Both inflections detected in chronological order from pure character interaction:
1. Weak Coupling (t=1, ep 0.02)
2. Non-Local Paths (t=9, ep 0.17)

No hardcoded behaviors. Pure emergence.
