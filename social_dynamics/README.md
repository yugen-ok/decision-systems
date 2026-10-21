# Geometric Flow Dynamics Framework

Modeling human dynamics and narrative structure as emergent geometric flows.

## Abstract

We propose a minimal dynamical framework for modeling human interaction and narrative structure as geometric flows in an abstract state space. System evolution is defined by a vector field F(x), decomposed into a weighted sum of directional modes, each derived from four primitive geometric functionals: (i) boundary interaction (violence), (ii) alignment with a target field (love), (iii) divergence or conservation (money), and (iv) local amplification of perturbations (respect). Character behavior is not explicitly scripted; instead, each "mode" corresponds to a distinct weighting over these primitives, yielding emergent directional preferences within a shared state space.

We define narrative inflection points as transitions in the geometric structure of the flow, formalized as compound conditions on local and global properties (e.g., simultaneous orthogonality to a target gradient and positive progression along the trajectory). These conditions capture regime shifts such as domain decoupling/coupling, constraint breakdown, and loss of stability.

To evaluate the framework, we construct a simulation of interacting modes corresponding to principal agents in The Blacklist Season 1. Without encoding any plot events, we track system observables and detect inflection points based solely on the defined geometric criteria. The model identifies 5 of 7 major narrative transitions within the empirically observed episode ranges.

The results suggest that complex narrative and social dynamics can be represented as transitions between regimes of a low-dimensional geometric system, and that such transitions are inherently multi-constraint phenomena rather than threshold-based events. This supports the hypothesis that a small set of primitive geometric interactions is sufficient to generate and analyze structured emergent behavior in human systems.


## Motivation Overview

 ## Motivation

   **Central claim**: Complex human dynamics and narrative structure can be modeled as geometric flows in an abstract state space, emerging from a small set of          
   primitive forces.                                                                                                                                                     
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

   **Core insight**: Just as physical systems evolve according to force fields, social and narrative systems can be modeled as flows in state space, where different     
   "characters" or "modes" contribute different directional preferences based on primitive geometric properties.

## Quick Start

**Read the documentation**:
- `docs/OVERVIEW.md` - Ultra-compressed summary
- `docs/FRAMEWORK.md` - Complete framework with detailed explanations

**Run the simulation**:
```bash
cd simulation
pip install -r requirements.txt
python run_simulation.py
```

## Repository Structure

```
docs/
  OVERVIEW.md         # Quick summary
  FRAMEWORK.md        # Complete documentation
simulation/
  core_dynamics.py    # Framework implementation
  run_simulation.py   # Simulation runner
  requirements.txt    # Dependencies
backup/               # Original formulation files
```

## Key Principles

- **Minimal**: One file per required thing
- **Emergent**: Inflection points arise naturally, not hardcoded
- **Consistent**: Conceptual → Mathematical → Implementation
- **Clear**: Everything explained, no jumps

## Results

Both inflection points detected in chronological order from pure character interaction:
1. Weak Coupling (t=1)
2. Non-Local Paths (t=9)

No behaviors hardcoded. Pure emergence from four primitive forces.
