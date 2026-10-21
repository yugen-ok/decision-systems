"""
Main simulation script.

Loads case study data, runs simulation with relational targets, saves results.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

from data_loader import load_case_study, get_available_case_studies
from simulation_engine import DynamicsEngine, InflectionDetector, SimulationState, InflectionEvent


def initialize_state(case_study) -> SimulationState:
    """Initialize simulation state with per-character positions."""
    np.random.seed(42)

    # Initialize positions for each character (each in R^n)
    positions = {}
    for char in case_study.characters:
        positions[char.name] = np.random.randn(case_study.n_dims) * case_study.initial_state_scale

    # Initial weights
    weights = {char.name: char.initial_weight for char in case_study.characters}

    # Create engine
    engine = DynamicsEngine(
        case_study.n_dims,
        case_study.boundary_normal,
        case_study.characters,
        case_study.jacobian_epsilon
    )

    # Initialize with zero Jacobian
    n_total = case_study.n_dims * len(case_study.characters)
    J = np.zeros((n_total, n_total))

    # Compute initial directions
    directions = {}
    for char in case_study.characters:
        directions[char.name] = engine.find_best_direction(char, positions, J)

    # Compute initial global flow
    F = engine.compute_flow(directions, weights)

    return SimulationState(
        t=0,
        positions=positions,
        F=F,
        J=J,
        directions=directions,
        weights=weights,
        episode=0.0
    )


def run_simulation(case_study, max_inflections=2, max_iterations=1000, log_interval=20):
    """Run simulation until inflections detected or max iterations reached.

    Args:
        case_study: CaseStudyData object
        max_inflections: Stop after detecting this many inflections
        max_iterations: Maximum number of iterations
        log_interval: Print status every N iterations

    Returns:
        (history, detected_inflections) tuple
    """
    # Create engine and detector
    engine = DynamicsEngine(
        case_study.n_dims,
        case_study.boundary_normal,
        case_study.characters,
        case_study.jacobian_epsilon
    )

    # Only check inflections up to max_inflections
    inflections_to_check = case_study.inflections[:max_inflections]

    detector = InflectionDetector(
        engine,
        inflections_to_check,
        case_study.time_per_episode,
        case_study.hidden_dims,
        case_study.visible_dims
    )

    # Initialize
    state = initialize_state(case_study)
    history = [state]
    detected_inflections = []
    detected_ids = set()

    print(f"Running simulation: {case_study.name}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Stop after: {max_inflections} inflections")
    print(f"  Characters: {len(case_study.characters)}")
    print(f"  Each position in R^{case_study.n_dims}")
    print()

    # Main loop
    for i in range(max_iterations):
        # Update weights based on performance (alignment with global flow)
        new_weights = {}
        for char in case_study.characters:
            # Get character's direction and flow contribution
            e_k = state.directions[char.name]
            char_idx = engine.char_names.index(char.name)
            F_char = state.F[char_idx * engine.n : (char_idx + 1) * engine.n]

            # Compute alignment with global flow
            alignment = np.dot(e_k, F_char)

            # Update weight based on effectiveness
            if alignment > 0:
                # Direction matches system movement -> increase influence
                new_weights[char.name] = state.weights[char.name] * 1.01
            else:
                # Direction opposes system movement -> decrease influence
                new_weights[char.name] = state.weights[char.name] * 0.99

        # Compute new Jacobian
        J_new = engine.compute_jacobian(state.positions, state.directions, state.weights)

        # Each character picks new direction
        new_directions = {}
        for char in case_study.characters:
            new_directions[char.name] = engine.find_best_direction(char, state.positions, J_new)

        # Compute new global flow
        F_new = engine.compute_flow(new_directions, new_weights)

        # Update positions with adaptive damping: pos_k += dt * a_k * e_k / (1 + ||F||)
        damping_factor = 1.0 / (1.0 + np.linalg.norm(F_new))
        new_positions = {}
        for char in case_study.characters:
            a_k = new_weights[char.name]
            e_k = new_directions[char.name]
            new_positions[char.name] = state.positions[char.name] + case_study.dt * a_k * e_k * damping_factor

        # Create new state
        t_new = state.t + 1
        ep_new = t_new * case_study.dt / case_study.time_per_episode

        state = SimulationState(
            t=t_new,
            positions=new_positions,
            F=F_new,
            J=J_new,
            directions=new_directions,
            weights=new_weights,
            episode=ep_new
        )

        history.append(state)

        # Check inflections
        if len(history) >= 2:
            prev_state = history[-2]

            for spec in inflections_to_check:
                if spec.id in detected_ids:
                    continue

                detected, details = detector.check_inflection(state, prev_state, spec.id)

                if detected:
                    event = InflectionEvent(
                        id=spec.id,
                        name=spec.name,
                        time=state.t,
                        episode=state.episode,
                        condition_details=details
                    )
                    detected_inflections.append(event)
                    detected_ids.add(spec.id)

                    print(f"[INFLECTION DETECTED] {spec.name} at t={state.t}, ep={state.episode:.2f}")

        # Log progress
        if (i + 1) % log_interval == 0:
            print(f"  Iteration {i+1}/{max_iterations}, ep={state.episode:.1f}, "
                  f"inflections={len(detected_inflections)}/{max_inflections}")

            # Detailed state
            total_pos_norm = sum(np.linalg.norm(p) for p in state.positions.values())
            print(f"    Total ||positions||={total_pos_norm:.4f}, ||F||={np.linalg.norm(state.F):.4f}")
            print(f"    Weights: " + ", ".join([f"{name}={w:.3f}" for name, w in state.weights.items()]))

        # Stop if we've detected enough inflections
        if len(detected_inflections) >= max_inflections:
            print(f"\nStopping: detected {max_inflections} inflections at iteration {i+1}")
            break

    print(f"\nSimulation complete: {len(history)} iterations, {len(detected_inflections)} inflections detected")

    return history, detected_inflections


def save_results(case_study, history, detected_inflections, output_dir="results"):
    """Save simulation results to output directory."""
    # Create output directory
    output_path = Path(output_dir) / case_study.id
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detected inflections
    inflections_data = []
    for event in detected_inflections:
        def convert_value(val):
            if isinstance(val, (np.floating, np.integer)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (bool, np.bool_)):
                return bool(val)
            return val

        inflections_data.append({
            "id": int(event.id),
            "name": event.name,
            "time": int(event.time),
            "episode": float(event.episode),
            "details": {k: {sub_k: convert_value(sub_v) for sub_k, sub_v in v.items()}
                       for k, v in event.condition_details.items()}
        })

    with open(output_path / "inflections.json", 'w') as f:
        json.dump(inflections_data, f, indent=2)

    # Save detailed numerical output
    with open(output_path / "trajectory.txt", 'w') as f:
        f.write(f"# Simulation Trajectory: {case_study.name}\n")
        f.write(f"# Time units per episode: {case_study.time_per_episode:.2f}\n")
        f.write(f"# Relational targets: positions per character\n\n")

        for i, state in enumerate(history):
            if i % 10 == 0:  # Log every 10 iterations
                f.write(f"=== Iteration {state.t}, Episode {state.episode:.2f} ===\n")
                for char_name, pos in state.positions.items():
                    f.write(f"{char_name} position: {pos}\n")
                    f.write(f"  ||pos|| = {np.linalg.norm(pos):.6f}\n")
                f.write(f"Global flow F: ||F|| = {np.linalg.norm(state.F):.6f}\n")
                f.write(f"Weights: {state.weights}\n")
                f.write("\n")

    # Create visualization
    create_visualization(case_study, history, detected_inflections, output_path)

    print(f"\nResults saved to: {output_path}")


def create_visualization(case_study, history, detected_inflections, output_path):
    """Create visualization of simulation results."""
    # Create engine for computing observables
    engine = DynamicsEngine(
        case_study.n_dims,
        case_study.boundary_normal,
        case_study.characters,
        case_study.jacobian_epsilon
    )

    # Extract time series
    times = [s.episode for s in history]

    # Compute observables for Red (primary character)
    alignments = []
    phi_values = []

    for state in history:
        # Alignment for Red
        grad_phi = engine.target_gradient("Red", state.positions)
        char_idx = engine.char_names.index("Red")
        F_red = state.F[char_idx * engine.n : (char_idx + 1) * engine.n]

        F_norm = np.linalg.norm(F_red)
        grad_norm = np.linalg.norm(grad_phi)

        if F_norm > 1e-10 and grad_norm > 1e-10:
            alignment = np.abs(np.dot(F_red, grad_phi)) / (F_norm * grad_norm)
        else:
            alignment = 0.0
        alignments.append(alignment)

        # Red's target value
        phi = engine.target_field("Red", state.positions)
        phi_values.append(phi)

    # Coupling (extract from Jacobian)
    couplings = []
    for state in history:
        if state.t > 0:
            n_chars = len(engine.char_names)
            n_dims = engine.n
            hidden_indices = []
            visible_indices = []
            for char_idx in range(n_chars):
                base = char_idx * n_dims
                for h_dim in case_study.hidden_dims:
                    hidden_indices.append(base + h_dim)
                for v_dim in case_study.visible_dims:
                    visible_indices.append(base + v_dim)

            J_cross = state.J[np.ix_(visible_indices, hidden_indices)]
            J_self = state.J[np.ix_(visible_indices, visible_indices)]

            cross_norm = np.linalg.norm(J_cross, 'fro')
            self_norm = np.linalg.norm(J_self, 'fro')

            if self_norm > 1e-10:
                coupling = cross_norm / self_norm
            else:
                coupling = 0.0
            couplings.append(coupling)
        else:
            couplings.append(0.0)

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Plot 1: Alignment
    ax = axes[0]
    ax.plot(times, alignments, 'b-', linewidth=1.5, label='Alignment')
    ax.axhline(0.3, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_ylabel('Alignment |<F,grad_phi>|/(|F||grad_phi|)')
    ax.set_title('Inflection 1: Alignment Check (should be < 0.3)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for event in detected_inflections:
        if event.name == "Non-Local Paths":
            ax.axvline(event.episode, color='green', linestyle=':', alpha=0.7, linewidth=2)

    # Plot 2: Coupling
    ax = axes[1]
    ax.plot(times, couplings, 'purple', linewidth=1.5, label='Domain Coupling')
    ax.axhline(0.9, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_ylabel('Domain Coupling ||dFv/dxh|| / ||dFv/dxv||')
    ax.set_title('Inflection 2: Coupling Check (should be < 0.9)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for event in detected_inflections:
        if event.name == "Weak Coupling":
            ax.axvline(event.episode, color='green', linestyle=':', alpha=0.7, linewidth=2)

    # Plot 3: Red's progress
    ax = axes[2]
    ax.plot(times, phi_values, 'darkred', linewidth=1.5, label='Red Target Value')
    ax.set_ylabel("Red Target Value phi(pos_Red)")
    ax.set_xlabel('Episode')
    ax.set_title("Red's Progress (relational target: pos_Liz - pos_Tom)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'observables.png', dpi=150)
    print(f"  Saved: {output_path / 'observables.png'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run geometric flow dynamics simulation')

    parser.add_argument('--case-study', type=str, default='blacklist_s1',
                       help='Case study ID to load (default: blacklist_s1)')
    parser.add_argument('--inflections', type=int, default=2,
                       help='Number of inflection points to detect (default: 2)')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Maximum number of iterations (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=20,
                       help='Log status every N iterations (default: 20)')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Directory containing case study JSON files')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Directory for saving results')
    parser.add_argument('--list', action='store_true',
                       help='List available case studies and exit')

    args = parser.parse_args()

    # List case studies if requested
    if args.list:
        available = get_available_case_studies(args.data_dir)
        print("Available case studies:")
        for cs_id in available:
            print(f"  - {cs_id}")
        return

    # Load case study
    print(f"Loading case study: {args.case_study}")
    case_study = load_case_study(args.case_study, args.data_dir)
    print(f"  {case_study.name}")
    print(f"  {case_study.description}")
    print(f"  Characters: {len(case_study.characters)}")
    print(f"  Relational targets: enabled")
    print()

    # Run simulation
    history, detected_inflections = run_simulation(
        case_study,
        max_inflections=args.inflections,
        max_iterations=args.iterations,
        log_interval=args.log_interval
    )

    # Save results
    save_results(case_study, history, detected_inflections, args.output_dir)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Detected {len(detected_inflections)} inflection(s):")
    for event in detected_inflections:
        print(f"  {event.name}: ep {event.episode:.2f} (t={event.time})")
    print("="*60)


if __name__ == "__main__":
    main()
