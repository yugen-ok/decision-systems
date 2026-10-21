"""
Simulation engine module.

Abstract simulation engine for geometric flow dynamics.
Now with relational targets and per-character positions.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from data_loader import CharacterSpec, InflectionSpec, InflectionCondition, compute_relational_target


@dataclass
class SimulationState:
    """Current state of the simulation."""
    t: int
    positions: Dict[str, np.ndarray]  # Per-character positions
    F: np.ndarray  # Global flow (for Jacobian/inflections)
    J: np.ndarray  # Jacobian
    directions: Dict[str, np.ndarray]
    weights: Dict[str, float]
    episode: float


@dataclass
class InflectionEvent:
    """Detected inflection point."""
    id: int
    name: str
    time: int
    episode: float
    condition_details: Dict[str, any]


class DynamicsEngine:
    """Core dynamics engine for flow evolution."""

    def __init__(self, n_dims: int, boundary_normal: np.ndarray,
                 characters: List[CharacterSpec], jacobian_eps: float = 1e-4):
        """Initialize dynamics engine.

        Args:
            n_dims: Dimensionality of each character position
            boundary_normal: Base normal vector to boundary (state-dependent)
            characters: List of character specifications
            jacobian_eps: Epsilon for finite difference Jacobian
        """
        self.n = n_dims
        self.n_boundary_base = boundary_normal  # Base boundary normal
        self.eps = jacobian_eps

        # Store character specs
        self.characters = {char.name: char for char in characters}
        self.char_names = [char.name for char in characters]

    def compute_boundary_normal(self, positions: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute state-dependent boundary normal.

        The boundary adapts based on system stability:
        - Higher overall energy → stronger constraints
        - Lower energy → weaker constraints

        Args:
            positions: Current positions of all characters

        Returns:
            Boundary normal vector
        """
        # Compute system energy (total position norm)
        total_norm = sum(np.linalg.norm(p) for p in positions.values())

        # Scale boundary strength based on system energy
        # If system is unstable (high energy), increase boundary strength
        strength_factor = 1.0 + 0.1 * np.tanh(total_norm - 0.1)

        # Return scaled boundary normal
        return strength_factor * self.n_boundary_base

    def compute_target(self, char_name: str, positions: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute target for character from relational specification.

        Args:
            char_name: Name of character
            positions: Current positions of all characters

        Returns:
            Target vector in R^n (or zeros if no target)
        """
        char = self.characters[char_name]
        relation = char.target_relation

        # If no target relation (alpha=0 characters), return zero vector
        if relation is None:
            return np.zeros(self.n)

        if relation.type == "center":
            # Return center directly (not center - pos_self)
            all_pos = np.array(list(positions.values()))
            center = np.mean(all_pos, axis=0)
            return center
        else:
            # Return relational target directly (don't normalize)
            target = compute_relational_target(relation, positions)
            return target

    def target_field(self, char_name: str, positions: Dict[str, np.ndarray]) -> float:
        """Target field value phi(pos_k) for character.

        Args:
            char_name: Name of character
            positions: Current positions of all characters

        Returns:
            Scalar field value
        """
        target = self.compute_target(char_name, positions)
        pos_k = positions[char_name]
        return -np.linalg.norm(pos_k - target)**2

    def target_gradient(self, char_name: str, positions: Dict[str, np.ndarray]) -> np.ndarray:
        """Gradient of target field for character.

        Args:
            char_name: Name of character
            positions: Current positions of all characters

        Returns:
            Gradient vector in R^n
        """
        target = self.compute_target(char_name, positions)
        pos_k = positions[char_name]
        return -2 * (pos_k - target)

    def score_direction(self, e: np.ndarray, char: CharacterSpec,
                       positions: Dict[str, np.ndarray], J: np.ndarray) -> float:
        """Score a candidate direction for a character.

        S(e) = alpha <e,grad_phi> + (gamma-beta)<e,Je> - eta|<e,n>|
        """
        score = 0.0

        # Love term
        if char.alpha != 0:
            grad_phi = self.target_gradient(char.name, positions)
            score += char.alpha * np.dot(e, grad_phi)

        # Money + Respect term
        jacobian_coeff = char.gamma - char.beta
        if jacobian_coeff != 0:
            Je = J @ e
            score += jacobian_coeff * np.dot(e, Je)

        # Violence term (state-dependent boundary)
        n_boundary = self.compute_boundary_normal(positions)
        boundary_component = np.dot(e, n_boundary)
        score -= char.eta * np.abs(boundary_component)

        return score

    def find_best_direction(self, char: CharacterSpec, positions: Dict[str, np.ndarray],
                           J: np.ndarray) -> np.ndarray:
        """Find direction that maximizes score for character."""
        jacobian_coeff = char.gamma - char.beta

        if jacobian_coeff == 0:
            # Simplified case: S = alpha<e,grad_phi> - eta|<e,n>|
            return self._optimize_simple(char, positions)
        else:
            # General case with Jacobian term
            return self._optimize_with_jacobian(char, positions, J)

    def _optimize_simple(self, char: CharacterSpec, positions: Dict[str, np.ndarray]) -> np.ndarray:
        """Optimize S = alpha<e,grad_phi> - eta|<e,n>| over unit sphere."""
        if char.alpha == 0:
            # No gradient term, return random direction
            e = np.random.randn(self.n)
            return e / np.linalg.norm(e)

        grad_phi = self.target_gradient(char.name, positions)
        n = self.compute_boundary_normal(positions)

        if char.eta > 0:
            # Avoid boundary: project grad_phi orthogonal to n
            grad_phi_perp = grad_phi - np.dot(grad_phi, n) * n
            e = char.alpha * grad_phi_perp
        elif char.eta < 0:
            # Favor boundary: add n component
            e = char.alpha * grad_phi - char.eta * n
        else:
            # No boundary preference
            e = char.alpha * grad_phi

        # Normalize
        norm = np.linalg.norm(e)
        if norm < 1e-10:
            e = np.random.randn(self.n)
            norm = np.linalg.norm(e)

        return e / norm

    def _optimize_with_jacobian(self, char: CharacterSpec, positions: Dict[str, np.ndarray],
                                J: np.ndarray) -> np.ndarray:
        """Optimize S with Jacobian term via gradient ascent on sphere."""
        # Initialize with simple direction
        e = self._optimize_simple(char, positions)

        # Extract relevant block of Jacobian for this character
        # J is (n_chars * n_dims) x (n_chars * n_dims)
        # We need the (n_dims x n_dims) block for this character
        char_idx = self.char_names.index(char.name)
        i_start = char_idx * self.n
        i_end = (char_idx + 1) * self.n
        J_block = J[i_start:i_end, i_start:i_end]

        # Gradient ascent on sphere
        lr = 0.1
        for _ in range(20):
            # Compute gradient of score
            grad = np.zeros(self.n)

            # Love term
            if char.alpha != 0:
                grad += char.alpha * self.target_gradient(char.name, positions)

            # Jacobian term (use extracted block)
            jacobian_coeff = char.gamma - char.beta
            if jacobian_coeff != 0:
                grad += jacobian_coeff * (J_block + J_block.T) @ e

            # Violence term (state-dependent boundary)
            if char.eta != 0:
                n_boundary = self.compute_boundary_normal(positions)
                boundary_comp = np.dot(e, n_boundary)
                if abs(boundary_comp) > 1e-8:
                    grad -= char.eta * np.sign(boundary_comp) * n_boundary

            # Project gradient onto tangent space
            grad_tangent = grad - np.dot(grad, e) * e

            # Update
            e_new = e + lr * grad_tangent
            e = e_new / np.linalg.norm(e_new)

        return e

    def compute_flow(self, directions: Dict[str, np.ndarray],
                    weights: Dict[str, float]) -> np.ndarray:
        """Compute global flow F = sum a_k e_k.

        This is used for Jacobian computation and inflection detection.

        Args:
            directions: Direction for each character
            weights: Weight for each character

        Returns:
            Global flow vector (concatenated from all character contributions)
        """
        # F as concatenation of per-character flows
        F_parts = []
        for char_name in self.char_names:
            a_k = weights[char_name]
            e_k = directions[char_name]
            F_parts.append(a_k * e_k)

        return np.concatenate(F_parts)

    def compute_jacobian(self, positions: Dict[str, np.ndarray],
                        directions: Dict[str, np.ndarray],
                        weights: Dict[str, float]) -> np.ndarray:
        """Compute Jacobian J = grad F(X) via finite differences.

        X = concat(pos_all)
        F(X) = concat(a_k e_k for all k)

        Args:
            positions: Current positions
            directions: Current directions
            weights: Current weights

        Returns:
            Jacobian matrix
        """
        # Total dimensionality
        n_total = self.n * len(self.char_names)
        J = np.zeros((n_total, n_total))

        # Current flow
        F0 = self.compute_flow(directions, weights)

        # Perturb each dimension of concatenated state
        for i in range(n_total):
            # Figure out which character and which dimension
            char_idx = i // self.n
            dim_idx = i % self.n
            char_name = self.char_names[char_idx]

            # Perturb position
            positions_perturb = {k: v.copy() for k, v in positions.items()}
            positions_perturb[char_name][dim_idx] += self.eps

            # Recompute directions with perturbed positions
            # (targets depend on positions, so directions change)
            directions_perturb = {}
            for cn in self.char_names:
                char = self.characters[cn]
                # Use zero Jacobian for this step (approximate)
                J_approx = np.zeros((n_total, n_total))
                directions_perturb[cn] = self.find_best_direction(char, positions_perturb, J_approx)

            # Compute perturbed flow
            F_perturb = self.compute_flow(directions_perturb, weights)

            # Finite difference
            J[:, i] = (F_perturb - F0) / self.eps

        return J


class InflectionDetector:
    """Detects inflection points based on geometric conditions."""

    def __init__(self, engine: DynamicsEngine, inflection_specs: List[InflectionSpec],
                 time_per_episode: float, hidden_dims: List[int], visible_dims: List[int]):
        """Initialize inflection detector.

        Args:
            engine: Dynamics engine for computing observables
            inflection_specs: List of inflection point specifications
            time_per_episode: Time units per episode
            hidden_dims: Indices of hidden dimensions in each position
            visible_dims: Indices of visible dimensions in each position
        """
        self.engine = engine
        self.specs = {spec.id: spec for spec in inflection_specs}
        self.time_per_episode = time_per_episode
        self.hidden_dims = hidden_dims
        self.visible_dims = visible_dims

    def check_inflection(self, state: SimulationState, prev_state: Optional[SimulationState],
                        inflection_id: int) -> Tuple[bool, Dict]:
        """Check if an inflection point is detected.

        Args:
            state: Current simulation state
            prev_state: Previous simulation state (or None)
            inflection_id: ID of inflection to check

        Returns:
            (detected, details) tuple
        """
        spec = self.specs[inflection_id]
        details = {}
        all_satisfied = True

        for condition in spec.conditions:
            satisfied, value = self._check_condition(condition, state, prev_state)
            details[condition.type] = {
                'satisfied': satisfied,
                'value': value,
                'threshold': condition.threshold
            }
            all_satisfied = all_satisfied and satisfied

        return all_satisfied, details

    def _check_condition(self, condition: InflectionCondition, state: SimulationState,
                        prev_state: Optional[SimulationState]) -> Tuple[bool, float]:
        """Check a single condition."""
        if condition.type == "low_alignment":
            char_name = condition.character
            grad_phi = self.engine.target_gradient(char_name, state.positions)

            # F is global flow (concatenated), extract this character's part
            char_idx = self.engine.char_names.index(char_name)
            F_char = state.F[char_idx * self.engine.n : (char_idx + 1) * self.engine.n]

            F_norm = np.linalg.norm(F_char)
            grad_norm = np.linalg.norm(grad_phi)

            if F_norm < 1e-10 or grad_norm < 1e-10:
                alignment = 0.0
            else:
                alignment = np.abs(np.dot(F_char, grad_phi)) / (F_norm * grad_norm)

            return alignment < condition.threshold, alignment

        elif condition.type == "positive_progress":
            if prev_state is None:
                return False, 0.0

            char_name = condition.character

            # Fix: compare both positions against SAME target (current one)
            target_current = self.engine.compute_target(char_name, state.positions)
            pos_current = state.positions[char_name]
            pos_previous = prev_state.positions[char_name]

            # Evaluate φ = -||pos - target||^2 for both positions
            phi_current = -np.linalg.norm(pos_current - target_current)**2
            phi_previous = -np.linalg.norm(pos_previous - target_current)**2

            delta_phi = phi_current - phi_previous

            return delta_phi > 0, delta_phi

        elif condition.type == "domain_coupling_ratio":
            # Extract submatrices based on hidden/visible dimensions
            # J is (n_chars * n_dims) x (n_chars * n_dims)
            # We treat hidden/visible as global dimensions within each character

            n_chars = len(self.engine.char_names)
            n_dims = self.engine.n

            # Extract rows/cols corresponding to visible/hidden dimensions
            # For simplicity, treat first half of state as hidden, second half as visible
            n_total = n_chars * n_dims
            n_hidden_total = len(self.hidden_dims) * n_chars
            n_visible_total = len(self.visible_dims) * n_chars

            # Build index lists
            hidden_indices = []
            visible_indices = []
            for char_idx in range(n_chars):
                base = char_idx * n_dims
                for h_dim in self.hidden_dims:
                    hidden_indices.append(base + h_dim)
                for v_dim in self.visible_dims:
                    visible_indices.append(base + v_dim)

            # Extract submatrices
            J_cross = state.J[np.ix_(visible_indices, hidden_indices)]
            J_self = state.J[np.ix_(visible_indices, visible_indices)]

            cross_norm = np.linalg.norm(J_cross, 'fro')
            self_norm = np.linalg.norm(J_self, 'fro')

            if self_norm < 1e-10:
                coupling = 0.0
            else:
                coupling = cross_norm / self_norm

            return coupling < condition.threshold, coupling

        else:
            raise ValueError(f"Unknown condition type: {condition.type}")
