"""
Data loader module.

Loads case study data from JSON files and constructs character and inflection specifications.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class TargetRelation:
    """Specification for relational target computation."""
    type: str  # "difference", "center"
    operands: List[str]  # Character names
    description: str


@dataclass
class CharacterSpec:
    """Character specification from data."""
    name: str
    alpha: float
    beta: float
    gamma: float
    eta: float
    target_relation: TargetRelation | None
    initial_weight: float
    growth_rate: float


@dataclass
class InflectionCondition:
    """Single condition for inflection detection."""
    type: str
    formula: str
    threshold: float
    character: str = None


@dataclass
class InflectionSpec:
    """Inflection point specification."""
    id: int
    name: str
    description: str
    conditions: List[InflectionCondition]
    expected_episodes: List[int]


@dataclass
class CaseStudyData:
    """Complete case study data."""
    id: str
    name: str
    description: str
    n_dims: int
    characters: List[CharacterSpec]
    inflections: List[InflectionSpec]
    dt: float
    time_per_episode: float
    initial_state_scale: float
    jacobian_epsilon: float
    boundary_normal: np.ndarray
    hidden_dims: List[int]
    visible_dims: List[int]


def load_case_study(case_study_id: str, data_dir: str = "data") -> CaseStudyData:
    """Load case study from JSON file.

    Args:
        case_study_id: ID of the case study (e.g., 'blacklist_s1')
        data_dir: Directory containing JSON data files

    Returns:
        CaseStudyData object with all specifications
    """
    # Load JSON file
    json_path = Path(data_dir) / f"{case_study_id}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Case study file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Parse state space
    state_space = data["state_space"]
    n_dims = state_space["n_dims"]

    # Parse characters
    characters = []
    for char_data in data["characters"]:
        # Parse target relation (may be null for alpha=0 characters)
        rel_data = char_data["target_relation"]
        if rel_data is None:
            target_rel = None
        else:
            target_rel = TargetRelation(
                type=rel_data["type"],
                operands=rel_data["operands"],
                description=rel_data["description"]
            )

        char = CharacterSpec(
            name=char_data["name"],
            alpha=char_data["alpha"],
            beta=char_data["beta"],
            gamma=char_data["gamma"],
            eta=char_data["eta"],
            target_relation=target_rel,
            initial_weight=char_data["initial_weight"],
            growth_rate=char_data["weight_growth_rate"]
        )
        characters.append(char)

    # Parse inflection points
    inflections = []
    for inf_data in data["inflection_points"]:
        conditions = []
        for cond_data in inf_data["conditions"]:
            cond = InflectionCondition(
                type=cond_data["type"],
                formula=cond_data["formula"],
                threshold=cond_data.get("threshold", 0.0),
                character=cond_data.get("character")
            )
            conditions.append(cond)

        inflection = InflectionSpec(
            id=inf_data["id"],
            name=inf_data["name"],
            description=inf_data["description"],
            conditions=conditions,
            expected_episodes=inf_data["expected_episodes"]
        )
        inflections.append(inflection)

    # Parse simulation parameters
    sim_params = data["simulation_params"]

    # Construct CaseStudyData
    case_study = CaseStudyData(
        id=data["case_study_id"],
        name=data["name"],
        description=data["description"],
        n_dims=n_dims,
        characters=characters,
        inflections=inflections,
        dt=sim_params["dt"],
        time_per_episode=sim_params["time_per_episode"],
        initial_state_scale=sim_params["initial_state_scale"],
        jacobian_epsilon=sim_params["jacobian_epsilon"],
        boundary_normal=np.array(sim_params["boundary_normal"]),
        hidden_dims=sim_params["hidden_dims"],
        visible_dims=sim_params["visible_dims"]
    )

    return case_study


def get_available_case_studies(data_dir: str = "data") -> List[str]:
    """Get list of available case study IDs.

    Args:
        data_dir: Directory containing JSON data files

    Returns:
        List of case study IDs
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    json_files = list(data_path.glob("*.json"))
    return [f.stem for f in json_files]


def compute_relational_target(relation: TargetRelation, positions: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute target from relational specification.

    Args:
        relation: TargetRelation specification
        positions: Dictionary of character positions

    Returns:
        Target vector
    """
    if relation.type == "difference":
        # target = pos_A - pos_B
        char_a, char_b = relation.operands
        return positions[char_a] - positions[char_b]

    elif relation.type == "center":
        # target = average(all positions) - pos_self
        # Note: pos_self will be subtracted by caller
        if relation.operands[0] == "all":
            all_pos = np.array(list(positions.values()))
            return np.mean(all_pos, axis=0)
        else:
            raise ValueError(f"Unknown center operand: {relation.operands}")

    else:
        raise ValueError(f"Unknown target relation type: {relation.type}")
