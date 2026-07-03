from dataclasses import dataclass


@dataclass
class PredatorPreyConfig:
    """
    Parameters governing the predator-prey interaction model.
    """

    predator_chase_weight: float = 0.75

    prey_escape_weight: float = 0.80
    prey_separation_weight: float = 0.15
    prey_alignment_weight: float = 0.15
    prey_cohesion_weight: float = 0.35

    wander_weight: float = 0.20

    predator_detection_radius: float = 6.0

    prey_detection_radius: float = 4.0
    separation_radius: float = 1.0
    alignment_radius: float = 1.0
    cohesion_radius: float = 0.5

    wander_radius: float = 1.5
    wander_noise: float = 0.05

    max_separation_strength: float = 0.5