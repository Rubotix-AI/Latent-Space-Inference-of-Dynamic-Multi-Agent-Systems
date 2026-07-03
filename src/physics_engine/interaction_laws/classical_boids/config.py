from dataclasses import dataclass


@dataclass
class BoidsConfig:
    """
    Parameters governing the classical Reynolds Boids model.
    """

    separation_weight: float = 0.15
    alignment_weight: float = 0.15
    cohesion_weight: float = 0.40
    wander_weight: float = 0.20

    separation_radius: float = 1.0
    alignment_radius: float = 1.0
    cohesion_radius: float = 0.5

    wander_radius: float = 1.5
    wander_noise: float = 0.05
    max_separation_strength: float = 0.5