from dataclasses import dataclass


@dataclass
class BoidsConfig:
    """
    Parameters governing the classical Reynolds Boids model.
    """

    separation_weight: float = 0.15
    alignment_weight: float = 0.20
    cohesion_weight: float = 0.65
    wander_weight: float = 0.2

    separation_radius: float = 1.0
    alignment_radius: float = 1.0
    cohesion_radius: float = 0.5

    wander_radius: float = 1.0
    wander_noise: float = 0.05
    max_separation_strength: float = 0.25