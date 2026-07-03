from dataclasses import dataclass


@dataclass
class SpringNetworkConfig:
    """
    Parameters governing the spring network interaction model.
    """

    spring_constant: float = 1.0
    rest_length: float = 1.0

    damping: float = 0.25

    wander_weight: float = 0.10

    wander_radius: float = 1.5
    wander_noise: float = 0.05