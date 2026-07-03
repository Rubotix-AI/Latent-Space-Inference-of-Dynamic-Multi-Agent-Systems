from dataclasses import dataclass


@dataclass
class SpringNetworkConfig:
    """
    Parameters governing the spring network interaction model.
    """

    spring_constant: float = 0.4
    rest_length: float = 2.0

    damping: float = 0.25

    wander_weight: float = 0.3

    wander_radius: float = 1.5
    wander_noise: float = 0.05