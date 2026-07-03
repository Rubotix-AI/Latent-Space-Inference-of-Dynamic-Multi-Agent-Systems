from dataclasses import dataclass


@dataclass
class VicsekConfig:
    """
    Parameters governing the Vicsek interaction model.
    """

    interaction_radius: float = 1.0

    preferred_speed: float = 2.0

    response_rate: float = 1.0

    angular_noise: float = 0.2