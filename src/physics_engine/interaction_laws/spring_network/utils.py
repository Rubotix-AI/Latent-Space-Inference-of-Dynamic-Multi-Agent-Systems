import numpy as np


def spring_force(
    diff: np.ndarray,
    rest_length: float,
    stiffness: float,
) -> np.ndarray:
    """
    Hooke's law.
    """

    distance = np.linalg.norm(diff)

    if distance < 1e-8:
        return np.zeros_like(diff)

    direction = diff / distance

    extension = distance - rest_length

    return stiffness * extension * direction


def damping_force(
    relative_velocity: np.ndarray,
    damping: float,
) -> np.ndarray:
    """
    Simple viscous damping.
    """

    return damping * relative_velocity