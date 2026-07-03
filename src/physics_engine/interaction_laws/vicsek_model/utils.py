import numpy as np


def average_heading(velocities: list[np.ndarray]) -> np.ndarray:
    """
    Computes the average heading of a set of velocity vectors.
    """

    if len(velocities) == 0:
        return np.zeros(2)

    heading = np.sum(velocities, axis=0)

    norm = np.linalg.norm(heading)

    if norm < 1e-8:
        return np.zeros(2)

    return heading / norm


def rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates a vector by the given angle.
    """

    c = np.cos(angle)
    s = np.sin(angle)

    rotation = np.array([
        [c, -s],
        [s,  c],
    ])

    return rotation @ vector