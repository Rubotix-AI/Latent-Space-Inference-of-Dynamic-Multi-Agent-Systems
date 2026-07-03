import numpy as np


def separation(diff: np.ndarray, minimum_norm: float) -> np.ndarray:
    norm = max(np.linalg.norm(diff), minimum_norm)
    return -diff / norm


def alignment(velocity: np.ndarray) -> np.ndarray:
    return velocity


def cohesion(position: np.ndarray) -> np.ndarray:
    return position