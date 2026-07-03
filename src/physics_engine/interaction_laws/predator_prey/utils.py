import numpy as np


def separation(diff: np.ndarray, minimum_norm: float) -> np.ndarray:
    norm = max(np.linalg.norm(diff), minimum_norm)
    return -diff / norm


def alignment(velocity: np.ndarray) -> np.ndarray:
    return velocity


def cohesion(position: np.ndarray) -> np.ndarray:
    return position


def chase(diff: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(diff)

    if norm < 1e-8:
        return np.zeros_like(diff)

    return diff / norm


def escape(diff: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(diff)

    if norm < 1e-8:
        return np.zeros_like(diff)

    return -diff / norm