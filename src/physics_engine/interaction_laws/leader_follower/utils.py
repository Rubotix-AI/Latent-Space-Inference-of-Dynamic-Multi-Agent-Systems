import numpy as np


def separation(diff: np.ndarray, minimum_norm: float) -> np.ndarray:
    norm = max(np.linalg.norm(diff), minimum_norm)
    return -diff / norm


def leader_attraction(diff: np.ndarray) -> np.ndarray:
    """
    Unit vector pointing toward the leader.
    """
    norm = np.linalg.norm(diff)

    if norm < 1e-8:
        return np.zeros_like(diff)

    return diff / norm


def leader_alignment(leader_velocity: np.ndarray) -> np.ndarray:
    return leader_velocity