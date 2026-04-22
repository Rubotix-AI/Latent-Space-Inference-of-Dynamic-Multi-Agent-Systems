import numpy as np

from boid_config import MAX_SEPERATION_VALUE, X_BOUND, Y_BOUND

def seperation(diff: np.ndarray) -> np.ndarray:
    return diff / max(np.linalg.norm(diff), MAX_SEPERATION_VALUE)

def alignment(velocity: np.ndarray) -> np.ndarray:
    return velocity

def cohesion(position: np.ndarray) -> np.ndarray:
    return position

def wrap(vec: np.ndarray) -> np.ndarray:
    # shift
    vec[0] += X_BOUND
    vec[1] += Y_BOUND
    # mod
    vec[0] %= (2 * X_BOUND + 1)
    vec[1] %= (2 * Y_BOUND + 1)
    # shift back 
    vec[0] -= X_BOUND
    vec[1] -= Y_BOUND

def dist_correction(pos: np.ndarray):
    x, y = pos
    if x > X_BOUND:
        x -= X_BOUND
    if x < -X_BOUND:
        x += X_BOUND

    if y > Y_BOUND:
        y -= Y_BOUND
    if y < -Y_BOUND:
        y += Y_BOUND

    return np.array([x, y])