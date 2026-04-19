import numpy as np

from simulation.boid_config import MAX_SEPERATION_VALUE, X_BOUND, Y_BOUND
from simulation.boid_models import Agent

def seperation(diff: np.ndarray) -> np.ndarray:
    return diff / max(np.linalg.norm(diff), MAX_SEPERATION_VALUE)

def alignment(boid: Agent) -> np.ndarray:
    return boid.velocity

def cohesion(boid: Agent) -> np.ndarray:
    return boid.position

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