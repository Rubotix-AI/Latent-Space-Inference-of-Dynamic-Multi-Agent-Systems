import numpy as np
from config import SEED, NUM_OF_COORDS, NUM_OF_BOIDS, MAX_SEPERATION_VALUE, X_BOUND, Y_BOUND, DELTA_T
from sim import Agent

def seperation(diff: np.ndarray):
    return diff / max(np.linalg.norm(diff), MAX_SEPERATION_VALUE)

def alignment(boid: Agent):
    return boid.velocity

def cohesion(boid: Agent):
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