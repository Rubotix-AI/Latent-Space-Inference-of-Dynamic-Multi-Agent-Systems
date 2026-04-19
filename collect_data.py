import numpy as np
np.set_printoptions(precision=2) # displays truncated floats for all numpy vectors

from config import SEED, NUM_OF_COORDS, NUM_OF_BOIDS, MAX_SEPERATION_VALUE, X_BOUND, Y_BOUND, DELTA_T
from sim import Agent, Simulation

rng = np.random.default_rng(seed=SEED)

def create_boids(num_boids: int):
    all_boids = []
    for i in range(num_boids):
        curr = Agent(
            id=i,
            pos=rng.uniform(-X_BOUND, X_BOUND, size=(NUM_OF_COORDS,)),
            vel=rng.uniform(-Y_BOUND, Y_BOUND, size=(NUM_OF_COORDS,)),
            wander_radius=rng.random()
        )
        all_boids.append(curr)
    
    return all_boids

boids = create_boids(5)
sim = Simulation(
    boids=boids,
    boid_count=NUM_OF_BOIDS,
    seperation=0.5,
    alignment=0.5,
    cohesion=0.5,
    wander=0.5,
    sep_radius=2.,
    coh_radius=2.,
    align_radius=2.,
    wander_radius=1.
)

for _ in range(10):
    print(sim)
    sim.update()