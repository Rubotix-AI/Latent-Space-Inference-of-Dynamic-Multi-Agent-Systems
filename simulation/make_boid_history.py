import numpy as np
import pandas as pd

np.set_printoptions(precision=2) # displays truncated floats for all numpy vectors

from boid_config import SEED, TOTAL_TIMESTEPS, NUM_OF_COORDS, NUM_OF_BOIDS,X_BOUND, Y_BOUND
from boid_config import SEPERATION_RADIUS, SEPERATION_WEIGHT, ALIGNEMENT_RADIUS, ALIGNMENT_WEIGHT, COHESION_RADIUS, COHESION_WEIGHT, WANDER_RADIUS, WANDER_WEIGHT
from boid_models import Agent, Simulation

rng = np.random.default_rng(seed=SEED)

def create_boids(num_boids: int) -> list[Agent]:
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

boids = create_boids(NUM_OF_BOIDS)
sim = Simulation(
    boids=boids,
    boid_count=NUM_OF_BOIDS,
    seperation=SEPERATION_WEIGHT,
    alignment=ALIGNMENT_WEIGHT,
    cohesion=COHESION_WEIGHT,
    wander=WANDER_WEIGHT,
    sep_radius=SEPERATION_RADIUS,
    coh_radius=COHESION_RADIUS,
    align_radius=ALIGNEMENT_RADIUS,
    wander_radius=WANDER_RADIUS
)

columns = ["time", "id", "x", "y", "vx", "vy"]
rows = []

for t in range(TOTAL_TIMESTEPS):
    rows.extend(sim)
    sim.update()

df = pd.DataFrame(data=rows, columns=columns)

df.to_csv('data/boid_history.csv', index=False)