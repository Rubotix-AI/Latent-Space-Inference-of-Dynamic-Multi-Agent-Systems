import numpy as np

from simulation.boid_config import SEED, NUM_OF_COORDS, DELTA_T
from simulation.sim_utils import cohesion, alignment, seperation, wrap

rng = np.random.default_rng(seed=SEED)

class Agent:
    def __init__(self, id: int, pos: np.ndarray, vel: np.ndarray, wander_radius: np.float32):
        self.id = id
        self.position = pos
        self.velocity = vel
        self.wander = wander_radius * ( rng.random((NUM_OF_COORDS,)) - 0.5 * np.ones((NUM_OF_COORDS,)) )
    
    def __str__(self):
        return f"""{self.id}
    Position: {self.position}
    Velocity: {self.velocity}
    Wander: {self.wander}"""


class Simulation:
    def __init__(self, 
            boids: list[Agent], 
            boid_count: np.uint32, 
            seperation: np.float32, 
            cohesion: np.float32, 
            alignment: np.float32,
            wander: np.float32, 
            sep_radius: np.float32,
            coh_radius: np.float32,
            align_radius: np.float32,
            wander_radius: np.float32
        ):
        self.boids = boids
        self.boid_count = boid_count
        self.w_s = seperation
        self.w_c = cohesion
        self.w_a = alignment
        self.w_w = wander

        self.r_s = sep_radius
        self.r_c = coh_radius
        self.r_a = align_radius
        self.r_w = wander_radius

        self.average_speed = ...
        self.flocking_radius = ...
        self.turn_rate = ...
        self.dispersion = ...

    def update(self):
        all_boids = self.boids

        for boid in all_boids:
            sep, alg, coh = self.calculate_vecs(boid, all_boids)
            
            a_new = self.w_s * sep + \
                self.w_a * alg + \
                self.w_c * coh + \
                self.w_w * boid.wander
            
            boid.velocity += DELTA_T * a_new
            boid.position += DELTA_T * boid.velocity
            wrap(boid.position)

    def calculate_vecs(self, boid: Agent, neighbours: list[Agent]) -> list[np.ndarray]:
        sep_vec = np.zeros_like(boid.velocity)
        align_vec = np.zeros_like(boid.velocity)
        coh_vec = np.zeros_like(boid.position) # same size as velocity so it doesn't matter

        count_a, count_c = 0, 0
        for neighbour in neighbours:
            if neighbour == boid: # skip the boid itself
                continue
            diff = neighbour.position - boid.position
            dist = np.linalg.norm(diff)
            if dist < self.r_s:
                sep_vec -= seperation(diff)
            if dist < self.r_a:
                align_vec += alignment(neighbour)
                count_a += 1
            if dist < self.r_c:
                coh_vec += cohesion(neighbour)
                count_c += 1
        align_vec = ( align_vec / count_a ) - boid.velocity if count_a else align_vec
        coh_vec = ( coh_vec / count_c ) - boid.position if count_c else coh_vec

        return sep_vec, align_vec, coh_vec
    
    def __str__(self):
        return "\n".join(str(boid) for boid in self.boids) + "\n-----------------------------------"
