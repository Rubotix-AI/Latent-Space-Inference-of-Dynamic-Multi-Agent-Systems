import numpy as np

SEED = 42

rng = np.random.default_rng(seed=SEED)

class Agent:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, wander_radius: np.float32):
        self.position = pos
        self.velocity = vel
        self.wander = rng.uniform(-wander_radius, wander_radius)

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


    def calculate_vecs(self, boid:Agent, neighbours: list[Agent]):
        sep_vec = np.zeros_like(boid.velocity)
        align_vec = np.zeros_like(boid.velocity)
        coh_vec = np.zeros_like(boid.position) # same size as velocity so it doesn't matter

        for neighbour in neighbours:
            diff = neighbour.position - boid.position
            dist = np.linalg.norm(diff)
            sep_vec = self.seperation(boid, neighbours, dist)
            align_vec = self.alignment(boid, neighbours, dist)
            coh_vec = self.cohesion(boid, neighbours, dist)

        return sep_vec, align_vec, coh_vec

    def seperation(self, boid: Agent, neighbours: list[Agent], dist: np.float32):
        sep_vec = np.zeros_like(boid.velocity)

        for neighbour_boid in neighbours:
            if dist < self.r_s:
                sep_vec -= diff / max(np.linalg.norm(diff), 1)
        
        return sep_vec
    
    def alignment(self, boid: Agent, neighbours: list[Agent], dist: np.float32):
        align_vec = np.zeros_like(boid.velocity)
        count = 0
        for neighbour_boid in neighbours:
            dist = np.linalg.norm(neighbour_boid.position - boid.position)
            if dist < self.r_a:
                align_vec += neighbour_boid.velocity
                count += 1

        try:
            return (align_vec / count) - boid.velocity
        except ZeroDivisionError as e:
            print(f"Alignment Count Null: {e}")

    def cohesion(self, boid: Agent, neighbours: list[Agent], dist: np.float32):
        coh_vec = np.zeros_like(boid.velocity)
        count = 0
        for neighbour_boid in neighbours:
            dist = np.linalg.norm(neighbour_boid.position - boid.position)
            if dist < self.r_c:
                coh_vec += neighbour_boid.position
                count += 1
        
        try:
            return (coh_vec / count) - boid.position
        except ZeroDivisionError as e:
            print(f"Cohesion Count Null : {e}")

    def update(self):
