import numpy as np

np.set_printoptions(precision=2) # displays truncated floats for all numpy vectors

SEED = 42
DELTA_T = 1
MAX_SEPERATION_VALUE = 1
NUM_OF_BOIDS = 5
NUM_OF_COORDS = 2
X_BOUND = 5
Y_BOUND = 5

rng = np.random.default_rng(seed=SEED)

class Agent:
    def __init__(self, id: int, pos: np.ndarray, vel: np.ndarray, wander_radius: np.float32):
        self.id = id
        self.position = pos
        self.velocity = vel
        self.wander = wander_radius * ( rng.random((NUM_OF_COORDS,)) - 0.5 * np.ones((NUM_OF_COORDS,)) )
    
    def __str__(self):
        return f"{self.id} --> Position: {self.position} Velocity: {self.velocity} Wander: {self.wander}"

def seperation(diff: np.ndarray):
    return diff / max(np.linalg.norm(diff), MAX_SEPERATION_VALUE)

def alignment(boid: Agent):
    return boid.velocity

def cohesion(boid: Agent):
    return boid.position

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

    def calculate_vecs(self, boid: Agent, neighbours: list[Agent]):
        sep_vec = np.zeros_like(boid.velocity)
        align_vec = np.zeros_like(boid.velocity)
        coh_vec = np.zeros_like(boid.position) # same size as velocity so it doesn't matter

        count_a, count_c = 0, 0
        for neighbour in neighbours:
            diff = neighbour.position - boid.position
            dist = np.linalg.norm(diff)
            if dist < self.r_s:
                sep_vec -= seperation(diff)
            if dist < self.r_a:
                align_vec += alignment(boid)
                count_a += 1
            if dist < self.r_c:
                coh_vec += cohesion(boid)
                count_c += 1
        
        try:
            align_vec = ( align_vec / count_a ) - boid.velocity
            coh_vec = ( coh_vec / count_c ) - boid.position
        except ZeroDivisionError as e:
            print(f"Zero Division Error with Count: count_a, count_c --> {count_a, count_c}")

        return sep_vec, align_vec, coh_vec
    
    def __str__(self):
        return "\n".join(str(boid) for boid in self.boids)


def create_boids(num_boids: int):
    all_boids = []
    for i in range(num_boids):
        curr = Agent(
            id=i,
            pos=np.random.uniform(-X_BOUND, X_BOUND, size=(NUM_OF_COORDS,)),
            vel=np.random.uniform(-Y_BOUND, Y_BOUND, size=(NUM_OF_COORDS,)),
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
    sep_radius=1.,
    coh_radius=1.,
    align_radius=1.,
    wander_radius=1.
)

print(sim)

sim.update()

print("\n", sim)