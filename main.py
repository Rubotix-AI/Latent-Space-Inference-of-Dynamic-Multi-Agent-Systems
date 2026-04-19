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

    def calculate_vecs(self, boid: Agent, neighbours: list[Agent]):
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
        return "\n".join(str(boid) for boid in self.boids)


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