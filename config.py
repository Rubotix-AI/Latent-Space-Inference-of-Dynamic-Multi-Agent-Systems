import numpy as np

num_boids = 20
timesteps = 200
width, height = 640, 480
max_speed = 4
neighborhood_radius = 75

positions = np.random.rand(num_boids, 2) * [width, height]
velocities = (np.random.rand(num_boids, 2) - 0.5) * max_speed