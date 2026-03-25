import numpy as np

from config import timesteps, neighborhood_radius, max_speed, width, height, positions, velocities

history = []

def limit_speed(v, max_speed):
    speed = np.linalg.norm(v)
    return (v / speed) * max_speed if speed > max_speed else v

def update_boids(pos, vel):
    new_pos, new_vel = pos.copy(), vel.copy()
    for i in range(len(pos)):
        neighbors = [j for j in range(len(pos)) if i != j and np.linalg.norm(pos[i] - pos[j]) < neighborhood_radius]
        if neighbors:
            center = np.mean(pos[neighbors], axis=0)
            separation = np.sum(pos[i] - pos[neighbors], axis=0)
            avg_vel = np.mean(vel[neighbors], axis=0)
            new_vel[i] += (center - pos[i]) * 0.01 + separation * 0.05 + (avg_vel - vel[i]) * 0.05
        new_vel[i] = limit_speed(new_vel[i], max_speed)
        new_pos[i] = np.mod(new_pos[i] + new_vel[i], [width, height])
    return new_pos, new_vel

for t in range(timesteps):
    state = np.hstack((positions, velocities))
    history.append(state)
    positions, velocities = update_boids(positions, velocities)

history = np.array(history)