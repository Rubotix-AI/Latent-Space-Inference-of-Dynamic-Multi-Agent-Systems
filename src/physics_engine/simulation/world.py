import numpy as np

from physics_engine.simulation.config import SimulationConfig


class World:
    """
    Stores world geometry and boundary conditions.
    """

    def __init__(self, config: SimulationConfig):
        self.width = config.world_width
        self.height = config.world_height
        self.periodic = config.periodic_boundary

    def wrap(self, position: np.ndarray):
        """
        Apply periodic boundary conditions.
        """

        if not self.periodic:
            return

        half_w = self.width / 2
        half_h = self.height / 2

        position[0] = ((position[0] + half_w) % self.width) - half_w
        position[1] = ((position[1] + half_h) % self.height) - half_h

    def shortest_displacement(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ):
        """
        Returns the shortest vector from source -> target
        under periodic boundaries.
        """

        diff = target - source

        if not self.periodic:
            return diff

        half_w = self.width / 2
        half_h = self.height / 2

        if diff[0] > half_w:
            diff[0] -= self.width
        elif diff[0] < -half_w:
            diff[0] += self.width

        if diff[1] > half_h:
            diff[1] -= self.height
        elif diff[1] < -half_h:
            diff[1] += self.height

        return diff