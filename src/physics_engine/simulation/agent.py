from dataclasses import dataclass, field
import numpy as np

@dataclass
class Agent:
    """
    Generic agent used by every interaction law.
    """

    id: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def reset_acceleration(self):
        self.acceleration.fill(0.0)

    @property
    def heading(self):
        speed = np.linalg.norm(self.velocity)

        if speed < 1e-8:
            return np.zeros_like(self.velocity)

        return self.velocity / speed