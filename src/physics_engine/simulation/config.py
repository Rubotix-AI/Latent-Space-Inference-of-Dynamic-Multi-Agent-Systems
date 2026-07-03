from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Configuration shared by every simulation.
    """

    dt: float = 0.05

    total_timesteps: int = 10000
    num_agents: int = 50

    seed: int = 42

    world_width: float = 10.0
    world_height: float = 10.0

    periodic_boundary: bool = True