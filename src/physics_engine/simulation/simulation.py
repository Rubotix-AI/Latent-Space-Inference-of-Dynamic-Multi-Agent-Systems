from agent import Agent
from config import SimulationConfig
from world import World


class Simulation:
    """
    Generic physics simulator.

    The simulator knows nothing about the interaction law.
    """

    def __init__(
        self,
        agents: list[Agent],
        world: World,
        interaction,
        config: SimulationConfig,
    ):
        self.agents = agents
        self.world = world
        self.interaction = interaction
        self.config = config

        self.timestep = 0

    def step(self):
        """
        Advance the simulation by one timestep.
        """
        for agent in self.agents:
            agent.reset_acceleration()

        # Interaction law computes all accelerations
        self.interaction.step(self.agents, self.world)

        dt = self.config.dt

        # Integrate motion
        for agent in self.agents:
            agent.velocity += dt * agent.acceleration
            agent.position += dt * agent.velocity

            self.world.wrap(agent.position)

        self.timestep += 1

    def __iter__(self):
        """
        Used for dataset generation.
        """

        for agent in self.agents:
            yield (
                self.timestep,
                agent.id,
                *agent.position,
                *agent.velocity,
            )