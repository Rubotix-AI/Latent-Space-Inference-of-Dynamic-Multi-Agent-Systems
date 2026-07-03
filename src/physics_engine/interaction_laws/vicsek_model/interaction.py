import numpy as np

from physics_engine.simulation.agent import Agent
from physics_engine.simulation.interaction import InteractionLaw
from physics_engine.simulation.world import World

from physics_engine.interaction_laws.vicsek_model.config import VicsekConfig
from physics_engine.interaction_laws.vicsek_model.utils import (
    average_heading,
    rotate,
)


class VicsekInteraction(InteractionLaw):

    def __init__(
        self,
        config: VicsekConfig,
        rng: np.random.Generator,
    ):
        super().__init__(
            rng=rng,
            wander_radius=0.0,
            wander_noise=0.0,
        )

        self.config = config

    def step(
        self,
        agents: list[Agent],
        world: World,
    ):

        for agent in agents:

            neighbour_velocities = []

            for neighbour in agents:

                if neighbour is agent:
                    continue

                diff = world.shortest_displacement(
                    agent.position,
                    neighbour.position,
                )

                dist = np.linalg.norm(diff)

                if dist < self.config.interaction_radius:
                    neighbour_velocities.append(
                        neighbour.velocity
                    )

            if neighbour_velocities:

                heading = average_heading(
                    neighbour_velocities
                )

            else:

                speed = np.linalg.norm(agent.velocity)

                if speed > 1e-8:
                    heading = agent.velocity / speed
                else:
                    heading = np.array([1.0, 0.0])

            angle = self.rng.uniform(
                -self.config.angular_noise,
                self.config.angular_noise,
            )

            heading = rotate(
                heading,
                angle,
            )

            desired_velocity = (
                self.config.preferred_speed
                * heading
            )

            agent.acceleration += (
                self.config.response_rate
                * (
                    desired_velocity
                    - agent.velocity
                )
            )