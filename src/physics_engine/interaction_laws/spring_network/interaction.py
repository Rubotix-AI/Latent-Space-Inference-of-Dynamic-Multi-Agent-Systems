import numpy as np

from physics_engine.simulation.agent import Agent
from physics_engine.simulation.interaction import InteractionLaw
from physics_engine.simulation.spring import Spring
from physics_engine.simulation.world import World

from physics_engine.interaction_laws.spring_network.config import SpringNetworkConfig
from physics_engine.interaction_laws.spring_network.utils import (
    spring_force,
    damping_force,
)


class SpringNetworkInteraction(InteractionLaw):

    def __init__(
        self,
        springs: list[Spring],
        config: SpringNetworkConfig,
        rng: np.random.Generator,
    ):
        super().__init__(
            rng=rng,
            wander_radius=config.wander_radius,
            wander_noise=config.wander_noise,
        )

        self.springs = springs
        self.config = config

    def step(
        self,
        agents: list[Agent],
        world: World,
    ):

        agent_lookup = {
            agent.id: agent
            for agent in agents
        }

        for spring in self.springs:

            agent_a = agent_lookup[spring.agent_a]
            agent_b = agent_lookup[spring.agent_b]

            diff = world.shortest_displacement(
                agent_a.position,
                agent_b.position,
            )

            spring_vec = spring_force(
                diff,
                spring.rest_length,
                spring.stiffness,
            )

            damping_vec = damping_force(
                agent_b.velocity - agent_a.velocity,
                self.config.damping,
            )

            force = spring_vec + damping_vec

            agent_a.acceleration += force
            agent_b.acceleration -= force

        for agent in agents:

            wander = self._update_wander(agent)

            agent.acceleration += (
                self.config.wander_weight
                * wander
            )