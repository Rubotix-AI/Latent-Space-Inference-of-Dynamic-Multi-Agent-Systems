import numpy as np

from physics_engine.simulation.agent import Agent
from physics_engine.simulation.interaction import InteractionLaw
from physics_engine.simulation.world import World

from physics_engine.interaction_laws.classical_boids.config import BoidsConfig
from physics_engine.interaction_laws.classical_boids.utils import alignment, cohesion, separation

class BoidsInteraction(InteractionLaw):

    def __init__(self, config: BoidsConfig, rng: np.random.Generator):
        super().__init__(
            rng=rng,
            wander_radius=config.wander_radius,
            wander_noise=config.wander_noise,
        )

        self.config = config

        self.r_sep = config.separation_radius
        self.r_align = config.alignment_radius
        self.r_coh = config.cohesion_radius

        self.w_sep = config.separation_weight
        self.w_align = config.alignment_weight
        self.w_coh = config.cohesion_weight
        self.w_wander = config.wander_weight

    def step(
        self,
        agents: list[Agent],
        world: World,
    ):

        for agent in agents:

            sep = np.zeros(2)
            align = np.zeros(2)
            coh = np.zeros(2)

            align_count = 0
            coh_count = 0

            for neighbour in agents:

                if neighbour is agent:
                    continue

                diff = world.shortest_displacement(
                    agent.position,
                    neighbour.position,
                )

                dist = np.linalg.norm(diff)

                if dist < self.r_sep:
                    sep += separation(
                        diff,
                        self.config.max_separation_strength,
                    )

                if dist < self.r_align:
                    align += alignment(neighbour.velocity)
                    align_count += 1

                if dist < self.r_coh:
                    coh += cohesion(neighbour.position)
                    coh_count += 1

            if align_count:
                align = align / align_count - agent.velocity

            if coh_count:
                coh = coh / coh_count - agent.position

            wander = self._update_wander(agent)

            agent.acceleration += (
                self.w_sep * sep
                + self.w_align * align
                + self.w_coh * coh
                + self.w_wander * wander
            )
