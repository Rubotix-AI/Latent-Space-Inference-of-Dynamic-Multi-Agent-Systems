import numpy as np

from physics_engine.simulation.agent import Agent, AgentType
from physics_engine.simulation.interaction import InteractionLaw
from physics_engine.simulation.world import World

from config import PredatorPreyConfig
from utils import (
    alignment,
    chase,
    cohesion,
    escape,
    separation,
)


class PredatorPreyInteraction(InteractionLaw):

    def __init__(
        self,
        config: PredatorPreyConfig,
        rng: np.random.Generator,
    ):
        super().__init__(
            rng=rng,
            wander_radius=config.wander_radius,
            wander_noise=config.wander_noise,
        )

        self.config = config

    def step(
        self,
        agents: list[Agent],
        world: World,
    ):

        predators = [
            a for a in agents
            if a.type == AgentType.PREDATOR
        ]

        prey = [
            a for a in agents
            if a.type == AgentType.PREY
        ]

        #############################
        # Predators
        #############################

        for predator in predators:

            chase_force = np.zeros(2)

            nearest_dist = np.inf

            for victim in prey:

                diff = world.shortest_displacement(
                    predator.position,
                    victim.position,
                )

                dist = np.linalg.norm(diff)

                if (
                    dist < nearest_dist
                    and dist < self.config.predator_detection_radius
                ):
                    nearest_dist = dist
                    chase_force = chase(diff)

            wander = self._update_wander(predator)

            predator.acceleration += (
                self.config.predator_chase_weight * chase_force
                + self.config.wander_weight * wander
            )

        #############################
        # Prey
        #############################

        for animal in prey:

            sep = np.zeros(2)
            align = np.zeros(2)
            coh = np.zeros(2)
            esc = np.zeros(2)

            align_count = 0
            coh_count = 0

            #############################
            # prey-prey interactions
            #############################

            for neighbour in prey:

                if neighbour is animal:
                    continue

                diff = world.shortest_displacement(
                    animal.position,
                    neighbour.position,
                )

                dist = np.linalg.norm(diff)

                if dist < self.config.separation_radius:
                    sep += separation(
                        diff,
                        self.config.max_separation_strength,
                    )

                if dist < self.config.alignment_radius:
                    align += alignment(
                        neighbour.velocity
                    )
                    align_count += 1

                if dist < self.config.cohesion_radius:
                    coh += cohesion(
                        neighbour.position
                    )
                    coh_count += 1

            if align_count:
                align = (
                    align / align_count
                    - animal.velocity
                )

            if coh_count:
                coh = (
                    coh / coh_count
                    - animal.position
                )

            #############################
            # escape nearest predator
            #############################

            nearest_dist = np.inf

            for predator in predators:

                diff = world.shortest_displacement(
                    animal.position,
                    predator.position,
                )

                dist = np.linalg.norm(diff)

                if (
                    dist < nearest_dist
                    and dist < self.config.prey_detection_radius
                ):
                    nearest_dist = dist
                    esc = escape(diff)

            wander = self._update_wander(animal)

            animal.acceleration += (
                self.config.prey_separation_weight * sep
                + self.config.prey_alignment_weight * align
                + self.config.prey_cohesion_weight * coh
                + self.config.prey_escape_weight * esc
                + self.config.wander_weight * wander
            )