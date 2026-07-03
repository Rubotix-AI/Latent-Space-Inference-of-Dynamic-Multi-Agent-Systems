import numpy as np

from physics_engine.simulation.agent import Agent
from physics_engine.simulation.interaction import InteractionLaw
from physics_engine.simulation.world import World

from config import LeaderFollowerConfig
from utils import (
    separation,
    leader_alignment,
    leader_attraction,
)


class LeaderFollowerInteraction(InteractionLaw):

    def __init__(
        self,
        config: LeaderFollowerConfig,
        rng: np.random.Generator,
    ):
        super().__init__(
            rng=rng,
            wander_radius=config.wander_radius,
            wander_noise=config.wander_noise,
        )

        self.config = config

        self.leader_id = config.leader_id

        self.r_sep = config.separation_radius
        self.r_leader = config.leader_radius

        self.w_sep = config.separation_weight
        self.w_pos = config.leader_attraction_weight
        self.w_vel = config.leader_alignment_weight
        self.w_wander = config.wander_weight

    def step(
        self,
        agents: list[Agent],
        world: World,
    ):

        leader = None

        for agent in agents:
            if agent.id == self.leader_id:
                leader = agent
                break

        if leader is None:
            raise ValueError(
                f"Leader with id={self.leader_id} not found."
            )

        for agent in agents:

            # Leader is externally controlled.
            if agent.id == self.leader_id:
                agent.acceleration = self._update_wander(agent)
                continue

            separation_force = np.zeros(2)

            for neighbour in agents:

                if neighbour is agent:
                    continue

                if neighbour.id == self.leader_id:
                    continue

                diff = world.shortest_displacement(
                    agent.position,
                    neighbour.position,
                )

                dist = np.linalg.norm(diff)

                if dist < self.r_sep:
                    separation_force += separation(
                        diff,
                        self.config.max_separation_strength,
                    )

            leader_diff = world.shortest_displacement(
                agent.position,
                leader.position,
            )

            leader_distance = np.linalg.norm(leader_diff)

            attraction_force = np.zeros(2)
            alignment_force = np.zeros(2)

            if leader_distance < self.r_leader:

                attraction_force = leader_attraction(
                    leader_diff
                )

                alignment_force = (
                    leader_alignment(leader.velocity)
                    - agent.velocity
                )

            wander = self._update_wander(agent)

            agent.acceleration += (
                self.w_sep * separation_force
                + self.w_pos * attraction_force
                + self.w_vel * alignment_force
                + self.w_wander * wander
            )