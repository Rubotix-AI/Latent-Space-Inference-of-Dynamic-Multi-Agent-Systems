from abc import ABC, abstractmethod

import numpy as np

from physics_engine.simulation.agent import Agent
from physics_engine.simulation.world import World


class InteractionLaw(ABC):
    """
    Base class for every interaction law.

    Provides common functionality shared across interaction laws,
    while enforcing the step() interface.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        wander_radius: float,
        wander_noise: float,
    ):
        self.rng = rng

        self.wander_radius = wander_radius
        self.wander_noise = wander_noise

        self.wander_vectors: dict[int, np.ndarray] = {}

    def _update_wander(
        self,
        agent: Agent,
    ) -> np.ndarray:
        """
        Returns the updated wander vector for an agent.

        Each agent maintains a persistent wander direction whose
        magnitude remains constant while its direction evolves
        gradually over time.
        """

        # Initialize on first encounter
        if agent.id not in self.wander_vectors:

            wander = self.rng.random(2) - 0.5

            norm = np.linalg.norm(wander)

            if norm > 1e-8:
                wander = (
                    wander / norm
                ) * self.wander_radius

            self.wander_vectors[agent.id] = wander

            return wander

        wander = self.wander_vectors[agent.id]

        # Small random perturbation
        wander += (
            self.wander_noise
            * (self.rng.random(2) - 0.5)
        )

        # Keep magnitude constant
        norm = np.linalg.norm(wander)

        if norm > 1e-8:
            wander = (
                wander / norm
            ) * self.wander_radius

        self.wander_vectors[agent.id] = wander

        return wander

    @abstractmethod
    def step(
        self,
        agents: list[Agent],
        world: World,
    ) -> None:
        """
        Compute one interaction step by updating each agent's
        acceleration.
        """
        pass