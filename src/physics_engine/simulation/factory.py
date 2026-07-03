import numpy as np

from physics_engine.simulation.agent import Agent, AgentType
from physics_engine.simulation.config import SimulationConfig
from physics_engine.simulation.simulation import Simulation
from physics_engine.simulation.world import World

from physics_engine.interaction_laws.classical_boids.interaction import BoidsInteraction
from physics_engine.interaction_laws.leader_follower.interaction import LeaderFollowerInteraction
from physics_engine.interaction_laws.vicsek_model.interaction import VicsekInteraction
from physics_engine.interaction_laws.predator_prey.interaction import PredatorPreyInteraction

from physics_engine.interaction_laws.spring_network.config import SpringNetworkConfig
from physics_engine.interaction_laws.spring_network.interaction import SpringNetworkInteraction

from physics_engine.simulation.spring import Spring


def _create_world(
    config: SimulationConfig,
) -> World:
    return World(config)


def _create_agents(
    interaction_name: str,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> list[Agent]:

    agents: list[Agent] = []

    for i in range(config.num_agents):

        agent_type = AgentType.PREY

        if interaction_name == "predator_prey":

            # First 10% become predators.
            if i < max(1, config.num_agents // 10):
                agent_type = AgentType.PREDATOR

        agents.append(
            Agent(
                id=i,
                type=agent_type,
                position=rng.uniform(
                    low=[
                        -config.world_width / 2,
                        -config.world_height / 2,
                    ],
                    high=[
                        config.world_width / 2,
                        config.world_height / 2,
                    ],
                ),
                velocity=rng.normal(size=2),
            )
        )

    return agents


def _create_springs(
    config: SimulationConfig,
    spring_config: SpringNetworkConfig,
) -> list[Spring]:
    """
    Connect every agent to the next, forming a chain.
    """

    springs = []

    for i in range(config.num_agents - 1):

        springs.append(
            Spring(
                agent_a=i,
                agent_b=i + 1,
                rest_length=spring_config.rest_length,
                stiffness=spring_config.spring_constant,
            )
        )

    return springs


def _create_interaction(
    interaction_name: str,
    interaction_config,
    rng: np.random.Generator,
    simulation_config: SimulationConfig,
):

    if interaction_name == "classical_boids":

        return BoidsInteraction(
            interaction_config,
            rng,
        )

    if interaction_name == "leader_follower":

        return LeaderFollowerInteraction(
            interaction_config,
            rng,
        )

    if interaction_name == "predator_prey":

        return PredatorPreyInteraction(
            interaction_config,
            rng,
        )

    if interaction_name == "spring_network":

        springs = _create_springs(
            simulation_config,
            interaction_config,
        )

        return SpringNetworkInteraction(
            springs,
            interaction_config,
            rng,
        )

    if interaction_name == "vicsek_model":

        return VicsekInteraction(
            interaction_config,
            rng,
        )

    raise ValueError(
        f"Unknown interaction law '{interaction_name}'."
    )


def create_simulation(
    interaction_name: str,
    simulation_config: SimulationConfig,
    interaction_config,
) -> Simulation:
    """
    Creates a fully initialized simulation.
    """

    rng = np.random.default_rng(
        simulation_config.seed
    )

    world = _create_world(
        simulation_config,
    )

    agents = _create_agents(
        interaction_name,
        simulation_config,
        rng,
    )

    interaction = _create_interaction(
        interaction_name,
        interaction_config,
        rng,
        simulation_config,
    )

    return Simulation(
        agents=agents,
        world=world,
        interaction=interaction,
        config=simulation_config,
    )