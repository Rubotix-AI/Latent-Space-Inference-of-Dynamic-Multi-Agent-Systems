from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from physics_engine.simulation.config import SimulationConfig
from physics_engine.simulation.factory import create_simulation


def create_dataset(
    interaction_name: str,
    simulation_config: SimulationConfig,
    interaction_config,
    trajectory_id: int,
    save: bool = True,
):
    """
    Runs a simulation and converts it into a dataset.

    Returns
    -------
    pd.DataFrame
        Trajectory data.
    """

    simulation = create_simulation(
        interaction_name=interaction_name,
        simulation_config=simulation_config,
        interaction_config=interaction_config,
    )

    rows = []

    for _ in range(simulation_config.total_timesteps):

        for agent in simulation.snapshot():

            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "timestep": simulation.timestep,
                    "interaction": interaction_name,
                    "agent_id": agent.id,
                    "agent_type": agent.type.value,
                    "x": agent.position[0],
                    "y": agent.position[1],
                    "vx": agent.velocity[0],
                    "vy": agent.velocity[1],
                    "ax": agent.acceleration[0],
                    "ay": agent.acceleration[1],
                }
            )

        simulation.step()

    df = pd.DataFrame(rows)

    if save:

        SRC_DIR = Path(__file__).resolve().parent.parent

        DATA_DIR = SRC_DIR / "data"

        trajectory_dir = DATA_DIR / "trajectories"
        metadata_dir = DATA_DIR / "metadata"

        trajectory_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        trajectory_path = (
            trajectory_dir
            / f"trajectory_{trajectory_id:06d}.csv"
        )

        metadata_path = (
            metadata_dir
            / f"trajectory_{trajectory_id:06d}.json"
        )

        df.to_csv(
            trajectory_path,
            index=False,
        )

        metadata = {
            "trajectory_id": trajectory_id,
            "interaction": interaction_name,
            "seed": simulation_config.seed,
            "timesteps": simulation_config.total_timesteps,
            "num_agents": simulation_config.num_agents,
            "world_width": simulation_config.world_width,
            "world_height": simulation_config.world_height,
            "interaction_config": vars(interaction_config),
        }

        with open(metadata_path, "w") as f:
            json.dump(
                metadata,
                f,
                indent=4,
            )

    return df


if __name__ == "__main__":

    from physics_engine.interaction_laws.spring_network.config import (
        SpringNetworkConfig,
    )

    simulation_config = SimulationConfig()

    leader_config = SpringNetworkConfig()

    create_dataset(
        interaction_name="spring_network",
        simulation_config=simulation_config,
        interaction_config=leader_config,
        trajectory_id=0,
    )