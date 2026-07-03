
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from physics_engine.simulation.config import SimulationConfig as simulation_config

def visualize(
    trajectory_path: str | Path,
):
    """
    Visualize a trajectory stored in a CSV file.
    """

    df = pd.read_csv(trajectory_path)

    fig, ax = plt.subplots(figsize=(7, 7))

    interaction_name = df["interaction"].iloc[0]

    ax.set_title(interaction_name.replace("_", " ").title())

    ax.set_xlim(
        -simulation_config.world_width / 2,
        simulation_config.world_width / 2,
    )

    ax.set_ylim(
        -simulation_config.world_height / 2,
        simulation_config.world_height / 2,
    )

    # ax.set_aspect("equal")

    scatter = ax.scatter([], [])

    timestep_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
    )

    timesteps = sorted(df["timestep"].unique())

    has_agent_type = "agent_type" in df.columns

    colour_map = {
        "prey": "tab:blue",
        "predator": "tab:red",
    }

    def init():

        scatter.set_offsets(np.zeros((0, 2)))

        timestep_text.set_text("")

        return scatter, timestep_text

    def update(frame):

        current = df[
            df["timestep"] == frame
        ]

        positions = current[
            ["x", "y"]
        ].to_numpy()

        scatter.set_offsets(positions)

        if has_agent_type:

            colours = [
                colour_map.get(t, "black")
                for t in current["agent_type"]
            ]

            scatter.set_color(colours)

        timestep_text.set_text(
            f"Timestep: {frame}"
        )

        return scatter, timestep_text

    animation = FuncAnimation(
        fig,
        update,
        frames=timesteps,
        init_func=init,
        interval=20,
        blit=True,
    )

    plt.show()


if __name__ == "__main__":

    visualize(
        "data/trajectories/trajectory_000000.csv"
    )