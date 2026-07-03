
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation


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

    x_min = df["x"].min()
    x_max = df["x"].max()

    y_min = df["y"].min()
    y_max = df["y"].max()

    padding = 0.5

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_aspect("equal")

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

        scatter.set_offsets([])

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