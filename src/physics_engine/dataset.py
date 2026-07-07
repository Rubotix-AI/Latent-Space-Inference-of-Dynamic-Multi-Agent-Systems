from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from physics_engine.simulation.config import SimulationConfig
from physics_engine.simulation.factory import create_simulation

# Interaction-law configs.
from physics_engine.interaction_laws.classical_boids.config import BoidsConfig
from physics_engine.interaction_laws.leader_follower.config import (
    LeaderFollowerConfig,
)
from physics_engine.interaction_laws.predator_prey.config import (
    PredatorPreyConfig,
)
from physics_engine.interaction_laws.vicsek_model.config import VicsekConfig
from physics_engine.interaction_laws.spring_network.config import (
    SpringNetworkConfig,
)


# ---------------------------------------------------------------------------
# Registry of interaction laws
# ---------------------------------------------------------------------------
#
# The models are asked to recover *which* hidden interaction law governs a
# system from trajectories alone. That is only possible if the dataset
# actually contains several laws. LAW_REGISTRY is the single source of truth
# that ties together (a) the string name used by the simulation factory and
# (b) the config class that parameterises that law.

LAW_REGISTRY: dict[str, type] = {
    "classical_boids": BoidsConfig,
    "leader_follower": LeaderFollowerConfig,
    "predator_prey": PredatorPreyConfig,
    "vicsek_model": VicsekConfig,
    "spring_network": SpringNetworkConfig,
}

# Stable, sorted ordering -> integer class labels for the classifiers.
LAW_NAMES: list[str] = sorted(LAW_REGISTRY.keys())
LAW_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(LAW_NAMES)}

# Columns written to every trajectory CSV (kept identical to the schema the
# original single-trajectory generator produced).
FEATURE_COLUMNS = ["x", "y", "vx", "vy", "ax", "ay"]


# ---------------------------------------------------------------------------
# Single-trajectory generation (unchanged public behaviour)
# ---------------------------------------------------------------------------


def create_dataset(
    interaction_name: str,
    simulation_config: SimulationConfig,
    interaction_config,
    trajectory_id: int,
    save: bool = True,
    data_dir: Path | None = None,
):
    """
    Runs a single simulation and converts it into a trajectory DataFrame.

    Parameters
    ----------
    interaction_name:
        Key into ``LAW_REGISTRY`` selecting the interaction law.
    simulation_config:
        Shared simulation parameters (dt, agents, world, seed, ...).
    interaction_config:
        Parameters specific to the chosen interaction law.
    trajectory_id:
        Global id used to name the output files.
    save:
        When True the trajectory CSV and a metadata JSON are written to disk.
    data_dir:
        Root directory for ``trajectories/`` and ``metadata/``. Defaults to
        ``src/data`` so the historical layout is preserved.

    Returns
    -------
    pd.DataFrame
        Trajectory data with one row per (timestep, agent).
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

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"

        trajectory_dir = data_dir / "trajectories"
        metadata_dir = data_dir / "metadata"

        trajectory_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        trajectory_path = (
            trajectory_dir / f"trajectory_{trajectory_id:06d}.csv"
        )
        metadata_path = metadata_dir / f"trajectory_{trajectory_id:06d}.json"

        df.to_csv(trajectory_path, index=False)

        metadata = {
            "trajectory_id": trajectory_id,
            "interaction": interaction_name,
            "label": LAW_TO_INDEX[interaction_name],
            "seed": simulation_config.seed,
            "timesteps": simulation_config.total_timesteps,
            "num_agents": simulation_config.num_agents,
            "world_width": simulation_config.world_width,
            "world_height": simulation_config.world_height,
            "interaction_config": vars(interaction_config),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    return df


# ---------------------------------------------------------------------------
# Parameter randomisation
# ---------------------------------------------------------------------------
#
# For the "discover the hidden law" question to be meaningful, every
# trajectory of a given law must look *different* at the surface level
# (different densities, noise, radii, agent counts, ...) while still obeying
# the same underlying rule. Otherwise a classifier can cheat on trivial
# statistics. We therefore randomise both the shared simulation config and
# the per-law interaction config.


def _jitter(rng: np.random.Generator, value: float, lo: float, hi: float) -> float:
    """Multiply ``value`` by a uniform factor in ``[lo, hi]``."""
    return float(value) * float(rng.uniform(lo, hi))


def sample_simulation_config(
    rng: np.random.Generator,
    timesteps: int,
    trajectory_seed: int,
) -> SimulationConfig:
    """Draw a randomised shared simulation config for one trajectory."""

    num_agents = int(rng.integers(12, 31))          # 12..30 agents
    world = float(rng.uniform(8.0, 12.0))           # square world

    return SimulationConfig(
        dt=0.05,
        total_timesteps=timesteps,
        num_agents=num_agents,
        seed=trajectory_seed,
        world_width=world,
        world_height=world,
        periodic_boundary=True,
    )


def sample_interaction_config(interaction_name: str, rng: np.random.Generator):
    """
    Return a randomised interaction config for ``interaction_name``.

    Only physically meaningful parameters are perturbed; structural fields
    (e.g. ``leader_id``) are left untouched. Ranges are centred on the
    hand-tuned defaults so simulations stay stable.
    """

    if interaction_name == "classical_boids":
        c = BoidsConfig()
        c.separation_weight = _jitter(rng, c.separation_weight, 0.6, 1.4)
        c.alignment_weight = _jitter(rng, c.alignment_weight, 0.6, 1.4)
        c.cohesion_weight = _jitter(rng, c.cohesion_weight, 0.6, 1.4)
        c.separation_radius = _jitter(rng, c.separation_radius, 0.7, 1.3)
        c.alignment_radius = _jitter(rng, c.alignment_radius, 0.7, 1.3)
        c.cohesion_radius = _jitter(rng, c.cohesion_radius, 0.7, 1.3)
        c.wander_noise = _jitter(rng, c.wander_noise, 0.5, 1.5)
        return c

    if interaction_name == "leader_follower":
        c = LeaderFollowerConfig()
        c.leader_attraction_weight = _jitter(rng, c.leader_attraction_weight, 0.6, 1.4)
        c.leader_alignment_weight = _jitter(rng, c.leader_alignment_weight, 0.6, 1.4)
        c.separation_weight = _jitter(rng, c.separation_weight, 0.6, 1.4)
        c.leader_radius = _jitter(rng, c.leader_radius, 0.7, 1.3)
        c.separation_radius = _jitter(rng, c.separation_radius, 0.7, 1.3)
        c.wander_noise = _jitter(rng, c.wander_noise, 0.5, 1.5)
        return c

    if interaction_name == "predator_prey":
        c = PredatorPreyConfig()
        c.predator_chase_weight = _jitter(rng, c.predator_chase_weight, 0.6, 1.4)
        c.prey_escape_weight = _jitter(rng, c.prey_escape_weight, 0.6, 1.4)
        c.prey_cohesion_weight = _jitter(rng, c.prey_cohesion_weight, 0.6, 1.4)
        c.predator_detection_radius = _jitter(rng, c.predator_detection_radius, 0.7, 1.3)
        c.prey_detection_radius = _jitter(rng, c.prey_detection_radius, 0.7, 1.3)
        c.wander_noise = _jitter(rng, c.wander_noise, 0.5, 1.5)
        return c

    if interaction_name == "vicsek_model":
        c = VicsekConfig()
        c.interaction_radius = _jitter(rng, c.interaction_radius, 0.6, 1.5)
        c.preferred_speed = _jitter(rng, c.preferred_speed, 0.7, 1.3)
        c.response_rate = _jitter(rng, c.response_rate, 0.7, 1.3)
        c.angular_noise = _jitter(rng, c.angular_noise, 0.5, 1.8)
        return c

    if interaction_name == "spring_network":
        c = SpringNetworkConfig()
        c.spring_constant = _jitter(rng, c.spring_constant, 0.6, 1.4)
        c.rest_length = _jitter(rng, c.rest_length, 0.7, 1.3)
        c.damping = _jitter(rng, c.damping, 0.6, 1.4)
        c.wander_noise = _jitter(rng, c.wander_noise, 0.5, 1.5)
        return c

    raise ValueError(f"Unknown interaction law '{interaction_name}'.")


# ---------------------------------------------------------------------------
# Dataset (many trajectories, many laws) + train/test manifest
# ---------------------------------------------------------------------------


def generate_dataset(
    trajectories_per_law: int = 40,
    timesteps: int = 300,
    laws: list[str] | None = None,
    test_fraction: float = 0.2,
    seed: int = 0,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Generate a benchmark dataset spanning every interaction law and write a
    stratified train/test manifest.

    The manifest (``data/manifest.csv``) is what the model data loader reads;
    it has one row per trajectory with columns::

        trajectory_id, interaction, label, split,
        num_agents, timesteps, csv_path, metadata_path

    Parameters
    ----------
    trajectories_per_law:
        How many trajectories to simulate for each law.
    timesteps:
        Length (in steps) of every trajectory.
    laws:
        Subset of ``LAW_NAMES`` to include. Defaults to all laws.
    test_fraction:
        Fraction of each law's trajectories held out for testing.
    seed:
        Master seed controlling all randomisation (reproducible datasets).
    data_dir:
        Output root. Defaults to ``src/data``.

    Returns
    -------
    pd.DataFrame
        The manifest.
    """

    if laws is None:
        laws = LAW_NAMES

    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir = Path(data_dir)

    master_rng = np.random.default_rng(seed)

    manifest_rows: list[dict] = []
    trajectory_id = 0

    for interaction_name in laws:

        if interaction_name not in LAW_REGISTRY:
            raise ValueError(f"Unknown interaction law '{interaction_name}'.")

        # Deterministic split: shuffle indices once per law and hold out the
        # tail as the test set. Stratified because we split within each law.
        n = trajectories_per_law
        n_test = max(1, int(round(n * test_fraction)))
        order = master_rng.permutation(n)
        test_positions = set(order[:n_test].tolist())

        for k in range(n):

            traj_seed = int(master_rng.integers(0, 2**31 - 1))
            law_rng = np.random.default_rng(traj_seed)

            sim_cfg = sample_simulation_config(law_rng, timesteps, traj_seed)
            law_cfg = sample_interaction_config(interaction_name, law_rng)

            create_dataset(
                interaction_name=interaction_name,
                simulation_config=sim_cfg,
                interaction_config=law_cfg,
                trajectory_id=trajectory_id,
                save=True,
                data_dir=data_dir,
            )

            split = "test" if k in test_positions else "train"

            manifest_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "interaction": interaction_name,
                    "label": LAW_TO_INDEX[interaction_name],
                    "split": split,
                    "num_agents": sim_cfg.num_agents,
                    "timesteps": sim_cfg.total_timesteps,
                    "csv_path": str(
                        (data_dir / "trajectories"
                         / f"trajectory_{trajectory_id:06d}.csv").resolve()
                    ),
                    "metadata_path": str(
                        (data_dir / "metadata"
                         / f"trajectory_{trajectory_id:06d}.json").resolve()
                    ),
                }
            )

            trajectory_id += 1

        print(f"[generate_dataset] {interaction_name}: {n} trajectories "
              f"({n - n_test} train / {n_test} test)")

    manifest = pd.DataFrame(manifest_rows)

    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    labels_path = data_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(
            {"law_names": LAW_NAMES, "law_to_index": LAW_TO_INDEX},
            f,
            indent=4,
        )

    print(f"[generate_dataset] wrote {len(manifest)} trajectories -> "
          f"{manifest_path}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the multi-law trajectory benchmark dataset."
    )
    parser.add_argument("--trajectories-per-law", type=int, default=40)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    generate_dataset(
        trajectories_per_law=args.trajectories_per_law,
        timesteps=args.timesteps,
        test_fraction=args.test_fraction,
        seed=args.seed,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )
