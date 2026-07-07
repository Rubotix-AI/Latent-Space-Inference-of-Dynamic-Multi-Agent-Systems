"""
Data loading for the "discover the hidden interaction law" task.

The physics engine (``physics_engine.dataset.generate_dataset``) writes one CSV
per trajectory plus a ``manifest.csv`` describing the train/test split. This
module turns those trajectories into fixed-shape tensors the models consume.

A single training example is a short **window** of ``window`` consecutive
timesteps of one trajectory:

    feats : (window, N_max, F)   per-agent dynamics features
    pos   : (window, N_max, 2)   per-agent positions (used to build graphs)
    mask  : (N_max,)             True for real agents, False for padding
    label : ()                   integer id of the governing interaction law

Because different trajectories have different agent counts, every window is
padded up to ``N_max`` agents and a boolean ``mask`` marks the real ones. All
models in this project are permutation invariant over agents and respect the
mask, so padding never leaks information.

Features (F = 5) are deliberately translation invariant so a classifier cannot
cheat on absolute location:

    [vx, vy, ax, ay, speed]

Positions are kept separately (and only used by the GNN to decide which agents
are neighbours); they are centred per window to remove global drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_NAMES = ["vx", "vy", "ax", "ay", "speed"]
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------


def _load_trajectory_arrays(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one trajectory CSV into dense arrays.

    Returns
    -------
    feats : (T, N, F) float32   dynamics features per (timestep, agent)
    pos   : (T, N, 2) float32   positions per (timestep, agent)
    """

    df = pd.read_csv(csv_path)

    n_agents = int(df["agent_id"].max()) + 1
    n_steps = int(df["timestep"].max()) + 1

    # Rows are written timestep-major, agent-minor, so a simple reshape
    # recovers the (T, N, cols) grid without an expensive pivot.
    cols = ["x", "y", "vx", "vy", "ax", "ay"]
    grid = df[cols].to_numpy(dtype=np.float32)
    grid = grid.reshape(n_steps, n_agents, len(cols))

    pos = grid[:, :, 0:2]                       # x, y
    vel = grid[:, :, 2:4]                       # vx, vy
    acc = grid[:, :, 4:6]                       # ax, ay
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)

    feats = np.concatenate([vel, acc, speed], axis=-1)  # (T, N, 5)

    return feats.astype(np.float32), pos.astype(np.float32)


@dataclass
class _Window:
    """One (trajectory, time-slice) example."""
    csv_path: str
    label: int
    start: int
    length: int


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LawTrajectoryDataset(Dataset):
    """
    Fixed-window classification dataset over trajectories.

    Parameters
    ----------
    manifest:
        Rows of the global manifest belonging to this split.
    window:
        Number of consecutive timesteps per example.
    n_max:
        Maximum agent count across the whole dataset (for padding).
    feat_mean, feat_std:
        Per-feature normalisation stats (computed on the training split and
        shared with the test split).
    stride:
        Gap between successive windows carved from a trajectory. Defaults to
        ``window`` (non-overlapping). A smaller stride yields more examples.
    cache:
        Keep parsed trajectory arrays in memory (fast for small datasets).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        window: int,
        n_max: int,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        stride: int | None = None,
        cache: bool = True,
    ):
        self.window = int(window)
        self.n_max = int(n_max)
        self.feat_mean = feat_mean.astype(np.float32)
        self.feat_std = feat_std.astype(np.float32)
        self.stride = int(stride) if stride else int(window)
        self.cache = cache
        self._array_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self.windows: list[_Window] = []
        for _, row in manifest.iterrows():
            n_steps = int(row["timesteps"])
            label = int(row["label"])
            path = str(row["csv_path"])
            last_start = n_steps - self.window
            if last_start < 0:
                continue
            for start in range(0, last_start + 1, self.stride):
                self.windows.append(
                    _Window(path, label, start, self.window)
                )

    def __len__(self) -> int:
        return len(self.windows)

    def _get_arrays(self, csv_path: str) -> tuple[np.ndarray, np.ndarray]:
        if self.cache and csv_path in self._array_cache:
            return self._array_cache[csv_path]
        arrays = _load_trajectory_arrays(csv_path)
        if self.cache:
            self._array_cache[csv_path] = arrays
        return arrays

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        feats_full, pos_full = self._get_arrays(w.csv_path)

        sl = slice(w.start, w.start + self.window)
        feats = feats_full[sl]                      # (W, N, F)
        pos = pos_full[sl].copy()                   # (W, N, 2)
        n = feats.shape[1]

        # Normalise dynamics features.
        feats = (feats - self.feat_mean) / self.feat_std

        # Centre positions per window (remove global translation/drift).
        pos = pos - pos.reshape(-1, 2).mean(axis=0, keepdims=True)

        # Pad agents up to n_max.
        F = feats.shape[-1]
        feats_p = np.zeros((self.window, self.n_max, F), dtype=np.float32)
        pos_p = np.zeros((self.window, self.n_max, 2), dtype=np.float32)
        mask = np.zeros((self.n_max,), dtype=bool)

        n_use = min(n, self.n_max)
        feats_p[:, :n_use] = feats[:, :n_use]
        pos_p[:, :n_use] = pos[:, :n_use]
        mask[:n_use] = True

        return {
            "feats": torch.from_numpy(feats_p),
            "pos": torch.from_numpy(pos_p),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(w.label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


@dataclass
class DatasetMeta:
    n_max: int
    n_features: int
    n_classes: int
    law_names: list[str]


def _compute_feature_stats(
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of each dynamics feature over the given (train) trajectories."""

    sums = np.zeros(N_FEATURES, dtype=np.float64)
    sq_sums = np.zeros(N_FEATURES, dtype=np.float64)
    count = 0

    for _, row in manifest.iterrows():
        feats, _ = _load_trajectory_arrays(str(row["csv_path"]))
        flat = feats.reshape(-1, N_FEATURES).astype(np.float64)
        sums += flat.sum(axis=0)
        sq_sums += (flat ** 2).sum(axis=0)
        count += flat.shape[0]

    mean = sums / max(count, 1)
    var = sq_sums / max(count, 1) - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    return mean.astype(np.float32), std.astype(np.float32)


def build_datasets(
    data_dir: str | Path,
    window: int = 50,
    stride: int | None = None,
    n_max: int | None = None,
):
    """
    Build train/test datasets from a generated benchmark.

    Returns
    -------
    train_ds, test_ds, meta
    """

    data_dir = Path(data_dir)
    manifest = pd.read_csv(data_dir / "manifest.csv")

    with open(data_dir / "labels.json") as f:
        labels = json.load(f)
    law_names = labels["law_names"]

    if n_max is None:
        n_max = int(manifest["num_agents"].max())

    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)

    feat_mean, feat_std = _compute_feature_stats(train_manifest)

    train_ds = LawTrajectoryDataset(
        train_manifest, window, n_max, feat_mean, feat_std, stride=stride
    )
    test_ds = LawTrajectoryDataset(
        test_manifest, window, n_max, feat_mean, feat_std, stride=stride
    )

    meta = DatasetMeta(
        n_max=n_max,
        n_features=N_FEATURES,
        n_classes=len(law_names),
        law_names=law_names,
    )

    return train_ds, test_ds, meta
