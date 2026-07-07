"""
Project entrypoints.

The library lives under ``src/`` (``physics_engine`` = simulator + dataset
generation, ``models`` = model architectures). Experiment orchestration lives in
the top-level ``experiments/`` package. Run everything from the repository root
with ``src`` on the path:

    # 1. Generate the multi-law benchmark dataset
    PYTHONPATH=src python -m physics_engine.dataset --trajectories-per-law 40 --timesteps 300

    # 2. Train + compare the models on the held-out test split
    PYTHONPATH=src python -m experiments.run_experiment --epochs 15 --window 50
"""

if __name__ == "__main__":
    print(__doc__)
