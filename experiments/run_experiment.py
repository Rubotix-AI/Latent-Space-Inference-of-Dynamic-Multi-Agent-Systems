"""
Run the interaction-law discovery benchmark.

Trains the LSTM, Transformer and GNN classifiers on the generated dataset and
compares them on the held-out test split. Results (per-model metrics, confusion
matrices, learning curves, and a summary table) are written to ``results/``.

Usage
-----
Run from the repository root with ``src`` on the path so the library packages
(``models``, ``physics_engine``) resolve.

First generate a dataset::

    PYTHONPATH=src python -m physics_engine.dataset --trajectories-per-law 40 --timesteps 300

Then run the benchmark::

    PYTHONPATH=src python -m experiments.run_experiment --epochs 15 --window 50

Add ``--models lstm gnn`` to run a subset, or ``--smoke`` for a fast check.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

from models.baseline_1_lstm import LSTMClassifier
from models.baseline_2_transformer import TransformerClassifier
from models.baseline_3_gnn import GNNClassifier

from experiments.data import build_datasets
from experiments.trainer import TrainResult, set_seed, train_model


MODEL_REGISTRY = {
    "lstm": LSTMClassifier,
    "transformer": TransformerClassifier,
    "gnn": GNNClassifier,
}


def _make_model(name: str, meta):
    cls = MODEL_REGISTRY[name]
    return cls(n_features=meta.n_features, n_classes=meta.n_classes)


def _repo_root() -> Path:
    # experiments/run_experiment.py -> <repo root>
    return Path(__file__).resolve().parent.parent


def _default_data_dir() -> Path:
    return _repo_root() / "src" / "data"


def _default_results_dir() -> Path:
    return _repo_root() / "results"


def print_summary(results: list[TrainResult], law_names: list[str]):
    print("\n" + "=" * 68)
    print("INTERACTION-LAW DISCOVERY - TEST-SET COMPARISON")
    print("=" * 68)
    header = f"{'model':<14}{'params':>10}{'train(s)':>10}{'test_acc':>10}{'inf(ms)':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        f = r.final
        print(
            f"{r.model_name:<14}{r.n_parameters:>10,}{r.train_seconds:>10.1f}"
            f"{f.accuracy:>10.3f}{f.mean_inference_ms:>10.3f}"
        )
    print("=" * 68)

    best = max(results, key=lambda r: r.final.accuracy)
    print(f"\nPer-class accuracy of best model ({best.model_name}):")
    for name in law_names:
        print(f"  {name:<18} {best.final.per_class_accuracy[name]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "gnn"])
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast settings for a pipeline sanity check.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.window = min(args.window, 20)
        args.epochs = min(args.epochs, 2)

    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest found at {manifest_path}.\n"
            "Generate a dataset first, e.g.:\n"
            "  PYTHONPATH=src python -m physics_engine.dataset "
            "--trajectories-per-law 40 --timesteps 300"
        )

    set_seed(args.seed)

    train_ds, test_ds, meta = build_datasets(
        data_dir, window=args.window, stride=args.stride
    )
    print(
        f"Dataset: {len(train_ds)} train / {len(test_ds)} test windows | "
        f"{meta.n_classes} laws | n_max={meta.n_max} | window={args.window}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    results: list[TrainResult] = []
    for name in args.models:
        if name not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
        print(f"\n>>> Training {name}")
        set_seed(args.seed)  # same init conditions per model
        model = _make_model(name, meta)
        result = train_model(
            model, name, train_loader, test_loader, meta.law_names,
            epochs=args.epochs, lr=args.lr, device=args.device,
        )
        results.append(result)

        with open(results_dir / f"{name}_result.json", "w") as f:
            json.dump(asdict(result), f, indent=2)

    print_summary(results, meta.law_names)

    summary = {
        "law_names": meta.law_names,
        "config": {
            "window": args.window,
            "stride": args.stride,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_max": meta.n_max,
        },
        "models": [
            {
                "model": r.model_name,
                "n_parameters": r.n_parameters,
                "train_seconds": r.train_seconds,
                "test_accuracy": r.final.accuracy,
                "mean_inference_ms": r.final.mean_inference_ms,
                "per_class_accuracy": r.final.per_class_accuracy,
                "confusion": r.final.confusion,
            }
            for r in results
        ],
    }
    with open(results_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {results_dir}")


if __name__ == "__main__":
    main()
