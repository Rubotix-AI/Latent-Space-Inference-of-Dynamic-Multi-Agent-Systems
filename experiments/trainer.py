"""
Training and evaluation loop shared by all baselines.

Model-agnostic: anything implementing the
``logits, latent = model(feats, pos, mask)`` contract can be trained and
evaluated here, so adding a new model never touches this file. Metric
definitions live in ``experiments/metrics/`` and are imported below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.common import count_parameters
from experiments.metrics.accuracy_traj_error import (
    accuracy,
    confusion_matrix,
    per_class_accuracy,
)
from experiments.metrics.time_taken import mean_inference_ms


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _move(batch, device):
    return (
        batch["feats"].to(device),
        batch["pos"].to(device),
        batch["mask"].to(device),
        batch["label"].to(device),
    )


@dataclass
class EvalResult:
    accuracy: float
    per_class_accuracy: dict[str, float]
    confusion: list[list[int]]
    mean_inference_ms: float
    n_examples: int


@dataclass
class TrainResult:
    model_name: str
    n_parameters: int
    train_seconds: float
    history: list[dict] = field(default_factory=list)
    final: EvalResult | None = None


@torch.no_grad()
def evaluate(model, loader: DataLoader, law_names: list[str], device) -> EvalResult:
    model.eval()
    n_classes = len(law_names)

    all_labels: list[int] = []
    all_preds: list[int] = []
    inf_times: list[float] = []

    for batch in loader:
        feats, pos, mask, label = _move(batch, device)

        start = time.perf_counter()
        logits, _ = model(feats, pos, mask)
        inf_times.append((time.perf_counter() - start) / feats.size(0))

        pred = logits.argmax(dim=-1)
        all_labels.extend(label.tolist())
        all_preds.extend(pred.tolist())

    cm = confusion_matrix(all_labels, all_preds, n_classes)

    return EvalResult(
        accuracy=accuracy(all_preds, all_labels),
        per_class_accuracy=per_class_accuracy(cm, law_names),
        confusion=cm.tolist(),
        mean_inference_ms=mean_inference_ms(inf_times),
        n_examples=len(all_labels),
    )


def train_model(
    model,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    law_names: list[str],
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    log_every: int = 1,
) -> TrainResult:

    device = torch.device(device)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    result = TrainResult(
        model_name=model_name,
        n_parameters=count_parameters(model),
        train_seconds=0.0,
    )

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen, correct = 0.0, 0, 0

        for batch in train_loader:
            feats, pos, mask, label = _move(batch, device)

            optim.zero_grad()
            logits, _ = model(feats, pos, mask)
            loss = criterion(logits, label)
            loss.backward()
            optim.step()

            running += loss.item() * label.size(0)
            seen += label.size(0)
            correct += (logits.argmax(-1) == label).sum().item()

        train_acc = correct / max(seen, 1)
        train_loss = running / max(seen, 1)
        test_eval = evaluate(model, test_loader, law_names, device)

        result.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_eval.accuracy,
            }
        )
        if log_every and (epoch % log_every == 0 or epoch == epochs):
            print(
                f"  [{model_name}] epoch {epoch:>3}/{epochs} "
                f"loss {train_loss:.4f} "
                f"train_acc {train_acc:.3f} test_acc {test_eval.accuracy:.3f}"
            )

    result.train_seconds = time.perf_counter() - t0
    result.final = evaluate(model, test_loader, law_names, device)
    return result
