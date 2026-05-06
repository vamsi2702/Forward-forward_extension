from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import optim

from .data import build_mnist_loaders
from .models import FFNetwork, SpatialFFNetwork
from .trainers import (
    evaluate_label_search,
    evaluate_label_search_spatial,
    train_ff_epoch,
    train_spatial_ff_epoch,
)
from .utils import seed_everything


def run_goodness_sweep(
    goodness_names: Sequence[str],
    activation_name: str = "relu",
    hidden_dims: Sequence[int] = (512, 512, 512),
    batch_size: int = 128,
    epochs: int = 2,
    lr: float = 1e-3,
    train_subset: int = 12000,
    test_subset: int = 2000,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    device: str | None = None,
    seed: int | None = 42,
) -> List[Dict[str, float]]:
    if seed is not None:
        seed_everything(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader = build_mnist_loaders(
        batch_size=batch_size,
        train_subset=train_subset,
        test_subset=test_subset,
    )

    rows: List[Dict[str, float]] = []
    for goodness_name in goodness_names:
        if seed is not None:
            seed_everything(seed)
        model = FFNetwork(
            input_dim=28 * 28,
            hidden_dims=hidden_dims,
            activation_name=activation_name,
            goodness_name=goodness_name,
        ).to(device_obj)
        optimizers = [optim.Adam(layer.parameters(), lr=lr) for layer in model.layers]

        last_train = {}
        for _ in range(epochs):
            last_train = train_ff_epoch(
                model,
                train_loader,
                optimizers,
                device=device_obj,
                max_batches=max_train_batches,
            )

        metrics = evaluate_label_search(
            model,
            test_loader,
            device=device_obj,
            max_batches=max_eval_batches,
        )
        rows.append(
            {
                "goodness": goodness_name,
                "activation": activation_name,
                "train_loss": last_train["train_loss"],
                "test_error": metrics["error_rate"],
                "test_accuracy": metrics["accuracy"],
            }
        )
    return rows


def run_activation_sweep(
    activation_names: Sequence[str],
    goodness_name: str = "sum_sq",
    hidden_dims: Sequence[int] = (512, 512, 512),
    batch_size: int = 128,
    epochs: int = 2,
    lr: float = 1e-3,
    train_subset: int = 12000,
    test_subset: int = 2000,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    device: str | None = None,
    seed: int | None = 42,
) -> List[Dict[str, float]]:
    if seed is not None:
        seed_everything(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader = build_mnist_loaders(
        batch_size=batch_size,
        train_subset=train_subset,
        test_subset=test_subset,
    )

    rows: List[Dict[str, float]] = []
    for activation_name in activation_names:
        if seed is not None:
            seed_everything(seed)
        model = FFNetwork(
            input_dim=28 * 28,
            hidden_dims=hidden_dims,
            activation_name=activation_name,
            goodness_name=goodness_name,
        ).to(device_obj)
        optimizers = [optim.Adam(layer.parameters(), lr=lr) for layer in model.layers]

        last_train = {}
        for _ in range(epochs):
            last_train = train_ff_epoch(
                model,
                train_loader,
                optimizers,
                device=device_obj,
                max_batches=max_train_batches,
            )

        metrics = evaluate_label_search(
            model,
            test_loader,
            device=device_obj,
            max_batches=max_eval_batches,
        )
        rows.append(
            {
                "activation": activation_name,
                "goodness": goodness_name,
                "train_loss": last_train["train_loss"],
                "test_error": metrics["error_rate"],
                "test_accuracy": metrics["accuracy"],
            }
        )
    return rows


def run_local_spatial_experiment(
    activation_name: str = "relu",
    goodness_name: str = "sum_sq",
    layer_specs: Sequence[dict] | None = None,
    batch_size: int = 128,
    epochs: int = 2,
    lr: float = 1e-3,
    train_subset: int = 12000,
    test_subset: int = 2000,
    jitter: bool = True,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    device: str | None = None,
    seed: int | None = 42,
) -> Dict[str, float]:
    if seed is not None:
        seed_everything(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader = build_mnist_loaders(
        batch_size=batch_size,
        train_subset=train_subset,
        test_subset=test_subset,
        jitter=jitter,
    )

    model = SpatialFFNetwork(
        in_channels=1,
        input_hw=(28, 28),
        layer_specs=layer_specs
        or [
            {"kernel_size": 7, "stride": 3, "out_channels": 32},
            {"kernel_size": 2, "stride": 1, "out_channels": 64},
        ],
        activation_name=activation_name,
        goodness_name=goodness_name,
    ).to(device_obj)
    optimizers = [optim.Adam(layer.parameters(), lr=lr) for layer in model.layers]

    history: Dict[str, float] = {}
    for epoch in range(epochs):
        train_metrics = train_spatial_ff_epoch(
            model,
            train_loader,
            optimizers,
            device=device_obj,
            max_batches=max_train_batches,
        )
        history = {
            "epoch": epoch + 1,
            **train_metrics,
        }

    # A compact proxy score: goodness margin after final epoch.
    history["goodness_margin"] = history["train_pos_goodness"] - history["train_neg_goodness"]
    metrics = evaluate_label_search_spatial(
        model,
        test_loader,
        device=device_obj,
        max_batches=max_eval_batches,
    )
    history["test_accuracy"] = metrics["accuracy"]
    history["test_error"] = metrics["error_rate"]
    return history
