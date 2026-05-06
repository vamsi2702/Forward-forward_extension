from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn

from .data import make_wrong_labels, overlay_label_channels
from .models import FFNetwork, SpatialFFNetwork


def _bce_from_goodness(goodness: torch.Tensor, theta: float, target: float) -> torch.Tensor:
    labels = torch.full_like(goodness, fill_value=target)
    logits = goodness - theta
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def train_ff_epoch(
    model: FFNetwork,
    train_loader: Iterable,
    optimizers: Sequence[torch.optim.Optimizer],
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_pos = 0.0
    running_neg = 0.0
    steps = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = images.view(images.size(0), -1).to(device)
        y = labels.to(device)

        wrong = make_wrong_labels(y)
        pos = overlay_label_channels(x, y)
        neg = overlay_label_channels(x, wrong)

        pos_h = pos
        neg_h = neg

        for layer, opt in zip(model.layers, optimizers):
            pos_out, pos_g, _ = layer(pos_h)
            neg_out, neg_g, _ = layer(neg_h)

            loss = _bce_from_goodness(pos_g, layer.theta, 1.0) + _bce_from_goodness(
                neg_g, layer.theta, 0.0
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.detach().cpu())
            running_pos += float(pos_g.mean().detach().cpu())
            running_neg += float(neg_g.mean().detach().cpu())
            steps += 1

            pos_h = pos_out.detach()
            neg_h = neg_out.detach()

    return {
        "train_loss": running_loss / max(steps, 1),
        "train_pos_goodness": running_pos / max(steps, 1),
        "train_neg_goodness": running_neg / max(steps, 1),
    }


@torch.no_grad()
def evaluate_label_search(
    model: FFNetwork,
    data_loader: Iterable,
    device: torch.device,
    max_batches: int | None = None,
    num_classes: int = 10,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = images.view(images.size(0), -1).to(device)
        y = labels.to(device)
        scores_per_label: List[torch.Tensor] = []

        for cls in range(num_classes):
            cls_labels = torch.full_like(y, cls)
            x_cls = overlay_label_channels(x, cls_labels)
            _, goodnesses = model(x_cls)
            total_goodness = torch.stack(goodnesses, dim=0).sum(dim=0)
            scores_per_label.append(total_goodness)

        scores = torch.stack(scores_per_label, dim=1)
        preds = scores.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return {"accuracy": correct / max(total, 1), "error_rate": 1.0 - (correct / max(total, 1))}


@torch.no_grad()
def evaluate_label_search_spatial(
    model: SpatialFFNetwork,
    data_loader: Iterable,
    device: torch.device,
    max_batches: int | None = None,
    num_classes: int = 10,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = images.to(device)
        y = labels.to(device)
        flat = x.view(x.size(0), -1)
        scores_per_label: List[torch.Tensor] = []

        for cls in range(num_classes):
            cls_labels = torch.full_like(y, cls)
            x_cls_flat = overlay_label_channels(flat, cls_labels)
            x_cls = x_cls_flat.view_as(x)

            _, goodnesses = model(x_cls)

            # For spatial layers goodness is [B, P].
            # Sum over patches to get per-sample layer goodness, then sum over layers.
            per_layer_total = [g.sum(dim=1) for g in goodnesses]
            total_goodness = torch.stack(per_layer_total, dim=0).sum(dim=0)
            scores_per_label.append(total_goodness)

        scores = torch.stack(scores_per_label, dim=1)
        preds = scores.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return {"accuracy": correct / max(total, 1), "error_rate": 1.0 - (correct / max(total, 1))}


def train_spatial_ff_epoch(
    model: SpatialFFNetwork,
    train_loader: Iterable,
    optimizers: Sequence[torch.optim.Optimizer],
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_pos = 0.0
    running_neg = 0.0
    steps = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = images.to(device)
        y = labels.to(device)

        flat = x.view(x.size(0), -1)
        wrong = make_wrong_labels(y)
        pos_flat = overlay_label_channels(flat, y)
        neg_flat = overlay_label_channels(flat, wrong)
        pos_h = pos_flat.view_as(x)
        neg_h = neg_flat.view_as(x)

        for layer, opt in zip(model.layers, optimizers):
            pos_out_flat, pos_g, _ = layer(pos_h)
            neg_out_flat, neg_g, _ = layer(neg_h)

            pos_loss = _bce_from_goodness(pos_g, layer.theta, 1.0)
            neg_loss = _bce_from_goodness(neg_g, layer.theta, 0.0)
            loss = pos_loss + neg_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += float(loss.detach().cpu())
            running_pos += float(pos_g.mean().detach().cpu())
            running_neg += float(neg_g.mean().detach().cpu())
            steps += 1

            c, h, w = layer.out_channels, layer.output_h, layer.output_w
            pos_h = pos_out_flat.detach().view(x.size(0), c, h, w)
            neg_h = neg_out_flat.detach().view(x.size(0), c, h, w)

    return {
        "train_loss": running_loss / max(steps, 1),
        "train_pos_goodness": running_pos / max(steps, 1),
        "train_neg_goodness": running_neg / max(steps, 1),
    }
