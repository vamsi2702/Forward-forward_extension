from __future__ import annotations

from typing import Callable, Dict

import torch


def _sum_sq(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return y.pow(2).sum(dim=dim)


def _neg_sum_sq(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -y.pow(2).sum(dim=dim)


def _sum_abs(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return y.abs().sum(dim=dim)


def _neg_sum_abs(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -y.abs().sum(dim=dim)


def _sum_linear(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return y.sum(dim=dim)


def _neg_sum_linear(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -y.sum(dim=dim)


def _mean_sq(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return y.pow(2).mean(dim=dim)


def _mean_abs(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return y.abs().mean(dim=dim)


def _log1p_sum_sq(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.log1p(y.pow(2).sum(dim=dim))


def _sqrt_sum_sq(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sqrt(y.pow(2).sum(dim=dim) + 1e-6)


def _sum_softplus(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nn.functional.softplus(y).sum(dim=dim)


def _sum_huber_like(y: torch.Tensor, dim: int = -1, delta: float = 1.0) -> torch.Tensor:
    abs_y = y.abs()
    quad = torch.minimum(abs_y, torch.full_like(abs_y, delta))
    lin = abs_y - quad
    return (0.5 * quad.pow(2) + delta * lin).sum(dim=dim)


def _pnorm(y: torch.Tensor, dim: int = -1, p: float = 1.5) -> torch.Tensor:
    return torch.clamp(y.abs(), min=1e-8).pow(p).sum(dim=dim)


GOODNESS_FUNCTIONS: Dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "sum_sq": _sum_sq,
    "neg_sum_sq": _neg_sum_sq,
    "sum_abs": _sum_abs,
    "neg_sum_abs": _neg_sum_abs,
    "sum_linear": _sum_linear,
    "neg_sum_linear": _neg_sum_linear,
    "mean_sq": _mean_sq,
    "mean_abs": _mean_abs,
    "log1p_sum_sq": _log1p_sum_sq,
    "sqrt_sum_sq": _sqrt_sum_sq,
    "sum_softplus": _sum_softplus,
    "sum_huber_like": _sum_huber_like,
    "pnorm_1_5": lambda y, dim=-1: _pnorm(y, dim=dim, p=1.5),
    "pnorm_3": lambda y, dim=-1: _pnorm(y, dim=dim, p=3.0),
}

# normalization_mode tracks the footnote requirement for unsquared activity objectives.
GOODNESS_CONFIGS = {
    "sum_sq": {"normalization_mode": "l2", "theta": 2.0},
    "neg_sum_sq": {"normalization_mode": "l2", "theta": -2.0},
    "sum_abs": {"normalization_mode": "l1", "theta": 1.0},
    "neg_sum_abs": {"normalization_mode": "l1", "theta": -1.0},
    "sum_linear": {"normalization_mode": "l1", "theta": 0.5},
    "neg_sum_linear": {"normalization_mode": "l1", "theta": -0.5},
    "mean_sq": {"normalization_mode": "l2", "theta": 0.05},
    "mean_abs": {"normalization_mode": "l1", "theta": 0.05},
    "log1p_sum_sq": {"normalization_mode": "l2", "theta": 1.0},
    "sqrt_sum_sq": {"normalization_mode": "l2", "theta": 1.0},
    "sum_softplus": {"normalization_mode": "l1", "theta": 2.0},
    "sum_huber_like": {"normalization_mode": "l1", "theta": 1.5},
    "pnorm_1_5": {"normalization_mode": "l1", "theta": 1.0},
    "pnorm_3": {"normalization_mode": "l2", "theta": 2.0},
}
