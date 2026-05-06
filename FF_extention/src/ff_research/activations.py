from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn
import torch.nn.functional as F


class StudentTNegLogPDF(nn.Module):
    """Negative log-density of a Student-t, used as a smooth heavy-tail nonlinearity."""

    def __init__(self, df: float = 3.0, scale: float = 1.0) -> None:
        super().__init__()
        if df <= 0:
            raise ValueError("df must be > 0")
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.df = float(df)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x / self.scale).pow(2) / self.df
        return 0.5 * (self.df + 1.0) * torch.log1p(z)


class SquarePlus(nn.Module):
    """Squareplus from Barron 2021: 0.5 * (x + sqrt(x^2 + 4))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x + torch.sqrt(x * x + 4.0))


class BentIdentity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.sqrt(x * x + 1.0) - 1.0) + x


ACTIVATION_FACTORIES: Dict[str, Callable[[], nn.Module]] = {
    "relu": lambda: nn.ReLU(),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
    "elu": lambda: nn.ELU(alpha=1.0),
    "selu": lambda: nn.SELU(),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(),
    "mish": lambda: nn.Mish(),
    "softplus": lambda: nn.Softplus(beta=1.0),
    "tanh": lambda: nn.Tanh(),
    "hardtanh": lambda: nn.Hardtanh(min_val=-2.0, max_val=2.0),
    "softsign": lambda: nn.Softsign(),
    "tanhshrink": lambda: nn.Tanhshrink(),
    "squareplus": lambda: SquarePlus(),
    "bent_identity": lambda: BentIdentity(),
    "student_t_nll_df2": lambda: StudentTNegLogPDF(df=2.0, scale=1.0),
    "student_t_nll_df3": lambda: StudentTNegLogPDF(df=3.0, scale=1.0),
    "student_t_nll_df5": lambda: StudentTNegLogPDF(df=5.0, scale=1.0),
}


def build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in ACTIVATION_FACTORIES:
        available = ", ".join(sorted(ACTIVATION_FACTORIES))
        raise KeyError(f"Unknown activation '{name}'. Available: {available}")
    return ACTIVATION_FACTORIES[key]()
