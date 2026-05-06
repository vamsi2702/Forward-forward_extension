from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn

from .activations import build_activation
from .goodness import GOODNESS_CONFIGS, GOODNESS_FUNCTIONS


@dataclass
class LayerStats:
    loss: float
    pos_goodness: float
    neg_goodness: float


class FFLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_name: str,
        goodness_name: str,
        theta: float | None = None,
        normalization_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = build_activation(activation_name)
        self.goodness_name = goodness_name

        config = GOODNESS_CONFIGS.get(goodness_name, {})
        self.normalization_mode = normalization_mode or config.get("normalization_mode", "l2")
        self.theta = float(theta if theta is not None else config.get("theta", 2.0))

        if goodness_name not in GOODNESS_FUNCTIONS:
            available = ", ".join(sorted(GOODNESS_FUNCTIONS))
            raise KeyError(f"Unknown goodness '{goodness_name}'. Available: {available}")
        self.goodness_fn = GOODNESS_FUNCTIONS[goodness_name]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_norm = self.activation(self.linear(x))
        goodness = self.goodness_fn(pre_norm, dim=-1)
        output = self._normalize(pre_norm)
        return output, goodness, pre_norm

    def _normalize(self, y: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        if self.normalization_mode == "l1":
            denom = y.abs().sum(dim=-1, keepdim=True) + eps
        else:
            denom = torch.sqrt(y.pow(2).sum(dim=-1, keepdim=True) + eps)
        return y / denom


class FFNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation_name: str,
        goodness_name: str,
        theta: float | None = None,
        normalization_mode: str | None = None,
    ) -> None:
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList(
            [
                FFLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    activation_name=activation_name,
                    goodness_name=goodness_name,
                    theta=theta,
                    normalization_mode=normalization_mode,
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        goodnesses: List[torch.Tensor] = []
        h = x
        for layer in self.layers:
            h, g, _ = layer(h)
            goodnesses.append(g)
        return h, goodnesses


class LocallyConnectedFFLayer(nn.Module):
    """Locally connected (no weight sharing) FF layer with per-patch goodness."""

    def __init__(
        self,
        in_channels: int,
        input_hw: Tuple[int, int],
        kernel_size: int,
        stride: int,
        out_channels: int,
        activation_name: str,
        goodness_name: str,
        theta: float | None = None,
        normalization_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_h, self.input_w = input_hw
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels

        self.output_h = (self.input_h - kernel_size) // stride + 1
        self.output_w = (self.input_w - kernel_size) // stride + 1
        self.num_patches = self.output_h * self.output_w
        self.patch_dim = in_channels * kernel_size * kernel_size

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
        self.weight = nn.Parameter(
            torch.randn(self.num_patches, self.patch_dim, out_channels) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(self.num_patches, out_channels))
        self.activation = build_activation(activation_name)

        config = GOODNESS_CONFIGS.get(goodness_name, {})
        self.normalization_mode = normalization_mode or config.get("normalization_mode", "l2")
        self.theta = float(theta if theta is not None else config.get("theta", 2.0))

        if goodness_name not in GOODNESS_FUNCTIONS:
            available = ", ".join(sorted(GOODNESS_FUNCTIONS))
            raise KeyError(f"Unknown goodness '{goodness_name}'. Available: {available}")
        self.goodness_fn = GOODNESS_FUNCTIONS[goodness_name]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [B, C, H, W]
        patches = self.unfold(x).transpose(1, 2)  # [B, P, D]
        pre = torch.einsum("bpd,pdo->bpo", patches, self.weight) + self.bias
        act = self.activation(pre)

        local_goodness = self.goodness_fn(act, dim=-1)  # [B, P]
        normed = self._normalize(act)
        flattened = normed.reshape(x.size(0), -1)
        return flattened, local_goodness, act

    def _normalize(self, y: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        if self.normalization_mode == "l1":
            denom = y.abs().sum(dim=-1, keepdim=True) + eps
        else:
            denom = torch.sqrt(y.pow(2).sum(dim=-1, keepdim=True) + eps)
        return y / denom


class SpatialFFNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_hw: Tuple[int, int],
        layer_specs: Sequence[dict],
        activation_name: str,
        goodness_name: str,
        theta: float | None = None,
        normalization_mode: str | None = None,
    ) -> None:
        super().__init__()
        layers: List[LocallyConnectedFFLayer] = []
        c = in_channels
        hw = input_hw

        for spec in layer_specs:
            layer = LocallyConnectedFFLayer(
                in_channels=c,
                input_hw=hw,
                kernel_size=spec["kernel_size"],
                stride=spec["stride"],
                out_channels=spec["out_channels"],
                activation_name=activation_name,
                goodness_name=goodness_name,
                theta=theta,
                normalization_mode=normalization_mode,
            )
            layers.append(layer)
            c = spec["out_channels"]
            hw = (layer.output_h, layer.output_w)

        self.layers = nn.ModuleList(layers)
        self.final_shape = (c, hw[0], hw[1])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        goodnesses: List[torch.Tensor] = []
        h = x
        for layer in self.layers:
            flat, g, _ = layer(h)
            goodnesses.append(g)
            c, hh, ww = layer.out_channels, layer.output_h, layer.output_w
            h = flat.view(x.size(0), c, hh, ww)
        return h, goodnesses
