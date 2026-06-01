"""NPU-friendly capacity blocks for online SFC variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.online_sfc_2d import RMSNorm2d, _runtime_assert


class PooledChannelCapacityMixer2d(nn.Module):
    """Large current-frame channel mixer with no streaming cache.

    The block pools over the band/frequency axis before running the large 1x1
    projections.  This adds useful per-frame channel capacity while keeping the
    expensive activations at width 1 and adding zero recurrent state.
    """

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.norm = RMSNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * hidden_channels, kernel_size=1, bias=True)
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        y = self.norm(x).mean(dim=3, keepdim=True)
        value, gate = self.expand(y).chunk(2, dim=1)
        y = value * torch.sigmoid(gate)
        return x + self.project(y) * self.residual_scale


def build_capacity_mixers(
    *,
    channels: int,
    hidden_channels: int = 0,
    n_layers: int = 0,
) -> nn.ModuleList:
    if n_layers < 0:
        raise ValueError(f"n_layers must be non-negative, got {n_layers}")
    if n_layers == 0:
        return nn.ModuleList()
    if hidden_channels <= 0:
        raise ValueError("hidden_channels must be positive when n_layers > 0")
    return nn.ModuleList(
        [PooledChannelCapacityMixer2d(channels=channels, hidden_channels=hidden_channels) for _ in range(n_layers)]
    )


def apply_capacity_mixers(x: torch.Tensor, mixers: nn.ModuleList) -> torch.Tensor:
    for mixer in mixers:
        x = mixer(x)
    return x
