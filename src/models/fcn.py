"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch
from torch import nn

from .common import ConvBlock, manual_pad


class FCN(nn.Module):
    """FCN feature extractor implementation for use in MILLET. Same as original architecture."""

    def __init__(self, n_in_channels: int, out_channels: int, padding_mode: str = "replicate"):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            ConvBlock(n_in_channels, 128, 8, padding_mode=padding_mode),
            ConvBlock(128, 256, 5, padding_mode=padding_mode),
            ConvBlock(256, out_channels, 3, padding_mode=padding_mode),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        x = x.float()
        x = x.transpose(1,2)
        min_len = 5
        if x.shape[-1] >= min_len:
            x = self.instance_encoder(x)
            x = self.gap(x)
            x = x.squeeze(-1)
            return x
        else:
            padded_x = manual_pad(x, min_len)
            x = self.instance_encoder(padded_x)
            x = self.gap(x)
            x = x.squeeze(-1)
            return x
