# file: models/harmamba_pytorch.py
"""
HARMamba: Efficient and Lightweight Wearable Sensor HAR based on Bidirectional Mamba
Paper: https://arxiv.org/abs/2403.20183 (v3, 2024-08-08)

This implementation follows Fig. 1 and Algorithm 1 in the paper:
- Channel-independent preprocessing
- RevIN (reversible instance normalization)
- Patching per channel (1D)
- Class token + learnable positional embedding
- HARMamba blocks (bidirectional selective SSM with ZOH discretization)
- MLP classifier head on the [CLS] token

Notes:
- The selective SSM uses a diagonal A (stable negative) and per-step B, C, Δ from data.
- ZOH discretization for diagonal A is implemented elementwise: A_bar = exp(Δ * A);
  B_bar = (A_bar - 1) / (A + eps) * B.
- Causal 1D convs are used before SSM (as in Algorithm 1) to extract local features.
- This is self-contained (no external CUDA kernels). It is faithful and readable, and serves as a
  solid baseline reference; performance can be improved by fusing scans or using official Mamba kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility layers
# -----------------------------

class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN).

    Normalizes per (B, C) across the time dimension. For classification we only need forward (normalize).

    Args:
        num_channels: number of sensor channels C.
        affine: whether to learn per-channel affine parameters.
        eps: numerical stability.
    """

    def __init__(self, num_channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_buffer("gamma", torch.ones(1, num_channels, 1))
            self.register_buffer("beta", torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.var(dim=-1, unbiased=False, keepdim=True).add(self.eps).sqrt()
        x_norm = (x - mean) / std
        # why: learn a per-channel rescaling to keep representation capacity
        return x_norm * self.gamma + self.beta


class PatchEmbed1D(nn.Module):
    """1D patch embedding per channel.

    Splits each channel's sequence into patches of length P with stride S,
    flattens, and projects to D.

    Input shape:  (B, C, L)
    Output shape: list[torch.Tensor] with length C, each tensor (B, N_patches, D)
    where N_patches = 1 + floor((L - P) / S)
    """

    def __init__(self, patch_size: int, stride: Optional[int], emb_dim: int):
        super().__init__()
        self.P = patch_size
        self.S = stride if stride is not None else patch_size
        self.emb_dim = emb_dim
        self.proj = nn.Linear(patch_size, emb_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, C, L = x.shape
        patches_per_channel = []
        for ch in range(C):
            seq = x[:, ch : ch + 1, :]  # (B,1,L)
            # Unfold -> (B, 1, P, N)
            n = 1 + (L - self.P) // self.S
            unfolded = seq.unfold(dimension=2, size=self.P, step=self.S)  # (B,1,N,P)
            unfolded = unfolded.squeeze(1)  # (B, N, P)
            tokens = self.proj(unfolded)  # (B, N, D)
            patches_per_channel.append(tokens)
        return torch.stack(patches_per_channel, dim=1), n  # (B, C, N, D), N


# -----------------------------
# Selective SSM (Mamba-style)
# -----------------------------

class DiagSelectiveSSM(nn.Module):
    """Bidirectional selective SSM with diagonal A and per-step B, C, Δ.

    Args:
        d_model: token feature dim fed into SSM (E in paper).
        d_state: hidden state size (equal to E works well; can be smaller).
        conv_kernel: causal conv kernel size before SSM.
    """

    def __init__(self, d_model: int, d_state: Optional[int] = None, conv_kernel: int = 3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state or d_model
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel - 1)
        # why: left-causal via trim; padding set to keep shape then we slice
        self.act = nn.SiLU()

        # Base diagonal A (stable negative)
        self.A = nn.Parameter(-torch.ones(self.d_state))

        # Projections to B, C, Δ for each time step
        self.to_params = nn.Linear(d_model, 3 * self.d_state)

    @staticmethod
    def _zoh_discretize(A: torch.Tensor, Delta: torch.Tensor, B: torch.Tensor, eps: float = 1e-6):
        # A: (..., d), Delta: (..., d), B: (..., d)
        A_bar = torch.exp(Delta * A)
        B_bar = (A_bar - 1.0) / (A + eps) * B
        return A_bar, B_bar

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """Run one directional selective SSM scan.

        x: (B, L, E)  -> returns y: (B, L, E)
        """
        B, L, E = x.shape

        # Pre-conv (causal) + SiLU
        xc = self.conv(x.transpose(1, 2))[:, :, :L]  # trim right-padding to be causal
        xc = self.act(xc.transpose(1, 2))  # (B, L, E)

        # Per-step parameters
        params = self.to_params(xc)  # (B, L, 3*D)
        B_t, C_t, Delta_t = params.chunk(3, dim=-1)
        # Gate/scale to keep dynamics stable
        Delta_t = F.softplus(Delta_t)
        C_t = torch.tanh(C_t)

        # Broadcast base A
        A = self.A.view(1, 1, -1).expand(B, L, -1)
        A_bar, B_bar = self._zoh_discretize(A, Delta_t, B_t)

        # Recurrent scan (elementwise)
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * xc[:, t]
            y_t = C_t[:, t] * h
            outs.append(y_t)
        y = torch.stack(outs, dim=1)  # (B, L, E)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward scan
        y_f = self._scan(x)
        # Backward scan
        y_b = torch.flip(self._scan(torch.flip(x, dims=[1])), dims=[1])
        return y_f + y_b


class HARMambaBlock(nn.Module):
    """One HARMamba block (Algorithm 1).

    Args:
        d_model: token dim D
        d_ssm: SSM working dim E
        conv_kernel: causal conv kernel size before SSM
    """

    def __init__(self, d_model: int, d_ssm: int, conv_kernel: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.to_x = nn.Linear(d_model, d_ssm)
        self.to_z = nn.Linear(d_model, d_ssm)
        self.ssm = DiagSelectiveSSM(d_model=d_ssm, d_state=d_ssm, conv_kernel=conv_kernel)
        self.out = nn.Linear(d_ssm, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x_norm = self.norm(x)
        x_in = self.to_x(x_norm)
        z = self.to_z(x_norm)
        g = self.act(z)
        y = self.ssm(x_in) * g
        return self.out(y)


class HARMambaEncoder(nn.Module):
    def __init__(self, depth: int, d_model: int, d_ssm: int, conv_kernel: int = 3, drop_path: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [HARMambaBlock(d_model=d_model, d_ssm=d_ssm, conv_kernel=conv_kernel) for _ in range(depth)]
        )
        self.stochastic_depth = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            # why: residual keeps information highway; paper’s Fig. 1 implies block returns same shape
            x = x + blk(x)
        return x


@dataclass
class HARMambaConfig:
    num_channels: int
    num_classes: int
    seq_len: int
    patch_size: int = 16
    patch_stride: Optional[int] = None
    d_model: int = 128  # token dim D
    d_ssm: int = 128    # SSM dim E
    depth: int = 12
    conv_kernel: int = 3
    cls_token: bool = True


class HARMamba(nn.Module):
    """HARMamba architecture.

    Input: (B, C, L) raw sensor sequences (e.g., IMU channels)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_channels,
                    num_classes,
                    patch_size=32,
                    patch_stride=32,
                    d_model=128,
                    d_ssm=128,
                    depth=2,
                    conv_kernel=3,
                    cls_token=True):
        super().__init__()
    
        self.revin = RevIN(num_channels, affine=True)
        self.patch = PatchEmbed1D(patch_size, patch_stride, d_model)

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.patch_size=patch_size
        self.patch_stride=patch_stride
        self.d_model=d_model
        self.d_ssm=d_ssm
        self.depth=depth
        self.conv_kernel=conv_kernel
        self.cls_token=cls_token

        # Class token & positional embeddings are over total token length across all channels (+1 if CLS)
        # We defer pos embedding init to first forward when length is known.
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) if cls_token else None
        self.pos: Optional[nn.Parameter] = None  # lazily initialized

        self.encoder = HARMambaEncoder(depth=depth, d_model=d_model, d_ssm=d_ssm, conv_kernel=conv_kernel)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, num_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif m is self.cls:
            nn.init.trunc_normal_(m, std=0.02)

    def _build_pos(self, total_tokens: int, device: torch.device, dtype: torch.dtype):
        if self.pos is None or self.pos.shape[1] != total_tokens:
            pos = nn.Parameter(torch.zeros(1, total_tokens, self.d_model, device=device, dtype=dtype))
            nn.init.trunc_normal_(pos, std=0.02)
            self.pos = pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        """
        x = x.transpose(1,2)
        B, C, L = x.shape
        assert C == self.num_channels, f"expected {self.num_channels} channels, got {C}"

        # Channel-independent RevIN
        x = self.revin(x)

        # Patch per channel → (B, C, N, D)
        tokens_per_channel, N = self.patch(x)

        # Concatenate channels along sequence dim
        tokens = tokens_per_channel.reshape(B, C * N, self.d_model)  # (B, L_tokens, D)

        # Append CLS and add pos emb
        if self.cls_token:
            cls = self.cls.expand(B, -1, -1)  # (B,1,D)
            tokens = torch.cat([cls, tokens], dim=1)
        total_tokens = tokens.shape[1]
        self._build_pos(total_tokens, tokens.device, tokens.dtype)
        tokens = tokens + self.pos

        # Encoder
        tokens = self.encoder(tokens)

        # Classification on CLS (or mean if no CLS)
        if self.cls_token:
            feat = tokens[:, 0]
        else:
            feat = tokens.mean(dim=1)
        feat = self.final_norm(feat)
        logits = self.head(feat)
        return logits
