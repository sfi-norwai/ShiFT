import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """1D Conv -> BN -> GELU with residual."""
    def __init__(self, in_ch, out_ch, k=5, p=2, d=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, dilation=d)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        # x: (B, C, T)
        y = self.drop(self.bn(self.conv(x)))
        y = F.gelu(y)
        return y + self.proj(x)
    
class PositionalEncoding(nn.Module):
    """Sinusoidal PE for Transformer."""
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)
    
class CNNTransformer(nn.Module):
    """
    Input:  (B, T, F)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        proj_dim=256,
        conv_channels=(128, 128),
        n_heads=4,
        n_layers=3,
        dim_ff=512,
        dropout=0.2,
        use_cls_token=True,
    ):
        super().__init__()
        self.use_cls = use_cls_token
        self.dropout = nn.Dropout(dropout)

        # Light temporal conv front-end (improves locality before attention)
        convs = []
        c_in = in_features
        for c_out in conv_channels:
            convs.append(TemporalConvBlock(c_in, c_out, k=5, p=2, dropout=dropout))
            c_in = c_out
        self.conv_stack = nn.Sequential(*convs)

        # Project to Transformer d_model
        self.proj = nn.Linear(conv_channels[-1], proj_dim)

        # Optional CLS token
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, proj_dim))
            nn.init.trunc_normal_(self.cls, std=0.02)

        self.pe = PositionalEncoding(proj_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(proj_dim)
        self.head = nn.Linear(proj_dim, num_classes)

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, F)
        x = x.transpose(1, 2)                # (B, F, T)
        x = self.conv_stack(x)               # (B, C, T)
        x = x.transpose(1, 2)                # (B, T, C)

        x = self.proj(x)                     # (B, T, D)

        if self.use_cls:
            B = x.size(0)
            cls = self.cls.expand(B, -1, -1) # (B, 1, D)
            x = torch.cat([cls, x], dim=1)   # prepend CLS
            if key_padding_mask is not None:
                # pad an extra False for CLS
                false_col = torch.zeros((key_padding_mask.size(0), 1), dtype=torch.bool, device=key_padding_mask.device)
                key_padding_mask = torch.cat([false_col, key_padding_mask], dim=1)

        x = self.pe(x)
        # key_padding_mask: (B, T) True=pad (optional)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if self.use_cls:
            pooled = x[:, 0]                 # CLS
        else:
            pooled = x.mean(dim=1)           # Mean over time

        pooled = self.dropout(pooled)
        return self.head(pooled)