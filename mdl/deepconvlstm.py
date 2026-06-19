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
    
class DeepConvLSTM(nn.Module):
    """
    Input:  (B, T, F)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        conv_channels=(64, 128, 128, 256),
        lstm_hidden=256,
        lstm_layers=1,
        bidirectional=True,
        dropout=0.3,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Temporal CNN stack
        convs = []
        c_in = in_features
        for c_out in conv_channels:
            convs.append(TemporalConvBlock(c_in, c_out, k=5, p=2, dropout=dropout))
            c_in = c_out
        self.conv_stack = nn.Sequential(*convs)

        # LSTM over time
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)

        # Head
        self.norm = nn.LayerNorm(lstm_out)
        self.fc = nn.Linear(lstm_out, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)            # (B, F, T)
        x = self.conv_stack(x)           # (B, C, T)
        x = x.transpose(1, 2)            # (B, T, C)

        # LSTM
        x, _ = self.lstm(x)              # (B, T, H)
        # Temporal pooling (mean + max works well; here mean for simplicity)
        x = x.mean(dim=1)                # (B, H)

        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)