import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCNEncoder(nn.Module):
    def __init__(
        self,
        n_in_channels,
        out_channels,
        hidden_channels=64,
        depth=1,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        self.input_fc = nn.Linear(n_in_channels, hidden_channels)

        layers = []
        for i in range(depth):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.output_fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        """
        x: (B, T, C)
        return: (B, T, out_channels)
        """
        x = x.float()
        x = self.input_fc(x)
        x = x.transpose(1, 2)         # (B, H, T)
        x = self.tcn(x)               # (B, H, T)
        x = x.transpose(1, 2)
        x = self.output_fc(x)
        x = x.transpose(1, 2)
        x = self.gap(x)             # (B, C, 1)
        x = x.squeeze(-1)

        return x
