import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Hardtanh(0, 20)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RNNGRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.bn = nn.BatchNorm1d(2 * hidden_size)

    def forward(self, x):
        # x: (B, L, input_size)
        x, _ = self.rnn(x)
        # x: (B, L, 2 * hidden_size)
        x = x.transpose(1, 2)  # bn takes (B, H, L)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x
