import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
    def __init__(self, input_size, hidden_size, normalization="batchnorm"):
        super().__init__()
        self.normalization = normalization
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        if normalization == "batchnorm":
            self.bn = nn.BatchNorm1d(2 * hidden_size)
        elif normalization == "layernorm":
            self.ln = nn.LayerNorm(2 * hidden_size)

    def forward(self, x, lengths):
        # x: (B, L, input_size)
        # total_length = x.shape[1]
        # x = pack_padded_sequence(
        #     x, lengths.cpu(), batch_first=True, enforce_sorted=False
        # )
        x, _ = self.rnn(x)
        # x, _ = pad_packed_sequence(x, batch_first=True, total_length=total_length)
        # x: (B, L, 2 * hidden_size)
        if self.normalization == "batchnorm":
            x = x.transpose(1, 2)  # bn takes (B, H, L)
            x = self.bn(x)
            x = x.transpose(1, 2)
        else:
            x = self.ln(x)
        return x
