import torch
from torch import Tensor, nn


class BaseWaveAug(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, wav):
        if torch.rand(()).item() < self.p:
            return self.augment(wav)
        return wav

    def augment(self, wav):
        pass
