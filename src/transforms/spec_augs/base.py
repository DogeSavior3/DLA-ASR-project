import torch
from torch import Tensor, nn


class BaseSpecAug(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, spec):
        if torch.rand(()).item() < self.p:
            return self.augment(spec)
        return spec

    def augment(self, spec):
        pass
