import torch
from torch import Tensor, nn


class BaseWaveAug(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0 and 1")
        self.p = p

    def forward(self, wav: Tensor) -> Tensor:
        if torch.rand(()).item() < self.p:
            return self.augment(wav)
        return wav

    def augment(self, wav: Tensor) -> Tensor:
        """Return an augmented waveform without modifying the input in place."""
        raise NotImplementedError
